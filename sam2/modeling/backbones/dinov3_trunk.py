"""DINOv3 (SurgeNetXL) trunk for BlooDet, a drop-in replacement of the Hiera trunk.

The SAM 2 ImageEncoder expects ``trunk(x)`` to return a list of 4 feature maps ordered
fine -> coarse (strides 4, 8, 16, 32) and ``trunk.channel_list`` to list their channels
coarse -> fine (same convention as Hiera, e.g. [896, 448, 224, 112] for B+).

A ViT only has stride 16, so the pyramid is rebuilt ViTDet/DPT-style from intermediate
blocks (the best dense features of a ViT are not only at its output):
    stride  4 <- block `layer_indices[0]`, 2x ConvTranspose (x4)
    stride  8 <- block `layer_indices[1]`, 1x ConvTranspose (x2)
    stride 16 <- block `layer_indices[2]` (last block), 1x1 conv
    stride 32 <- stride-16 map, max-pool (dropped later by ImageEncoder scalp=1)
At 512 px with patch 16 the stride-16 grid is 32x32, i.e. exactly the main feature size
of Hiera-B+ in BlooDet, so nothing downstream (memory, decoders, point branch) changes.

The ViT weights come from SurgeNetDINO (DINOv3 ViT-B/16 pretrained on SurgeNetXL, 4.7M
surgical frames; github.com/rlpddejong/SurgeNetDINO, CC-BY-NC). That checkpoint uses the
official DINOv3 layout; it is converted to timm's ``vit_base_patch16_dinov3_qkvb``:
  * no register tokens in the checkpoint -> ``num_reg_tokens=0``;
  * fused qkv bias -> q_bias / v_bias. The key bias is dropped: it adds the same constant
    to every score of a query, so the softmax is unchanged (the conversion is exact);
  * RoPE periods are copied from the checkpoint (stored in bf16).

The pyramid layers (``pyramid.*``) are new and randomly initialised.
Freezing / unfreezing of the ViT blocks is decided by the training script through
``requires_grad``; if no ViT parameter requires grad the ViT runs under no_grad.
"""

from typing import List, Sequence

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def convert_dinov3_checkpoint(sd: dict, embed_dim: int) -> dict:
    """Official DINOv3 state_dict -> timm ``vit_*_patch16_dinov3_qkvb`` state_dict."""
    if "teacher" in sd and isinstance(sd["teacher"], dict):
        sd = sd["teacher"]
    out = {}
    for k, v in sd.items():
        if k in ("mask_token", "rope_embed.periods"):
            continue
        if k.endswith("attn.qkv.bias"):
            p = k[: -len("qkv.bias")]
            out[p + "q_bias"] = v[:embed_dim]
            out[p + "v_bias"] = v[2 * embed_dim:]
        else:
            out[k] = v
    return out


class DinoV3Trunk(nn.Module):
    # build_sam._load_checkpoint must not load SAM 2 Hiera weights into this encoder
    skip_sam2_ckpt = True

    def __init__(
        self,
        weights_path: str,
        model_name: str = "vit_base_patch16_dinov3_qkvb",
        img_size: int = 512,
        layer_indices: Sequence[int] = (3, 6, 11),
        pyramid_dims: Sequence[int] = (192, 384, 768, 768),  # strides 4, 8, 16, 32
    ):
        super().__init__()
        import timm  # local import: only needed for this encoder

        self.vit = timm.create_model(
            model_name, pretrained=False, img_size=img_size, num_classes=0, num_reg_tokens=0
        )
        d = self.vit.embed_dim
        self.layer_indices = list(layer_indices)
        assert len(self.layer_indices) == 3

        if weights_path:
            raw = torch.load(weights_path, map_location="cpu", weights_only=False)
            if "teacher" in raw and isinstance(raw["teacher"], dict):
                raw = raw["teacher"]
            self.vit.load_state_dict(convert_dinov3_checkpoint(raw, d), strict=True)
            periods = raw.get("rope_embed.periods", None)
            if periods is not None and hasattr(self.vit.rope, "periods"):
                with torch.no_grad():
                    self.vit.rope.periods.copy_(periods.float())
            print(f"[DinoV3Trunk] loaded {weights_path} into {model_name} (strict)")

        c4, c8, c16, c32 = pyramid_dims
        self.pyramid = nn.ModuleDict({
            "s4": nn.Sequential(
                nn.ConvTranspose2d(d, d // 2, 2, 2), LayerNorm2d(d // 2), nn.GELU(),
                nn.ConvTranspose2d(d // 2, c4, 2, 2), LayerNorm2d(c4)),
            "s8": nn.Sequential(nn.ConvTranspose2d(d, c8, 2, 2), LayerNorm2d(c8)),
            "s16": nn.Sequential(nn.Conv2d(d, c16, 1), LayerNorm2d(c16)),
            "s32": nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(c16, c32, 1), LayerNorm2d(c32)),
        })
        # Hiera convention: coarse -> fine
        self.channel_list = [c32, c16, c8, c4]

    def _vit_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.vit.forward_intermediates(
            x, indices=self.layer_indices, output_fmt="NCHW", intermediates_only=True
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if any(p.requires_grad for p in self.vit.parameters()):
            f_s4, f_s8, f_s16 = self._vit_features(x)
        else:
            with torch.no_grad():
                f_s4, f_s8, f_s16 = self._vit_features(x)
        s16 = self.pyramid["s16"](f_s16)
        return [
            self.pyramid["s4"](f_s4),
            self.pyramid["s8"](f_s8),
            s16,
            self.pyramid["s32"](s16),
        ]
