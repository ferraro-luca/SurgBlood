"""v8: bleeding-point heatmap utilities (shared by sam2_base, training and evaluation).

The point decoder (a copy of SAM's mask decoder) already outputs a 128x128 map that the
original BlooDet discards. v8 trains it as a heatmap of the bleeding point:
  * target  : Gaussian centred on the GT point (peak = 1), all zeros if there is no point;
  * loss    : penalty-reduced focal loss (CenterNet), the standard loss for point heatmaps;
  * decoding: argmax + local soft-argmax (7x7 window) -> sub-cell (x, y) in [0, 1].
A global soft-argmax is NOT used: thousands of low-probability background cells would
pull the estimate towards the image centre.

Point convention (same as the dataset and the MLP head 'iou'): (x, y) normalised to [0, 1],
x = column, y = row; (0, 0) = no point.

Prior bias: the map has never been trained, so at the start it predicts ~50% everywhere and
the focal loss (summed over 128x128 cells, normalised by the few peaks) would be in the
thousands. As in CenterNet, a fixed negative bias is added to the logits (prior p ~ 1%):
the loss starts at normal values. The same bias is used for decoding, so the peak position
is unchanged; it only matters for the fusion weight (peak probability).
"""

import torch
import torch.nn.functional as F

HEATMAP_LOGIT_BIAS = -4.6  # sigmoid(-4.6) ~ 0.01


def gaussian_heatmap_target(points: torch.Tensor, size: int = 128, sigma: float = 2.0) -> torch.Tensor:
    """points: [N, 2] normalised (x, y); (0, 0) = no point. Returns [N, 1, size, size] in [0, 1]."""
    n = points.shape[0]
    device = points.device
    ys = torch.arange(size, device=device, dtype=torch.float32).view(1, size, 1)
    xs = torch.arange(size, device=device, dtype=torch.float32).view(1, 1, size)
    # Centre of the peak on the nearest cell, so that exactly one cell is 1.0
    cx = (points[:, 0] * size - 0.5).round().clamp(0, size - 1).view(n, 1, 1)
    cy = (points[:, 1] * size - 0.5).round().clamp(0, size - 1).view(n, 1, 1)
    g = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
    has_point = ~(points == 0).all(dim=1)
    g = g * has_point.view(n, 1, 1).float()
    return g.unsqueeze(1)


def heatmap_focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 2.0,
                       beta: float = 4.0, eps: float = 1e-6) -> torch.Tensor:
    """CenterNet penalty-reduced focal loss. logits/target: [N, 1, H, W]. Normalised by #peaks."""
    p = torch.sigmoid(logits.float() + HEATMAP_LOGIT_BIAS).clamp(eps, 1 - eps)
    pos = target.eq(1).float()
    neg = 1 - pos
    pos_loss = -((1 - p) ** alpha) * torch.log(p) * pos
    neg_loss = -((1 - target) ** beta) * (p ** alpha) * torch.log(1 - p) * neg
    num_pos = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos


def decode_heatmap(logits: torch.Tensor, window: int = 7):
    """logits: [N, 1, H, W] (or [N, H, W]). Returns (points [N, 2] normalised (x, y), peak [N] in [0, 1])."""
    if logits.dim() == 3:
        logits = logits.unsqueeze(1)
    prob = torch.sigmoid(logits[:, :1].float() + HEATMAP_LOGIT_BIAS)
    n, _, h, w = prob.shape
    flat = prob.view(n, -1)
    peak, idx = flat.max(dim=1)
    iy = (idx // w).float()
    ix = (idx % w).float()
    # local soft-argmax around the peak
    r = window // 2
    offs = torch.arange(-r, r + 1, device=prob.device, dtype=torch.float32)
    k = offs.numel()
    yy = (iy.view(n, 1, 1) + offs.view(1, -1, 1)).clamp(0, h - 1).expand(n, k, k)
    xx = (ix.view(n, 1, 1) + offs.view(1, 1, -1)).clamp(0, w - 1).expand(n, k, k)
    lin = (yy * w + xx).long().view(n, -1)
    wts = flat.gather(1, lin).clamp(min=0)
    wts = wts / wts.sum(dim=1, keepdim=True).clamp(min=1e-8)
    cy = (wts * yy.reshape(n, -1)).sum(dim=1)
    cx = (wts * xx.reshape(n, -1)).sum(dim=1)
    pts = torch.stack([(cx + 0.5) / w, (cy + 0.5) / h], dim=1)
    return pts, peak


def fuse_points(mlp_points: torch.Tensor, heat_points: torch.Tensor, peak: torch.Tensor) -> torch.Tensor:
    """Confidence-weighted fusion: the sharper the heatmap peak, the more it counts."""
    w = peak.clamp(0, 1).view(-1, 1)
    return w * heat_points + (1 - w) * mlp_points
