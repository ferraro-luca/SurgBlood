"""BlooDet with a DINOv3 encoder pretrained on surgical data (SurgeNetXL) instead of Hiera-B+.

Experiments D1 / D2 (same recipe as "v7 from baseline", only the image encoder changes):
  * encoder: DINOv3 ViT-B/16 SurgeNetXL + a new multi-scale pyramid/FPN neck
    (sam2/modeling/backbones/dinov3_trunk.py, config sam2_dinov3_b_surg.yaml);
  * every non-encoder module (memories, mask/edge decoders, point branch) is initialised from
    BlooDet-Pretrain, exactly like v7-from-baseline -> the comparison isolates the encoder;
  * phase 1 (--warmup_epochs): ONLY the new pyramid + neck are trained (everything else
    frozen), so the heads are not disrupted by random neck features;
  * phase 2: neck + mask decoder + edge generator + point branches (v7 LRs), plus the last
    --unfreeze_blocks ViT blocks at --vit_lr (0 = D1, frozen encoder; e.g. 3 = D2).
    Pyramid LR schedule over the phase-2 epochs;
  * data, --balance, losses, gate OFF, validation on both domains, best ckpt = mean Dice: as v7.

model_config.json (written next to the weights) stores model_cfg + gate, so Evaluate2.py /
Evaluate3.py rebuild the right architecture automatically.
"""
import os
import datetime
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from tensorboardX import SummaryWriter
import torch.optim as optim
import wandb

from sam2_video_baseline import SAM2_Video_baseline as network
from sam2_video_baseline import save_model_config
from datasetVideoTrain import CholeBPTrain
from datasetVideoTest import CholeBPTest
from metrics.maskIouFloder import SegmentationMetric


# Define focal loss
def focal_loss(logits, mask, alpha=0.75, gamma=2.0):
    mask = F.interpolate(mask, size=logits.size()[2:], mode='bilinear')
    mask = mask.float()
    bce_loss = F.binary_cross_entropy_with_logits(logits, mask, reduction='none')
    probas = torch.sigmoid(logits)
    pt = mask * probas + (1 - mask) * (1 - probas)
    focal_weight = (1 - pt) ** gamma
    alpha_weight = mask * alpha + (1 - mask) * (1 - alpha)
    loss_focal = focal_weight * alpha_weight * bce_loss
    return loss_focal.mean()


# Define Dice Loss
def dice_loss(logits, mask, smooth=1e-6):
    mask = F.interpolate(mask, size=logits.size()[2:], mode='bilinear').float()
    logits = torch.sigmoid(logits)
    intersection = (logits * mask).sum(dim=(2, 3))
    union = logits.sum(dim=(2, 3)) + mask.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--save_path', default="/work/cvcs2026/blood_overflow/weights/Hemo_Cholec_SurgBlood_dino_D1", type=str)
    parser.add_argument('--model_cfg', default="sam2_dinov3_b_surg.yaml", type=str,
                        help='SAM2 config with the DINOv3 SurgeNetXL trunk')
    parser.add_argument('--unfreeze_blocks', type=int, default=0,
                        help='last N ViT blocks trained in phase 2 (0 = D1 frozen encoder, 3 = D2)')
    parser.add_argument('--vit_lr', type=float, default=2e-6, help='lr of the unfrozen ViT blocks')
    parser.add_argument('--neck_lr', type=float, default=1e-4, help='lr of the new pyramid + FPN neck')
    parser.add_argument('--warmup_epochs', type=int, default=2,
                        help='phase 1: epochs where ONLY pyramid + neck are trained')
    parser.add_argument('--init_ckpt', type=str,
                        default="/work/cvcs2026/blood_overflow/weights/BlooDet-Pretrain",
                        help='BlooDet weights for every NON-encoder module (as v7 from baseline)')
    parser.add_argument('--point_gate', choices=['off', 'on'], default='off',
                        help="off = v7 (interruttore punto->maschera spento); "
                             "on = esperimento di controllo 'v6 + 20 epoche' (interruttore acceso, come v1-v6)")
    parser.add_argument('--data_path', default="/work/cvcs2026/blood_overflow/03_processed/Hemo_Cholec_splitted/", type=str,
                        help='Hemo_Cholec root (train/ used for training)')
    parser.add_argument('--test_data_path', default="/work/cvcs2026/blood_overflow/03_processed/Hemo_Cholec_splitted/", type=str,
                        help='Hemo_Cholec root for validation (test/ split, as in v3/v5)')
    parser.add_argument('--surgblood_path', default="/work/cvcs2026/blood_overflow/03_processed/SurgBlood_splitted/", type=str,
                        help='SurgBlood root prepared by src/datasets/surgblood_prepare.py (train/ and test/ = validation split)')
    parser.add_argument('--max_epochs', type=int, default=25, help='total epochs (warmup + phase 2)')
    parser.add_argument('--base_lr', type=float, default=6e-5, help='mask decoder / edge learning rate')
    parser.add_argument('--point_lr', type=float, default=6e-5, help='lr for bleeding-point branches')
    parser.add_argument('--encoder_lr', type=float, default=0.0, help='(unused here, see --vit_lr / --neck_lr)')
    parser.add_argument('--memory_lr', type=float, default=0.0, help='lr for SAM2 memory_attention + memory_encoder (0 = frozen)')
    parser.add_argument('--edge_weight', type=float, default=0.5, help='weight for edge loss')
    parser.add_argument('--exist_weight', type=float, default=1.0, help='weight for point-existence loss (SurgBlood only)')
    parser.add_argument('--point_weight', type=float, default=1.0, help='weight for point-coordinate loss (SurgBlood only)')
    parser.add_argument('--no_augment', action='store_true', help='Disable clip-consistent photometric augmentation')
    parser.add_argument('--balance', action='store_true',
                        help='Sample 50/50 per dataset (with replacement). Default: every training clip of both datasets once per epoch')
    parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')
    parser.add_argument('--run_name', type=str, default="dino_D1(dinov3_b_surgenetxl_frozen)", help='nome della run su W&B')

    parser.add_argument('--debug_steps', type=int, default=0,
                        help='>0: limit train steps per epoch and validation videos (smoke test only)')

    parser.add_argument('--warmup', action='store_true', help='(unused, kept for compatibility with training.sh)')
    parser.add_argument('--AdamW', action='store_true', help='(unused, kept for compatibility with training.sh)')
    parser.add_argument('--snapshot', type=str, default=None, help='(unused)')
    return parser.parse_args()


# Bleeding-point branches (see sam2/modeling/sam2_base.py). 'estimator' (PWC-Net optical
# flow) and the prompt encoders stay frozen.
POINT_KEYS = ['sam_point_decoder', 'point_memory_attention', 'offset_x_proj', 'offset_y_proj']
NECK_KEYS = ['image_encoder.trunk.pyramid', 'image_encoder.neck']
# Leading dot avoids matching 'point_memory_attention'
MEMORY_KEYS = ['.memory_attention.', '.memory_encoder.']


def vit_keys(n_blocks, depth=12):
    return [f'image_encoder.trunk.vit.blocks.{i}.' for i in range(depth - n_blocks, depth)]


def build_param_groups(net, args, phase):
    """phase 'warmup': only the new pyramid + neck. phase 'main': neck + heads (+ last ViT blocks)."""
    for param in net.parameters():
        param.requires_grad = False
    depth = len(net.predictor.image_encoder.trunk.vit.blocks)
    vkeys = vit_keys(args.unfreeze_blocks, depth) if args.unfreeze_blocks > 0 else []

    groups = {'neck': [], 'decoder': [], 'point': [], 'vit': [], 'memory': []}
    for name, param in net.named_parameters():
        if 'estimator' in name:
            continue
        if any(k in name for k in NECK_KEYS):
            groups['neck'].append(param)
        elif phase == 'warmup':
            continue
        elif any(k in name for k in POINT_KEYS):
            groups['point'].append(param)
        elif 'mask_decoder' in name or 'edge_generator' in name:
            groups['decoder'].append(param)
        elif vkeys and any(k in name for k in vkeys):
            groups['vit'].append(param)
        elif args.memory_lr > 0 and any(k in name for k in MEMORY_KEYS):
            groups['memory'].append(param)

    lrs = {'neck': args.neck_lr, 'decoder': args.base_lr, 'point': args.point_lr,
           'vit': args.vit_lr, 'memory': args.memory_lr}
    names, param_groups = [], []
    for g, params in groups.items():
        if not params:
            continue
        for p in params:
            p.requires_grad = True
        names.append(g)
        param_groups.append({'params': params, 'lr': lrs[g]})
        print(f"[{phase}] Parametri sbloccati {g:<8}: {sum(p.numel() for p in params):>12,}  (lr {lrs[g]:.1e})")

    required = ('neck',) if phase == 'warmup' else ('neck', 'decoder', 'point') + (('vit',) if vkeys else ())
    for g in required:
        assert groups[g], f"ERRORE: nessun parametro sbloccato per il gruppo '{g}'. Controlla i nomi dei layer."
    return names, param_groups, [lrs[g] for g in names]


def evaluate(net, loader, with_points, max_videos=0):
    metric = SegmentationMetric(2)
    frames = exist_tp = point_total = pck2 = pck5 = pck10 = 0
    empty_frames = empty_fp = 0   # GT without blood / ... where the model still predicts blood

    with torch.no_grad():
        for v_idx, sample in enumerate(loader):
            if max_videos and v_idx >= max_videos:
                break
            images = sample['images'].cuda().view(-1, 3, 512, 512)
            masks = sample['masks'].squeeze(0).cpu()
            points = sample['points'].squeeze(0).cpu()
            video_segments = net(images)
            frames += images.size(0)

            for i in range(images.size(0)):
                out = video_segments[i][1]
                sam_pre = torch.sigmoid(out['mask_logits'].unsqueeze(1))
                sam_pre = F.interpolate(sam_pre, size=(512, 512), mode='bilinear', align_corners=False).squeeze(1)
                pred_bin, gt_bin = (sam_pre > 0.5).int(), (masks[i] > 0.5).int()
                metric.addBatch(pred_bin, gt_bin, [255])
                if gt_bin.sum() == 0:
                    empty_frames += 1
                    empty_fp += int(pred_bin.sum() > 0)

                if not with_points:
                    continue
                existed_label = (points[i][0] != 0 and points[i][1] != 0)
                existed_pred = (torch.sigmoid(out['object_score']).cpu().numpy() > 0.5).astype(int)
                exist_tp += (existed_pred == existed_label.cpu().numpy()).all()
                if existed_label:
                    point_total += 1
                    dist = torch.sqrt(torch.sum((points[i] * 512 - out['iou'].cpu() * 512) ** 2, dim=-1))
                    pck2 += int(dist < 512 * 0.02)
                    pck5 += int(dist < 512 * 0.05)
                    pck10 += int(dist < 512 * 0.1)

    iou, dice = metric.IntersectionOverUnion(), metric.DiceCoefficient()
    res = {'IoU_blood': float(iou[1]), 'Dice_blood': float(dice[1]),
           'empty_frames': empty_frames,
           'empty_fp_rate': empty_fp / empty_frames if empty_frames else 0.0}
    if with_points:
        res.update({
            'precision_exist': exist_tp / frames if frames else 0.0,
            'PCK2': pck2 / point_total if point_total else 0.0,
            'PCK5': pck5 / point_total if point_total else 0.0,
            'PCK10': pck10 / point_total if point_total else 0.0,
        })
    return res


def train(Network):
    args = parser()
    assert args.batch_size == 1, "Il codice assume una clip per batch (come v3/v5)."
    os.makedirs(args.save_path, exist_ok=True)

    run = wandb.init(
        entity="mgualtieri01-unimore",
        project="BlooDet",
        name=args.run_name,
        job_type="fine-tuning",
        tags=["fine-tuning", "Hemo_Cholec", "SurgBlood", "joint", "point_branches", "point_gate_" + args.point_gate, "dinov3_surgenetxl", f"unfreeze_{args.unfreeze_blocks}"],
        notes="DINOv3 SurgeNetXL encoder + BlooDet heads from BlooDet-Pretrain (v7-from-baseline recipe).",
        config=vars(args) | {
            "architecture": "BlooDet-SAM2",
            "frames_length": 8,
            "overlap": 7,
            "clip_augmentation": not args.no_augment,
            "balanced_sampling": args.balance,
            "selection_metric": "mean(Dice Hemo_Cholec val, Dice SurgBlood val)",
            "point_gates_mask": args.point_gate == 'on',
        }
    )

    augment = not args.no_augment
    hemo_train = CholeBPTrain(args.data_path, args.img_size, mode='train', frames_length=8, overlap=7,
                              augment=augment, has_point_labels=False)
    surg_train = CholeBPTrain(args.surgblood_path, args.img_size, mode='train', frames_length=8, overlap=7,
                              augment=augment, has_point_labels=True)
    train_dataset = ConcatDataset([hemo_train, surg_train])
    print(f"Clip Hemo_Cholec: {len(hemo_train)} | Clip SurgBlood: {len(surg_train)} | Totale: {len(train_dataset)}")

    if args.balance:
        # Optional: each dataset contributes ~50% of the clips seen in an epoch.
        # Epoch length = 2 x Hemo_Cholec clips: each Hemo_Cholec clip is seen ~once per epoch
        # (as in v5), plus the same number of randomly drawn SurgBlood clips.
        weights = [0.5 / len(hemo_train)] * len(hemo_train) + [0.5 / len(surg_train)] * len(surg_train)
        sampler = WeightedRandomSampler(weights, num_samples=2 * len(hemo_train), replacement=True)
        print(f"Campionamento bilanciato: {2 * len(hemo_train)} clip per epoca (50% Hemo_Cholec, 50% SurgBlood)")
        trainloader = DataLoader(train_dataset, batch_size=1, sampler=sampler, num_workers=4, pin_memory=True)
    else:
        # Default: all training clips of both datasets, each once per epoch, shuffled together
        trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    val_loaders = {
        'hemo': DataLoader(CholeBPTest(args.test_data_path, mode='test'), batch_size=1, shuffle=False, num_workers=0),
        'surg': DataLoader(CholeBPTest(args.surgblood_path, mode='test'), batch_size=1, shuffle=False, num_workers=0),
    }

    gate_on = (args.point_gate == 'on')
    net = Network(point_gates_mask=gate_on, model_cfg=args.model_cfg)
    init_sd = torch.load(args.init_ckpt, map_location='cpu')
    init_sd = {k: v for k, v in init_sd.items() if '.image_encoder.' not in k}  # Hiera encoder: not used
    res = net.load_state_dict(init_sd, strict=False)
    bad_missing = [k for k in res.missing_keys if '.image_encoder.' not in k]
    assert not bad_missing and not res.unexpected_keys, (bad_missing[:5], res.unexpected_keys[:5])
    net.cuda()
    print(f"Encoder: {args.model_cfg} | moduli non-encoder da: {args.init_ckpt}")
    print(f"Point-existence gate on the blood mask: {'ON (controllo v6 + 20 epoche)' if gate_on else 'OFF (v7)'}")
    # The gate option is not in the state_dict: store it next to the checkpoints
    save_model_config(args.save_path, point_gates_mask=gate_on, init_ckpt=args.init_ckpt,
                      model_cfg=args.model_cfg, unfreeze_blocks=args.unfreeze_blocks,
                      warmup_epochs=args.warmup_epochs,
                      version=f"dino_D{1 if args.unfreeze_blocks == 0 else 2}")

    assert args.max_epochs > args.warmup_epochs >= 0
    phase = 'warmup' if args.warmup_epochs > 0 else 'main'
    group_names, param_groups, group_base_lrs = build_param_groups(net, args, phase)
    optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999), weight_decay=0.1)

    sw = SummaryWriter(args.save_path)
    global_step = 0
    best_score = 0.0

    for epoch in range(args.max_epochs):
        net.train(True)

        if phase == 'warmup' and epoch == args.warmup_epochs:
            phase = 'main'
            print(f"===== Fine warm-up del neck: inizio fase 2 (epoca {epoch + 1}) =====")
            group_names, param_groups, group_base_lrs = build_param_groups(net, args, phase)
            optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999), weight_decay=0.1)

        if phase == 'warmup':
            lr_scale = 1.0  # neck only, constant lr
        else:
            # Scheduler piramidale sulle epoche della fase 2, proporzionale per tutti i gruppi
            n_main = args.max_epochs - args.warmup_epochs
            e = epoch - args.warmup_epochs
            lr_scale = 1 - abs((e + 1) / (n_main + 1) * 2 - 1)
        for group, group_lr in zip(optimizer.param_groups, group_base_lrs):
            group['lr'] = lr_scale * group_lr

        for step, sample in enumerate(trainloader):
            if args.debug_steps and step >= args.debug_steps:
                break
            image, mask, edge, point = sample['image'], sample['mask'], sample['edge'], sample['point']
            point_labeled = bool(sample['point_labeled'].item())

            images = image.float().cuda().view(-1, 3, args.img_size, args.img_size)
            points = point.float().cuda().view(-1, 2)
            frame_length = images.size(0)

            video_segments = net(images)

            pre_masks, pre_edges, pre_point, pre_exist = [], [], [], []
            for i in range(frame_length):
                out = video_segments[i][1]
                pre_masks.append(out['mask_logits'])
                pre_point.append(out['iou'])
                pre_exist.append(out['object_score'])
                pre_edges.append(out['edge_logits'])

            mask_pred = torch.stack(pre_masks, dim=0)
            edge_pred = torch.stack(pre_edges, dim=0)
            pre_point = torch.stack(pre_point, dim=0)
            existed_score_logits = torch.stack(pre_exist, dim=0)

            mask = mask.squeeze(0).float().cuda()
            edge = edge.squeeze(0).float().cuda()

            # Maschera + edge: su tutte le clip (Focal + Dice, come v3/v5)
            lossmask = focal_loss(mask_pred, mask) + dice_loss(mask_pred, mask)
            lossmask_edge = focal_loss(edge_pred, edge) + dice_loss(edge_pred, edge)
            loss = lossmask + args.edge_weight * lossmask_edge

            # Punto: solo sulle clip con punti annotati (SurgBlood)
            if point_labeled:
                existed_label = (points != 0).any(dim=1).float()
                loss_exist = F.binary_cross_entropy_with_logits(existed_score_logits, existed_label.unsqueeze(1))
                loss_point = torch.tensor(0.0, device=points.device)
                for i in range(points.size(0)):
                    if not torch.all(points[i] == 0):
                        loss_point = loss_point + F.smooth_l1_loss(pre_point[i].unsqueeze(0), points[i].unsqueeze(0))
                loss = loss + args.exist_weight * loss_exist + args.point_weight * loss_point

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            log = {
                "train/total_loss": loss.item(),
                "train/mask_loss": lossmask.item(),
                "train/edge_loss": lossmask_edge.item(),
                "train/is_surgblood": float(point_labeled),
            }
            if point_labeled:
                log["train/exist_loss"] = loss_exist.item()
                log["train/point_loss"] = loss_point.item()
            for name, group in zip(group_names, optimizer.param_groups):
                log[f"train/lr_{name}"] = group['lr']
            run.log(log, step=global_step)
            for k, v in log.items():
                sw.add_scalar(k, v, global_step=global_step)

            if step % 200 == 0:
                lr_str = ', '.join(f'{n}={g["lr"]:.2e}' for n, g in zip(group_names, optimizer.param_groups))
                pt_str = (f', Exist={loss_exist.item():.4f}, Point={loss_point.item():.4f}' if point_labeled else '')
                print(f'{datetime.datetime.now()} | Step: {global_step}/{epoch + 1}/{args.max_epochs} | '
                      f'[{"SurgBlood" if point_labeled else "Hemo_Cholec"}] Total={loss.item():.4f}, '
                      f'Mask={lossmask.item():.4f}, Edge={lossmask_edge.item():.4f}{pt_str} | LR: {lr_str}')

        ############################# Validation (both domains) #############################
        net.eval()
        max_videos = 2 if args.debug_steps else 0
        res_hemo = evaluate(net, val_loaders['hemo'], with_points=False, max_videos=max_videos)
        res_surg = evaluate(net, val_loaders['surg'], with_points=True, max_videos=max_videos)
        score = 0.5 * (res_hemo['Dice_blood'] + res_surg['Dice_blood'])

        print("evaluate_epoch---------------:", epoch + 1)
        print(f"Hemo_Cholec val -> IoU: {res_hemo['IoU_blood']:.4f} | Dice: {res_hemo['Dice_blood']:.4f}")
        print(f"SurgBlood   val -> IoU: {res_surg['IoU_blood']:.4f} | Dice: {res_surg['Dice_blood']:.4f} | "
              f"precision_exist: {res_surg['precision_exist']:.4f} | PCK2: {res_surg['PCK2']:.4f} | "
              f"PCK5: {res_surg['PCK5']:.4f} | PCK10: {res_surg['PCK10']:.4f} | "
              f"FP su frame senza sangue: {res_surg['empty_fp_rate']:.3f} ({res_surg['empty_frames']} frame)")
        print(f"Selection score (media Dice): {score:.4f}")

        if score > best_score:
            best_score = score
            save_file = os.path.join(args.save_path, 'best_dino.pth')
            torch.save(net.state_dict(), save_file)
            run.save(save_file)
            print(f"\n*** Modello migliorato! Salvato in {save_file} (score: {best_score:.4f}) ***\n")

        last_file = os.path.join(args.save_path, 'last_dino.pth')
        torch.save(net.state_dict(), last_file)
        print(f"Checkpoint aggiornato salvato in: {last_file}")

        val_log = {"epoch": epoch + 1, "val/score": score, "val/best_score": best_score}
        val_log.update({f"val_hemo/{k}": v for k, v in res_hemo.items()})
        val_log.update({f"val_surg/{k}": v for k, v in res_surg.items()})
        run.log(val_log, step=global_step)

    run.finish()


if __name__ == '__main__':
    train(network)
