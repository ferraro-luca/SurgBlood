"""BlooDet fine-tuning v7: v6 recipe WITHOUT the point-existence gate on the blood mask.

In the original BlooDet the bleeding-POINT decoder's existence score (object_score_logits)
controls two things on the blood-mask pathway (sam2/modeling/sam2_base.py):
  1. if "no bleeding point", the whole blood mask is set to NO_OBJ_SCORE (-1024) -> black;
  2. the memory object pointer is replaced by `no_obj_ptr` ("no object" for next frames).
v7 disables both (point_gates_mask=False): the blood mask and its memory are decided by the
mask pathway only; the point branch only predicts the bleeding point.

Everything else is IDENTICAL to v6 (data, --balance, point losses on SurgBlood only, LRs,
frozen memory, validation on both domains, best ckpt = mean Dice). Weights are initialised
from the v6 best checkpoint (--init_ckpt), so v7 vs v6 measures the effect of the gate.

The option is not part of the state_dict: model_config.json is written next to the weights
and read automatically by Evaluate2.py / Evaluate3.py (apply_model_config).
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
    parser.add_argument('--save_path', default="/work/cvcs2026/blood_overflow/weights/Hemo_Cholec_SurgBlood_finetune_v7", type=str)
    parser.add_argument('--init_ckpt', type=str,
                        default="/work/cvcs2026/blood_overflow/weights/Hemo_Cholec_SurgBlood_finetune_v6/best_finetuned_v6.pth",
                        help='weights to start from (v6 best)')
    parser.add_argument('--data_path', default="/work/cvcs2026/blood_overflow/03_processed/Hemo_Cholec_splitted/", type=str,
                        help='Hemo_Cholec root (train/ used for training)')
    parser.add_argument('--test_data_path', default="/work/cvcs2026/blood_overflow/03_processed/Hemo_Cholec_splitted/", type=str,
                        help='Hemo_Cholec root for validation (test/ split, as in v3/v5)')
    parser.add_argument('--surgblood_path', default="/work/cvcs2026/blood_overflow/03_processed/SurgBlood_splitted/", type=str,
                        help='SurgBlood root prepared by src/datasets/surgblood_prepare.py (train/ and test/ = validation split)')
    parser.add_argument('--max_epochs', type=int, default=20, help='maximum epoch number to train')
    parser.add_argument('--base_lr', type=float, default=6e-5, help='mask decoder / edge learning rate')
    parser.add_argument('--point_lr', type=float, default=6e-5, help='lr for bleeding-point branches')
    parser.add_argument('--encoder_lr', type=float, default=2e-6, help='lr for unfrozen image encoder (blocks 18-23 + neck)')
    parser.add_argument('--memory_lr', type=float, default=0.0, help='lr for SAM2 memory_attention + memory_encoder (0 = frozen)')
    parser.add_argument('--edge_weight', type=float, default=0.5, help='weight for edge loss')
    parser.add_argument('--exist_weight', type=float, default=1.0, help='weight for point-existence loss (SurgBlood only)')
    parser.add_argument('--point_weight', type=float, default=1.0, help='weight for point-coordinate loss (SurgBlood only)')
    parser.add_argument('--no_augment', action='store_true', help='Disable clip-consistent photometric augmentation')
    parser.add_argument('--balance', action='store_true',
                        help='Sample 50/50 per dataset (with replacement). Default: every training clip of both datasets once per epoch')
    parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')
    parser.add_argument('--run_name', type=str, default="finetuning_v7(v6_no_point_gate)", help='nome della run su W&B')

    parser.add_argument('--debug_steps', type=int, default=0,
                        help='>0: limit train steps per epoch and validation videos (smoke test only)')

    parser.add_argument('--warmup', action='store_true', help='(unused, kept for compatibility with training.sh)')
    parser.add_argument('--AdamW', action='store_true', help='(unused, kept for compatibility with training.sh)')
    parser.add_argument('--snapshot', type=str, default=None, help='(unused)')
    return parser.parse_args()


# Bleeding-point branches (see sam2/modeling/sam2_base.py). 'estimator' (PWC-Net optical
# flow) and the prompt encoders stay frozen.
POINT_KEYS = ['sam_point_decoder', 'point_memory_attention', 'offset_x_proj', 'offset_y_proj']
ENCODER_KEYS = ['image_encoder.neck'] + [f'image_encoder.trunk.blocks.{i}.' for i in range(18, 24)]
# Leading dot avoids matching 'point_memory_attention'
MEMORY_KEYS = ['.memory_attention.', '.memory_encoder.']


def build_param_groups(net, args):
    for param in net.parameters():
        param.requires_grad = False

    groups = {'decoder': [], 'point': [], 'encoder': [], 'memory': []}
    for name, param in net.named_parameters():
        if 'estimator' in name:
            continue
        if any(k in name for k in POINT_KEYS):
            groups['point'].append(param)
        elif 'mask_decoder' in name or 'edge_generator' in name:
            groups['decoder'].append(param)
        elif any(k in name for k in ENCODER_KEYS):
            groups['encoder'].append(param)
        elif args.memory_lr > 0 and any(k in name for k in MEMORY_KEYS):
            groups['memory'].append(param)

    lrs = {'decoder': args.base_lr, 'point': args.point_lr, 'encoder': args.encoder_lr, 'memory': args.memory_lr}
    names, param_groups = [], []
    for g, params in groups.items():
        if not params:
            continue
        for p in params:
            p.requires_grad = True
        names.append(g)
        param_groups.append({'params': params, 'lr': lrs[g]})
        print(f"Parametri sbloccati {g:<8}: {sum(p.numel() for p in params):>12,}  (lr {lrs[g]:.1e})")

    for g in ('decoder', 'point', 'encoder'):
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
        tags=["fine-tuning", "Hemo_Cholec", "SurgBlood", "joint", "point_branches", "no_point_gate", "v7"],
        notes="v7: v6 recipe from v6 weights, point-existence score no longer gates the blood mask / memory pointer.",
        config=vars(args) | {
            "architecture": "BlooDet-SAM2",
            "frames_length": 8,
            "overlap": 7,
            "clip_augmentation": not args.no_augment,
            "balanced_sampling": args.balance,
            "selection_metric": "mean(Dice Hemo_Cholec val, Dice SurgBlood val)",
            "point_gates_mask": False,
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

    net = Network(point_gates_mask=False)
    net.load_state_dict(torch.load(args.init_ckpt))
    net.cuda()
    print(f"Pesi iniziali: {args.init_ckpt}")
    print(f"Point-existence gate on the blood mask: OFF (v7)")
    # The gate option is not in the state_dict: store it next to the checkpoints
    save_model_config(args.save_path, point_gates_mask=False, init_ckpt=args.init_ckpt, version='v7')

    group_names, param_groups, group_base_lrs = build_param_groups(net, args)
    optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999), weight_decay=0.1)

    sw = SummaryWriter(args.save_path)
    global_step = 0
    best_score = 0.0

    for epoch in range(args.max_epochs):
        net.train(True)

        # Scheduler piramidale applicato proporzionalmente a tutti i gruppi
        lr_scale = 1 - abs((epoch + 1) / (args.max_epochs + 1) * 2 - 1)
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
            save_file = os.path.join(args.save_path, 'best_finetuned_v7.pth')
            torch.save(net.state_dict(), save_file)
            run.save(save_file)
            print(f"\n*** Modello migliorato! Salvato in {save_file} (score: {best_score:.4f}) ***\n")

        last_file = os.path.join(args.save_path, 'last_finetuned_v7.pth')
        torch.save(net.state_dict(), last_file)
        print(f"Checkpoint aggiornato salvato in: {last_file}")

        val_log = {"epoch": epoch + 1, "val/score": score, "val/best_score": best_score}
        val_log.update({f"val_hemo/{k}": v for k, v in res_hemo.items()})
        val_log.update({f"val_surg/{k}": v for k, v in res_surg.items()})
        run.log(val_log, step=global_step)

    run.finish()


if __name__ == '__main__':
    train(network)
