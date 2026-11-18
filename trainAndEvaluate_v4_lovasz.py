import os
import datetime
import argparse
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.optim as optim
import wandb

from sam2_video_baseline import SAM2_Video_baseline as network
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


# Define Lovasz-Hinge Loss (ottimizzazione diretta della IoU)
def _lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_loss(logits, mask):
    mask = F.interpolate(mask.float(), size=logits.size()[2:], mode='bilinear', align_corners=False)
    labels = (mask > 0.5).float()

    losses = []
    # Calcolo per singolo frame nella clip per stabilità anche sui frame vuoti
    for logit, label in zip(logits, labels):
        logit_flat = logit.view(-1)
        label_flat = label.view(-1)
        signs = 2.0 * label_flat - 1.0
        errors = 1.0 - logit_flat * signs
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        gt_sorted = label_flat[perm]
        grad = _lovasz_grad(gt_sorted)
        losses.append(torch.dot(F.relu(errors_sorted), grad))
    return torch.stack(losses).mean()


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--save_path', default="/work/cvcs2026/blood_overflow/weights/Hemo_Cholec_finetune_v4_lovasz", type=str)
    parser.add_argument('--data_path', default="/work/cvcs2026/blood_overflow/03_processed/Hemo_Cholec_splitted/", type=str)
    parser.add_argument('--max_epochs', type=int, default=20, help='maximum epoch number to train')
    parser.add_argument('--base_lr', type=float, default=6e-5, help='decoder/edge learning rate')
    parser.add_argument('--encoder_lr_ratio', type=float, default=0.1, help='multiplier for unfrozen encoder lr')
    parser.add_argument('--edge_weight', type=float, default=0.5, help='weight for edge loss')
    parser.add_argument('--lovasz_weight', type=float, default=0.5, help='weight for lovasz hinge loss on mask')
    parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')
    parser.add_argument('--run_name', type=str, default="finetuning_v4_lovasz(hemo_cholec_unfrozen_encoder_s3)", help='nome della run su W&B')

    parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
    parser.add_argument('--AdamW', action='store_true', help='If activated, use AdamW to finetune SAM model')
    parser.add_argument('--snapshot', type=str, default=None, help='path to the snapshot')
    parser.add_argument('--test_data_path', default="/work/cvcs2026/blood_overflow/03_processed/Hemo_Cholec_splitted/", type=str)
    return parser.parse_args()


def train(Network):
    args = parser()
    os.makedirs(args.save_path, exist_ok=True)

    run = wandb.init(
        entity="mgualtieri01-unimore",
        project="BlooDet",
        name=args.run_name,
        job_type="fine-tuning",
        tags=["fine-tuning", "Hemo_Cholec", "mask_decoder", "edge_generator", "unfrozen_encoder", "lovasz_loss"],
        notes="Fine-tuning v4 of BlooDet on Hemo_Cholec with partial unfreezing of Image Encoder (blocks 18-23 + neck) + Lovasz-Hinge Loss.",
        config={
            "experiment_type": "fine-tuning",
            "base_lr": args.base_lr,
            "encoder_lr": args.base_lr * args.encoder_lr_ratio,
            "encoder_lr_ratio": args.encoder_lr_ratio,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "img_size": args.img_size,
            "frames_length": 8,
            "overlap": 7,
            "architecture": "BlooDet-SAM2",
            "dataset": "Hemo_Cholec_splitted",
            "trainable_modules": ["mask_decoder", "edge_generator", "image_encoder.neck", "image_encoder.trunk.blocks.18-23"],
            "loss_weights": {"mask_focal_dice": 1.0, "lovasz": args.lovasz_weight, "edge": args.edge_weight},
            "optimizer": "AdamW"
        }
    )

    train_dataset = CholeBPTrain(args.data_path, args.img_size, mode='train', frames_length=8, overlap=7)
    test_dataset = CholeBPTest(args.test_data_path, mode='test')

    print("Total clips:", len(train_dataset))

    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    net = Network()

    # A. Carica i pesi originali
    pretrained_path = '/work/cvcs2026/blood_overflow/weights/BlooDet-Pretrain'
    net.load_state_dict(torch.load(pretrained_path))
    net.cuda()

    # B. Congela l'intera rete
    for param in net.parameters():
        param.requires_grad = False

    # C. Sblocca Decoder/Edge (LR pieno) + ultimi blocchi Encoder & Neck (LR ridotto)
    decoder_params = []
    encoder_params = []

    unfrozen_encoder_keys = [
        'image_encoder.neck',
        'image_encoder.trunk.blocks.18.',
        'image_encoder.trunk.blocks.19.',
        'image_encoder.trunk.blocks.20.',
        'image_encoder.trunk.blocks.21.',
        'image_encoder.trunk.blocks.22.',
        'image_encoder.trunk.blocks.23.',
    ]

    for name, param in net.named_parameters():
        # Sicurezza: la testa del punto di sanguinamento resta rigorosamente congelata
        if 'estimator' in name:
            continue

        if 'mask_decoder' in name or 'edge_generator' in name:
            param.requires_grad = True
            decoder_params.append(param)
        elif any(k in name for k in unfrozen_encoder_keys):
            param.requires_grad = True
            encoder_params.append(param)

    num_dec_params = sum(p.numel() for p in decoder_params)
    num_enc_params = sum(p.numel() for p in encoder_params)
    print(f"Parametri sbloccati Decoder/Edge: {num_dec_params:,}")
    print(f"Parametri sbloccati Encoder:      {num_enc_params:,}")

    assert num_enc_params > 0, "ERRORE: Nessun parametro dell'encoder è stato sbloccato! Controlla i nomi dei layer."

    # D. Passa all'ottimizzatore i due gruppi con Differential Learning Rate
    optimizer = optim.AdamW([
        {'params': decoder_params, 'lr': args.base_lr},
        {'params': encoder_params, 'lr': args.base_lr * args.encoder_lr_ratio}
    ], betas=(0.9, 0.999), weight_decay=0.1)

    sw = SummaryWriter(args.save_path)
    global_step = 0
    best_dice = 0.0

    max_epoch = args.max_epochs
    for epoch in range(max_epoch):
        net.train(True)

        # Scheduler piramidale applicato proporzionalmente a entrambi i gruppi
        lr_scale = 1 - abs((epoch + 1) / (args.max_epochs + 1) * 2 - 1)
        optimizer.param_groups[0]['lr'] = lr_scale * args.base_lr
        optimizer.param_groups[1]['lr'] = lr_scale * (args.base_lr * args.encoder_lr_ratio)

        for step, sample in enumerate(trainloader):
            image, mask, edge, point, video_meta_dict = (
                sample['image'], sample['mask'], sample['edge'], sample['point'], sample['video_meta_dict']
            )

            images, masks, edges = image.float().cuda(), mask.float().cuda(), edge.float().cuda()
            points = point.float().cuda()

            images = images.view(-1, 3, args.img_size, args.img_size)
            masks = masks.view(-1, 1, args.img_size, args.img_size)
            edges = edges.view(-1, 1, args.img_size, args.img_size)
            points = points.view(-1, 2)

            frame_length = images.size(0)

            # Forward pass
            video_segments = net(images)

            pre_masks = []
            pre_edges = []
            pre_point = []
            pre_existed_score_logits = []

            for i in range(frame_length):
                if 1 in video_segments[i]:
                    pre_masks.append(video_segments[i][1]['mask_logits'])
                    pre_point.append(video_segments[i][1]['iou'])
                    pre_existed_score_logits.append(video_segments[i][1]['object_score'])
                    pre_edges.append(video_segments[i][1]['edge_logits'])
                else:
                    print(f"Key 1 not found in video_segments[{i}]")

            mask_pred = torch.stack(pre_masks, dim=0)
            pre_point = torch.stack(pre_point, dim=0)
            existed_score_logits = torch.stack(pre_existed_score_logits, dim=0)
            edge_pred = torch.stack(pre_edges, dim=0)

            mask = mask.squeeze(0).cuda()
            edge = edge.squeeze(0).cuda()

            # Focal loss + Dice loss + Lovasz-Hinge loss per Maschera
            loss_focal = focal_loss(mask_pred, mask)
            loss_dice = dice_loss(mask_pred, mask)
            loss_lovasz = lovasz_hinge_loss(mask_pred, mask)
            lossmask = loss_focal + loss_dice + args.lovasz_weight * loss_lovasz

            # Focal loss + Dice loss per Edge
            loss_focal_edge = focal_loss(edge_pred, edge)
            loss_dice_edge = dice_loss(edge_pred, edge)
            lossmask_edge = loss_focal_edge + loss_dice_edge

            # Calcolo Loss dei punti (solo logging, non usate per backpropagation)
            existed_label = (points != 0).any(dim=1).float().cuda()
            lossExsited = F.binary_cross_entropy_with_logits(existed_score_logits, existed_label.unsqueeze(1))

            lossPoint = torch.tensor(0.0, device=points.device)
            for i in range(points.size(0)):
                if not torch.all(points[i] == 0):
                    lossPoint += F.smooth_l1_loss(pre_point[i].unsqueeze(0), points[i].unsqueeze(0))

            # Loss totale
            loss = lossmask + args.edge_weight * lossmask_edge

            if lossmask > 10 and epoch > 17:
                print("lossmask:", lossmask)
                print("video_meta_dict:", video_meta_dict)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            # Log TensorBoard
            sw.add_scalar('lr/decoder', optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalar('lr/encoder', optimizer.param_groups[1]['lr'], global_step=global_step)
            sw.add_scalars('losses', {
                'total_loss': loss.item(),
                'mask_loss': lossmask.item(),
                'lovasz_loss': loss_lovasz.item(),
                'edge_loss': lossmask_edge.item()
            }, global_step=global_step)

            # Log Weights & Biases (Step di Training)
            run.log({
                "train/total_loss": loss.item(),
                "train/mask_loss": lossmask.item(),
                "train/lovasz_loss": loss_lovasz.item(),
                "train/edge_loss": lossmask_edge.item(),
                "train/lr": optimizer.param_groups[0]['lr'],
                "train/encoder_lr": optimizer.param_groups[1]['lr']
            }, step=global_step)

            if step % 200 == 0:
                print(
                    f'{datetime.datetime.now()} | Step: {global_step}/{epoch + 1}/{max_epoch} | '
                    f'Total Loss={loss.item():.6f}, Mask Loss={lossmask.item():.6f} (Lovasz={loss_lovasz.item():.6f}), '
                    f'Edge Loss={lossmask_edge.item():.6f} | '
                    f'LR_dec={optimizer.param_groups[0]["lr"]:.2e}, LR_enc={optimizer.param_groups[1]["lr"]:.2e}'
                )

        ############################# Evaluate / Test ############################################
        net.eval()
        ignore_labels = [255]
        metric = SegmentationMetric(2)

        frame_length_total = 0
        point_exist_TP = 0
        point_exist_total = 0
        point_MPCK5_TP = 0
        point_MPCK10_TP = 0
        point_MPCK20_TP = 0

        with torch.no_grad():
            for step, sample in enumerate(testloader):
                images, masks, points, video_name_dict, frame_names = (
                    sample['images'], sample['masks'], sample['points'], sample['video_meta_dict'], sample['frame_names']
                )

                images = images.cuda()
                images = images.view(-1, 3, 512, 512)
                shape = (512, 512)
                frame_length = images.size(0)
                video_segments = net(images)

                masks = masks.squeeze(0).cpu()
                points = points.squeeze(0).cpu()

                frame_length_total += frame_length

                for i in range(frame_length):
                    current_pred_mask = video_segments[i][1]['mask_logits'].unsqueeze(1)
                    sam_pre = torch.sigmoid(current_pred_mask)

                    sam_gt = masks[i].cpu()
                    sam_pre = F.interpolate(sam_pre, size=shape, mode='bilinear', align_corners=False)
                    sam_pre = sam_pre.squeeze(1)

                    sam_pre = (sam_pre > 0.5).int()
                    sam_gt = (sam_gt > 0.5).int()
                    metric.addBatch(sam_pre, sam_gt, ignore_labels)

                    existed_label = (points[i][0] != 0 and points[i][1] != 0)
                    exited_score_logits = video_segments[i][1]['object_score']
                    exited_score_logits = torch.sigmoid(exited_score_logits).cpu().numpy()
                    existed_pred = (exited_score_logits > 0.5).astype(int)

                    point_exist_TP += (existed_pred == existed_label.cpu().numpy()).all()

                    if existed_label:
                        point_exist_total += 1
                        pre_point = video_segments[i][1]['iou']
                        point_gt = points[i] * 512
                        point_pre = pre_point * 512

                        distances = torch.sqrt(torch.sum((point_gt.cpu() - point_pre.cpu()) ** 2, dim=-1))
                        if distances < (512 * 0.05):
                            point_MPCK5_TP += 1
                        if distances < (512 * 0.1):
                            point_MPCK10_TP += 1
                        if distances < (512 * 0.02):
                            point_MPCK20_TP += 1

        precision_exist = point_exist_TP / frame_length_total if frame_length_total > 0 else 0.0
        MPCK5 = point_MPCK5_TP / point_exist_total if point_exist_total > 0 else 0.0
        MPCK10 = point_MPCK10_TP / point_exist_total if point_exist_total > 0 else 0.0
        MPCK20 = point_MPCK20_TP / point_exist_total if point_exist_total > 0 else 0.0

        raw_iou = metric.IntersectionOverUnion()
        raw_dice = metric.DiceCoefficient()

        if hasattr(raw_dice, 'numel') and raw_dice.numel() > 1:
            current_dice = raw_dice[1].item()
            current_iou = raw_iou[1].item()
        elif hasattr(raw_dice, '__len__') and len(raw_dice) > 1:
            current_dice = float(raw_dice[1])
            current_iou = float(raw_iou[1])
        else:
            current_dice = float(raw_dice)
            current_iou = float(raw_iou)

        print("evaluate_epoch---------------:", epoch + 1)
        print(f"IoU (Background, Sangue): {raw_iou} -> Sangue: {current_iou:.4f}")
        print(f"Dice (Background, Sangue): {raw_dice} -> Sangue: {current_dice:.4f}")
        print("precision_exist:", precision_exist)
        print("MPCK2:", MPCK20)
        print("MPCK5:", MPCK5)
        print("MPCK10:", MPCK10)

        if current_dice > best_dice:
            best_dice = current_dice
            save_file = os.path.join(args.save_path, 'best_finetuned_Hemo_Cholec.pth')
            torch.save(net.state_dict(), save_file)
            run.save(save_file)
            print(f"\n*** Modello migliorato! Salvato in {save_file} (Dice Sangue: {best_dice:.4f}) ***\n")

        # Salvataggio Garantito dell'ultima epoca (Last)
        last_file = os.path.join(args.save_path, 'last_finetuned_Hemo_Cholec.pth')
        torch.save(net.state_dict(), last_file)
        print(f"Checkpoint aggiornato salvato in: {last_file}")

        run.log({
            "epoch": epoch + 1,
            "val/IoU_blood": current_iou,
            "val/Dice_blood": current_dice,
            "val/best_Dice_blood": best_dice,
            "val/precision_exist": precision_exist,
            "val/MPCK2": MPCK20,
            "val/MPCK5": MPCK5,
            "val/MPCK10": MPCK10
        }, step=global_step)

    run.finish()


if __name__ == '__main__':
    train(network)