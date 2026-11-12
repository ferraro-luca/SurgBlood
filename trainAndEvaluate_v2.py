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

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--save_path', default="/work/cvcs2026/blood_overflow/weights/cholec_finetune_v1", type=str)
    parser.add_argument('--data_path', default="/work/cvcs2026/blood_overflow/03_processed/cholec_splitted/", type=str)
    parser.add_argument('--max_epochs', type=int, default=20, help='maximum epoch number to train')
    parser.add_argument('--base_lr', type=float, default=6e-5, help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')
    parser.add_argument('--run_name', type=str, default="finetune_mask_edge_v1", help='nome della run su W&B')

    parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
    parser.add_argument('--AdamW', action='store_true', help='If activated, use AdamW to finetune SAM model')
    parser.add_argument('--snapshot', type=str, default=None, help='path to the snapshot')
    parser.add_argument('--test_data_path', default="/work/cvcs2026/blood_overflow/03_processed/cholec_splitted/", type=str)

    return parser.parse_args()


def train(Network):
    args = parser()
    os.makedirs(args.save_path, exist_ok=True)

    # --- INIZIALIZZAZIONE WEIGHTS & BIASES ---
    run = wandb.init(
        entity="mgualtieri01-unimore",
        project="BlooDet",
        name=args.run_name,
        job_type="fine-tuning",
        tags=["fine-tuning", "CholecSeg8k_filtered", "mask_decoder", "edge_generator"],
        notes="Fine-tuning di BlooDet su CholecSeg8k_filtered (sbloccati solo mask_decoder e edge_generator)",
        config={
            "experiment_type": "fine-tuning",
            "base_lr": args.base_lr,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "img_size": args.img_size,
            "frames_length": 8,
            "overlap": 7,
            "architecture": "BlooDet-SAM2",
            "dataset": "CholecSeg8k_filtered",
            "trainable_modules": ["mask_decoder", "edge_generator"],
            "loss_weights": {"mask": 1.0, "edge": 0.5},
            "optimizer": "AdamW"
        }
    )

    train_dataset = CholeBPTrain(args.data_path, args.img_size, mode='train', frames_length=8, overlap=7)
    test_dataset = CholeBPTest(args.test_data_path, mode='test')

    print("Total clips:", len(train_dataset))

    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    net = Network()

    # A. Carica i pesi originali (sola lettura)
    pretrained_path = '/work/cvcs2026/blood_overflow/weights/BlooDet-Pretrain'
    net.load_state_dict(torch.load(pretrained_path))
    net.cuda()

    # B. Congela l'intera rete
    for param in net.parameters():
        param.requires_grad = False

    # C. Sblocca solo il decoder delle maschere e l'edge generator
    mask_params = []
    for name, param in net.named_parameters():
        if 'mask_decoder' in name or 'edge_generator' in name:
            param.requires_grad = True
            mask_params.append(param)

    # D. Passa all'ottimizzatore SOLO i parametri sbloccati
    optimizer = optim.AdamW(mask_params, lr=args.base_lr, betas=(0.9, 0.999), weight_decay=0.1)

    sw = SummaryWriter(args.save_path)
    global_step = 0
    best_dice = 0.0

    max_epoch = args.max_epochs
    for epoch in range(max_epoch):
        # Fondamentale rimettere train(True) all'inizio di ogni epoca dopo net.eval()
        net.train(True)

        optimizer.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (args.max_epochs + 1) * 2 - 1)) * args.base_lr

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

            # Focal loss + Dice loss per Maschera
            loss_focal = focal_loss(mask_pred, mask)
            loss_dice = dice_loss(mask_pred, mask)
            lossmask = loss_focal + loss_dice

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

            loss = lossmask + 0.5 * lossmask_edge

            if lossmask > 10 and epoch > 17:
                print("lossmask:", lossmask)
                print("video_meta_dict:", video_meta_dict)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            # Log TensorBoard
            sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('losses', {
                'total_loss': loss.item(),
                'mask_loss': lossmask.item(),
                'edge_loss': lossmask_edge.item(),
                'lossPoint': lossPoint.item(),
                'existed_loss': lossExsited.item()
            }, global_step=global_step)

            # Log Weights & Biases (Step di Training)
            run.log({
                "train/total_loss": loss.item(),
                "train/mask_loss": lossmask.item(),
                "train/edge_loss": lossmask_edge.item(),
                "train/point_loss": lossPoint.item(),
                "train/existed_loss": lossExsited.item(),
                "train/lr": optimizer.param_groups[0]['lr']
            }, step=global_step)

            if step % 200 == 0:
                print(
                    f'{datetime.datetime.now()} | Step: {global_step}/{epoch + 1}/{max_epoch} | '
                    f'Total Loss={loss.item():.6f}, Mask Loss={lossmask.item():.6f}, '
                    f'Edge Loss={lossmask_edge.item():.6f}, Point Loss={lossPoint.item():.6f}'
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

        # Estrazione scalare float per la classe 1 (Sanguinamento)
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

        # Salvataggio Miglior Modello (Best) + Upload su W&B
        if current_dice > best_dice:
            best_dice = current_dice
            save_file = os.path.join(args.save_path, 'best_finetuned_cholec.pth')
            torch.save(net.state_dict(), save_file)
            run.save(save_file)
            print(f"\n*** Modello migliorato! Salvato in {save_file} (Dice Sangue: {best_dice:.4f}) ***\n")

        # Salvataggio Garantito dell'ultima epoca (Last)
        last_file = os.path.join(args.save_path, 'last_finetuned_cholec.pth')
        torch.save(net.state_dict(), last_file)
        print(f"Checkpoint aggiornato salvato in: {last_file}")

        # Log Weights & Biases (Fine Epoca)
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