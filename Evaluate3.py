import datetime
import argparse
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.optim as optim
import wandb
import numpy as np
from sam2_video_baseline import SAM2_Video_baseline as network

from datasetVideoTrain import CholeBPTrain
from datasetVideoTest import CholeBPTest

from metrics.maskIouFloder import SegmentationMetric


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--save_path', default="/work/cvcs2026/blood_overflow/04_results/Hemo_Cholec_evaluate", type=str)
    parser.add_argument('--data_path', default="/work/cvcs2026/blood_overflow/03_processed/Hemo_Cholec_test/", type=str)
    parser.add_argument('--max_epochs', type=int, default=20, help='maximum epoch number to train')
    parser.add_argument('--base_lr', type=float, default=6e-5, help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')

    parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
    parser.add_argument('--AdamW', action='store_true', help='If activated, use AdamW to finetune SAM model')
    parser.add_argument('--snapshot', type=str, default='/work/cvcs2026/blood_overflow/weights/Hemo_Cholec_finetune_v3_encoder/best_finetuned_Hemo_Cholec.pth', help='path to the snapshot')

    parser.add_argument('--test_data_path', default="/work/cvcs2026/blood_overflow/03_processed/Hemoset_test/", type=str)

    # Nuovi argomenti per W&B
    parser.add_argument('--run_name', type=str, default='Hemo_Cholec_metrics_Finetuned_Hemo_Cholec_v3_no_TTA', help='Nome della run su W&B')
    parser.add_argument('--model_variant', type=str, default='finetuned_Hemo_Cholec_v3', help='baseline o finetuned_hemoset')

    # Opzione TTA (Horizontal Flip)
    parser.add_argument('--no_tta', action='store_true', help='Disabilita Horizontal Flip TTA')

    return parser.parse_args()


def extract_scores(metric_obj):
    iou_scores = metric_obj.IntersectionOverUnion()
    dice_scores = metric_obj.DiceCoefficient()
    iou_arr = iou_scores.detach().cpu().numpy() if torch.is_tensor(iou_scores) else np.asarray(iou_scores)
    dice_arr = dice_scores.detach().cpu().numpy() if torch.is_tensor(dice_scores) else np.asarray(dice_scores)

    iou_blood = float(iou_arr[1]) if iou_arr.ndim > 0 and len(iou_arr) > 1 else float(iou_arr)
    dice_blood = float(dice_arr[1]) if dice_arr.ndim > 0 and len(dice_arr) > 1 else float(dice_arr)
    miou = float(iou_arr.mean())
    mdice = float(dice_arr.mean())
    return iou_blood, dice_blood, miou, mdice


def train(Network, testpath):
    args = parser()
    use_tta = not args.no_tta
    thresholds = [0.35, 0.40, 0.45, 0.50, 0.55]

    print(f"Processing {testpath} | TTA (Horizontal Flip): {use_tta} | Thresholds: {thresholds}")

    run = wandb.init(
        entity="mgualtieri01-unimore",
        project="BlooDet",
        name=args.run_name,
        dir="/work/cvcs2026/blood_overflow/temp_wandb",
        job_type="evaluation",
        tags=["evaluation", args.model_variant, testpath, "TTA" if use_tta else "no_TTA"],
        config={
            "checkpoint": args.snapshot,
            "test_data_path": args.test_data_path,
            "split": testpath,
            "model_variant": args.model_variant,
            "img_size": args.img_size,
            "tta": use_tta,
            "thresholds": thresholds,
        }
    )

    test_dataset = CholeBPTest(args.test_data_path, mode=testpath)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    net = Network()
    net.load_state_dict(torch.load(args.snapshot))
    net.eval()
    net.cuda()

    ignore_labels = [255]
    # Un calcolatore di metriche per ciascuna soglia in un solo passaggio
    metrics_by_th = {th: SegmentationMetric(2) for th in thresholds}

    frame_length_total = 0

    with torch.no_grad():
        for step, sample in enumerate(testloader):
            images = sample['images'].cuda()
            masks = sample['masks']

            images = images.view(-1, 3, 512, 512)
            shape = (512, 512)
            frame_length = images.size(0)

            # 1. Forward pass originale
            video_segments = net(images)

            # 2. Forward pass con Horizontal Flip TTA (se attivo)
            if use_tta:
                images_flip = torch.flip(images, dims=[-1])
                video_segments_flip = net(images_flip)

            masks = masks.squeeze(0).cpu()
            frame_length_total += frame_length

            for i in range(frame_length):
                # Probabilità predizione originale
                pred_logits = video_segments[i][1]['mask_logits'].unsqueeze(1)
                prob_orig = torch.sigmoid(pred_logits)

                if use_tta:
                    # Probabilità predizione flippata, poi ri-ribaltata orizzontalmente
                    pred_logits_flip = video_segments_flip[i][1]['mask_logits'].unsqueeze(1)
                    prob_flip = torch.flip(torch.sigmoid(pred_logits_flip), dims=[-1])
                    sam_pre = 0.5 * (prob_orig + prob_flip)
                else:
                    sam_pre = prob_orig

                sam_gt = masks[i].cpu()
                sam_pre = F.interpolate(sam_pre, size=shape, mode='bilinear', align_corners=False)
                sam_pre = sam_pre.squeeze(1)

                sam_gt_int = (sam_gt > 0.5).int()

                # Valutazione su tutte le soglie nello stesso forward pass
                for th in thresholds:
                    sam_pre_int = (sam_pre > th).int()
                    metrics_by_th[th].addBatch(sam_pre_int, sam_gt_int, ignore_labels)

    print("\n================ RISULTATI PER SOGLIA ================")
    metrics_to_log = {"test/total_frames": frame_length_total}

    for th in thresholds:
        iou_blood, dice_blood, miou, mdice = extract_scores(metrics_by_th[th])
        print(f"Threshold {th:.2f} -> IoU_blood: {iou_blood:.4f} | Dice_blood: {dice_blood:.4f} | mIoU: {miou:.4f} | mDice: {mdice:.4f}")

        th_key = f"{int(round(th * 100))}"
        metrics_to_log[f"sweep/IoU_blood_th{th_key}"] = iou_blood
        metrics_to_log[f"sweep/Dice_blood_th{th_key}"] = dice_blood

        # Mantiene i nomi standard a soglia 0.50 per confronto diretto su W&B
        if th == 0.50:
            metrics_to_log["test/IoU_blood"] = iou_blood
            metrics_to_log["test/Dice_blood"] = dice_blood
            metrics_to_log["test/mIoU"] = miou
            metrics_to_log["test/mDice"] = mdice

    run.log(metrics_to_log)
    run.summary.update(metrics_to_log)
    run.finish()


if __name__ == '__main__':
    test_paths = [
        "test",
    ]
    for test_path in test_paths:
        train(network, testpath=test_path)