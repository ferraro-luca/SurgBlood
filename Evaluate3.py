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
from sam2_video_baseline import apply_model_config, load_model_config

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

    parser.add_argument('--test_data_path', default="/work/cvcs2026/blood_overflow/03_processed/Hemo_Cholec_test/", type=str)

    # Nuovi argomenti per W&B
    parser.add_argument('--run_name', type=str, default='Hemo_Cholec_metrics_Finetuned_Hemo_Cholec_v3_no_TTA', help='Nome della run su W&B')
    parser.add_argument('--model_variant', type=str, default='finetuned_Hemo_Cholec_v3', help='baseline o finetuned_Hemo_Cholec')
    parser.add_argument('--point_gate', choices=['auto', 'on', 'off'], default='auto',
                        help="auto = legge model_config.json accanto al checkpoint (v7: off; v1-v6: on)")
    parser.add_argument('--soft_gate', choices=['auto', 'on', 'off'], default='auto',
                        help="on = nessun click al mask decoder quando lo score dice 'nessun punto'")
    parser.add_argument('--click_source', choices=['mlp', 'heatmap', 'fusion', 'oracle'], default='mlp',
                        help="v8: da dove arriva il click alla maschera (mlp = come sempre; oracle = punto vero, solo SurgBlood)")

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
    threshold = 0.50

    print(f"Processing {testpath} | Threshold: {threshold}")

    run = wandb.init(
        entity="mgualtieri01-unimore",
        project="BlooDet",
        name=args.run_name,
        dir="/work/cvcs2026/blood_overflow/temp_wandb",
        job_type="evaluation",
        tags=["evaluation", args.model_variant, testpath, "no_TTA"],
        config={
            "checkpoint": args.snapshot,
            "test_data_path": args.test_data_path,
            "split": testpath,
            "model_variant": args.model_variant,
            "img_size": args.img_size,
            "tta": False,
            "point_gate": args.point_gate,
            "soft_gate": args.soft_gate,
            "click_source": args.click_source,
            "threshold": threshold,
        }
    )

    test_dataset = CholeBPTest(args.test_data_path, mode=testpath)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Architettura dal model_config.json (default Hiera-B+ per v1-v7)
    model_cfg = load_model_config(args.snapshot).get('model_cfg', 'sam2_hiera_b+.yaml')
    print(f"Model config: {model_cfg}")
    net = Network(model_cfg=model_cfg)
    net.load_state_dict(torch.load(args.snapshot))
    apply_model_config(net, args.snapshot, args.point_gate, args.soft_gate)
    if args.click_source == 'oracle':
        raise SystemExit("click_source=oracle richiede i punti annotati: usarlo solo con Evaluate2 (SurgBlood)")
    net.predictor.click_source = args.click_source
    print(f"Click alla maschera da: {args.click_source}")
    net.eval()
    net.cuda()

    ignore_labels = [255]
    metric = SegmentationMetric(2)

    frame_length_total = 0

    with torch.no_grad():
        for step, sample in enumerate(testloader):
            images = sample['images'].cuda()
            masks = sample['masks']

            images = images.view(-1, 3, 512, 512)
            shape = (512, 512)
            frame_length = images.size(0)

            # Forward pass singolo (nessuna TTA)
            video_segments = net(images)

            masks = masks.squeeze(0).cpu()
            frame_length_total += frame_length

            for i in range(frame_length):
                pred_logits = video_segments[i][1]['mask_logits'].unsqueeze(1)
                sam_pre = torch.sigmoid(pred_logits)

                sam_gt = masks[i].cpu()
                sam_pre = F.interpolate(sam_pre, size=shape, mode='bilinear', align_corners=False)
                sam_pre = sam_pre.squeeze(1)

                # Binarizzazione a soglia 0.50
                sam_pre = (sam_pre > threshold).int()
                sam_gt = (sam_gt > 0.5).int()
                metric.addBatch(sam_pre, sam_gt, ignore_labels)

    iou_blood, dice_blood, miou, mdice = extract_scores(metric)
    print(f"IoU: {metric.IntersectionOverUnion()}")
    print(f"Dice: {metric.DiceCoefficient()}")
    print(f"Threshold {threshold:.2f} -> IoU_blood: {iou_blood:.4f} | Dice_blood: {dice_blood:.4f} | mIoU: {miou:.4f} | mDice: {mdice:.4f}")

    metrics_to_log = {
        "test/IoU_blood": iou_blood,
        "test/Dice_blood": dice_blood,
        "test/mIoU": miou,
        "test/mDice": mdice,
        "test/total_frames": frame_length_total,
    }
    run.log(metrics_to_log)
    run.summary.update(metrics_to_log)
    run.finish()


if __name__ == '__main__':
    test_paths = [
        "test",
    ]
    for test_path in test_paths:
        train(network, testpath=test_path)