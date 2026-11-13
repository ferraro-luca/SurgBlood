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
# import SAM2_Video_baseline as network
from sam2_video_baseline import SAM2_Video_baseline as network

from datasetVideoTrain import CholeBPTrain
from datasetVideoTest  import CholeBPTest

from metrics.maskIouFloder import SegmentationMetric


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--save_path', default="/work/cvcs2026/blood_overflow/04_results/SurgBlood_evaluate_baseline", type=str)
    parser.add_argument('--data_path', default="/work/cvcs2026/blood_overflow/02_extracted/SurgBlood/SurgBlood-Dataset-test/", type=str)
    parser.add_argument('--max_epochs', type=int, default=20, help='maximum epoch number to train')
    parser.add_argument('--base_lr', type=float, default=6e-5, help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')
    # parser.add_argument('--img_size', type=int, default=128, help='input patch size of network input')

    parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
    parser.add_argument('--AdamW', action='store_true', help='If activated, use AdamW to finetune SAM model')
    parser.add_argument('--snapshot', type=str, default=None, help='path to the snapshot')

    parser.add_argument('--test_data_path', default="/work/cvcs2026/blood_overflow/02_extracted/SurgBlood/SurgBlood-Dataset-test/",type=str)

    parser.add_argument('--run_name', type=str, default='SurgBlood_metrics_baseline', help='Nome della run su W&B')
    parser.add_argument('--model_variant', type=str, default='baseline', help='baseline o finetuned_cholec')

    parser.parse_args()
    return parser.parse_args()



def train(Network,testpath):
    args = parser()
    print(f"Processing {testpath}.............................................................")

    # Inizializzazione W&B (salva i file temporanei su /work per non sporcare la cartella del progetto)
    run = wandb.init(
        entity="mgualtieri01-unimore",
        project="BlooDet",
        name=args.run_name,
        dir="/work/cvcs2026/blood_overflow/temp_wandb",  # Directory temporanea per i file W&B
        job_type="evaluation",
        tags=["evaluation", "SurgBlood", args.model_variant, testpath],
        config={
            "checkpoint": args.snapshot,
            "dataset": "SurgBlood",
            "split": testpath,
            "model_variant": args.model_variant,
            "img_size": args.img_size,
            "threshold": 0.5,
        }
    )

    test_dataset = CholeBPTest(args.test_data_path, mode=testpath)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)


    net = Network()

    path = '/work/cvcs2026/blood_overflow/weights/BlooDet-Pretrain'  # 替换为实际权重文件路径
    net.load_state_dict(torch.load(path))
    net.eval()
    net.cuda()


    #############################test############################################
    net.eval()
    ignore_labels = [255]
    metric = SegmentationMetric(2)

    frame_length_total = 0
    iou_list = []#iou
    dice_list = []#dice
    point_exist_TP = 0#count the TP point_exist score
    point_exist_total = 0#calculate the MPCK if the point is existed
    point_MPCK5_TP = 0
    point_MPCK10_TP = 0
    point_MPCK20_TP = 0

    with torch.no_grad():
        for step, sample in enumerate(testloader):
            images, masks, points, video_name_dict, frame_names = sample['images'], sample['masks'], sample['points'], sample['video_meta_dict'],sample['frame_names']
            sample['frame_names']
            images = images.cuda()
            images = images.view(-1, 3, 512, 512)
            shape = (512, 512)
            frame_length = images.size(0)
            video_segments = net(images)

            video_name = video_name_dict['filename_or_obj'][0]
            masks = masks.squeeze(0).cpu()
            points = points.squeeze(0).cpu()

            frame_length_total+=frame_length

            for i in range(frame_length):
                # mask
                current_pred_mask = video_segments[i][1]['mask_logits'].unsqueeze(1)
                frame_name = frame_names[i][0]
                # calculate iou
                sam_pre = current_pred_mask
                sam_pre = torch.sigmoid(sam_pre)

                sam_gt = masks[i].cpu()
                sam_pre = F.interpolate(sam_pre, size=shape, mode='bilinear', align_corners=False)
                sam_pre= sam_pre.squeeze(1)


                sam_pre = (sam_pre > 0.5).float()#
                sam_gt = (sam_gt > 0.5).float()

                sam_pre = (sam_pre > 0.5).int()
                sam_gt = (sam_gt > 0.5).int()
                hist = metric.addBatch(sam_pre, sam_gt, ignore_labels)

                # GT
                existed_label = ( points[i][0] != 0 and points[i][1] != 0)
                # point_exist_TP
                exited_score_logits = video_segments[i][1]['object_score']
                # Apply sigmoid and convert to numpy array
                exited_score_logits = torch.sigmoid(exited_score_logits).cpu().numpy()
                # >0.5 is 1, <0.5 is 0
                existed_pred = (exited_score_logits > 0.5).astype(int)
                # Calculate TP
                point_exist_TP += (existed_pred == existed_label.cpu().numpy()).all()

                if existed_label:
                    point_exist_total += 1

                    # point
                    pre_point = video_segments[i][1]['iou']
                    point_gt = points[i]*512
                    point_pre = pre_point*512

                    # Calculate MPCK10_TP
                    distances = torch.sqrt(torch.sum((point_gt.cpu() - point_pre.cpu()) ** 2, dim=-1))
                    if distances < (512*0.05):
                        point_MPCK5_TP += 1
                    if distances < (512*0.1):
                        point_MPCK10_TP += 1
                    if distances < (512*0.02):
                        point_MPCK20_TP += 1

    # Calculate metrics
    precision_exist = point_exist_TP / frame_length_total

    print(point_exist_total)

    MPCK5 = point_MPCK5_TP / point_exist_total
    MPCK10 = point_MPCK10_TP / point_exist_total
    MPCK20 = point_MPCK20_TP / point_exist_total
    iou_scores = metric.IntersectionOverUnion()
    dice_scores = metric.DiceCoefficient()
    print(f"IoU: {iou_scores}")
    print(f"Dice: {dice_scores}")

    print("precision_exist:", precision_exist)
    print("MPCK2:", MPCK20)
    print("MPCK5:", MPCK5)
    print("MPCK10:", MPCK10)

    # Log delle metriche su W&B
    iou_arr = iou_scores.detach().cpu().numpy() if torch.is_tensor(iou_scores) else np.asarray(iou_scores)
    dice_arr = dice_scores.detach().cpu().numpy() if torch.is_tensor(dice_scores) else np.asarray(dice_scores)

    iou_blood = float(iou_arr[1]) if iou_arr.ndim > 0 and len(iou_arr) > 1 else float(iou_arr)
    dice_blood = float(dice_arr[1]) if dice_arr.ndim > 0 and len(dice_arr) > 1 else float(dice_arr)
    miou = float(iou_arr.mean())
    mdice = float(dice_arr.mean())

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
"""
def train(Network,testpath):
    args = parser()
    print(f"Processing {testpath}.............................................................")

    test_dataset = CholeBPTest(args.test_data_path, mode=testpath)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    net = Network()

    path = '/work/cvcs2026/blood_overflow/weights/cholec_finetune_v1/best_finetuned_cholec.pth'
    net.load_state_dict(torch.load(path))
    net.eval()
    net.cuda()

    #############################test############################################
    ignore_labels = [255]
    metric = SegmentationMetric(2)

    frame_length_total = 0

    with torch.no_grad():
        for step, sample in enumerate(testloader):
            # Estrazione esclusiva di immagini, maschere e metadati (punti ignorati)
            images = sample['images']
            masks = sample['masks']
            video_name_dict = sample['video_meta_dict']
            frame_names = sample['frame_names']
            
            images = images.cuda()
            images = images.view(-1, 3, 512, 512)
            shape = (512, 512)
            frame_length = images.size(0)
            
            # Inferenza del modello
            video_segments = net(images)

            video_name = video_name_dict['filename_or_obj'][0]
            masks = masks.squeeze(0).cpu()

            frame_length_total += frame_length

            for i in range(frame_length):
                # Estrazione maschera predetta
                current_pred_mask = video_segments[i][1]['mask_logits'].unsqueeze(1)
                frame_name = frame_names[i][0]
                
                # Calcolo IoU e Dice
                sam_pre = current_pred_mask
                sam_pre = torch.sigmoid(sam_pre)

                sam_gt = masks[i].cpu()
                sam_pre = F.interpolate(sam_pre, size=shape, mode='bilinear', align_corners=False)
                sam_pre = sam_pre.squeeze(1)

                # Binarizzazione a soglia 0.5
                sam_pre = (sam_pre > 0.5).int()
                sam_gt = (sam_gt > 0.5).int()
                
                # Aggiunta al calcolatore di metriche
                hist = metric.addBatch(sam_pre, sam_gt, ignore_labels)

    # Stampa esclusiva delle metriche di segmentazione
    print(f"IoU: {metric.IntersectionOverUnion()}")
    print(f"Dice: {metric.DiceCoefficient()}")
"""

if __name__ == '__main__':
    # test_paths = [
    #     "testHemoSet",
    # ]
    test_paths = [
        "test",
    ]
    # test_paths = [
    #     "test751",
    # ]
    # test_paths = [
    #     "NIPSRebuttalDataset",
    # ]
    # test_paths = [
    #     "testGallbladder",
    #     "testCysticTriangle",
    #     "testVessel",
    #     "testGallbladderBed",
    # ]

    # 遍历测试路径并执行训练或测试函数
    for test_path in test_paths:
        train(network, testpath=test_path)  # 调用 train 函数