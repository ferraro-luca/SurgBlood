import torch
import torch.nn as nn
import numpy as np
from sam2.build_sam import build_sam2_video_predictor
import time
import cv2
import os
import json


# ---------------------------------------------------------------------------------------
# Architecture options that are NOT stored in the state_dict travel with the checkpoint in
# a small 'model_config.json' saved in the same folder (written by the training script).
# Evaluation scripts call apply_model_config(net, ckpt_path) after loading the weights.
# ---------------------------------------------------------------------------------------
MODEL_CONFIG_NAME = 'model_config.json'


def save_model_config(save_dir, **config):
    with open(os.path.join(save_dir, MODEL_CONFIG_NAME), 'w') as f:
        json.dump(config, f, indent=2)


def load_model_config(ckpt_path):
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), MODEL_CONFIG_NAME)
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            return json.load(f)
    return {}


def apply_model_config(net, ckpt_path, point_gate='auto'):
    """Apply the architecture options of a checkpoint.

    point_gate: 'auto' -> read model_config.json next to the checkpoint (default: gate ON,
                i.e. original BlooDet, for v1-v6 checkpoints that have no config file);
                'on' / 'off' -> force it (e.g. to test v6 without the gate at inference).
    """
    cfg = load_model_config(ckpt_path)
    if point_gate == 'auto':
        gate = cfg.get('point_gates_mask', True)
        source = 'model_config.json' if 'point_gates_mask' in cfg else 'default (no model_config.json)'
    else:
        gate = (point_gate == 'on')
        source = f'forced --point_gate {point_gate}'
    net.predictor.point_gates_mask = gate
    print(f"Point-existence gate on the blood mask: {'ON (original BlooDet)' if gate else 'OFF (v7)'}  [{source}]")
    return gate


class SAM2_Video_baseline(nn.Module):

    def __init__(self, point_gates_mask=True):
        super(SAM2_Video_baseline, self).__init__()

        # # #sam2_hiera_large
        # self.sam2_checkpoint = "/home/siat-hci/ZhoZhangJun/SAM2-Codes/checkpoints/sam2_hiera_large.pt"
        # self.model_cfg = "sam2_hiera_l.yaml"

        # #sam2_hiera_large
        # self.sam2_checkpoint = "/home/siat-hci/ZhoZhangJun/SAM2-Codes/checkpoints/sam2_hiera_base_plus.pt"
        # self.model_cfg = "sam2_hiera_b+.yaml"
        #sam2_hiera_large
        self.sam2_checkpoint = "/work/cvcs2026/blood_overflow/weights/sam2_hiera_base_plus.pt"
        self.model_cfg = "sam2_hiera_b+.yaml"

        # # # #sam2_hiera_large
        # self.sam2_checkpoint = "/home/siat-hci/ZhoZhangJun/SAM2-Codes/checkpoints/sam2_hiera_small.pt"
        # self.model_cfg = "sam2_hiera_s.yaml"
        # # # #sam2_hiera_large
        # sam2_checkpoint = "/home/siat-hci/ZhoZhangJun/SAM2-Codes/checkpoints/sam2_hiera_tiny.pt"
        # model_cfg = "sam2_hiera_t.yaml"

        #video net 
        self.predictor = build_sam2_video_predictor(self.model_cfg, self.sam2_checkpoint , device="cuda")
        self.predictor.point_gates_mask = point_gates_mask

        # for n, p in self.sam2image.named_parameters():
        #     p.requires_grad = False
        #
        # for n, p in self.sam2image.named_parameters():
        #     # print(n)
        #     if "mask_decoder" in n:
        #
        #         p.requires_grad = True
        #     if "prompt" in n:  # prompt
        #
        #         p.requires_grad = True
        #     # memory_attention
        #     if "memory_attention" in n:
        #
        #         p.requires_grad = True
        #     # memory_encoder
        #     if "memory_encoder" in n:
        #
        #         p.requires_grad = True


        for name, param in self.predictor.named_parameters():
            if 'estimator' in name:
                param.requires_grad = False


        total_params = sum(p.numel() for p in self.predictor.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(p.numel() for p in self.predictor.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

        # if self.cfg is not None and self.cfg.snapshot:
        #     print('load checkpoint')
        #     self.load_state_dict(torch.load(self.cfg.snapshot))


    def forward(self, images ):

        inference_state = self.predictor.train_init_state(images)#inital the state  for every batch of images

        # print(inference_state)#inference_state is a dict
        # input(1111111111)
        # for k, v in inference_state.items():
        #     print(k)
        #     print(v)
        #     # print(v.size())
        #     print('-----------------')
        # input(1111111111)


    
        ann_frame_idx = 0
        ann_obj_id = 1
        # points = np.array([[800, 100]], dtype=np.float32)
        points = None
        labels = np.array([1], np.int32)

        # _, out_obj_ids, out_mask_logits = self.predictor.train_add_new_points_or_box(


        frame_idx, obj_ids, video_res_masks, iou, object_score_logits = self.predictor.train_add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
        
        # run propagation throughout the video and collect the results in a dict
        video_segments = {}

        total_time = 0.0
        total_frames = 0

        start_time = time.time()  # Record the start time for this frame

        # frame_idx, obj_ids, video_res_masks, iou, object_score_logits
        for out_frame_idx, out_obj_ids, out_mask_logits, iou, object_score_logits,out_edge_logits in self.predictor.train_propagate_in_video(inference_state):
            # start_time = time.time()  # Record the start time for this frame
            total_frames += 1
            video_segments[out_frame_idx] = {
                out_obj_id: {
                    'mask_logits': out_mask_logits[i],
                    'iou': iou[i],
                    'object_score': object_score_logits[i],
                    'edge_logits': out_edge_logits[i]
                } for i, out_obj_id in enumerate(out_obj_ids)
            }

            # end_time = time.time()  # Record the end time for this frame
            # elapsed_time = end_time - start_time  # Time taken to process this frame
            # print(f"Frame {out_frame_idx}: {elapsed_time:.6f} seconds")
            # total_time += elapsed_time

        end_time = time.time()  # Record the end time for this frame
        elapsed_time = end_time - start_time  # Time taken to process this frame
        total_time += elapsed_time

        # # Calculate average FPS after processing all frames
        # print(f"Total number of frames: {total_frames}")
        # avg_fps = total_frames / total_time if total_time > 0 else float('inf')
        # print(f"Average FPS: {avg_fps:.2f}")
        # input(1111111111)


        return video_segments






















