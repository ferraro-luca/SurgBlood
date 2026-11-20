import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF
import numpy as np


class CholeBPTrain(Dataset):
    def __init__(self, data_path, img_size, frames_length, overlap, mode='train', augment=False,
                 brightness=0.15, contrast=0.15, saturation=0.15,
                 color_p=0.8, blur_p=0.2, blur_sigma=(0.1, 1.0), blur_kernel=5):
        self.data_path = data_path
        self.mode = mode
        self.img_size = img_size
        self.frames_length = frames_length
        self.overlap = overlap

        # Clip-consistent photometric augmentation (v5). Disabled by default so that
        # older training scripts (v1-v4) keep their original behaviour.
        self.augment = augment
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.color_p = color_p
        self.blur_p = blur_p
        self.blur_sigma = blur_sigma
        self.blur_kernel = blur_kernel

        # Initialize transforms
        self.resize = T.Resize((self.img_size, self.img_size))
        self.to_tensor_norm = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transforms = T.Compose([self.resize, self.to_tensor_norm])

        self.sample_lists = self.generate_clips(data_path, mode)

    def __len__(self):
        return len(self.sample_lists)

    def __getitem__(self, index):
        clip_info = self.sample_lists[index]
        name = clip_info['name']
        frame_list = clip_info['frames']  # List of frame names in the clip

        video_path = os.path.join(self.data_path, self.mode, 'videos-image', name)
        mask_path = os.path.join(self.data_path, self.mode, 'videos-mask', name)
        point_path = os.path.join(self.data_path, self.mode, 'videos-point', name)

        edge_path = os.path.join(self.data_path, self.mode, 'videos-mask-edge', name)

        images = []
        masks = []
        points = []
        edges = []

        # Sample the augmentation parameters ONCE per clip, so every frame gets the
        # exact same photometric perturbation (temporal consistency).
        aug_params = self.sample_clip_aug_params() if self.augment else None

        for frame_name in frame_list:
            img = Image.open(os.path.join(video_path, f'{frame_name}.jpg')).convert('RGB')
            mask = Image.open(os.path.join(mask_path, f'{frame_name}.jpg')).convert('L')
            edge = Image.open(os.path.join(edge_path, f'{frame_name}.jpg')).convert('L')

            # Apply transformations (photometric only on the RGB image: masks, edges and
            # bleeding-point coordinates are left untouched)
            img = self.resize(img)
            if aug_params is not None:
                img = self.apply_clip_aug(img, aug_params)
            img = self.to_tensor_norm(img)
            mask = T.ToTensor()(T.Resize((self.img_size, self.img_size))(mask))
            edge = T.ToTensor()(T.Resize((self.img_size, self.img_size))(edge))

            point = self.load_point(point_path, frame_name)

            images.append(img)  # Append image tensor
            masks.append(mask)  # Append mask tensor
            edges.append(edge)  # Append mask tensor

            points.append(torch.tensor(point).float())  # Append point tensor

        video_meta_dict = {'filename': name, 'frames': frame_list}
        return {
            'image': torch.stack(images),  # Stack list of tensors into one tensor
            'mask': torch.stack(masks),    # Stack list of tensors into one tensor
            'edge': torch.stack(edges),    # Stack list of tensors into one tensor
            'point': torch.stack(points),  # Stack point tensors
            'video_meta_dict': video_meta_dict,
        }

    def sample_clip_aug_params(self):
        """Draw one set of photometric parameters for a whole clip.

        torch's RNG is used (instead of numpy) because DataLoader re-seeds it
        correctly in every worker.
        """
        def uniform(low, high):
            return float(torch.empty(1).uniform_(low, high))

        params = {'color': None, 'blur_sigma': None}
        if float(torch.rand(1)) < self.color_p:
            params['color'] = {
                'brightness': uniform(1 - self.brightness, 1 + self.brightness),
                'contrast': uniform(1 - self.contrast, 1 + self.contrast),
                'saturation': uniform(1 - self.saturation, 1 + self.saturation),
            }
        if float(torch.rand(1)) < self.blur_p:
            params['blur_sigma'] = uniform(*self.blur_sigma)
        return params

    def apply_clip_aug(self, img, params):
        """Apply the same (pre-sampled) photometric perturbation to a PIL frame."""
        color = params['color']
        if color is not None:
            img = TF.adjust_brightness(img, color['brightness'])
            img = TF.adjust_contrast(img, color['contrast'])
            img = TF.adjust_saturation(img, color['saturation'])
        if params['blur_sigma'] is not None:
            img = TF.gaussian_blur(img, kernel_size=self.blur_kernel, sigma=params['blur_sigma'])
        return img

    def generate_clips(self, data_path, mode):
        sample_lists = []
        video_dir = os.path.join(data_path, mode, 'videos-image')
        video_list = sorted(os.listdir(video_dir))  # Sort video folders

        for name in video_list:
            video_path = os.path.join(video_dir, name)
            frame_names = sorted(f.split('.')[0] for f in os.listdir(video_path) if f.endswith('.jpg'))

            start_idx = 0
            while start_idx < len(frame_names)-self.frames_length+1:# to viod the last clip is not enough
                end_idx = min(start_idx + self.frames_length, len(frame_names))
                frame_list = frame_names[start_idx:end_idx]
                sample_lists.append({
                    'name': name,
                    'frames': frame_list
                })
                start_idx += (self.frames_length - self.overlap)

        return sample_lists

    def load_point(self, point_path, frame_name):
        """Load the point information for a given frame."""
        point_file_path = os.path.join(point_path, f'{frame_name}.txt')
        point = (0, 0)  # Default point if not found

        # Se il file del punto esiste proviamo a leggerlo, altrimenti restituisce (0.0, 0.0) in silenzio
        if os.path.exists(point_file_path):
            try:
                with open(point_file_path, 'r') as f:
                    line = f.readline().strip()
                    if line:
                        x, y = map(float, line.split())
                        return (x, y)
            except Exception:
                pass

        return point


# # Usage
# data_path = '/home/pjl307/ZZJ/CholeBleedingPoint/CholeBP51-Videos'
# train_dataset = CholeBPTrain( data_path,clip_length,overlap, mode='train'):

