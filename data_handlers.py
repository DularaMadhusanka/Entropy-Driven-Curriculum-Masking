import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import transforms

class OxfordPetsDataset(Dataset):
    def __init__(self, root_dir, transformations=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.transformations = transformations
        
        self.image_files = []
        self.classes = []
        self.class_to_idx = {}
        
        # 1. Quick File Scan
        if not os.path.exists(self.image_dir):
            # Fallback if images are in root
            if os.path.exists(os.path.join(root_dir, 'Abyssinian_1.jpg')): 
                self.image_dir = root_dir
            else: 
                raise FileNotFoundError(f"Image directory not found at {self.image_dir}")

        valid_classes = set()
        with os.scandir(self.image_dir) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith(('.jpg', '.jpeg', '.png')) and not entry.name.startswith('.'):
                    parts = entry.name.split("_")
                    if len(parts) >= 2:
                        class_name = "_".join(parts[:-1])
                        self.image_files.append(entry.name)
                        valid_classes.add(class_name)

        self.classes = sorted(list(valid_classes))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.image_files.sort()

    def get_label_from_filename(self, filename):
        class_name = "_".join(filename.split("_")[:-1])
        return self.class_to_idx[class_name]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_name = self.image_files[index]
        img_path = os.path.join(self.image_dir, img_name)
        
        try:
            # Load as RGB
            with open(img_path, 'rb') as f:
                img_pil = Image.open(f).convert('RGB')
            
            label = self.get_label_from_filename(img_name)
            
            if self.transformations:
                img_tensor = self.transformations(img_pil)
                dummy_prob = torch.zeros(16, dtype=torch.float32)
                return img_tensor, label, dummy_prob
            
            dummy_prob = torch.zeros(16, dtype=torch.float32)
            return img_pil, label, dummy_prob

        except Exception as e:
            # Skip corrupt images safely
            return self.__getitem__((index + 1) % len(self))

class GPUDifficultyScorer(nn.Module):
    def __init__(self, metric='gradient', patch_count=4):
        super().__init__()
        self.metric = metric
        self.patch_count = patch_count
        
        # Sobel Kernels for Gradient calculation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def get_gradient_map(self, images):
        # RGB to Grayscale
        gray = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
        
        # Convolve
        g_x = F.conv2d(gray, self.sobel_x, padding=1)
        g_y = F.conv2d(gray, self.sobel_y, padding=1)
        
        return torch.sqrt(g_x**2 + g_y**2 + 1e-8)

    def get_complexity_map(self, images):
        # Local Standard Deviation (Differentiable Entropy Approximation)
        gray = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
        
        k = 3
        avg = F.avg_pool2d(gray, k, stride=1, padding=1)
        avg_sq = F.avg_pool2d(gray**2, k, stride=1, padding=1)
        return torch.sqrt(torch.clamp(avg_sq - avg**2, min=1e-8))

    def forward(self, images):
        if self.metric == 'entropy':
            score_map = self.get_complexity_map(images)
        else:
            score_map = self.get_gradient_map(images)
            
        # Downsample to patches (e.g. 4x4)
        patches = F.adaptive_avg_pool2d(score_map, (self.patch_count, self.patch_count))
        
        # Flatten and Normalize
        B = images.size(0)
        patches_flat = patches.view(B, -1)
        return patches_flat / (patches_flat.sum(dim=1, keepdim=True) + 1e-6)

def get_oxford_loaders(dataset_path, batch_size=32, resize=224):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.RandomCrop((resize, resize)), 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Init Dataset
    ds = OxfordPetsDataset(dataset_path, transformations=train_transform)
    
    # Split
    total = len(ds)
    train_sz = int(0.8 * total)
    val_sz = int(0.1 * total)
    test_sz = total - train_sz - val_sz
    
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        ds, [train_sz, val_sz, test_sz], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader