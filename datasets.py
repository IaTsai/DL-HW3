import torch
import os
import numpy as np
import skimage.io as sio  # 用來讀取 class1.tif ~ class4.tif
from PIL import Image    # 只用來讀彩色圖片 image.tif
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from scipy.ndimage import label  # 放在開頭 split instance each class

from utils import merge_masks, mask_to_bbox


# Train/Val Dataset
class TrainValDataset(Dataset):
    def __init__(self, base_dir, transforms=None):
        self.base_dir = base_dir
        self.sample_ids = sorted(os.listdir(base_dir))  # UUID目錄名稱列表
        self.transforms = transforms

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        sample_dir = os.path.join(self.base_dir, sample_id)

        # --- 讀取原始 image.tif ---
        img_path = os.path.join(sample_dir, "image.tif")
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)  # 轉成numpy array

        mask_list = []
        labels = []

        # --- 讀取 class1.tif ~ class4.tif（可能有缺失，要保護）---
        for cls_idx in range(1, 5):
            mask_path = os.path.join(sample_dir, f"class{cls_idx}.tif")
            if os.path.exists(mask_path):
                try:
                    m = sio.imread(mask_path)
                    if m.max() > 0:  # 這個mask有有效像素
                        mask_list.append(m)
                        labels.append(cls_idx)
                except Exception as e:
                    print(f"⚠️ 讀取失敗，跳過 {mask_path}，原因：{e}")

        # --- 如果這個sample根本沒有任何mask ---
        if len(mask_list) == 0:
            masks = np.zeros((0, img.shape[0], img.shape[1]), dtype=np.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            masks = []
            boxes = []
            labels_out = []

            for mask_np, cls_idx in zip(mask_list, labels):
                labeled_mask, num = label(mask_np > 0)  # 對每張 mask 做連通元件分析

                for instance_id in range(1, num + 1):
                    instance_mask = (
                        labeled_mask == instance_id
                    ).astype(np.uint8)
                    bbox = mask_to_bbox(instance_mask)
                    x_min, y_min, x_max, y_max = bbox

                    if x_max > x_min and y_max > y_min:
                        masks.append(instance_mask)
                        boxes.append(bbox)
                        labels_out.append(cls_idx)  # 保留 class label
                    else:
                        # print(f"忽略無效 bbox: {bbox}")
                        if x_max <= x_min or y_max <= y_min:
                            continue  # 無效的 mask，跳過即可，不印太多 log

            # Trans to Tensor
            masks = np.stack(masks, axis=0)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels_out, dtype=torch.int64)

        img = F.to_tensor(img)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target


# Test Dataset
class TestDataset(Dataset):
    def __init__(self, base_dir, transforms=None):
        self.base_dir = base_dir
        self.img_list = sorted(os.listdir(base_dir))  # test_dataset name
        self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.base_dir, img_name)

        # image of test data is RGB graph ---
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        img = F.to_tensor(img)

        if self.transforms:
            img = self.transforms(img)

        return img, img_name


# DataAugument
class TransformWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, idx):
        img, target = self.base_dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.base_dataset)
