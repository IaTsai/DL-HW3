import os
import random
import json
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from datasets import TrainValDataset
from transforms import get_train_transform, get_val_transform
from tqdm import tqdm

from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models import resnet101, resnet50, resnext50_32x4d
import numpy as np
from torch.cuda.amp import autocast, GradScaler  # AMP
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.backbone_utils import BackboneWithFPN


from efficientnet_pytorch import EfficientNet
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from collections import OrderedDict  # efficientnet_b0
import timm
import torch.nn as nn
from torchvision.ops import FeaturePyramidNetwork
from torch.optim.lr_scheduler import LambdaLR
from transforms import get_train_transform, get_val_transform


# Define SE module
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(
            self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x))
        )


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


# Define CBAM module
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x)) * x


# Patch CBAM for models
def patch_cbam(model):
    print("🔧 Applying CBAM...")
    for name, module in model.backbone.body.named_children():
        print(f"[DEBUG] CBAM target: {name}, Type: {type(module).__name__}")
        if isinstance(module, nn.Sequential):
            for i, block in enumerate(module):
                if hasattr(block, "conv3"):
                    channels = block.conv3.out_channels
                    block.add_module("cbam", CBAM(channels))
                    print(f"CBAM added to {name}[{i}]")
        else:
            print(f"Skip non-sequential module: {name}")


# Patch SE for models
def patch_se(model):
    print("🔧 Applying SE...")
    for name, module in model.backbone.body.named_children():
        print(f"[DEBUG] SE target: {name}, Type: {type(module).__name__}")
        if isinstance(module, nn.Sequential):
            for i, block in enumerate(module):
                if hasattr(block, "conv3"):
                    channels = block.conv3.out_channels
                    block.add_module("se", SELayer(channels))
                    print(f"✅ SE added to {name}[{i}]")
        else:
            print(f"Skip non-sequential module: {name}")


# efficientnetb0/efficientnetb1
class TimmBackboneWithFPN(nn.Module):
    def __init__(self, backbone, in_channels_list, out_channels):
        super().__init__()
        self.backbone = backbone
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool()
        )
        self.out_channels = out_channels

    def forward(self, x):
        features = self.backbone(x)  # list of feature maps
        out = {str(i): feat for i, feat in enumerate(features)}
        return self.fpn(out)


def create_efficientnet_fpn(version='b0', out_channels=256):
    model_name = f'tf_efficientnet_{version}'
    out_indices = (1, 2, 3, 4)  # C2 ~ C5
    efficientnet = timm.create_model(
        model_name,
        pretrained=True,
        features_only=True,
        out_indices=out_indices
    )

    in_channels_list = [
        efficientnet.feature_info[i]['num_chs']
        for i in out_indices
    ]

    backbone = TimmBackboneWithFPN(
        efficientnet, in_channels_list, out_channels
        )
    return backbone


def get_model(
    backbone_name="maskrcnn_resnet50_fpn_v2",
    pretrained=True,
    num_classes=5,
    box_score_thresh=0.001
):
    """
    Suppurt backbone_name:
        - 'maskrcnn_resnet50_fpn_v2'
        - 'resnet50'
        - 'resnet101'
        - 'resnext50_32x4d'
        - 'efficientnet_b0'
        - 'efficientnet_b1'
    """

    if backbone_name == 'maskrcnn_resnet50_fpn_v2':
        model = maskrcnn_resnet50_fpn_v2(
            weights="DEFAULT" if pretrained else None
        )

    # --- EfficientNet ---
    elif backbone_name in ['efficientnet_b0', 'efficientnet_b1']:
        version = backbone_name.split('_')[-1]  # 'b0' or 'b1'
        backbone = create_efficientnet_fpn(version)
        model = MaskRCNN(backbone, num_classes=num_classes)

    # --- ResNet family ---
    elif backbone_name in ['resnet50', 'resnet101', 'resnext50_32x4d']:
        if backbone_name == 'resnet50':
            backbone_raw = resnet50(weights="DEFAULT" if pretrained else None)
        elif backbone_name == 'resnet101':
            backbone_raw = resnet101(weights="DEFAULT" if pretrained else None)
        elif backbone_name == 'resnext50_32x4d':
            backbone_raw = resnext50_32x4d(
                weights="DEFAULT" if pretrained else None
            )

        return_layers = {
            'layer1': '0',
            'layer2': '1',
            'layer3': '2',
            'layer4': '3',
        }
        in_channels_list = [256, 512, 1024, 2048]
        out_channels = 256

        body = IntermediateLayerGetter(
            backbone_raw, return_layers=return_layers
            )
        backbone = BackboneWithFPN(
            body,
            return_layers=return_layers,
            in_channels_list=in_channels_list,
            out_channels=out_channels
        )

        model = MaskRCNN(backbone, num_classes=num_classes)

    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    # --- replace box predictor and confidence threshold ---
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.score_thresh = box_score_thresh

    return model


# --- 1. Environment Setup ---
seed = 42
torch.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)

# --- 2. Parameters ---
BASE_PATH = "/mnt/sdb1/ia313553058/maskRCNN/DL-HW3/data/train"
GT_JSON_PATH = "train_gt.json"
BATCH_SIZE = 4
NUM_EPOCHS = 400

LR = 0.001
BEST_MAP = 0.0
# Warmup+ReduceLRonPlateau decay+Early stop
initial_lr = 1e-4
base_lr = 1e-3
patience = 20
lr_decay_factor = 0.05
max_lr_decay = 4
no_improve_count = 0
lr_decay_count = 0

SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 3. Device Setup ---
device = (
    torch.device('cuda')
    if torch.cuda.is_available()
    else torch.device('cpu')
)

# --- 4. Dataset and DataLoader ---
full_dataset = TrainValDataset(BASE_PATH)
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset_raw, val_dataset_raw = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(seed)
    )


def export_val_gt_json(
    val_dataset_raw, full_gt_path="train_gt.json",
    save_path="val_gt.json"
):
    val_ids = [int(target["image_id"].item()) for _, target in val_dataset_raw]

    with open(full_gt_path) as f:
        full_json = json.load(f)

    val_json = {
        "images": [img for img in full_json["images"] if img["id"] in val_ids],
        "annotations": [
            anno
            for anno in full_json["annotations"]
            if anno["image_id"] in val_ids
        ],
        "categories": full_json["categories"]
    }

    with open(save_path, "w") as f:
        json.dump(val_json, f, indent=2)

    print(f"val_gt.json create!, "
          f"include {len(val_json['images'])} pics, "
          f"{len(val_json['annotations'])} annotations")


export_val_gt_json(val_dataset_raw)

# --- chaeck train_dataset_raw and train_gt.json same or not ---
coco_gt = COCO("val_gt.json")  # val subset's ground truth

raw_image_ids = set(
    [int(sample["image_id"].item()) for _, sample in train_dataset_raw]
    )
json_image_ids = set(coco_gt.getImgIds())

common_ids = raw_image_ids & json_image_ids
missing_in_json = raw_image_ids - json_image_ids
missing_in_dataset = json_image_ids - raw_image_ids

print(f"Train Dataset image count: {len(raw_image_ids)}")
print(f"COCO GT JSON image count: {len(json_image_ids)}")
print(f"Overlapping image_ids: {len(common_ids)}")
if missing_in_json:
    print(f"Missing in train_gt.json: {sorted(missing_in_json)}")
if missing_in_dataset:
    print(f"Missing in train_dataset_raw: {sorted(missing_in_dataset)}")

# dataAugument: transform


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
# DataAugument OFF CHCH
# train_dataset = TransformWrapper(train_dataset_raw, transform=None)
# val_dataset = TransformWrapper(val_dataset_raw, transform=None)


# DataAugument ON
train_dataset = TransformWrapper(train_dataset_raw, get_train_transform())
val_dataset = TransformWrapper(val_dataset_raw, get_val_transform())

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True, num_workers=2,
    collate_fn=lambda x: tuple(zip(*x))
    )
val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    collate_fn=lambda x: tuple(zip(*x))
    )

# --- 5. Model -----
model = get_model(
    # 'resnet50', 'resnet101', 'resnext50_32x4d',
    # 'maskrcnn_resnet50_fpn_v2', 'efficientnet_b0', 'efficientnet_b1'
    backbone_name="resnext50_32x4d",
    pretrained=True,
    num_classes=5,
    box_score_thresh=0.001
)

patch_se(model)
# patch_cbam(model)
model.to(device)


params = [p for p in model.parameters() if p.requires_grad]

# Warmup+ReduceLRonPlateau decay+Early stop
optimizer = torch.optim.AdamW(params, lr=initial_lr, weight_decay=1e-4)


def lr_lambda(epoch):
    return 0.1 if epoch < 3 else 1.0


lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

# --- 7. Training Loop ---
for epoch in range(NUM_EPOCHS):
    print(f"\n[Epoch {epoch+1}/{NUM_EPOCHS}]")

    # --- Train ---
    model.train()
    running_loss = 0.0

    scaler = GradScaler()  # AMP
    for batch_idx, (images, targets) in enumerate(
        tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
    ):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        try:
            with autocast():  # AMP ON
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
        except Exception as e:
            print(f"[ERROR] Skipping batch due to: {e}")
            continue

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        # empty GPU mem
        torch.cuda.empty_cache()
        running_loss += losses.item()

    lr_scheduler.step()
    print(f"Training Loss: {running_loss:.4f}")

    print(
        f"[Epoch {epoch+1}] CUDA max memory allocated: "
        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
        )
    torch.cuda.reset_peak_memory_stats()
    # --- Eval ---
    model.eval()
    coco_dt_list = []

    first_batch = True

    with torch.no_grad():
        for images, targets in tqdm(
            val_loader, desc=f"Evaluating Epoch {epoch+1}"
        ):
            images = list(img.to(device) for img in images)
            outputs = model(images)

            # Only debug first batch
            if first_batch:   # only print once
                for output in outputs:
                    print("sss[DEBUG] output keys:", output.keys())
                    print("sss[DEBUG] output['boxes']:", output['boxes'])
                    print("sss[DEBUG] output['scores']:", output['scores'])
                    print("sss[DEBUG] output['labels']:", output['labels'])
                first_batch = False

            # collect all predict result togather
            for img, output, target in zip(images, outputs, targets):
                image_id = int(target["image_id"].cpu().item())

                if image_id not in coco_gt.imgs:
                    print(
                        f"[WARNING] image_id {image_id} not in coco_gt!"
                        )
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                print(
                    f"[DEBUG] Validation - image_id: {image_id}, "
                    f"output_boxes: {boxes.shape}, "
                    f"scores: {scores.shape}"
                    )

                for box, score, label in zip(boxes, scores, labels):
                    # while Eval only keep >=0.05 add into coco_dt_lis
                    if score >= 0.05:
                        coco_dt_list.append({
                            "image_id": image_id,
                            "category_id": int(label),
                            "bbox": [
                                float(box[0]),
                                float(box[1]),
                                float(box[2] - box[0]),
                                float(box[3] - box[1])
                            ],
                            "score": float(score)
                            })
            # empty GPU mem
            torch.cuda.empty_cache()
    # --- COCO API to Eval---
    print(f"[DEBUG] coco_dt_list length: {len(coco_dt_list)}")
    if len(coco_dt_list) > 0:
        coco_dt = coco_gt.loadRes(coco_dt_list)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        if len(coco_dt_list) == 0:
            print("❗ No predictions to evaluate. Skipping COCOeval.")
            continue

        coco_dt_list = [
            d for d in coco_dt_list if np.isfinite(d['bbox']).all()
            ]
        if len(coco_dt_list) == 0:
            print("All bbox invalid. Skipping eval.")
            continue
        # Begin coco Eval
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        mAP = coco_eval.stats[0]

        print(f"Validation mAP: {mAP:.4f}")
        print(
            f"Validation AP50: {coco_eval.stats[1]:.4f}, "
            f"AP75: {coco_eval.stats[2]:.4f}"
            )

        if mAP > BEST_MAP:
            BEST_MAP = mAP
            model_save_path = os.path.join(
                SAVE_DIR, f"best_model_epoch{epoch+1}_mAP{mAP:.4f}.pth"
                )
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model to {model_save_path}")
            no_improve_count = 0
        else:
            no_improve_count += 1
        # lower lr or early stop
        if no_improve_count >= patience:
            if lr_decay_count < max_lr_decay:
                new_lr = optimizer.param_groups[0]['lr'] * lr_decay_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                lr_decay_count += 1
                no_improve_count = 0
                print(
                    f"mAP not improve {patience} times "
                    f"→ Lower LR to {new_lr}"
                    )
            else:  # Early stop
                print("mAP not improve > patience times, trigger Early stop")
                break
    else:
        print("Warning: No detection results after evaluation.")
print(
    f"[Memory] Max allocated: "
    f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
    )

print("\nTraining completed.")
print(f"Best validation mAP achieved: {BEST_MAP:.4f}")

print(
    f"[Memory] Max allocated: "
    f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
    )
# Debug for avoid coredump
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
