import os
import json
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from datasets import TestDataset
from pycocotools import mask as mask_utils
import numpy as np
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models import resnet101, resnet50, resnext50_32x4d
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
from collections import Counter


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


# Define SE module
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
            print(f"⚠️ Skip non-sequential module: {name}")


# Patch SE for models
def patch_se(model):
    print("Applying SE...")
    for name, module in model.backbone.body.named_children():
        print(f"[DEBUG] SE target: {name}, Type: {type(module).__name__}")
        if isinstance(module, nn.Sequential):
            for i, block in enumerate(module):
                if hasattr(block, "conv3"):
                    channels = block.conv3.out_channels
                    block.add_module("se", SELayer(channels))
                    print(f"SE added to {name}[{i}]")
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
        efficientnet.feature_info[i]['num_chs'] for i in out_indices
        ]

    backbone = TimmBackboneWithFPN(
        efficientnet,
        in_channels_list, out_channels
        )
    return backbone


def get_model(
    backbone_name="maskrcnn_resnet50_fpn_v2",
    pretrained=True,
    num_classes=5,
    box_score_thresh=0.001
):
    """
    Support backbone_name:
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
            backbone_raw,
            return_layers=return_layers)
        backbone = BackboneWithFPN(
            body,
            return_layers=return_layers,   # For Debug
            in_channels_list=in_channels_list,
            out_channels=out_channels
        )

        model = MaskRCNN(backbone, num_classes=num_classes)

    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.score_thresh = box_score_thresh

    return model


# --- 1. Env setting ---
device = (
    torch.device('cuda')
    if torch.cuda.is_available()
    else torch.device('cpu')
)

# --- 2. Path setting ---
TEST_DIR = "/mnt/sdb1/ia313553058/maskRCNN/DL-HW3/data/test_release"
TEST_JSON = "/mnt/sdb1/ia313553058/maskRCNN/DL-HW3/test_image_name_to_ids.json"
# BEST_MODEL_PATH = "best_model.pth"
BEST_MODEL_PATH = (
                  "./saved_models/ReduceLR_50V2-cbam/"
                  "best_model_epoch91_mAP0.2856.pth"
)
SAVE_RESULTS_PATH = "test-results.json"

# --- 3. load Test Dataset ---
test_dataset = TestDataset(TEST_DIR)
test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=lambda x: tuple(zip(*x))
    )


# --- 4. Load model ---

model = get_model(
    # or 'resnet101', 'resnext50_32x4d', 'maskrcnn_resnet50_fpn_v2'
    backbone_name="maskrcnn_resnet50_fpn_v2",
    pretrained=True,
    num_classes=5,
    box_score_thresh=0.001
)

patch_cbam(model)  # patch attention
# patch_se(model)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
model = model.to(device)
model.roi_heads.score_thresh = 0.001
model.eval()
# --- 5. Load test image name
with open(TEST_JSON, "r") as f:
    image_name_to_id = {item["file_name"]: item["id"] for item in json.load(f)}

# --- 6. Start Eval ---
results = []
threshold_score = 0.001  # Same as train threshold_score

for images, img_names in test_loader:
    images = list(img.to(device) for img in images)  # move to device

    with torch.no_grad():
        try:
            outputs = model(images)
        except Exception as e:
            print(f"Error on image {img_names[0]}: {e}")
            continue

    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

    for output, img_name in zip(outputs, img_names):
        image_id = image_name_to_id[img_name]
        boxes = output["boxes"]
        scores = output["scores"]
        labels = output["labels"]
        masks = output["masks"]

        for box, score, label, mask in zip(boxes, scores, labels, masks):
            if score < threshold_score:
                continue

            # bbox:trans to [x_min, y_min, width, height]
            box = box.numpy()
            x_min, y_min, x_max, y_max = box
            bbox = [
                float(x_min),
                float(y_min),
                float(x_max - x_min),
                float(y_max - y_min)
                ]

            #  mask：Binarization + trans to RLE format
            mask = mask[0].numpy()
            binary_mask = (mask > 0.5).astype('uint8')

            rle = mask_utils.encode(np.asfortranarray(binary_mask))
            rle["counts"] = rle["counts"].decode("utf-8")

            # assemblr dict
            result = {
                "image_id": image_id,
                "bbox": bbox,
                "score": float(score),
                "category_id": int(label),  # 1~4類
                "segmentation": rle,
            }

            results.append(result)

print(f"{len(results)} instance predicts")

invalid_cats = [r for r in results if r['category_id'] not in [1, 2, 3, 4]]
if len(invalid_cats) > 0:
    print(f"Found {len(invalid_cats)} num category_id not in [1,2,3,4]！")
    print(f"Example：{invalid_cats[:3]}")
else:
    print("All category_id All legal (1~4)")

empty_boxes = [r for r in results if r['bbox'][2] <= 1 or r['bbox'][3] <= 1]
if empty_boxes:
    print(f"{len(empty_boxes)} num bbox W or H pretty small（<=1）,may a error")

img_id_counts = Counter([r["image_id"] for r in results])
missing_imgs = [
    img_id for img_id in image_name_to_id.values()
    if img_id not in img_id_counts
    ]
print(f"{len(missing_imgs)} num pic predict is null（No instance）")
if missing_imgs:
    print(f"null predict pic image_id：{missing_imgs[:3]}")

with open(SAVE_RESULTS_PATH, "w") as f:
    json.dump(results, f)

print(f"OK：{SAVE_RESULTS_PATH}！We can uplaod result")
