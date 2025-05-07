import numpy as np
import skimage.io as sio
from pycocotools import mask as mask_utils


def merge_masks(masks, labels):
    """
    Merge multiple class-specific masks into a single instance mask,
    and return the corresponding class label for each instance.

    Args:
        masks: list of ndarray, each is a mask for one class (shape: H x W)
        labels: list of int, class label for each mask

    Returns:
        merged_mask: ndarray, combined instance mask
        (each instance marked with unique ID)
        instance_labels: list of int, class label for each instance
    """
    instance_id = 1
    merged_mask = np.zeros_like(masks[0], dtype=np.uint16)
    instance_labels = []

    for m, label in zip(masks, labels):
        unique_instances = np.unique(m)
        for uid in unique_instances:
            if uid == 0:
                continue  # 0 is background
            merged_mask[m == uid] = instance_id
            instance_labels.append(label)
            instance_id += 1

    return merged_mask, instance_labels


def mask_to_bbox(mask):
    """
    Compute bounding box from a binary mask.

    Args:
        mask: (H, W) ndarray, binary mask

    Returns:
        list of 4 floats: [x_min, y_min, x_max, y_max]
    """
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return [0, 0, 0, 0]
    x_min = np.min(x_indices)
    x_max = np.max(x_indices)
    y_min = np.min(y_indices)
    y_max = np.max(y_indices)
    return [x_min, y_min, x_max, y_max]


def decode_maskobj(mask_obj):
    return mask_utils.decode(mask_obj)


def encode_mask(binary_mask):
    arr = np.asfortranarray(binary_mask).astype(np.uint8)
    rle = mask_utils.encode(arr)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def read_maskfile(filepath):
    mask_array = sio.imread(filepath)
    return mask_array
