import torchvision.transforms as T


def get_train_transform():
    return T.Compose([
        # --- Augmentation (enhanced version) ---
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])


def get_val_transform():
    return T.Compose([
    ])
