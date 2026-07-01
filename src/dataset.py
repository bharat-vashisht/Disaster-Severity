"""
dataset.py
Loads the AIDERv2 dataset (Train/Val/Test) using torchvision.ImageFolder
and applies appropriate transforms (augmentation for train, plain resize for val/test).
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ---- Config ----
DATA_DIR = "data"          # contains Train/, Val/, Test/
IMG_SIZE = 224              # ResNet18 default input size
BATCH_SIZE = 16              # small batch size since training on CPU
NUM_WORKERS = 0              # keep 0 on Windows to avoid multiprocessing issues

# ImageNet normalization stats (required since we use a pretrained ResNet18)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transforms():
    """Returns (train_transform, eval_transform)."""

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    return train_transform, eval_transform


def get_dataloaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE):
    """
    Builds train/val/test datasets and dataloaders.
    Returns: train_loader, val_loader, test_loader, class_names
    """
    train_transform, eval_transform = get_transforms()

    train_dataset = datasets.ImageFolder(f"{data_dir}/Train", transform=train_transform)
    val_dataset   = datasets.ImageFolder(f"{data_dir}/Val",   transform=eval_transform)
    test_dataset  = datasets.ImageFolder(f"{data_dir}/Test",  transform=eval_transform)

    class_names = train_dataset.classes  # e.g. ['Earthquake', 'Fire', 'Flood', 'Normal']

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    # Quick sanity check when running this file directly:
    # python src/dataset.py
    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    print("Classes:", class_names)
    print("Train batches:", len(train_loader), "| Train images:", len(train_loader.dataset))
    print("Val batches:  ", len(val_loader),   "| Val images:  ", len(val_loader.dataset))
    print("Test batches: ", len(test_loader),  "| Test images: ", len(test_loader.dataset))

    # Pull one batch to confirm shapes are correct
    images, labels = next(iter(train_loader))
    print("Batch image shape:", images.shape)   # [batch_size, 3, 224, 224]
    print("Batch label shape:", labels.shape)    # [batch_size]