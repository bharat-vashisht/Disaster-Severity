"""
train_mobilenet.py
Trains the MobileNetV2 disaster severity classifier on AIDERv2.
Mirrors train.py exactly — same data, same epochs, same LR —
so results are directly comparable to ResNet18.

Run from project root:
    python src/train_mobilenet.py
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model_mobilenet import get_mobilenet, count_trainable_params

NUM_EPOCHS      = 8
LEARNING_RATE   = 1e-3
MODEL_SAVE_PATH = "models/mobilenet_best.pth"


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted  = outputs.max(1)
        correct      += predicted.eq(labels).sum().item()
        total        += labels.size(0)

        if (batch_idx + 1) % 50 == 0:
            print(f"    batch {batch_idx+1}/{len(loader)} "
                  f"- running loss: {running_loss/total:.4f}")

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs        = model(images)
            loss           = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted  = outputs.max(1)
            correct      += predicted.eq(labels).sum().item()
            total        += labels.size(0)

    return running_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    print(f"Classes: {class_names}")

    model = get_mobilenet(num_classes=len(class_names), freeze_backbone=True)
    model = model.to(device)
    trainable, total = count_trainable_params(model)
    print(f"MobileNetV2 — Trainable: {trainable:,} / {total:,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    best_val_loss  = float("inf")
    overall_start  = time.time()

    print(f"Starting training for {NUM_EPOCHS} epochs...\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        print(f"Epoch {epoch}/{NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(
            model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start
        print(f"  Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"  Val loss:   {val_loss:.4f} | Val acc:   {val_acc:.4f}")
        print(f"  Epoch time: {epoch_time/60:.1f} min\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names":      class_names,
                "epoch":            epoch,
                "val_acc":          val_acc,
                "architecture":     "MobileNetV2",
            }, MODEL_SAVE_PATH)
            print(f"  -> Saved best model (val_loss={val_loss:.4f})\n")

    total_time = time.time() - overall_start
    print(f"Training complete in {total_time/60:.1f} minutes.")

    print("\nEvaluating on test set...")
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"MobileNetV2 — Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")
    print(f"\nCompare with ResNet18 test accuracy: 0.9710")
    print(f"Winner: {'MobileNetV2' if test_acc > 0.9710 else 'ResNet18'} by accuracy")


if __name__ == "__main__":
    main()
