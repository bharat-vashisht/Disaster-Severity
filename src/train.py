"""
train.py
Trains the ResNet18 disaster severity classifier on the AIDERv2 dataset.
Saves the best model (lowest validation loss) to models/best_model.pth.

Run from project root:
    python src/train.py
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import get_model, count_trainable_params

# ---- Config ----
NUM_EPOCHS = 8          # keep modest since CPU-only; bump up later if time allows
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = "models/best_model.pth"


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        # Print progress every 50 batches so you can see it's alive
        if (batch_idx + 1) % 50 == 0:
            print(f"    batch {batch_idx + 1}/{len(loader)} "
                  f"- running loss: {running_loss / total:.4f}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Data ----
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    print(f"Classes: {class_names}")
    print(f"Train images: {len(train_loader.dataset)} | Val images: {len(val_loader.dataset)}")

    # ---- Model ----
    model = get_model(num_classes=len(class_names), freeze_backbone=True)
    model = model.to(device)
    trainable, total = count_trainable_params(model)
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    # ---- Loss & Optimizer ----
    criterion = nn.CrossEntropyLoss()
    # Only optimize parameters that require gradients (the unfrozen ones)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LEARNING_RATE)

    best_val_loss = float("inf")

    print(f"\nStarting training for {NUM_EPOCHS} epochs...\n")
    overall_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        print(f"Epoch {epoch}/{NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start
        print(f"  Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"  Val loss:   {val_loss:.4f} | Val acc:   {val_acc:.4f}")
        print(f"  Epoch time: {epoch_time/60:.1f} min\n")

        # Save the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "epoch": epoch,
                "val_acc": val_acc,
            }, MODEL_SAVE_PATH)
            print(f"  -> Saved new best model (val_loss={val_loss:.4f}) to {MODEL_SAVE_PATH}\n")

    total_time = time.time() - overall_start
    print(f"Training complete in {total_time/60:.1f} minutes.")
    print(f"Best model saved to: {MODEL_SAVE_PATH}")

    # ---- Final test evaluation using the best saved model ----
    print("\nEvaluating best model on test set...")
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()