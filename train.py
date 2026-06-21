import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from configs import config


def plot_training_diagnostics(history):
    """
    Plots a 2-panel dashboard showing training/validation convergence
    and a side-by-side comparison of the 0.50 vs 0.25 probability thresholds.

    Parameters:
    - history: A dictionary containing lists of metrics collected across epochs.
    """
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    # Initialize a clean, wide dual-panel figure matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ----------------------------------------------------
    # PANEL 1: LOSS CONVERGENCE CURVES
    # ----------------------------------------------------
    ax1.plot(epochs, history["train_loss"], label="Train Loss (Stabilized LPBF)", color="tab:blue", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss (Stabilized LPBF)", color="tab:orange", linewidth=2,
             linestyle="--")

    # Highlight learning rate reductions if they occurred
    for i in range(1, len(history["lr"])):
        if history["lr"][i] < history["lr"][i - 1]:
            ax1.axvline(x=i + 1, color="red", linestyle=":", alpha=0.7)
            ax1.text(i + 1, ax1.get_ylim()[1] * 0.9, f"LR Drop\nto {history['lr'][i]:.6f}",
                     color="red", fontsize=9, ha="center", bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax1.set_title("Model Optimization & Objective Convergence", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Training Epoch Iterations", fontsize=10)
    ax1.set_ylabel("Stabilized Loss Magnitude", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    # ----------------------------------------------------
    # PANEL 2: PRECISION, RECALL, & F1 THRESHOLD TRAJECTORIES
    # ----------------------------------------------------
    # 0.50 Static Threshold Lines
    ax2.plot(epochs, history["f1_050"], label="F1-Score (0.50 Threshold)", color="tab:green", alpha=0.4, linestyle=":")
    ax2.plot(epochs, history["rec_050"], label="Recall (0.50 Threshold)", color="tab:orange", alpha=0.4, linestyle=":")

    # 0.25 Optimized Sensitive Threshold Lines (Bold to focus attention)
    ax2.plot(epochs, history["f1_025"], label="F1-Score (0.25 Threshold)", color="tab:green", linewidth=2.5, marker="D",
             markevery=5)
    ax2.plot(epochs, history["rec_025"], label="Recall (0.25 Threshold)", color="tab:orange", linewidth=2, marker="s",
             markevery=5)
    ax2.plot(epochs, history["prec_025"], label="Precision (0.25 Threshold)", color="tab:blue", linewidth=2, marker="o",
             markevery=5)

    ax2.set_title("Micro-Defect Resolution Performance Trajectory", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Training Epoch Iterations", fontsize=10)
    ax2.set_ylabel("Evaluation Performance Score (0.0 - 1.0)", fontsize=10)
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower left", fontsize=9, ncol=2)

    plt.tight_layout()
    plt.show()

# ==========================================
# 1. ARCHITECTURE & LOG-LOG DICE LOSS
# ==========================================
def get_unet_efficientnet(encoder_name="efficientnet-b3", pretrained="imagenet"):
    """Creates a U-Net++ architecture with an EfficientNet encoder for 1-channel grayscale input."""
    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,
        encoder_weights=pretrained,
        in_channels=1,
        classes=1,
        activation=None  # Raw logits passed straight to loss functions
    )
    return model


class StabilizedLPBFLoss(nn.Module):
    """
    Optimized for microscopic features.
    Uses log-loss Dice to smooth out small-scale overlaps and Asymmetric Focal Loss.
    """

    def __init__(self):
        super(StabilizedLPBFLoss, self).__init__()
        # alpha=0.40 prevents the model from hallucinating defects everywhere
        self.focal = smp.losses.FocalLoss(mode='binary', alpha=0.40, gamma=2.0)
        # log_loss=True scales better when defect structures are only a few pixels wide
        self.dice = smp.losses.DiceLoss(mode='binary', log_loss=True)

    def forward(self, logits, targets):
        return 0.6 * self.focal(logits, targets) + 0.4 * self.dice(logits, targets)


# ==========================================
# 2. DATA PIPELINE WITH OVERSAMPLING RETRY
# ==========================================
class LPBFUnetDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.masks_dir = self.data_dir / "masks"
        self.transforms = transforms

        self.image_filenames = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.tif')])
        self.mask_filenames = sorted([f for f in os.listdir(self.masks_dir) if f.endswith('.png')])

        assert len(self.image_filenames) == len(self.mask_filenames), \
            f"Mismatch: Found {len(self.image_filenames)} images but {len(self.mask_filenames)} masks."

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = self.images_dir / self.image_filenames[idx]
        mask_path = self.masks_dir / self.mask_filenames[idx]

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

        if image is None or mask is None:
            raise FileNotFoundError(f"Error loading data at index {idx}: {img_path}")

        image = np.expand_dims(image, axis=-1)

        if self.transforms:
            # Step 1: Initialize with a baseline crop transformation
            augmented = self.transforms(image=image, mask=mask)
            image_out = augmented['image']
            mask_out = augmented['mask']

            # Step 2: Retry loop for positive class oversampling
            # If the crop didn't capture a defect, search up to 2 more times
            if not (torch.any(mask_out == 1) if isinstance(mask_out, torch.Tensor) else np.any(mask_out == 1)):
                for _ in range(2):
                    alt_augmented = self.transforms(image=image, mask=mask)
                    alt_mask = alt_augmented['mask']

                    has_defect = torch.any(alt_mask == 1) if isinstance(alt_mask, torch.Tensor) else np.any(
                        alt_mask == 1)
                    if has_defect:
                        image_out = alt_augmented['image']
                        mask_out = alt_mask
                        break

            image = image_out
            mask = mask_out

        # Step 3: Enforce proper tensor configurations
        if isinstance(mask, torch.Tensor):
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            mask = mask.float()
        else:
            mask = torch.from_numpy(mask).float().unsqueeze(0)

        return image, mask


def get_lpbf_dataloaders(data_dir, patch_size=512, batch_size=8, num_workers=4):
    """
    Loads pre-partitioned subdirectories directly to prevent transform and data leakage.
    """
    data_path = Path(data_dir)

    train_transforms = A.Compose([
        A.RandomCrop(width=patch_size, height=patch_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # Alternative unified block using standard keyword-free range mapping
        A.RandomBrightnessContrast((-0.2, 0.2), (-0.2, 0.2), p=0.5),
        A.GaussianBlur((3, 5), p=0.3),
        A.GaussNoise(p=0.3),  # Defaults to standard image sensor white-noise profile
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])

    val_transforms = A.Compose([
        A.CenterCrop(width=patch_size, height=patch_size),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])

    # Point datasets directly to their respective pre-split subdirectories
    train_dataset = LPBFUnetDataset(data_dir=data_path / "train", transforms=train_transforms)
    val_dataset = LPBFUnetDataset(data_dir=data_path / "val", transforms=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


# ==========================================
# 3. CORE TRAIN & VALIDATION ENGINE
# ==========================================
def train_epoch(model, loader, optimizer, scaler, loss_fn, device):
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = loss_fn(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate_epoch_with_metrics(model, loader, loss_fn, device, threshold=0.5):
    model.eval()
    running_loss = 0.0
    tp_list, fp_list, fn_list, tn_list = [], [], [], []

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = loss_fn(outputs, masks)

        running_loss += loss.item() * images.size(0)

        # Calculate confusion matrix entries
        probs = torch.sigmoid(outputs)
        preds = (probs > threshold).long()
        targets = masks.long()

        tp, fp, fn, tn = smp.metrics.get_stats(preds, targets, mode='binary', threshold=threshold)
        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)
        tn_list.append(tn)

    all_tp = torch.cat(tp_list).sum(dim=0)
    all_fp = torch.cat(fp_list).sum(dim=0)
    all_fn = torch.cat(fn_list).sum(dim=0)
    all_tn = torch.cat(tn_list).sum(dim=0)

    precision = smp.metrics.precision(all_tp, all_fp, all_fn, all_tn, reduction="micro")
    recall = smp.metrics.recall(all_tp, all_fp, all_fn, all_tn, reduction="micro")
    f1_score = smp.metrics.f1_score(all_tp, all_fp, all_fn, all_tn, reduction="micro")

    return {
        "val_loss": running_loss / len(loader.dataset),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1_score": f1_score.item()
    }


# ==========================================
# 4. UNIFIED MAIN EXECUTION BLOCK
# ==========================================
def main():
    """
    Load configuration parameters
    """
    device = config.DEVICE
    print(f"Initializing optimization environment on: {device}")
    if torch.cuda.is_available():
        print(f"Target Hardware Identified: {torch.cuda.get_device_name(0)}")

    DATASET_DIR = config.UNET_DATASET_DIR
    CHECKPOINT_DIR = Path(config.CHECKPOINT_DIR)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    BATCH_SIZE = config.BATCH_SIZE
    PATCH_SIZE = config.PATCH_SIZE
    NUM_WORKERS = config.NUM_WORKERS
    LEARNING_RATE = config.LEARNING_RATE
    EPOCHS = config.NUM_EPOCHS
    ENCODER_BACKBONE = "efficientnet-b3"

    print("Binding datasets to parallel memory pipelines...")
    train_loader, val_loader = get_lpbf_dataloaders(
        data_dir=DATASET_DIR, patch_size=PATCH_SIZE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    # Initialize model
    model = get_unet_efficientnet(encoder_name=ENCODER_BACKBONE, pretrained="imagenet").to(device)
    loss_function = StabilizedLPBFLoss()

    # ----------------------------------------------------
    # PHASE 1: FREEZE ENCODER (Epochs 1-10)
    # ----------------------------------------------------
    print("Freezing pre-trained encoder weights for Phase 1 structural regularization...")
    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=1e-2  # Aggressive weight decay to fight overfitting
    )

    scaler = torch.amp.GradScaler('cuda')
    best_f1_score = -1.0
    encoder_unfrozen = False
    history = {
        "train_loss": [], "val_loss": [], "lr": [],
        "prec_050": [], "rec_050": [], "f1_050": [],
        "prec_025": [], "rec_025": [], "f1_025": []
    }

    print(f"\n--- Launching Two-Phase Execution Loop for {ENCODER_BACKBONE} ---")

    for epoch in range(1, EPOCHS + 1):
        # PHASE 2 TRIGGER: Unfreeze encoder at Epoch 11
        if epoch == 11 and not encoder_unfrozen:
            print("\n>>> PHASE 2: Unfreezing encoder weights for gentle joint fine-tuning... <<<")
            for param in model.encoder.parameters():
                param.requires_grad = True

            # Reinitialize optimizer with ALL parameters at a significantly lower learning rate
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
            encoder_unfrozen = True

        train_loss = train_epoch(model, train_loader, optimizer, scaler, loss_function, device)

        metrics_050 = validate_epoch_with_metrics(model, val_loader, loss_function, device, threshold=0.50)
        metrics_025 = validate_epoch_with_metrics(model, val_loader, loss_function, device, threshold=0.25)

        current_lr = optimizer.param_groups[0]['lr']

        print(f"\n=================== EPOCH {epoch:02d}/{EPOCHS:02d} ===================")
        print(f"Losses -> Train: {train_loss:.4f} | Val: {metrics_050['val_loss']:.4f} | LR: {current_lr:.7f}")
        print(
            f"Metrics (At 0.50 Threshold) -> Prec: {metrics_050['precision']:.4f} | Rec: {metrics_050['recall']:.4f} | F1: {metrics_050['f1_score']:.4f}")
        print(
            f"Metrics (At 0.25 Threshold) -> Prec: {metrics_025['precision']:.4f} | Rec: {metrics_025['recall']:.4f} | F1: {metrics_025['f1_score']:.4f}")

        if metrics_025["f1_score"] > best_f1_score:
            best_f1_score = metrics_025["f1_score"]
            checkpoint_path = CHECKPOINT_DIR / f"best_unetplusplus_{ENCODER_BACKBONE}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics_025,
            }, checkpoint_path)
            print(f" => Saved new optimum sensitive checkpoint.")

        # APPEND METRICS TO HISTORY MATRIX AT THE END OF EVERY EPOCH
        history["train_loss"].append(train_loss)
        history["val_loss"].append(metrics_050["val_loss"])
        history["lr"].append(current_lr)

        history["prec_050"].append(metrics_050["precision"])
        history["rec_050"].append(metrics_050["recall"])
        history["f1_050"].append(metrics_050["f1_score"])

        history["prec_025"].append(metrics_025["precision"])
        history["rec_025"].append(metrics_025["recall"])
        history["f1_025"].append(metrics_025["f1_score"])

    print("Optimization run complete. Plotting training metrics dashboard...")
    plot_training_diagnostics(history)


if __name__ == "__main__":
    main()