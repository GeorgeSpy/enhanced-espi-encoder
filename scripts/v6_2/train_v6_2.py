#!/usr/bin/env python3
"""
v6.2-A: Domain-Mix Training for Averaged/Noisy Robustness
Based on v6.1 architecture (1-channel, frozen backbone)

Key changes from v6.1:
- DomainStratifiedBatchSampler: 20 clean + 10 avg + 10 pseudo per batch
- Targeted augmentations: clean to noisy-like (p=0.7)
- Holdout evaluation with zero-recall gates
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import json
import sys
import os

# Add tools directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))
from asymmetric_focal_loss import create_loss_function
from domain_stratified_sampler import DomainStratifiedBatchSampler, get_domain


class NoisyAugmentation1Ch:
    """
    Lightweight augmentation to make clean samples look noisy/averaged.
    Designed for 1-channel grayscale input.
    """
    def __init__(self, p=0.7):
        self.p = p

    def __call__(self, image: np.ndarray, domain: str) -> np.ndarray:
        """
        Apply if domain=='clean' and random < p.

        Args:
            image: numpy array [H, W] (already z-scored)
            domain: 'clean', 'averaged', or 'pseudo'

        Returns:
            Augmented image (or original)
        """
        if domain != 'clean' or np.random.random() > self.p:
            return image

        # Mild Gaussian blur
        if np.random.random() < 0.8:
            sigma = np.random.uniform(0.3, 0.7)
            from scipy.ndimage import gaussian_filter
            image = gaussian_filter(image, sigma=sigma)

        # Gamma correction (flatten contrast)
        if np.random.random() < 0.6:
            gamma = np.random.uniform(0.9, 1.1)
            # Normalize to [0, 1]
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                norm = (image - img_min) / (img_max - img_min)
                norm = np.power(norm, gamma)
                image = norm * (img_max - img_min) + img_min

        # Additive Gaussian noise
        if np.random.random() < 0.7:
            noise_std = np.random.uniform(0.01, 0.03)
            image = image + np.random.randn(*image.shape) * noise_std

        # Speckle noise (multiplicative)
        if np.random.random() < 0.5:
            speckle_std = np.random.uniform(0.01, 0.02)
            image = image + image * np.random.randn(*image.shape) * speckle_std

        return image


class ESPIDatasetV62(Dataset):
    """Dataset with domain awareness for v6.2."""
    def __init__(self, csv_file, root_dir, augment=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.augment = augment

        # Add domain column if not present
        if 'domain' not in self.df.columns:
            self.df['domain'] = self.df['board'].apply(get_domain)

        print(f"Dataset loaded: {len(self.df)} samples")
        print(f"Label distribution: {self.df['label'].value_counts().sort_index().to_dict()}")
        print(f"Domain distribution: {self.df['domain'].value_counts().to_dict()}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root_dir / row['path']
        image = Image.open(img_path).convert('L')

        # Convert to numpy and apply per-image z-score
        image = np.array(image.resize((256, 256)), dtype=np.float32) / 255.0
        mu, sigma = image.mean(), image.std() + 1e-6
        image = (image - mu) / sigma

        # Apply domain-aware augmentation
        domain = row['domain']
        if self.augment is not None:
            image = self.augment(image, domain=domain)

        # Convert to tensor
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # [1, H, W]
        label = torch.tensor(row['label'], dtype=torch.long)

        return image_tensor, label


class ResNet18_V62(nn.Module):
    """Same architecture as v6.1: frozen backbone + simple head."""
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        # Load pretrained ResNet-18
        if pretrained:
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = resnet18(weights=None)

        # Modify for 1-channel input
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove final classifier
        self.backbone.fc = nn.Identity()

        # New classifier head (same as v6.1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # Freeze backbone
        self._freeze_backbone()

    def _freeze_backbone(self):
        """Freeze all backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen - only classifier will be trained")

    def forward(self, x):
        # Extract features through backbone layers
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Global average pooling
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        return self.classifier(x)


def evaluate_with_gates(model, loader, criterion, device, split_name="val"):
    """
    Evaluate with zero-recall detection.

    Returns metrics + zero_recall_classes list.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            loss = criterion(logits, target)

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    acc = 100.0 * correct / total
    avg_loss = total_loss / len(loader)

    # Per-class recall
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    recalls = {}
    zero_recall_classes = []

    for c in range(5):
        mask = all_labels == c
        if mask.sum() > 0:
            recall = (all_preds[mask] == c).mean()
            recalls[c] = recall
            if recall == 0.0:
                zero_recall_classes.append(c)
        else:
            recalls[c] = None

    print(f"\n{split_name} Results:")
    print(f"  Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")
    print(f"  Per-class Recall: {recalls}")
    if zero_recall_classes:
        print(f"  WARNING: ZERO RECALL in classes: {zero_recall_classes}")

    return {
        'loss': avg_loss,
        'acc': acc,
        'recalls': recalls,
        'zero_recall_classes': zero_recall_classes
    }


def main():
    parser = argparse.ArgumentParser(description='v6.2-A: Domain-Mix Training')
    parser.add_argument("--csv", default="labels_modes.v6.train.csv", help="Train CSV")
    parser.add_argument("--val_csv", default="labels_modes.v6.val.csv", help="Val CSV")
    parser.add_argument("--img_root", default=".", help="Image root")
    parser.add_argument("--checkpoint", default="checkpoints/espi_modes_v6_1_head.pt", help="v6.1 checkpoint to start from")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs")
    parser.add_argument("--bs", type=int, default=40, help="Batch size")
    parser.add_argument("--clean_per_batch", type=int, default=20, help="Clean samples per batch")
    parser.add_argument("--avg_per_batch", type=int, default=10, help="Averaged samples per batch")
    parser.add_argument("--pseudo_per_batch", type=int, default=10, help="Pseudo samples per batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--aug_p", type=float, default=0.7, help="Augmentation probability for clean samples")
    parser.add_argument("--loss_type", default="afl", choices=["afl", "cb", "ldam_cb", "balanced_softmax"], help="Loss type")
    parser.add_argument("--label_smoothing", type=float, default=0.05, help="Label smoothing")
    parser.add_argument("--save", default="checkpoints/espi_modes_v6_2.pt", help="Save path")
    parser.add_argument("--export_onnx", default="models/espi_modes_v6_2.onnx", help="ONNX export path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create augmentation
    augment = NoisyAugmentation1Ch(p=args.aug_p)

    # Create datasets
    train_dataset = ESPIDatasetV62(args.csv, args.img_root, augment=augment)
    val_dataset = ESPIDatasetV62(args.val_csv, args.img_root, augment=None)  # No aug on val

    # Calculate class counts for loss function
    class_counts = train_dataset.df['label'].value_counts().sort_index().values
    print(f"Class counts: {dict(enumerate(class_counts))}")

    # Load priors
    log_priors = None
    if os.path.exists("training_priors.json"):
        with open("training_priors.json", 'r') as f:
            priors_data = json.load(f)
        log_priors = [priors_data['log_prior'][str(i)] for i in range(5)]
        print(f"Loaded log_priors: {log_priors}")

    # Create domain-stratified batch sampler
    train_sampler = DomainStratifiedBatchSampler(
        train_dataset,
        batch_size=args.bs,
        clean_per_batch=args.clean_per_batch,
        avg_per_batch=args.avg_per_batch,
        pseudo_per_batch=args.pseudo_per_batch,
        shuffle=True
    )

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Model
    model = ResNet18_V62(5, pretrained=True).to(device)

    # Load v6.1 checkpoint if available
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        print("Checkpoint loaded successfully")
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint}, starting from ImageNet weights")

    # Create loss function
    criterion = create_loss_function(args.loss_type, class_counts, log_priors=log_priors)
    print(f"Using loss function: {args.loss_type}")

    # Optimizer (only for classifier parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)

    # Warmup + Cosine scheduler
    warmup_epochs = 1
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs)

    # Training loop
    best_acc = 0.0
    best_noisy_acc = 0.0
    early_stop_counter = 0
    max_early_stop = 3  # Stop if zero-recall persists

    for epoch in range(args.epochs):
        # Warmup LR
        if epoch < warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * (epoch + 1) / warmup_epochs

        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            logits = model(data)

            # Apply label smoothing if specified
            if args.label_smoothing > 0:
                loss = F.cross_entropy(logits, target, label_smoothing=args.label_smoothing)
            else:
                loss = criterion(logits, target)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = logits.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)

            if batch_idx % 20 == 0:
                print(f'Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, Acc: {100.*train_correct/train_total:.2f}%')

        # Validation
        val_metrics = evaluate_with_gates(model, val_loader, criterion, device, split_name="Val")

        train_acc = 100. * train_correct / train_total

        print(f'\nEpoch {epoch+1}/{args.epochs} Summary:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')

        # Check gates
        if val_metrics['zero_recall_classes']:
            early_stop_counter += 1
            print(f"  WARNING: Early stop counter: {early_stop_counter}/{max_early_stop}")
            if early_stop_counter >= max_early_stop:
                print(f"  EARLY STOP: Zero recall persisted for {max_early_stop} epochs")
                break
        else:
            early_stop_counter = 0

        # Save best model
        if val_metrics['acc'] > best_acc:
            best_acc = val_metrics['acc']
            torch.save(model.state_dict(), args.save)
            print(f'  New best model saved. Val Acc: {val_metrics["acc"]:.2f}%')

        if epoch >= warmup_epochs:
            scheduler.step()

    # Final evaluation on clean holdout (LOBO 90db boards)
    print("\n" + "=" * 50)
    print("FINAL EVALUATION ON CLEAN HOLDOUT")
    print("=" * 50)

    # TODO: Load clean holdout data and evaluate

    # Export to ONNX
    print("\nExporting to ONNX...")
    model.eval()
    dummy_input = torch.randn(1, 1, 256, 256).to(device)

    try:
        torch.onnx.export(
            model, dummy_input, args.export_onnx,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=11
        )
        print(f"Model exported to {args.export_onnx}")
    except Exception as e:
        print(f"ONNX export failed: {e}")

    print(f"\nTraining completed! Best validation accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
