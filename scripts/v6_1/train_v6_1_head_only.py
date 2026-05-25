#!/usr/bin/env python3
"""
v6.1 Head-Only Fine-tuning Script
Fixes C4 bias by retraining only the classifier head
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

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from stratified_batch_sampler import StratifiedBatchSampler
from asymmetric_focal_loss import create_loss_function

class ESPIDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.df = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        print(f"Dataset loaded: {len(self.df)} samples")
        print(f"Label distribution: {self.df['label'].value_counts().sort_index().to_dict()}")

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
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # [1, H, W]
        label = torch.tensor(row['label'], dtype=torch.long)
        return image_tensor, label

class ResNet18_HeadOnly(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        # Load pretrained ResNet-18
        if pretrained:
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = resnet18(weights=None)
        
        # Modify for 1-channel input (will be changed if loading checkpoint)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove final classifier
        self.backbone.fc = nn.Identity()
        
        # New classifier head
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

def load_checkpoint(model, checkpoint_path):
    """Load checkpoint and handle missing keys"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Handle 3-channel to 1-channel conversion
        if 'backbone.conv1.weight' in state_dict:
            conv1_weight = state_dict['backbone.conv1.weight']
            if conv1_weight.shape[1] == 3:  # 3-channel input
                print("Converting 3-channel checkpoint to 1-channel model")
                # Take only the first channel (grayscale)
                conv1_weight_1ch = conv1_weight[:, 0:1, :, :]
                state_dict['backbone.conv1.weight'] = conv1_weight_1ch
        
        # Load with strict=False to handle missing keys
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys (will be randomly initialized): {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys (ignored): {unexpected_keys}")
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        return True
        
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='v6.1 Head-Only Fine-tuning')
    parser.add_argument("--csv", default="labels_modes.v5.csv", help="Train CSV")
    parser.add_argument("--val_csv", default="labels_modes.v6.val.csv", help="Val CSV")
    parser.add_argument("--img_root", default=".", help="Image root")
    parser.add_argument("--checkpoint", default="checkpoints/espi_modes_v5.pt", help="Pretrained checkpoint")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs")
    parser.add_argument("--bs", type=int, default=40, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--loss_type", default="afl", choices=["afl", "cb", "ldam_cb", "balanced_softmax"], help="Loss type")
    parser.add_argument("--label_smoothing", type=float, default=0.05, help="Label smoothing")
    parser.add_argument("--save", default="checkpoints/espi_modes_v6_1_head.pt", help="Save path")
    parser.add_argument("--export_onnx", default="models/espi_modes_v6_1.onnx", help="ONNX export path")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = ESPIDataset(args.csv, args.img_root)
    val_dataset = ESPIDataset(args.val_csv, args.img_root)
    
    # Calculate class counts for loss function
    class_counts = train_dataset.df['label'].value_counts().sort_index().values
    print(f"Class counts: {dict(enumerate(class_counts))}")
    
    # Load priors from training_priors.json
    log_priors = None
    if os.path.exists("training_priors.json"):
        import json
        with open("training_priors.json", 'r') as f:
            priors_data = json.load(f)
        log_priors = [priors_data['log_prior'][str(i)] for i in range(5)]
        print(f"Loaded log_priors: {log_priors}")
    else:
        print("training_priors.json not found, using default priors")
    
    # Create stratified batch sampler
    train_sampler = StratifiedBatchSampler(
        train_dataset, 
        batch_size=args.bs, 
        samples_per_class=8,  # 8 samples from each class
        class_4_multiplier=1.0,  # Balanced
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
    model = ResNet18_HeadOnly(5, pretrained=True).to(device)
    
    # Load checkpoint if available
    if os.path.exists(args.checkpoint):
        load_checkpoint(model, args.checkpoint)
    else:
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Training from scratch...")
    
    # Create loss function
    criterion = create_loss_function(args.loss_type, class_counts, log_priors=log_priors)
    print(f"Using loss function: {args.loss_type}")
    
    # Optimizer (only for classifier parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_acc = 0.0
    for epoch in range(args.epochs):
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
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                logits = model(data)
                loss = criterion(logits, target)
                
                val_loss += loss.item()
                pred = logits.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.save)
            print(f'  New best model saved! Val Acc: {val_acc:.2f}%')
        
        scheduler.step()
    
    # Export to ONNX
    print("Exporting to ONNX...")
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
    
    print(f"Training completed! Best validation accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()