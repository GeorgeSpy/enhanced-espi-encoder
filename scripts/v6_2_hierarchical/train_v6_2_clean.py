#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train ESPI V6.2 - Hierarchical Training Strategy (Optimized for RTX 3060)
=======================================================================
Phase 1: Gatekeeper (Signal vs Noise)
Phase 2: Expert (Modes 0-4) with ArcFace
Phase 3: Fine-tuning

Optimizations:
- Mixed Precision (AMP)
- Persistent Workers
- Pin Memory
- Gradient Accumulation (if needed)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import logging
from pathlib import Path
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **_kwargs):
        return iterable
import json
from torch.amp import autocast, GradScaler

EnhancedHybridPhysicsESPI_V6_2 = None
EnhancedESPIDataset = None
setup_logging = None
label_from_frequency = None
_RESOLVE_ERRORS = []

# Keep a default logger until runtime dependencies are resolved in main().
logger = logging.getLogger("train_v6_2_clean")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
LOG_DIR = Path("logs") / "train_v6.2_opt"


def _resolve_runtime_dependencies():
    """Resolve optional local modules lazily to keep the script import-safe."""
    global EnhancedHybridPhysicsESPI_V6_2, EnhancedESPIDataset, setup_logging
    global label_from_frequency, _RESOLVE_ERRORS

    _RESOLVE_ERRORS = []

    # Dataset/utils source
    try:
        from enhanced_espi_pipeline import EnhancedESPIDataset as _DatasetCls, setup_logging as _setup_logging_fn, label_from_frequency as _label_from_frequency
        EnhancedESPIDataset = _DatasetCls
        setup_logging = _setup_logging_fn
        label_from_frequency = _label_from_frequency
    except Exception as e_main_pipe:
        try:
            from enhanced_espi_pipeline_clean_FIXED import EnhancedESPIDataset as _DatasetCls, setup_logging as _setup_logging_fn, label_from_frequency as _label_from_frequency
            EnhancedESPIDataset = _DatasetCls
            setup_logging = _setup_logging_fn
            label_from_frequency = _label_from_frequency
        except Exception as e_fixed_pipe:
            _RESOLVE_ERRORS.append(f"dataset/utils import failed: {e_main_pipe} | {e_fixed_pipe}")

    # Model source (strict): require v6.2 architecture, no silent fallback.
    try:
        # Allow importing from the parent ESPI project root when launched from a packaged experiment folder.
        project_root = Path(__file__).resolve().parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from espi_v6_2 import EnhancedHybridPhysicsESPI_V6_2 as _ModelCls
        EnhancedHybridPhysicsESPI_V6_2 = _ModelCls
    except Exception as e_v62:
        _RESOLVE_ERRORS.append(
            "model import failed: espi_v6_2.EnhancedHybridPhysicsESPI_V6_2 "
            f"is required for gate/expert training | err={e_v62}"
        )

    if EnhancedESPIDataset is None or setup_logging is None or EnhancedHybridPhysicsESPI_V6_2 is None:
        msg = " ; ".join(_RESOLVE_ERRORS) if _RESOLVE_ERRORS else "unknown dependency resolution error"
        raise RuntimeError(f"Failed to resolve runtime dependencies: {msg}")

# ============================================================================
# Configuration
# ============================================================================
CONFIG = {
    'data': {
        'root_dir': os.environ.get('ESPI_DATA_ROOT', 'data_256'),
        'batch_size': 12, # Optimized for RTX 3060 (12GB VRAM)
        'num_workers': 4, # Parallel loading
    },
    'labels': {
        'num_classes': 6,              # 0..4 expert modes + 5 noise
        'noise_class_id': 5,
        'strict_required': True,
        'class_names': [
            "mode_1_1H",
            "mode_1_1T",
            "mode_1_2",
            "mode_2_1",
            "mode_higher",
            "noise_unknown",
        ],
        'by_frequency': {
            'enabled': True,
            'other_class_id': 5,
            'mode_higher_class_id': 4,
            'use_pseudonoisy_as_noise': True,
            # Material-aware bins aligned with v6.1/v6.2 labeling scripts.
            'windows_by_material': {
                'wood': [
                    {'name': 'mode_1_1H', 'lo': 165.0, 'hi': 185.0, 'class_id': 0},
                    {'name': 'mode_1_1T', 'lo': 320.0, 'hi': 350.0, 'class_id': 1},
                    {'name': 'mode_1_2',  'lo': 500.0, 'hi': 530.0, 'class_id': 2},
                    {'name': 'mode_2_1',  'lo': 540.0, 'hi': 570.0, 'class_id': 3},
                    {'name': 'mode_higher', 'lo': 680.0, 'hi': 1500.0, 'class_id': 4},
                ],
                'carbon': [
                    {'name': 'mode_1_1H', 'lo': 175.0, 'hi': 205.0, 'class_id': 0},
                    {'name': 'mode_1_1T', 'lo': 340.0, 'hi': 370.0, 'class_id': 1},
                    {'name': 'mode_1_2',  'lo': 530.0, 'hi': 560.0, 'class_id': 2},
                    {'name': 'mode_2_1',  'lo': 570.0, 'hi': 610.0, 'class_id': 3},
                    {'name': 'mode_higher', 'lo': 700.0, 'hi': 1500.0, 'class_id': 4},
                ],
            },
        },
    },
    'training': {
        'epochs_gate': 5,
        'epochs_expert': 15,
        'epochs_finetune': 5,
        'lr_gate': 1e-3,
        'lr_expert': 1e-3,
        'lr_finetune': 1e-4,
    },
    'model': {
        'feat_channels': 32,
        'fast_cpu': False,
        'num_expert_classes': 5,
    }
}

# ============================================================================
# Collate Functions (Must be top-level for pickling on Windows)
# ============================================================================
def gate_collate(batch):
    # batch is list of dicts
    # item['image'] is [1, H, W]. We want [B, 1, H, W].
    images = torch.stack([item['image'] for item in batch])
    original_labels = torch.tensor([item['label'] for item in batch])
    gate_labels = (original_labels == int(CONFIG['labels']['noise_class_id'])).long()
    return images, gate_labels

def expert_collate(batch):
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    return images, labels


def _resolve_sample_label(sample: dict, cfg: dict) -> int:
    labels_cfg = cfg.get('labels', {})
    byf_cfg = labels_cfg.get('by_frequency', {})
    noise_id = int(labels_cfg.get('noise_class_id', 5))
    strict_required = bool(labels_cfg.get('strict_required', True))
    use_pseudo_as_noise = bool(byf_cfg.get('use_pseudonoisy_as_noise', True))

    if use_pseudo_as_noise and bool(sample.get('is_pseudo_noisy', False)):
        return noise_id

    y = label_from_frequency(sample.get('frequency'), cfg, material=sample.get('material'))
    if y is None:
        if strict_required:
            raise RuntimeError(
                "labels.by_frequency is required for v6.2 but returned None. "
                "Verify CONFIG['labels']['by_frequency']."
            )
        y = 0 if str(sample.get('material', '')).lower() == 'wood' else 1
    return int(y)

# ============================================================================
# Loader Creation
# ============================================================================
def create_hierarchical_loaders(dataset, batch_size, num_workers=0):
    indices = list(range(len(dataset)))
    
    # Efficient label extraction
    if hasattr(dataset, 'samples') and isinstance(dataset.samples, list) and isinstance(dataset.samples[0], dict):
        labels = [_resolve_sample_label(s, dataset.config) for s in dataset.samples]
    else:
        labels = [dataset[i]['label'] for i in indices]
    
    num_expert_classes = int(CONFIG['model']['num_expert_classes'])
    noise_id = int(CONFIG['labels']['noise_class_id'])
    signal_indices = [i for i, label in enumerate(labels) if int(label) < num_expert_classes]
    noise_indices = [i for i, label in enumerate(labels) if int(label) == noise_id]
    
    logger.info(f"Dataset Split: {len(signal_indices)} Signal samples, {len(noise_indices)} Noise samples")
    
    gate_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=gate_collate
    )
    
    expert_subset = Subset(dataset, signal_indices)
    
    expert_loader = DataLoader(
        expert_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=expert_collate
    )
    
    return gate_loader, expert_loader

# ============================================================================
# Training Loop
# ============================================================================
def train_epoch(model, loader, optimizer, scaler, criterion, phase_name, epoch, device, train_gate=False, train_expert=False):
    model.train()
    
    # Freeze/Unfreeze
    for param in model.classifier.gatekeeper_head.parameters(): param.requires_grad = train_gate
    for param in model.classifier.expert_embedding.parameters(): param.requires_grad = train_expert
    for param in model.classifier.arcface.parameters(): param.requires_grad = train_expert
    
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"{phase_name} Epoch {epoch}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast(device_type=device.type):
            try:
                try:
                    outputs = model(images, targets=labels if train_expert else None)
                except TypeError:
                    # Fallback model from enhanced_espi_pipeline_clean_FIXED does not accept targets=
                    outputs = model(images)
                
                if isinstance(outputs, dict):
                    if train_gate and 'gate_logits' in outputs:
                        logits = outputs['gate_logits']
                    elif train_expert and 'expert_logits' in outputs:
                        logits = outputs['expert_logits']
                    else:
                        raise KeyError(
                            f"Model outputs do not include expected hierarchical keys for phase "
                            f"(train_gate={train_gate}, train_expert={train_expert}): {list(outputs.keys())}"
                        )
                else:
                    raise TypeError("v6.2 training expects dict outputs with gate_logits/expert_logits")
                    
                loss = criterion(logits, labels)
            except RuntimeError as e:
                print(f"CRITICAL ERROR in forward pass: {e}")
                print(f"Input shape: {images.shape}")
                if hasattr(model, "classifier") and hasattr(model.classifier, "backbone"):
                    try:
                        print(f"Model layer 0 weight shape: {model.classifier.backbone[0].weight.shape}")
                    except Exception:
                        pass
                raise e
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
        
    return total_loss / len(loader), correct / total

# ============================================================================
# Main
# ============================================================================
def main():
    global logger, LOG_DIR

    _resolve_runtime_dependencies()
    logger, LOG_DIR = setup_logging(log_dir="logs", experiment_name="train_v6.2_opt")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("CUDA is NOT available. Training on CPU.")

    data_root = Path(CONFIG['data']['root_dir'])
    if not data_root.exists():
        logger.error(
            f"Dataset root does not exist: {data_root}. "
            "Set ESPI_DATA_ROOT env var or update CONFIG['data']['root_dir']."
        )
        return 1
    
    dataset = EnhancedESPIDataset(
        root_dir=CONFIG['data']['root_dir'],
        mode='train',
        augment=True,
        config=CONFIG
    )
    
    gate_loader, expert_loader = create_hierarchical_loaders(
        dataset, 
        CONFIG['data']['batch_size'],
        num_workers=CONFIG['data']['num_workers']
    )
    
    model = EnhancedHybridPhysicsESPI_V6_2(
        num_expert_classes=int(CONFIG['model']['num_expert_classes']),
        feat_channels=CONFIG['model']['feat_channels'],
        fast_cpu=CONFIG['model']['fast_cpu']
    ).to(device)
    if hasattr(model, "classifier") and hasattr(model.classifier, "backbone"):
        try:
            print(f"DEBUG: Model Backbone Layer 0: {model.classifier.backbone[0]}")
            print(f"DEBUG: Model Backbone Layer 0 Weight Shape: {model.classifier.backbone[0].weight.shape}")
        except Exception:
            pass
    
    scaler = GradScaler()
    
    # Phase 1: Gatekeeper
    logger.info("=== STARTING PHASE 1: GATEKEEPER TRAINING ===")
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['training']['lr_gate'])
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(CONFIG['training']['epochs_gate']):
        loss, acc = train_epoch(model, gate_loader, optimizer, scaler, criterion, "Phase 1", epoch, device, train_gate=True)
        logger.info(f"Epoch {epoch}: Gate Loss={loss:.4f}, Gate Acc={acc:.4f}")
        torch.save(model.state_dict(), LOG_DIR / "ckpt_phase1_gate.pt")
        
    # Phase 2: Expert
    logger.info("=== STARTING PHASE 2: EXPERT TRAINING ===")
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['training']['lr_expert'])
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(CONFIG['training']['epochs_expert']):
        loss, acc = train_epoch(model, expert_loader, optimizer, scaler, criterion, "Phase 2", epoch, device, train_expert=True)
        logger.info(f"Epoch {epoch}: Expert Loss={loss:.4f}, Expert Acc={acc:.4f}")
        torch.save(model.state_dict(), LOG_DIR / "ckpt_phase2_expert.pt")
        
    # Phase 3: Fine-tune (Simplified)
    logger.info("=== STARTING PHASE 3: FINE-TUNING ===")
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['training']['lr_finetune'])
    
    for epoch in range(CONFIG['training']['epochs_finetune']):
        l1, a1 = train_epoch(model, expert_loader, optimizer, scaler, criterion, "Phase 3 (Exp)", epoch, device, train_expert=True)
        l2, a2 = train_epoch(model, gate_loader, optimizer, scaler, criterion, "Phase 3 (Gate)", epoch, device, train_gate=True)
        logger.info(f"Epoch {epoch}: Expert Acc={a1:.4f}, Gate Acc={a2:.4f}")
        
    torch.save(model.state_dict(), LOG_DIR / "model_v6.2_final.pt")
    logger.info("Training Complete! Model saved.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
