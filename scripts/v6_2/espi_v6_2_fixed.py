#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced ESPI Pipeline V6.2 - Hierarchical Architecture with ArcFace
==================================================================
Key Improvements over v6.1/E18:
1. Hierarchical Classification (Gatekeeper -> Expert)
2. ArcFace Metric Learning for robust Mode separation
3. Explicit handling of C5 (Unknown) as a separate binary task
4. Physics-based augmentation for C0 recovery

Author: ESPI Team
Version: 6.2.1 (Optimized)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import logging

# Import base components from the main pipeline
try:
    from enhanced_espi_pipeline import (
        EnhancedAttentionDnCNN,
        EnhancedLearnableFourierTransformPrior,
        QualityGuidedPhaseUnwrapper,
        MultiScalePhysicsFeatures,
        PhysicsConstraints
    )
    BASE_IMPORTED = True
except ImportError:
    BASE_IMPORTED = False
    print("Warning: Could not import base components. Using placeholders.")
    class EnhancedAttentionDnCNN(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()
        def forward(self, x): return x
    class EnhancedLearnableFourierTransformPrior(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()
        def forward(self, x): return x, x, x
    class QualityGuidedPhaseUnwrapper(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()
        def forward(self, x, a): return x
    class MultiScalePhysicsFeatures(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()
        def forward(self, x, a=None): return x
    class PhysicsConstraints(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()
        def forward(self, *args): return {}

logger = logging.getLogger("ESPI-v6.2")

# ============================================================================
# ArcFace Head (Metric Learning)
# ============================================================================
class ArcFaceHead(nn.Module):
    def __init__(self, in_features: int, out_classes: int, s: float = 30.0, m: float = 0.50):
        super().__init__()
        self.in_features = in_features
        self.out_classes = out_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(features, dim=1), F.normalize(self.weight, dim=1))
        if targets is None:
            return cosine * self.s

        targets = targets.view(-1).to(device=features.device, dtype=torch.long)
        valid_mask = (targets >= 0) & (targets < self.out_classes)
        output = cosine.clone()

        if valid_mask.any():
            valid_cosine = cosine[valid_mask]
            valid_targets = targets[valid_mask]
            theta = torch.acos(torch.clamp(valid_cosine, -1.0 + 1e-7, 1.0 - 1e-7))
            one_hot = torch.zeros_like(valid_cosine)
            one_hot.scatter_(1, valid_targets.view(-1, 1), 1)
            target_logit = torch.cos(theta + self.m)
            output[valid_mask] = one_hot * target_logit + (1.0 - one_hot) * valid_cosine

        return output * self.s

# ============================================================================
# Hierarchical Classifier
# ============================================================================
class HierarchicalClassifier(nn.Module):
    def __init__(self, input_channels: int, num_expert_classes: int = 5):
        super().__init__()
        logger.debug("HierarchicalClassifier initialized with input_channels=%s", input_channels)
        
        # Shared Feature Extractor (Backbone)
        self.backbone = nn.Sequential(
            # Input projection
            nn.Conv2d(input_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            
            # Block 1
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            
            # Block 2
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            
            # Block 3
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Head A: Gatekeeper (Binary: Signal vs Noise)
        self.gatekeeper_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 2) # [Signal, Noise]
        )
        
        # Head B: Expert (Multi-class: Modes 0-4) with ArcFace
        self.expert_embedding = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4)
        )
        self.arcface = ArcFaceHead(in_features=512, out_classes=num_expert_classes)

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        gate_logits = self.gatekeeper_head(features)
        embeddings = self.expert_embedding(features)
        expert_logits = self.arcface(embeddings, targets)
        return {
            'gate_logits': gate_logits,
            'expert_logits': expert_logits,
            'embeddings': embeddings
        }

# ============================================================================
# Enhanced ESPI Pipeline V6.2
# ============================================================================
class EnhancedHybridPhysicsESPI_V6_2(nn.Module):
    def __init__(self,
                 num_expert_classes: int = 5,
                 feat_channels: int = 32,
                 use_unwrapping: bool = True,
                 use_multiscale: bool = True,
                 on_axis_mode: bool = False,
                 dtype: torch.dtype = torch.float32,
                 fast_cpu: bool = False):
        super().__init__()
        
        # --- Physics Components ---
        n_layers = 5 if fast_cpu else 17
        use_attention = False if fast_cpu else True
        
        self.denoiser = EnhancedAttentionDnCNN(
            num_layers=n_layers,
            num_features=64,
            use_attention=use_attention,
            use_multiscale=use_multiscale
        ).to(dtype)
        
        self.leftp = EnhancedLearnableFourierTransformPrior(on_axis_mode=on_axis_mode).to(dtype)
        
        self.use_unwrapping = use_unwrapping
        if use_unwrapping:
            self.unwrapper = QualityGuidedPhaseUnwrapper()
            
        pf_scales = [1] if fast_cpu else [1, 2, 4]
        self.physics_features = MultiScalePhysicsFeatures(
            scales=pf_scales,
            out_channels=feat_channels
        ).to(dtype)

        logger.debug("EnhancedHybridPhysicsESPI_V6_2 init with feat_channels=%s", feat_channels)
        # --- New Classifier V6.2 ---
        # We pass feat_channels directly. If physics_features returns something else, we'll crash,
        # but at least we know what we passed.
        self.classifier = HierarchicalClassifier(
            input_channels=feat_channels,
            num_expert_classes=num_expert_classes
        ).to(dtype)
        
        self.physics_constraints = PhysicsConstraints().to(dtype)

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        denoised = self.denoiser(x)
        wrapped_phase, amplitude, spectral_filter = self.leftp(denoised)
        
        if self.use_unwrapping:
            unwrapped_phase = self.unwrapper(wrapped_phase, amplitude)
        else:
            unwrapped_phase = wrapped_phase
            
        physics_feats = self.physics_features(unwrapped_phase, amplitude)
        
        expected_channels = self.classifier.backbone[0].in_channels
        if physics_feats.shape[1] != expected_channels:
            raise ValueError(
                f"Channel mismatch between physics features and classifier: "
                f"got {physics_feats.shape[1]}, expected {expected_channels}"
            )
        
        cls_outputs = self.classifier(physics_feats, targets)
        
        return {
            'denoised': denoised,
            'wrapped_phase': wrapped_phase,
            'unwrapped_phase': unwrapped_phase,
            'amplitude': amplitude,
            'gate_logits': cls_outputs['gate_logits'],
            'expert_logits': cls_outputs['expert_logits'],
            'embeddings': cls_outputs['embeddings']
        }

    def predict(self, x: torch.Tensor, gate_threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            gate_probs = torch.softmax(outputs['gate_logits'], dim=1)
            is_noise = gate_probs[:, 1] > gate_threshold
            expert_probs = torch.softmax(outputs['expert_logits'], dim=1)
            expert_preds = expert_probs.argmax(dim=1)
            expert_conf = expert_probs.max(dim=1)[0]

            final_preds = torch.where(is_noise, torch.full_like(expert_preds, 5), expert_preds)
            signal_conf = gate_probs[:, 0] * expert_conf
            final_conf = torch.where(is_noise, gate_probs[:, 1], signal_conf)
            return final_preds, final_conf, is_noise

if __name__ == "__main__":
    print("Testing ESPI V6.2 Architecture...")
    model = EnhancedHybridPhysicsESPI_V6_2(fast_cpu=True)
    x = torch.randn(2, 1, 256, 256)
    out = model(x)
    print("Forward pass successful!")
