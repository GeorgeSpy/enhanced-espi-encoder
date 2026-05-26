#!/usr/bin/env python3
"""
Export v6.1 model to ONNX for production deployment
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from pathlib import Path
import argparse

class ResNet18_HeadOnly(nn.Module):
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

        # New classifier head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

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

def export_to_onnx(checkpoint_path, onnx_path, input_shape=(1, 1, 256, 256)):
    """
    Export PyTorch model to ONNX

    Args:
        checkpoint_path: Path to .pt checkpoint
        onnx_path: Path to save .onnx model
        input_shape: Input tensor shape (batch, channels, height, width)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create model
    model = ResNet18_HeadOnly(num_classes=5, pretrained=False)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Handle 3-channel to 1-channel conversion if needed
    if 'backbone.conv1.weight' in state_dict:
        conv1_weight = state_dict['backbone.conv1.weight']
        if conv1_weight.shape[1] == 3:  # 3-channel input
            print("Converting 3-channel checkpoint to 1-channel")
            conv1_weight_1ch = conv1_weight[:, 0:1, :, :]
            state_dict['backbone.conv1.weight'] = conv1_weight_1ch

    # Load state dict
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: Missing keys: {missing}")
    if unexpected:
        print(f"Warning: Unexpected keys: {unexpected}")

    model.to(device)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)

    # Create output directory if needed
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

    # Export to ONNX
    print(f"Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print("ONNX export successful.")
    print(f"   Model saved to: {onnx_path}")

    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed")

        # Print model info
        print(f"\nModel info:")
        print(f"  Input shape: {input_shape}")
        print(f"  Output shape: (batch_size, 5)")
        print(f"  Opset version: 14")

    except ImportError:
        print("WARNING: onnx package not installed - skipping verification")
    except Exception as e:
        print(f"WARNING: ONNX verification failed: {e}")

def main():
    parser = argparse.ArgumentParser(description='Export v6.1 to ONNX')
    parser.add_argument('--checkpoint', default='checkpoints/espi_modes_v6_1_head.pt',
                       help='Path to checkpoint')
    parser.add_argument('--onnx', default='models/espi_modes_v6_1.onnx',
                       help='Path to save ONNX model')
    parser.add_argument('--height', type=int, default=256, help='Input height')
    parser.add_argument('--width', type=int, default=256, help='Input width')
    args = parser.parse_args()

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        onnx_path=args.onnx,
        input_shape=(1, 1, args.height, args.width)
    )

if __name__ == '__main__':
    main()
