# EfficientNet for BirdNET-Analyzer

This directory contains a PyTorch implementation of EfficientNet models, based on the Flax implementation from [google-research/perch](https://github.com/google-research/perch/blob/main/chirp/models/efficientnet.py). The implementation is specifically designed to integrate with the BirdNET-Analyzer framework for bird classification tasks.

## Features

- **Complete EfficientNet Family**: Support for all standard variants (B0-B8, L2)
- **Perch Compatibility**: Matches the interface and architecture from the reference implementation
- **BirdNET Integration**: Compatible with existing BirdNET dual mel spectrogram frontend
- **Configurable Activations**: Support for different activation configurations (default, QAT, ReLU-only)
- **Stochastic Depth**: Proper implementation of stochastic depth for regularization
- **Squeeze-and-Excitation**: Full SE module implementation with configurable reduction ratios
- **PyTorch Native**: Pure PyTorch implementation with no external dependencies

## Files

- `efficientnet.py`: Main EfficientNet implementation
- `torch_model.py`: Integration with existing BirdNET models
- `efficientnet_examples.py`: Usage examples and demonstrations
- `test_efficientnet_integration.py`: Integration tests

## Quick Start

### Standalone Usage

```python
from birdnet_analyzer.efficientnet import EfficientNet, EfficientNetModel
import torch

# Create EfficientNet-B0 for classification
model = EfficientNet(EfficientNetModel.B0, num_classes=1000)

# Forward pass with BirdNET-style dual spectrograms
x = torch.randn(4, 2, 96, 511)  # (batch, channels, height, width)
predictions = model(x)
print(predictions.shape)  # torch.Size([4, 1000])

# Feature extraction (no classification head)
backbone = EfficientNet(EfficientNetModel.B0, include_top=False)
features = backbone(x)
print(features.shape)  # Feature maps with spatial dimensions
```

### Integration with BirdNET Frontend

```python
from birdnet_analyzer.torch_model import BirdNetEfficientNet
from birdnet_analyzer.efficientnet import EfficientNetModel

# Create integrated model (audio -> spectrograms -> classification)
model = BirdNetEfficientNet(
    model_variant=EfficientNetModel.B4,
    num_classes=1000
)

# Forward pass with raw audio
audio = torch.randn(2, 144000)  # 3 seconds at 48kHz
predictions = model(audio)
print(predictions.shape)  # torch.Size([2, 1000])
```

### Different Model Variants

```python
from birdnet_analyzer.efficientnet import create_efficientnet

# Create different variants
models = {
    'b0': create_efficientnet('b0', num_classes=100),
    'b4': create_efficientnet('b4', num_classes=100), 
    'b7': create_efficientnet('b7', num_classes=100),
}

for name, model in models.items():
    params = sum(p.numel() for p in model.parameters())
    print(f"EfficientNet-{name.upper()}: {params:,} parameters")
```

## Model Variants

| Model | Width Coeff | Depth Coeff | Dropout | Params (approx) |
|-------|------------|-------------|---------|-----------------|
| B0    | 1.0        | 1.0         | 0.2     | 5.3M           |
| B1    | 1.0        | 1.1         | 0.2     | 7.8M           |
| B2    | 1.1        | 1.2         | 0.3     | 9.1M           |
| B3    | 1.2        | 1.4         | 0.3     | 12M            |
| B4    | 1.4        | 1.8         | 0.4     | 19M            |
| B5    | 1.6        | 2.2         | 0.4     | 30M            |
| B6    | 1.8        | 2.6         | 0.5     | 43M            |
| B7    | 2.0        | 3.1         | 0.5     | 66M            |
| B8    | 2.2        | 3.6         | 0.5     | 87M            |
| L2    | 4.3        | 5.3         | 0.5     | 480M           |

## Advanced Configuration

### Custom Activation Functions

```python
from birdnet_analyzer.efficientnet import EfficientNet, EfficientNetModel, ActivationConfig
import torch.nn.functional as F

# Use ReLU-only configuration
model = EfficientNet(
    EfficientNetModel.B0,
    num_classes=100,
    activation_config="relu"
)

# Custom activation configuration
custom_config = ActivationConfig(
    activation=F.gelu,
    sigmoid=torch.sigmoid,
    stem_activation=F.mish,
    head_activation=F.mish
)

model = EfficientNet(
    EfficientNetModel.B2,
    activation_config=custom_config
)
```

### Custom Input Channels

```python
# For RGB images instead of dual spectrograms
model = EfficientNet(
    EfficientNetModel.B0,
    num_classes=1000,
    in_channels=3  # RGB instead of dual spectrograms
)

# Forward pass with RGB images
x = torch.randn(4, 3, 224, 224)
output = model(x)
```

## Architecture Details

The implementation follows the EfficientNet paper architecture:

1. **Stem**: 3x3 conv with stride 2
2. **MBConv Blocks**: Inverted residual blocks with squeeze-and-excitation
3. **Head**: 1x1 conv + global average pooling + dropout + linear classification

### MBConv Block Structure

Each MBConv block consists of:
1. Expansion convolution (1x1, if expand_ratio > 1)
2. Depthwise convolution (3x3 or 5x5)
3. Squeeze-and-excitation module
4. Projection convolution (1x1)
5. Skip connection (if same input/output dimensions)
6. Stochastic depth (during training)

### Scaling Rules

EfficientNet uses compound scaling:
- **Width**: Number of channels scaled by width_coefficient
- **Depth**: Number of layers scaled by depth_coefficient  
- **Resolution**: Input resolution scaled by resolution_coefficient (handled by frontend)

## Integration with BirdNET

The implementation is designed to work seamlessly with BirdNET's audio preprocessing:

1. **Input Format**: Expects dual mel spectrograms (2 channels, 96 mel bins, 511 time frames)
2. **Preprocessing**: Uses existing `BirdNETMelSpecLayer` for audio-to-spectrogram conversion
3. **Output**: Classification logits for bird species identification

## Testing

Run the integration tests to verify the implementation:

```bash
python test_efficientnet_integration.py
```

Run the examples (requires PyTorch):

```bash
python efficientnet_examples.py
```

## Compatibility

- **PyTorch**: 1.9+ (uses F.silu which requires PyTorch 1.9+)
- **Python**: 3.7+
- **BirdNET**: Compatible with existing BirdNET-Analyzer frontend

## References

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- [Perch EfficientNet Implementation](https://github.com/google-research/perch/blob/main/chirp/models/efficientnet.py)
- [BirdNET-Analyzer](https://github.com/kahst/BirdNET-Analyzer)

## License

This implementation follows the same license as the parent BirdNET-Analyzer project.