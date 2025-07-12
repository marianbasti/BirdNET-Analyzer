"""EfficientNet implementation in PyTorch.

Implementation of the EfficientNet model family based on the Flax implementation
from google-research/perch/chirp/models/efficientnet.py. This implementation provides
configurable EfficientNet variants (B0-B8, L2) with proper scaling, MBConv blocks,
squeeze-and-excitation, and stochastic depth.

The model is designed to work with the existing BirdNET frontend for audio preprocessing,
expecting precomputed features as input rather than raw audio.

Example usage:
    >>> import torch
    >>> from birdnet_analyzer.efficientnet import EfficientNet, EfficientNetModel
    >>> 
    >>> # Create EfficientNet-B0 model
    >>> model = EfficientNet(EfficientNetModel.B0, num_classes=1000)
    >>> 
    >>> # Forward pass with batch of features (B, C, H, W)
    >>> x = torch.randn(4, 2, 96, 511)  # BirdNET-style dual mel spectrograms
    >>> output = model(x)
    >>> print(output.shape)  # torch.Size([4, 1000])
    >>> 
    >>> # Create model without classification head for embeddings
    >>> backbone = EfficientNet(EfficientNetModel.B0, include_top=False)
    >>> embeddings = backbone(x)
    >>> print(embeddings.shape)  # torch.Size([4, C, H', W'])
"""

import enum
import math
from typing import Optional, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from dataclasses import dataclass


class EfficientNetModel(enum.Enum):
    """Different variants of EfficientNet."""
    B0 = "b0"
    B1 = "b1"
    B2 = "b2"
    B3 = "b3"
    B4 = "b4"
    B5 = "b5"
    B6 = "b6"
    B7 = "b7"
    B8 = "b8"
    L2 = "l2"


# Stage definition for EfficientNet architecture
EfficientNetStage = namedtuple('EfficientNetStage', [
    'num_blocks', 'features', 'kernel_size', 'strides', 'expand_ratio'
])

# Base configuration for EfficientNet-B0
STEM_FEATURES = 32
STAGES = [
    EfficientNetStage(1, 16, (3, 3), 1, 1),
    EfficientNetStage(2, 24, (3, 3), 2, 6),
    EfficientNetStage(2, 40, (5, 5), 2, 6),
    EfficientNetStage(3, 80, (3, 3), 2, 6),
    EfficientNetStage(3, 112, (5, 5), 1, 6),
    EfficientNetStage(4, 192, (5, 5), 2, 6),
    EfficientNetStage(1, 320, (3, 3), 1, 6),
]
HEAD_FEATURES = 1280

# Scaling configurations for different model variants
EfficientNetScaling = namedtuple('EfficientNetScaling', [
    'width_coefficient', 'depth_coefficient', 'dropout_rate'
])

SCALINGS = {
    EfficientNetModel.B0: EfficientNetScaling(1.0, 1.0, 0.2),
    EfficientNetModel.B1: EfficientNetScaling(1.0, 1.1, 0.2),
    EfficientNetModel.B2: EfficientNetScaling(1.1, 1.2, 0.3),
    EfficientNetModel.B3: EfficientNetScaling(1.2, 1.4, 0.3),
    EfficientNetModel.B4: EfficientNetScaling(1.4, 1.8, 0.4),
    EfficientNetModel.B5: EfficientNetScaling(1.6, 2.2, 0.4),
    EfficientNetModel.B6: EfficientNetScaling(1.8, 2.6, 0.5),
    EfficientNetModel.B7: EfficientNetScaling(2.0, 3.1, 0.5),
    EfficientNetModel.B8: EfficientNetScaling(2.2, 3.6, 0.5),
    EfficientNetModel.L2: EfficientNetScaling(4.3, 5.3, 0.5),
}


@dataclass
class ActivationConfig:
    """Configuration for activation functions, similar to Perch OpSet.
    
    This allows for configurable activation functions throughout the network,
    enabling compatibility with different training paradigms (e.g., QAT).
    
    Args:
        activation: Main activation function for MBConv blocks
        sigmoid: Sigmoid activation for squeeze-and-excitation
        stem_activation: Activation function for stem
        head_activation: Activation function for head
    """
    activation: Callable = F.relu  
    sigmoid: Callable = torch.sigmoid
    stem_activation: Callable = F.silu  # Swish/SiLU is default
    head_activation: Callable = F.silu


# Predefined activation configurations
ACTIVATION_CONFIGS = {
    "default": ActivationConfig(),
    "qat": ActivationConfig(  # Quantization-aware training compatible
        activation=F.relu,
        sigmoid=F.hardsigmoid,
        stem_activation=F.hardswish,
        head_activation=F.hardswish
    ),
    "relu": ActivationConfig(  # All ReLU for simplicity
        activation=F.relu,
        sigmoid=torch.sigmoid,
        stem_activation=F.relu,
        head_activation=F.relu
    )
}


def round_features(features: int, width_coefficient: float, depth_divisor: int = 8) -> int:
    """Round number of filters based on width multiplier.
    
    Args:
        features: Base number of features
        width_coefficient: Width scaling coefficient
        depth_divisor: Depth divisor for rounding
        
    Returns:
        Rounded number of features
    """
    features *= width_coefficient
    new_features = max(
        depth_divisor,
        int(features + depth_divisor / 2) // depth_divisor * depth_divisor,
    )
    if new_features < 0.9 * features:
        new_features += depth_divisor
    return int(new_features)


def round_num_blocks(num_blocks: int, depth_coefficient: float) -> int:
    """Round number of blocks based on depth multiplier.
    
    Args:
        num_blocks: Base number of blocks
        depth_coefficient: Depth scaling coefficient
        
    Returns:
        Rounded number of blocks
    """
    return int(math.ceil(depth_coefficient * num_blocks))


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation module.
    
    Args:
        in_channels: Number of input channels
        reduction_ratio: Reduction ratio for the squeeze layer
        sigmoid_fn: Sigmoid activation function to use
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 4, sigmoid_fn: Callable = torch.sigmoid):
        super().__init__()
        reduced_channels = max(1, in_channels // reduction_ratio)
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, 1, bias=True)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, 1, bias=True)
        self.sigmoid_fn = sigmoid_fn
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply squeeze-and-excitation.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        # Global average pooling
        s = F.adaptive_avg_pool2d(x, 1)
        # Squeeze
        s = F.relu(self.fc1(s))
        # Excitation
        s = self.sigmoid_fn(self.fc2(s))
        # Scale
        return x * s


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution block.
    
    This is the core building block of EfficientNet, consisting of:
    1. Expansion convolution (if expand_ratio > 1)
    2. Depthwise convolution
    3. Squeeze-and-excitation
    4. Projection convolution
    5. Skip connection (if applicable)
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size for depthwise convolution
        stride: Stride for depthwise convolution
        expand_ratio: Expansion ratio for bottleneck
        reduction_ratio: Reduction ratio for squeeze-and-excitation
        drop_rate: Drop rate for stochastic depth
        activation_fn: Activation function to use
        sigmoid_fn: Sigmoid function for SE module
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple] = 3,
        stride: int = 1,
        expand_ratio: int = 6,
        reduction_ratio: int = 4,
        drop_rate: float = 0.0,
        activation_fn: Callable = F.silu,
        sigmoid_fn: Callable = torch.sigmoid
    ):
        super().__init__()
        
        self.stride = stride
        self.use_skip = stride == 1 and in_channels == out_channels
        self.drop_rate = drop_rate
        self.activation_fn = activation_fn
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, expanded_channels, 1, bias=False)
            self.expand_bn = nn.BatchNorm2d(expanded_channels)
        else:
            self.expand_conv = None
            
        # Depthwise convolution
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        padding = tuple(k // 2 for k in kernel_size)
        
        self.depthwise_conv = nn.Conv2d(
            expanded_channels, expanded_channels, kernel_size,
            stride=stride, padding=padding, groups=expanded_channels, bias=False
        )
        self.depthwise_bn = nn.BatchNorm2d(expanded_channels)
        
        # Squeeze-and-excitation
        self.se = SqueezeExcitation(expanded_channels, reduction_ratio, sigmoid_fn)
        
        # Projection phase
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Forward pass through MBConv block.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            training: Whether in training mode (affects stochastic depth)
            
        Returns:
            Output tensor
        """
        identity = x
        
        # Expansion
        if self.expand_conv is not None:
            x = self.activation_fn(self.expand_bn(self.expand_conv(x)))
        
        # Depthwise convolution
        x = self.activation_fn(self.depthwise_bn(self.depthwise_conv(x)))
        
        # Squeeze-and-excitation
        x = self.se(x)
        
        # Projection
        x = self.project_bn(self.project_conv(x))
        
        # Skip connection with stochastic depth
        if self.use_skip:
            if self.drop_rate > 0 and training:
                # Stochastic depth: randomly drop the residual branch
                keep_prob = 1.0 - self.drop_rate
                if torch.rand(1).item() < keep_prob:
                    x = x + identity
            else:
                x = x + identity
                
        return x


class EfficientNetStem(nn.Module):
    """Stem layer for EfficientNet.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        activation_fn: Activation function to use
    """
    
    def __init__(self, in_channels: int, out_channels: int, activation_fn: Callable = F.silu):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation_fn = activation_fn
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through stem.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor
        """
        x = self.conv(x)
        x = self.bn(x)
        return self.activation_fn(x)


class EfficientNetHead(nn.Module):
    """Head layer for EfficientNet.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        activation_fn: Activation function to use
    """
    
    def __init__(self, in_channels: int, out_channels: int, activation_fn: Callable = F.silu):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation_fn = activation_fn
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through head.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor
        """
        x = self.conv(x)
        x = self.bn(x)
        return self.activation_fn(x)


class EfficientNet(nn.Module):
    """EfficientNet model implementation.
    
    This implementation follows the architecture from the EfficientNet paper
    and matches the interface from the Perch implementation. It supports all
    standard EfficientNet variants (B0-B8, L2) with configurable scaling.
    
    Args:
        model: EfficientNet model variant
        num_classes: Number of output classes (for classification head)
        in_channels: Number of input channels (default: 2 for BirdNET dual spectrograms)
        include_top: Whether to include classification head and global pooling
        survival_probability: Survival probability for stochastic depth
        activation_config: Activation configuration name or ActivationConfig object
        
    Example:
        >>> # Create EfficientNet-B0 for classification
        >>> model = EfficientNet(EfficientNetModel.B0, num_classes=1000)
        >>> x = torch.randn(4, 2, 96, 511)
        >>> logits = model(x)
        >>> 
        >>> # Create EfficientNet-B4 for feature extraction
        >>> backbone = EfficientNet(EfficientNetModel.B4, include_top=False)
        >>> features = backbone(x)
    """
    
    def __init__(
        self,
        model: EfficientNetModel,
        num_classes: int = 1000,
        in_channels: int = 2,
        include_top: bool = True,
        survival_probability: float = 0.8,
        activation_config: str = "default"
    ):
        super().__init__()
        
        self.model = model
        self.include_top = include_top
        self.survival_probability = survival_probability
        
        # Get scaling parameters and activation config
        scaling = SCALINGS[model]
        if isinstance(activation_config, str):
            activations = ACTIVATION_CONFIGS[activation_config]
        else:
            activations = activation_config
        
        # Stem
        stem_features = round_features(STEM_FEATURES, scaling.width_coefficient)
        self.stem = EfficientNetStem(in_channels, stem_features, activations.stem_activation)
        
        # Stages
        self.stages = nn.ModuleList()
        current_features = stem_features
        
        for stage_idx, stage in enumerate(STAGES):
            stage_blocks = nn.ModuleList()
            num_blocks = round_num_blocks(stage.num_blocks, scaling.depth_coefficient)
            stage_features = round_features(stage.features, scaling.width_coefficient)
            
            for block_idx in range(num_blocks):
                # First block in stage may have stride > 1
                stride = stage.strides if block_idx == 0 else 1
                in_ch = current_features if block_idx == 0 else stage_features
                
                # Calculate drop rate for stochastic depth
                total_blocks = sum(round_num_blocks(s.num_blocks, scaling.depth_coefficient) for s in STAGES)
                block_num = sum(round_num_blocks(STAGES[i].num_blocks, scaling.depth_coefficient) for i in range(stage_idx)) + block_idx
                drop_rate = (1.0 - survival_probability) * block_num / total_blocks if block_idx > 0 else 0.0
                
                block = MBConvBlock(
                    in_channels=in_ch,
                    out_channels=stage_features,
                    kernel_size=stage.kernel_size,
                    stride=stride,
                    expand_ratio=stage.expand_ratio,
                    drop_rate=drop_rate,
                    activation_fn=activations.activation,
                    sigmoid_fn=activations.sigmoid
                )
                stage_blocks.append(block)
                
            self.stages.append(stage_blocks)
            current_features = stage_features
            
        # Head
        head_features = round_features(HEAD_FEATURES, scaling.width_coefficient)
        self.head = EfficientNetHead(current_features, head_features, activations.head_activation)
        
        if include_top:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(scaling.dropout_rate)
            self.classifier = nn.Linear(head_features, num_classes)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through EfficientNet.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
               For BirdNET: (B, 2, 96, 511) - dual mel spectrograms
               
        Returns:
            If include_top=True: logits of shape (B, num_classes)
            If include_top=False: feature maps of shape (B, head_features, H', W')
        """
        # Stem
        x = self.stem(x)
        
        # Stages
        for stage_blocks in self.stages:
            for block in stage_blocks:
                x = block(x, training=self.training)
                
        # Head
        x = self.head(x)
        
        if self.include_top:
            # Global average pooling
            x = self.global_pool(x)
            x = x.flatten(1)
            # Dropout and classification
            x = self.dropout(x)
            x = self.classifier(x)
            
        return x
    
    def get_classifier(self) -> Optional[nn.Module]:
        """Get the classifier layer.
        
        Returns:
            Classifier layer if include_top=True, else None
        """
        return self.classifier if self.include_top else None
    
    def reset_classifier(self, num_classes: int):
        """Reset the classifier layer with new number of classes.
        
        Args:
            num_classes: New number of classes
        """
        if self.include_top:
            head_features = self.classifier.in_features
            self.classifier = nn.Linear(head_features, num_classes)


def create_efficientnet(
    model_name: str,
    num_classes: int = 1000,
    pretrained: bool = False,
    **kwargs
) -> EfficientNet:
    """Create an EfficientNet model.
    
    Args:
        model_name: Name of the model variant (e.g., 'b0', 'b4', 'l2')
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights (not implemented)
        **kwargs: Additional arguments passed to EfficientNet
        
    Returns:
        EfficientNet model
        
    Example:
        >>> model = create_efficientnet('b0', num_classes=100)
        >>> model = create_efficientnet('b4', include_top=False)
    """
    model_name = model_name.lower()
    if model_name not in [m.value for m in EfficientNetModel]:
        raise ValueError(f"Unknown model variant: {model_name}")
    
    model_enum = EfficientNetModel(model_name)
    
    if pretrained:
        raise NotImplementedError("Pretrained weights are not yet supported")
    
    return EfficientNet(model_enum, num_classes=num_classes, **kwargs)


# Convenience functions for common model variants
def efficientnet_b0(num_classes: int = 1000, **kwargs) -> EfficientNet:
    """Create EfficientNet-B0 model."""
    return create_efficientnet('b0', num_classes=num_classes, **kwargs)


def efficientnet_b4(num_classes: int = 1000, **kwargs) -> EfficientNet:
    """Create EfficientNet-B4 model."""
    return create_efficientnet('b4', num_classes=num_classes, **kwargs)


if __name__ == "__main__":
    # Simple test/example
    print("Testing EfficientNet implementation...")
    
    # Test different model variants
    for model_variant in [EfficientNetModel.B0, EfficientNetModel.B4]:
        print(f"\nTesting {model_variant.value.upper()}:")
        
        # Classification model
        model = EfficientNet(model_variant, num_classes=100)
        x = torch.randn(2, 2, 96, 511)  # BirdNET-style input
        output = model(x)
        print(f"  Classification output shape: {output.shape}")
        
        # Feature extraction model
        backbone = EfficientNet(model_variant, include_top=False)
        features = backbone(x)
        print(f"  Feature extraction output shape: {features.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
    
    print("\nAll tests passed!")