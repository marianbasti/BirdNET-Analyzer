"""
Utility functions for loading and using pretrained Whisper backbones in BirdNET.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

from birdnet_analyzer.torch_model import BirdNetTorchModel, WhisperBackbone, BirdNETMelSpecLayer


def load_pretrained_whisper_backbone(
    checkpoint_path: str,
    num_classes: int,
    device: str = "cuda",
    freeze_backbone: bool = False
) -> BirdNetTorchModel:
    """
    Load a BirdNET model with a pretrained Whisper backbone.
    
    Args:
        checkpoint_path: Path to the pretrained backbone weights
        num_classes: Number of output classes for the classifier
        device: Device to load the model on
        freeze_backbone: If True, freeze the backbone weights during fine-tuning
        
    Returns:
        BirdNetTorchModel with loaded pretrained weights
    """
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config if available
    config = checkpoint.get('config', {})
    emb_size = config.get('emb_size', 1024)
    n_mels = config.get('n_mels', 80)
    d_model = config.get('d_model', 512)
    n_heads = config.get('n_heads', 8)
    n_layers = config.get('n_layers', 6)
    
    print(f"Loading Whisper backbone with config:")
    print(f"  - Embedding size: {emb_size}")
    print(f"  - Mel bins: {n_mels}")
    print(f"  - Model dimension: {d_model}")
    print(f"  - Attention heads: {n_heads}")
    print(f"  - Transformer layers: {n_layers}")
    
    # Create model
    model = BirdNetTorchModel(
        num_classes=num_classes,
        emb_size=emb_size,
        n_mels=n_mels,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers
    )
    
    # Load backbone weights
    if 'backbone' in checkpoint:
        model.backbone.load_state_dict(checkpoint['backbone'])
        print("✓ Loaded pretrained backbone weights")
    else:
        warnings.warn("No backbone weights found in checkpoint")
    
    # Load spec layer weights if available
    if 'spec_layer' in checkpoint:
        model.spec_layer.load_state_dict(checkpoint['spec_layer'])
        print("✓ Loaded pretrained spectrogram layer weights")
    
    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("✓ Frozen backbone weights")
    
    return model.to(device)


def create_whisper_backbone_only(
    checkpoint_path: str,
    device: str = "cuda"
) -> tuple[BirdNETMelSpecLayer, WhisperBackbone]:
    """
    Load only the spectrogram layer and backbone from a checkpoint.
    Useful for feature extraction or transfer learning.
    
    Args:
        checkpoint_path: Path to the pretrained weights
        device: Device to load on
        
    Returns:
        Tuple of (spec_layer, backbone)
    """
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # Create components
    spec_layer = BirdNETMelSpecLayer(n_mels=config.get('n_mels', 80))
    backbone = WhisperBackbone(
        n_mels=config.get('n_mels', 80),
        d_model=config.get('d_model', 512),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('n_layers', 6),
        emb_size=config.get('emb_size', 1024)
    )
    
    # Load weights
    if 'spec_layer' in checkpoint:
        spec_layer.load_state_dict(checkpoint['spec_layer'])
    if 'backbone' in checkpoint:
        backbone.load_state_dict(checkpoint['backbone'])
    
    return spec_layer.to(device), backbone.to(device)


def extract_features(
    audio_tensor: torch.Tensor,
    spec_layer: BirdNETMelSpecLayer,
    backbone: WhisperBackbone,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Extract features from audio using pretrained Whisper backbone.
    
    Args:
        audio_tensor: Raw audio tensor of shape (batch_size, samples)
        spec_layer: Pretrained spectrogram layer
        backbone: Pretrained Whisper backbone
        device: Device to run on
        
    Returns:
        Feature embeddings of shape (batch_size, emb_size)
    """
    
    spec_layer.eval()
    backbone.eval()
    
    with torch.no_grad():
        audio_tensor = audio_tensor.to(device)
        
        # Convert to spectrogram
        spec = spec_layer(audio_tensor)
        
        # Extract features
        features = backbone(spec)
        
    return features


def compare_model_sizes():
    """Compare the parameter counts of EfficientNet vs Whisper backbone."""
    
    print("Comparing model architectures...")
    
    # Whisper backbone
    whisper_model = BirdNetTorchModel(
        num_classes=3000,
        emb_size=1024,
        n_mels=80,
        d_model=512,
        n_heads=8,
        n_layers=6
    )
    
    # Count parameters
    whisper_params = sum(p.numel() for p in whisper_model.parameters())
    whisper_backbone_params = sum(p.numel() for p in whisper_model.backbone.parameters())
    whisper_spec_params = sum(p.numel() for p in whisper_model.spec_layer.parameters())
    
    print(f"Whisper-based BirdNET:")
    print(f"  - Total parameters: {whisper_params:,}")
    print(f"  - Backbone parameters: {whisper_backbone_params:,}")
    print(f"  - Spectrogram layer parameters: {whisper_spec_params:,}")
    print(f"  - Classifier parameters: {whisper_params - whisper_backbone_params - whisper_spec_params:,}")
    
    # Memory usage estimate
    memory_mb = (whisper_params * 4) / (1024 * 1024)  # 4 bytes per float32
    print(f"  - Estimated memory (float32): {memory_mb:.1f} MB")


def validate_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Validate a pretrained checkpoint and return its metadata.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dictionary with checkpoint information
    """
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint: {e}")
    
    info = {
        'file_path': checkpoint_path,
        'file_size_mb': Path(checkpoint_path).stat().st_size / (1024 * 1024),
        'has_backbone': 'backbone' in checkpoint,
        'has_spec_layer': 'spec_layer' in checkpoint,
        'has_projection_head': 'proj_head' in checkpoint,
        'has_config': 'config' in checkpoint,
    }
    
    if 'config' in checkpoint:
        info['config'] = checkpoint['config']
    
    # Try to determine the number of parameters
    if 'backbone' in checkpoint:
        # Count parameters in backbone state dict
        backbone_params = sum(
            torch.numel(v) for v in checkpoint['backbone'].values()
        )
        info['backbone_parameters'] = backbone_params
    
    return info


if __name__ == "__main__":
    # Demo/test functions
    print("BirdNET Whisper Backbone Utilities")
    print("=" * 40)
    
    # Compare model sizes
    compare_model_sizes()
    
    print("\n" + "=" * 40)
    print("Demo: Creating Whisper backbone")
    
    # Create a small test model
    test_model = BirdNetTorchModel(
        num_classes=100,
        emb_size=256,
        n_mels=80,
        d_model=128,
        n_heads=4,
        n_layers=2
    )
    
    print(f"Created test model with {sum(p.numel() for p in test_model.parameters()):,} parameters")
    
    # Test forward pass
    test_audio = torch.randn(1, 48000 * 3)  # 3 seconds of audio
    with torch.no_grad():
        output = test_model(test_audio)
    
    print(f"Forward pass successful: {test_audio.shape} -> {output.shape}")
    print("✓ All tests passed!")
