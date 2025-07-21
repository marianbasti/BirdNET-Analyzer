#!/usr/bin/env python3
"""
Example script demonstrating how to pretrain a Whisper-based BirdNET model.

This script shows how to use the new Whisper backbone instead of the original EfficientNet
for self-supervised pretraining on unlabeled audio data.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import argparse
from pathlib import Path

# Add the parent directory to path so we can import birdnet_analyzer
import sys
sys.path.append(str(Path(__file__).parent.parent))

from birdnet_analyzer.torch_pretrain_utils import (
    UnlabeledAudioDataset, 
    SimCLRPretrainer, 
    collate_fn
)

def main():
    parser = argparse.ArgumentParser(description="Pretrain Whisper-based BirdNET model")
    parser.add_argument("--audio_dir", type=str, required=True, 
                        help="Directory containing unlabeled audio files")
    parser.add_argument("--epochs", type=int, default=100, 
                        help="Number of pretraining epochs")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--emb_size", type=int, default=1024, 
                        help="Embedding size")
    parser.add_argument("--d_model", type=int, default=512, 
                        help="Transformer model dimension")
    parser.add_argument("--n_heads", type=int, default=8, 
                        help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=6, 
                        help="Number of transformer layers")
    parser.add_argument("--save_path", type=str, default="whisper_pretrained_backbone.pt", 
                        help="Path to save the pretrained backbone")
    parser.add_argument("--checkpoint_every", type=int, default=10, 
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume_from", type=str, default=None, 
                        help="Resume training from checkpoint")
    parser.add_argument("--use_amp", action="store_true", 
                        help="Use automatic mixed precision")
    
    args = parser.parse_args()
    
    # Check if audio directory exists
    if not os.path.exists(args.audio_dir):
        raise ValueError(f"Audio directory {args.audio_dir} does not exist")
    
    print(f"Setting up Whisper-based pretraining...")
    print(f"Audio directory: {args.audio_dir}")
    print(f"Device: {args.device}")
    print(f"Model config: d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}")
    
    # Create dataset and dataloader
    dataset = UnlabeledAudioDataset(
        audio_dir=args.audio_dir,
        sample_rate=48000,
        min_len=1.0,
        max_len=10.0
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True if args.device == "cuda" else False
    )
    
    print(f"Created dataset with {len(dataset)} audio files")
    print(f"Batch size: {args.batch_size}, Batches per epoch: {len(dataloader)}")
    
    # Initialize pretrainer with Whisper backbone
    pretrainer = SimCLRPretrainer(
        emb_size=args.emb_size,
        proj_dim=128,
        n_mels=80,  # Whisper standard
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        device=args.device,
        seed=42,
        log_wandb=False,  # Set to True if you want to use wandb logging
        run_name=f"whisper_pretrain_{args.d_model}d_{args.n_layers}l"
    )
    
    # Start pretraining
    print("Starting pretraining...")
    try:
        pretrainer.train(
            dataloader=dataloader,
            epochs=args.epochs,
            lr=args.lr,
            save_path=args.save_path,
            checkpoint_every=args.checkpoint_every,
            resume_from=args.resume_from,
            use_amp=args.use_amp
        )
        print("Pretraining completed successfully!")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
        # Save current state
        torch.save({
            'spec_layer': pretrainer.spec_layer.state_dict(),
            'backbone': pretrainer.backbone.state_dict(),
            'proj_head': pretrainer.proj_head.state_dict(),
        }, 'interrupted_checkpoint.pt')
        print("Saved interrupted checkpoint to 'interrupted_checkpoint.pt'")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise

def test_model():
    """Test function to verify the Whisper backbone works correctly."""
    print("Testing Whisper backbone...")
    
    from birdnet_analyzer.torch_model import BirdNetTorchModel
    
    # Create a test model
    model = BirdNetTorchModel(
        num_classes=3000,  # Example number of bird species
        emb_size=512,
        n_mels=80,
        d_model=256,  # Smaller for testing
        n_heads=4,
        n_layers=3
    )
    
    # Create test input (3 seconds of audio at 48kHz)
    batch_size = 2
    audio_length = 48000 * 3  # 3 seconds
    test_input = torch.randn(batch_size, audio_length)
    
    print(f"Input shape: {test_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, 3000)")
    
    assert output.shape == (batch_size, 3000), f"Unexpected output shape: {output.shape}"
    print("✓ Model test passed!")
    
    # Test individual components
    spec_output = model.spec_layer(test_input)
    print(f"Spectrogram output shape: {spec_output.shape}")
    
    backbone_output = model.backbone(spec_output)
    print(f"Backbone output shape: {backbone_output.shape}")
    
    print("✓ All component tests passed!")

if __name__ == "__main__":
    # Add test mode
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_model()
    else:
        main()
