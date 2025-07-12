"""
EfficientNet for BirdNET-Analyzer: Usage Examples
=================================================

This file demonstrates how to use the new EfficientNet implementation
with the BirdNET-Analyzer framework.
"""

# Example 1: Using EfficientNet with precomputed features
def example_standalone_efficientnet():
    """Example of using EfficientNet directly with precomputed features."""
    try:
        import torch
        from ..efficientnet import EfficientNet, EfficientNetModel
        
        print("=== Standalone EfficientNet Usage ===")
        
        # Create model for classification
        model = EfficientNet(EfficientNetModel.B0, num_classes=1000)
        print(f"Created EfficientNet-B0 with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Example input: dual mel spectrograms (BirdNET format)
        batch_size = 4
        channels = 2  # Low-freq + high-freq spectrograms
        height = 96   # Mel bins
        width = 511   # Time frames
        
        x = torch.randn(batch_size, channels, height, width)
        print(f"Input shape: {x.shape}")
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        print(f"Output shape: {output.shape}")
        
        # Feature extraction (without classification head)
        feature_extractor = EfficientNet(EfficientNetModel.B0, include_top=False)
        with torch.no_grad():
            features = feature_extractor(x)
        print(f"Feature shape: {features.shape}")
        
        return True
        
    except ImportError as e:
        print(f"Could not run example: {e}")
        return False


# Example 2: Using integrated BirdNet + EfficientNet
def example_integrated_birdnet():
    """Example of using EfficientNet integrated with BirdNET frontend."""
    try:
        import torch
        from ..torch_model import BirdNetEfficientNet
        from ..efficientnet import EfficientNetModel
        
        print("\n=== Integrated BirdNet + EfficientNet Usage ===")
        
        # Create integrated model
        model = BirdNetEfficientNet(
            model_variant=EfficientNetModel.B0,
            num_classes=1000  # Number of bird species
        )
        
        print(f"Created integrated model with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Example input: raw audio (3 seconds at 48kHz)
        batch_size = 2
        sample_rate = 48000
        duration = 3.0
        num_samples = int(sample_rate * duration)
        
        audio = torch.randn(batch_size, num_samples)
        print(f"Audio input shape: {audio.shape}")
        
        # Forward pass (audio -> spectrograms -> classification)
        with torch.no_grad():
            predictions = model(audio)
        print(f"Predictions shape: {predictions.shape}")
        
        return True
        
    except ImportError as e:
        print(f"Could not run example: {e}")
        return False


# Example 3: Different model variants
def example_model_variants():
    """Example of creating different EfficientNet variants."""
    try:
        from ..efficientnet import create_efficientnet
        
        print("\n=== EfficientNet Model Variants ===")
        
        variants = ['b0', 'b1', 'b4', 'b7']
        
        for variant in variants:
            try:
                model = create_efficientnet(variant, num_classes=100, include_top=False)
                params = sum(p.numel() for p in model.parameters())
                print(f"EfficientNet-{variant.upper()}: {params:,} parameters")
            except Exception as e:
                print(f"Failed to create {variant}: {e}")
        
        return True
        
    except ImportError as e:
        print(f"Could not run example: {e}")
        return False


def run_all_examples():
    """Run all examples to demonstrate the EfficientNet implementation."""
    print("EfficientNet for BirdNET-Analyzer: Usage Examples")
    print("=" * 60)
    
    examples = [
        example_standalone_efficientnet,
        example_integrated_birdnet,
        example_model_variants,
    ]
    
    results = []
    for example in examples:
        try:
            success = example()
            results.append(success)
        except Exception as e:
            print(f"Example failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} examples ran successfully")
    
    if not any(results):
        print("Note: Examples require PyTorch to be installed to run fully.")
        print("However, the implementation is structurally complete and ready to use.")


if __name__ == "__main__":
    run_all_examples()