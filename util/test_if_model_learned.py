import torch

def check_model_training_status(model, whole_params=True, each_layer=True):
    """Function to check if model is trained"""


    # Statistics for all parameters
    if whole_params:
        print("=== Model Parameter Statistics ===")
        all_params = []
        for name, param in model.named_parameters():
            all_params.append(param.data.flatten())

        if all_params:
            all_params_tensor = torch.cat(all_params)

            print(f"Total number of parameters: {len(all_params_tensor)}")
            print(f"Mean value: {all_params_tensor.mean():.6f}")
            print(f"Standard deviation: {all_params_tensor.std():.6f}")
            print(f"Minimum value: {all_params_tensor.min():.6f}")
            print(f"Maximum value: {all_params_tensor.max():.6f}")

            # Proportion of parameters close to zero
            zero_like_params = (torch.abs(all_params_tensor) < 1e-6).float().mean()
            print(f"Proportion of parameters close to zero: {zero_like_params:.4f}")

            # Determine if trained
            if zero_like_params > 0.002:
                print("⚠️  Warning: Many parameters are close to zero. Model may be untrained.")
            elif all_params_tensor.std() < 1e-3:
                print("⚠️  Warning: Parameter variance is very small.")
            else:
                print("✅ Parameter distribution is normal. Model is likely trained.")

    if each_layer:
        print("\n=== Layer-wise Parameter Statistics ===")
        for name, param in model.named_parameters():
            if param.numel() > 0:
                print(f"{name}:")
                print(f"  Shape: {param.shape}")
                print(f"  Mean: {param.data.mean():.6f}")
                print(f"  Std: {param.data.std():.6f}")
                print(f"  Range: [{param.data.min():.6f}, {param.data.max():.6f}]")

def check_specific_layers(model):
    """Detailed check of important layer weights"""

    print("=== Important Layer Weight Check ===")

    # Check linear layer weights
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(f"\nLinear layer: {name}")
            weight = module.weight.data
            bias = module.bias.data if module.bias is not None else None

            print(f"  Weight shape: {weight.shape}")
            print(f"  Weight mean: {weight.mean():.6f}")
            print(f"  Weight std: {weight.std():.6f}")

            if bias is not None:
                print(f"  Bias mean: {bias.mean():.6f}")
                print(f"  Bias std: {bias.std():.6f}")

    # Check convolution layer weights
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv1d):
            print(f"\nConvolution layer: {name}")
            weight = module.weight.data
            print(f"  Weight shape: {weight.shape}")
            print(f"  Weight mean: {weight.mean():.6f}")
            print(f"  Weight std: {weight.std():.6f}")

def test_model_inference(model, device):
    """Test if model can perform inference normally"""

    print("=== Inference Test ===")

    # Test with dummy data
    batch_size = 2
    seq_len = 336  # Adjust according to configuration
    num_features = 7  # Adjust according to configuration

    dummy_input = torch.randn(batch_size, seq_len, num_features).to(device)

    try:
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)

        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output mean: {output.mean():.6f}")
        print(f"Output std: {output.std():.6f}")

        # Check if output contains NaN or Inf
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("❌ Output contains NaN or Inf")
        else:
            print("✅ Inference test successful")

    except Exception as e:
        print(f"❌ Inference test failed: {e}")


def check_training_history(checkpoint):
    """Check training history from checkpoint"""

    print("=== Training History Check ===")

    if 'args' in checkpoint:
        args = checkpoint['args']
        print("Training configuration:")
        for key, value in vars(args).items():
            if not key.startswith('_'):
                print(f"  {key}: {value}")

    if 'epoch' in checkpoint:
        print(f"Training epoch: {checkpoint['epoch']}")

    if 'best_model_loss' in checkpoint:
        print(f"Best model loss: {checkpoint['best_model_loss']}")

    if 'train_losses' in checkpoint:
        train_losses = checkpoint['train_losses']
        print(f"Training loss history: {len(train_losses)} epochs")
        if len(train_losses) > 0:
            print(f"  Final loss: {train_losses[-1]:.6f}")
            print(f"  Minimum loss: {min(train_losses):.6f}")


def comprehensive_model_check(model, checkpoint_path=None):
    """Comprehensive model check"""

    print("🔍 Comprehensive Model Training Status Check")
    print("=" * 50)

    # 1. Check checkpoint file
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print("1. Checkpoint file:")
        print(f"   Keys: {list(checkpoint.keys())}")

    # 2. Check parameter statistics
    print("\n2. Parameter statistics:")
    check_model_training_status(model, whole_params=True, each_layer=False)

    # 3. Inference test
    # print("\n3. Inference test:")
    # test_model_inference(model, model.device if hasattr(model, 'device') else 'cpu')

    # 4. Check training history
    if checkpoint_path is not None:
        print("\n4. Training history:")
        check_training_history(checkpoint)

    print("\n" + "=" * 50)
    # print("Check complete")
