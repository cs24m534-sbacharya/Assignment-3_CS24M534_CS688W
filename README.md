# Assignment-3_CS24M534_CS688W

Project Overview: MobileNet-v2 on CIFAR-10 with quantization-based compression.

Features: Baseline training, configurable quantization, W&B integration.

File Structure: Explains each modular file (main.py, dataloader.py, mobilenetv2.py, quantize.py, utils.py, test.py).

    main.py
        Purpose: Handles baseline training of MobileNet-v2 on CIFAR-10.
        Key Features:
        
        Initializes W&B for logging.
        Implements training loop with SGD optimizer and CosineAnnealingLR scheduler.
        Saves trained model as mobilenetv2_cifar10.pth.
    
    dataloader.py
        Purpose: Provides data loading utilities for CIFAR-10.
        Key Features:
        
        Applies normalization and data augmentation (RandomCrop, HorizontalFlip).
        Returns PyTorch DataLoader objects for training and testing sets.
        
    mobilenetv2.py
        Purpose: Defines MobileNet-v2 architecture.
        Key Features:
        
        Wrapper around torchvision.models.mobilenet_v2.
        Configurable number of output classes (default: 10 for CIFAR-10).
    
    quantize.py
        Purpose: Implements custom quantization logic for weights and activations.
        Key Features:
        
        quantize_tensor(): Performs uniform quantization and dequantization.
        apply_weight_quantization(): Quantizes model weights.
        compress_model(): Registers forward hooks for activation quantization.
    
    utils.py
        Purpose: Contains utility functions for evaluation and analysis.
        Key Features:
        
        evaluate(): Computes accuracy on a given dataset.
        calculate_model_size(): Estimates model size based on bitwidth.
        print_compression(): Displays compression ratio and size details.
    
    test.py
        Purpose: CLI script to apply quantization and evaluate performance.
        Key Features:
        
        Accepts --weight_quant_bits and --activation_quant_bits as arguments.
        Loads baseline model, applies quantization, and prints accuracy + compression ratio.

Installation Instructions: Python version and dependencies.

Training Command:
    python main.py

Testing Command:
    python test.py --weight_quant_bits 8 --activation_quant_bits 8

Expected Outputs: Accuracy and compression ratio.
