# Assignment-3_CS24M534_CS688W

Project Overview: MobileNet-v2 on CIFAR-10 with quantization-based compression.

Features: Baseline training, configurable quantization, W&B integration.

File Structure: Explains each modular file (main.py, dataloader.py, mobilenetv2.py, quantize.py, utils.py, test.py).

Installation Instructions: Python version and dependencies.

Training Command:
    python main.py

Testing Command:
    python test.py --weight_quant_bits 8 --activation_quant_bits 8

Expected Outputs: Accuracy and compression ratio.
