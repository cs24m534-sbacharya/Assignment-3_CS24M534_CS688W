import torch
import argparse
from mobilenetv2 import mobilenet_v2
from dataloader import get_cifar10
from utils import evaluate, print_compression
from quantize import compress_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantization Test')
    parser.add_argument('--weight_quant_bits', type=int, default=8, help='Bits for weight quantization')
    parser.add_argument('--activation_quant_bits', type=int, default=8, help='Bits for activation quantization')
    args = parser.parse_args()

    train_loader, test_loader = get_cifar10(batch_size=128)
    model = mobilenet_v2(num_classes=10)
    model.load_state_dict(torch.load('mobilenetv2_cifar10.pth', map_location=device))
    model.to(device)

    print("Evaluating baseline model...")
    baseline_acc = evaluate(model, test_loader, device)
    print(f"Baseline Accuracy: {baseline_acc:.2f}%")

    print("Applying quantization...")
    compressed_model = compress_model(model, weight_bits=args.weight_quant_bits, activation_bits=args.activation_quant_bits)
    compressed_model.to(device)
    quant_acc = evaluate(compressed_model, test_loader, device)
    print(f"Quantized Accuracy: {quant_acc:.2f}%")
    print_compression(model, weight_bits=args.weight_quant_bits)
