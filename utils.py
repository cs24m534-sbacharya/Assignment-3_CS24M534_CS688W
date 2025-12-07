import torch

def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

def calculate_model_size(model, bits_per_param=32):
    total_params = sum(p.numel() for p in model.parameters())
    size_bytes = total_params * (bits_per_param / 8)
    return size_bytes / (1024 * 1024)

def print_compression(model, weight_bits=8):
    baseline_size = calculate_model_size(model, bits_per_param=32)
    compressed_size = calculate_model_size(model, bits_per_param=weight_bits)
    compression_ratio = baseline_size / compressed_size
    print(f"Baseline Model Size: {baseline_size:.2f} MB")
    print(f"Compressed Model Size: {compressed_size:.2f} MB")
    print(f"Compression Ratio: {compression_ratio:.2f}x")
