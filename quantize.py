import torch

def quantize_tensor(tensor, num_bits=8):
    qmin, qmax = 0, 2 ** num_bits - 1
    min_val, max_val = tensor.min(), tensor.max()
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - min_val / scale
    quantized = ((tensor / scale) + zero_point).round().clamp(qmin, qmax)
    dequantized = (quantized - zero_point) * scale
    return dequantized, scale, zero_point

def apply_weight_quantization(model, weight_bits=8):
    for name, param in model.named_parameters():
        if 'weight' in name:
            param.data, _, _ = quantize_tensor(param.data, num_bits=weight_bits)
    return model

def compress_model(model, weight_bits=8, activation_bits=8):
    def activation_hook(module, input, output):
        quantized_output, _, _ = quantize_tensor(output, num_bits=activation_bits)
        return quantized_output

    model = apply_weight_quantization(model, weight_bits=weight_bits)
    for layer in model.features:
        layer.register_forward_hook(activation_hook)
    return model
