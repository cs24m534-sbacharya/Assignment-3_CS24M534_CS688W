from torchvision import models
import torch.nn as nn

def mobilenet_v2(num_classes=10):
    model = models.mobilenet_v2(num_classes=num_classes)
    return model
