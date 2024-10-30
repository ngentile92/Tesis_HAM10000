# models/vision_transformer.py

import timm
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

def create_model(model_name, pretrained, num_classes):
    if model_name.startswith('vit'):
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Modelo {model_name} no soportado.")
    return model
