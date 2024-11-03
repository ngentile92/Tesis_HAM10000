# models/vision_transformer.py

import timm
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


def create_model(model_name, pretrained, num_classes, use_additional_features=False):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    
    if use_additional_features:
        # Add a new fully connected layer to handle the additional features
        model.additional_fc = nn.Sequential(
            nn.Linear(3, 32),  # 3 variables: age, sex, localization
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # Update the final classification layer to include the output of additional_fc
        model.fc = nn.Sequential(
            nn.Linear(model.num_features + 16, num_classes)
        )
    return model
