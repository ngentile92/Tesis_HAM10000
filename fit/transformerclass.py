# Description: Clase VisionTransformerClassifier que hereda de nn.Module y que se encarga de definir la arquitectura de la red
import torch
import torch.nn as nn


# Definición de la clase VisionTransformerClassifier
class VisionTransformerClassifier(nn.Module):
    def __init__(self, vit, num_classes):
        super(VisionTransformerClassifier, self).__init__()
        self.vit = vit
        self.classifier = nn.Linear(vit.config.hidden_size, num_classes)
    
    def forward(self, x):
        outputs = self.vit(x)
        cls_output = outputs.last_hidden_state[:, 0]  # Token [CLS]
        logits = self.classifier(cls_output)
        return logits