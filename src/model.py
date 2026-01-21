import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_model(num_classes=102, pretrained=True):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    print(f'Using ResNet18 pretrained: {pretrained}')
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model