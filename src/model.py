import torch.nn as nn
from torchvision import models

_RESNET_CONFIG = {
    "resnet18":  (models.resnet18,  models.ResNet18_Weights),
    "resnet34":  (models.resnet34,  models.ResNet34_Weights),
    "resnet50":  (models.resnet50,  models.ResNet50_Weights),
    "resnet101": (models.resnet101, models.ResNet101_Weights),
    "resnet152": (models.resnet152, models.ResNet152_Weights),
}

def get_model(model_name="resnet18", num_classes=102, pretrained=True):
    if model_name not in _RESNET_CONFIG:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Available models: {list(_RESNET_CONFIG.keys())}"
        )

    model_fn, weights_enum = _RESNET_CONFIG[model_name]
    weights = weights_enum.DEFAULT if pretrained else None

    print(f"Using {model_name}, pretrained={pretrained}")

    model = model_fn(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
