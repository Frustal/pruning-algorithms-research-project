import torch.nn as nn
from torchvision import models

_MODEL_CONFIG = {
    "resnet18":  (models.resnet18,  models.ResNet18_Weights),
    "resnet34":  (models.resnet34,  models.ResNet34_Weights),
    "resnet50":  (models.resnet50,  models.ResNet50_Weights),
    "resnet101": (models.resnet101, models.ResNet101_Weights),
    "resnet152": (models.resnet152, models.ResNet152_Weights),
    "resnet18_custom": (models.resnet18, models.ResNet18_Weights),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights),
}

def get_model(model_name="resnet18", num_classes=102, pretrained=True):
    if model_name not in _MODEL_CONFIG:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Available models: {list(_MODEL_CONFIG.keys())}"
        )

    model_fn, weights_enum = _MODEL_CONFIG[model_name]
    weights = weights_enum.DEFAULT if pretrained else None

    print(f"Using {model_name}, pretrained={pretrained}")

    model = model_fn(weights=weights)
    
    if model_name == "resnet18_custom":
        # [2, 2, 1, 1] architecture
        model.layer3 = nn.Sequential(model.layer3[0])
        model.layer4 = nn.Sequential(model.layer4[0])
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        # EfficientNet classifier
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        # Standard ResNets
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
