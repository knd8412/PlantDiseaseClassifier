import torch
import torch.nn as nn

from src.models.resnet import ResNet18Classifier


def test_resnet18_output_match():
    num_classes = 17
    model = ResNet18Classifier(
        num_classes=num_classes,
        pretrained=False,
        dropout=0.0,
        train_backbone=True,
    )

    fc = model.model.fc
    if isinstance(fc, nn.Sequential):
        linear = fc[1]
    else:
        linear = fc

    assert isinstance(linear, nn.Linear)
    assert linear.out_features == num_classes
