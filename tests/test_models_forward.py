import torch

from models.convnet_scratch import build_model as build_scratch_model
from models.resnet import ResNet18Classifier


def test_scratch_cnn_forward_shape():
    num_classes = 39
    model = build_scratch_model(
        num_classes=num_classes,
        channels=[16, 32],
        regularisation="batchnorm",
        dropout=0.0,
    )
    x = torch.randn(4, 3, 128, 128)
    out = model(x)
    assert out.shape == (4, num_classes)


def test_resnet18_forward_shape_no_pretrain():
    num_classes = 39
    # pretrained=False so the test does not try to download weights
    model = ResNet18Classifier(
        num_classes=num_classes, pretrained=False, dropout=0.0, train_backbone=True
    )
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, num_classes)
