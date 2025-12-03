import torch

from src.models.resnet import ResNet18Classifier

ckpt_path = "src/models/checkpoints/resnet18_best.pt"
device = torch.device("cpu")

ckpt = torch.load(ckpt_path, map_location=device)
state = ckpt.get("model_state", ckpt)

# Build a dummy model using your YAML num_classes
model = ResNet18Classifier(
    num_classes=39,
    pretrained=False,
    dropout=0.2,
    train_backbone=True,
)

# Load weights (will fail if wrong number of classes)
missing, unexpected = model.load_state_dict(state, strict=False)

print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

# Now inspect the final layer shape
for name, param in model.named_parameters():
    if "fc.weight" in name or "classifier.2.weight" in name:
        print("Head weight shape:", param.shape)
