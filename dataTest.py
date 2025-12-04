# clearml_cnn_test.py
import torch
import torch.nn as nn
import torch.optim as optim
from clearml import Logger, Task

from data.dataset import load_dataset_and_dataloaders, show_batch

# -----------------------------
# CREATE CLEARML TASK
# -----------------------------
task = Task.init(
    project_name="PlantDisease_Test",
    task_name="QuickCNN_Run",
    task_type=Task.TaskTypes.training,
)

# -----------------------------
# SELECT DATASET
# -----------------------------
SELECTED_DATASET = "large"  # Options: 'tiny', 'medium', 'large'

train_loader, val_loader, test_loader, class_names = load_dataset_and_dataloaders(
    dataset_size=SELECTED_DATASET
)

# -----------------------------
# DEFINE SIMPLE CNN
# -----------------------------
IMAGE_SIZE = 224  # Set to match your dataset images


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224 -> 112
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112 -> 56
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56 -> 28
        )

        h, w = IMAGE_SIZE, IMAGE_SIZE
        self.flatten_size = 64 * (h // 8) * (w // 8)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -----------------------------
# SETUP MODEL, LOSS, OPTIMIZER
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"🚀 Training on {device} for 5 epochs...")

# -----------------------------
# TRAINING LOOP
# -----------------------------
for epoch in range(50):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:
        inputs = batch[0].to(device)
        labels = batch[1].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    print(f"Epoch [{epoch+1}/5] Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

    # Log to ClearML
    task.get_logger().report_scalar("Loss", "train", epoch_loss, epoch)
    task.get_logger().report_scalar("Accuracy", "train", epoch_acc, epoch)

print("🏁 Training finished successfully!")

# -----------------------------
# VISUALIZE SOME BATCHES
# -----------------------------
print("Visualizing Training Batch (with Augmentations):")
show_batch(train_loader, class_names, num_images=8, denorm=True)

# Log batch images to ClearML
logger = task.get_logger()
for i, batch in enumerate(train_loader):
    imgs = batch["image"][:8]  # first 8 images
    logger.report_image(
        title=f"Train Batch {i+1}", series="batch_samples", image=imgs, iteration=i
    )
    break  # Only log first batch
