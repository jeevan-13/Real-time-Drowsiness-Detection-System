import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import sys

class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.8)
        self.fc1 = nn.Linear(32 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 2)  

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = x.view(-1, 32 * 6 * 6)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

DATA_DIR    = "train"     
BATCH_SIZE  = 32
NUM_CLASSES = 2
NUM_EPOCHS  = 5
LR          = 1e-3
VAL_SPLIT   = 0.8
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((24, 24)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])
val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((24, 24)),
    transforms.ToTensor()
])


def train():
    if not os.path.isdir(DATA_DIR):
        print(f"Error: DATA_DIR '{DATA_DIR}' not found.", file=sys.stderr)
        sys.exit(1)

    full_ds = datasets.ImageFolder(DATA_DIR, transform=None)
    total = len(full_ds)
    val_n = int(total * VAL_SPLIT)
    train_n = total - val_n
    print(f"Found {total} images → {train_n} train | {val_n} val")
    print("Classes:", full_ds.classes)

    train_ds, val_ds = random_split(
        full_ds,
        [train_n, val_n],
        generator=torch.Generator().manual_seed(42)
    )
    train_ds.dataset.transform = train_transforms
    val_ds.dataset.transform   = val_transforms

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = BasicCNN().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS+1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        model.train()
        running_loss, running_corrects = 0.0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            running_corrects += (outputs.argmax(1) == labels).sum().item()
        train_loss = running_loss / train_n
        train_acc  = running_corrects / train_n

        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                val_corrects += (outputs.argmax(1) == labels).sum().item()
        val_loss = val_loss / val_n
        val_acc  = val_corrects / val_n

        print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f" Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "basic_cnn_eye_best.pth")
            print(f"→ New best model saved (val acc {best_val_acc:.4f})")

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    train()
