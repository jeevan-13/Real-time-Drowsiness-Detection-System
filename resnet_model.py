import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os
import sys

DATA_DIR    = "train"      
BATCH_SIZE  = 32
NUM_CLASSES = 2
NUM_EPOCHS  = 5
LR          = 1e-3
VAL_SPLIT   = 0.5         
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

common_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    *common_transforms.transforms
])
val_transforms = common_transforms

def train():
    print("Starting ResNet-18 training...", flush=True)

    full_ds = datasets.ImageFolder(DATA_DIR, transform=None)
    total = len(full_ds)
    val_n = int(total * VAL_SPLIT)
    train_n = total - val_n
    print(f"Found {total} images → {train_n} train | {val_n} val", flush=True)
    print("Classes:", full_ds.classes, flush=True)

    train_ds, val_ds = random_split(
        full_ds, [train_n, val_n],
        generator=torch.Generator().manual_seed(42)
    )
    train_ds.dataset.transform = train_transforms
    val_ds.dataset.transform   = val_transforms

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = models.resnet18(weights=None)
    in_feat = model.fc.in_features
    model.fc = nn.Linear(in_feat, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===", flush=True)
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            running_corrects += (outputs.argmax(1) == labels).sum().item()

        epoch_loss = running_loss / train_n
        epoch_acc  = running_corrects / train_n

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                val_corrects += (outputs.argmax(1) == labels).sum().item()

        val_loss /= val_n
        val_acc  = val_corrects / val_n

        print(f"Train: loss {epoch_loss:.4f}, acc {epoch_acc:.4f}", flush=True)
        print(f" Val : loss {val_loss:.4f}, acc {val_acc:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "resnet18_best.pth")
            print(f"→ Saved new best model (val acc {best_val_acc:.4f})", flush=True)

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}", flush=True)

if __name__ == "__main__":
    if not os.path.isdir(DATA_DIR):
        print(f"Error: DATA_DIR '{DATA_DIR}' not found.", file=sys.stderr)
        sys.exit(1)
    train()
