import argparse, os, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from collections import Counter
import numpy as np

CLASSES = ["paper", "plastic", "metal", "glass", "other"]

def build_dataloaders(data_dir: Path, img_size=224, batch_size=16):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = datasets.ImageFolder(str(data_dir/"train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(data_dir/"val"),   transform=val_tf)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    return train_dl, val_dl, train_ds.classes

def build_model(num_classes=5, backbone="resnet18", pretrained=True):
    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
    elif backbone == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)
        in_f = model.heads.head.in_features
        model.heads.head = nn.Linear(in_f, num_classes)
    else:
        raise ValueError("Unsupported backbone")
    return model

def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        loss_sum += loss.item()*x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18","vit_b_16"])
    ap.add_argument("--model", type=str, default="models/recycle_resnet18.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl, val_dl, classes = build_dataloaders(Path(args.data), batch_size=args.batch)
    print(f"Classes: {classes}")

    # 클래스 가중치 계산
    train_ds = datasets.ImageFolder(str(Path(args.data)/"train"))
    counts = Counter([y for _, y in train_ds.samples])
    num_classes = len(train_ds.classes)
    freq = np.array([counts.get(i, 1) for i in range(num_classes)], dtype=np.float32)
    class_weights = torch.tensor(freq.sum() / (freq + 1e-6), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    model = build_model(num_classes=len(classes), backbone=args.backbone).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_acc, best_path = 0.0, args.model
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, device, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_dl, device, criterion)
        print(f"[{epoch}/{args.epochs}] train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {val_loss:.4f} acc {val_acc:.3f}")
        scheduler.step(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(Path(best_path).parent, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "classes": classes, "backbone": args.backbone}, best_path)
            print(f"  -> saved: {best_path} (acc={best_acc:.3f})")

if __name__ == "__main__":
    main()
