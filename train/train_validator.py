import os
import random
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import classification_report, roc_auc_score

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ========== 1. Конфигурация ==========
DATA_DIR = "./datasets/MIDV2020"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5
BATCH_SIZE = 16
IMG_SIZE = 224
LR = 1e-4
MODEL_PATH = "validator_model.pth"


# ========== 2. Датасет с синтетикой ==========
class DocumentDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment_prob=0.5):
        self.images = []
        self.labels = []  # 0 = valid, 1 = fake

        for mode in ["photo", "scan_upright"]:
            base_dir = os.path.join(root_dir, mode, "images")
            if not os.path.exists(base_dir):
                continue

            # теперь обходим все подпапки, например alb_id, rus_internalpassport и т.д.
            for subdir, _, files in os.walk(base_dir):
                for fname in files:
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.images.append(os.path.join(subdir, fname))
                        self.labels.append(0)

        self.transform = transform
        self.augment_prob = augment_prob

        # "Поддельные" эффекты
        self.augment = A.Compose([
            A.OneOf([
                A.MotionBlur(p=0.3),
                A.GaussNoise(p=0.3),
                A.RandomBrightnessContrast(p=0.5),
                A.ImageCompression(quality_range=(30, 70), p=0.4),
                A.Perspective(scale=(0.05, 0.1), p=0.4)
            ], p=1.0),
            A.OneOf([
                A.CLAHE(p=0.5),
                A.Sharpen(p=0.5)
            ], p=0.3),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        label = self.labels[idx]

        img = cv2.imread(path)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if random.random() < self.augment_prob:
            img = self.augment(image=img)["image"]
            label = 1  # fake

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, torch.tensor(label, dtype=torch.long)

# ========== 3. Трансформации ==========
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])


# ========== 4. DataLoader ==========
dataset = DocumentDataset(DATA_DIR, transform=transform, augment_prob=0.4)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)


# ========== 5. Модель ==========
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# ========== 6. Тренировка ==========
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f"Train loss: {train_loss/len(train_loader):.4f}")

    # Валидация
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(out, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_prob.extend(probs)
            y_true.extend(labels.numpy())

    # Метрики
    print(classification_report(y_true, y_pred, target_names=["valid", "fake"], digits=4))

    try:
        auc = roc_auc_score(y_true, y_prob)
        print(f"🔹 AUC: {auc:.4f}")
    except Exception as e:
        print(f"AUC error: {e}")

torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Модель сохранена в {MODEL_PATH}")

