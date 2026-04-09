import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split

# ==============================
# CONFIG
# ==============================
DATA_DIR = "data"   #  user should place dataset here
IMG_DIR = os.path.join(DATA_DIR, "images")
LABELS_PATH = os.path.join(DATA_DIR, "labels.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# DATASET
# ==============================
class CTICHDataset(Dataset):
    def __init__(self, img_dir, dataframe, transform=None):
        self.data = dataframe
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

# ==============================
# TRANSFORMS
# ==============================
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(LABELS_PATH)

train_df, test_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=42)
train_df, val_df  = train_test_split(train_df, test_size=0.176, stratify=train_df["label"], random_state=42)

train_loader = DataLoader(CTICHDataset(IMG_DIR, train_df, train_transform), batch_size=16, shuffle=True)
val_loader   = DataLoader(CTICHDataset(IMG_DIR, val_df, val_transform), batch_size=16)
test_loader  = DataLoader(CTICHDataset(IMG_DIR, test_df, val_transform), batch_size=16)

# ==============================
# MODEL
# ==============================
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=2)
model = model.to(DEVICE)

# ==============================
# LOSS + OPTIMIZER
# ==============================
weights = torch.tensor([1.0, 3.0]).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.3, patience=2
)

# ==============================
# TRAIN
# ==============================
def train_one_epoch(model, loader):
    model.train()
    running_loss = 0
    
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss / len(loader)

# ==============================
# EVALUATE
# ==============================
def evaluate(model, loader, threshold=0.6):
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            
            probs = torch.softmax(outputs, dim=1)[:,1]
            preds = (probs > threshold).int()
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    return y_true, y_pred

def compute_metrics(y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn+fp)>0 else 0
    else:
        spec = 0

    return acc, prec, rec, f1, spec

# ==============================
# TRAIN LOOP
# ==============================
EPOCHS = 10
best_f1 = 0

for epoch in range(EPOCHS):
    loss = train_one_epoch(model, train_loader)
    
    y_true, y_pred = evaluate(model, val_loader)
    acc, prec, rec, f1, spec = compute_metrics(y_true, y_pred)

    scheduler.step(f1)

    print(f"\nEpoch {epoch+1}")
    print(f"Loss: {loss:.4f}")
    print(f"Val → Acc:{acc:.3f} Prec:{prec:.3f} Rec:{rec:.3f} F1:{f1:.3f}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "best_model.pth")

# ==============================
# FINAL TEST
# ==============================
model.load_state_dict(torch.load("best_model.pth"))

y_true, y_pred = evaluate(model, test_loader)
acc, prec, rec, f1, spec = compute_metrics(y_true, y_pred)

print("\n FINAL TEST RESULTS:")
print({
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1": f1,
    "Specificity": spec
})