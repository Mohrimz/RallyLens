"""
src/train_video.py - train the RGB stroke classifier.

Ported from VIDEO.ipynb into a script:
  - TemporalPoolingResNet (per-frame ResNet18 features + temporal mean pool)
  - weighted cross-entropy (inverse-frequency class weights)
  - group split by match (no clip from one match leaks across splits)
  - early stopping on validation macro-F1
  - saves video_model.pth + label_mapping.json + confusion matrix to MODELS

The model class and the preprocessing helpers are imported by pipeline.py,
so importing this module must NOT start training - everything runs from main().

Run (from the repo root):
    python src/train_video.py
"""

import os
import sys
import json

# Make the repo root importable so `import config` works when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================================
# Model
# ===========================================================================
class TemporalPoolingResNet(nn.Module):
    """Per-frame ResNet18 features + temporal mean pooling.

    Layer names `features` and `fc` are kept so pipeline.py can load the
    checkpoint with this exact class definition.
    """

    def __init__(self, num_classes=config.NUM_CLASSES, pretrained=False):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(base.children())[:-1])  # drop fc
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = self.features(x).view(b, t, 512).mean(dim=1)  # temporal mean pool
        return self.fc(x)


# ===========================================================================
# Preprocessing (shared with pipeline.py)
# ===========================================================================
def load_video_frames(video_path, num_frames=config.NUM_FRAMES):
    """Open a clip and uniformly sample `num_frames` RGB frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        raise ValueError(f"Video has 0 frames: {video_path}")

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        elif frames:
            frames.append(frames[-1].copy())
    cap.release()

    while len(frames) < num_frames:
        frames.append(
            frames[-1].copy() if frames
            else np.zeros((config.IMG, config.IMG, 3), dtype=np.uint8)
        )
    return frames[:num_frames]


def resize_with_letterbox(image, target_size=config.IMG):
    """Resize preserving aspect ratio, pad to a square canvas."""
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y, x = (target_size - new_h) // 2, (target_size - new_w) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def preprocess_frames(frames, target_size=config.IMG):
    """List of RGB frames -> normalised tensor (T, C, H, W)."""
    normalize = transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    processed = []
    for frame in frames:
        frame = resize_with_letterbox(frame, target_size)
        frame = frame.astype(np.float32) / 255.0
        frame = torch.from_numpy(frame).permute(2, 0, 1)  # (H,W,C)->(C,H,W)
        processed.append(normalize(frame))
    return torch.stack(processed)


def preprocess_clip(video_path):
    """Full path -> model-ready tensor (1, T, C, H, W)."""
    frames = load_video_frames(video_path, config.NUM_FRAMES)
    tensor = preprocess_frames(frames, config.IMG)
    return tensor.unsqueeze(0)


# ===========================================================================
# Dataset
# ===========================================================================
class VideoClipDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            frames = load_video_frames(row["file_path"], config.NUM_FRAMES)
            tensor = preprocess_frames(frames, config.IMG)
            return tensor, int(row["label_id"])
        except Exception as e:  # corrupt clip -> zeros, keeps the batch alive
            print(f"Error loading {row['file_path']}: {e}")
            return torch.zeros(config.NUM_FRAMES, 3, config.IMG, config.IMG), int(row["label_id"])


def build_dataset_index(dataset_dir):
    """Scan the per-class folders into a DataFrame with a match group_id."""
    rows = []
    for folder in config.CLASS_FOLDERS:
        class_path = os.path.join(dataset_dir, folder)
        if not os.path.isdir(class_path):
            print(f"Warning: folder not found - {class_path}")
            continue
        label_id = config.FOLDER_TO_ID[folder]
        for fname in os.listdir(class_path):
            if not fname.lower().endswith(".mp4"):
                continue
            # group_id = the match, so clips from one match stay in one split.
            if "_dataset_set" in fname:
                group_id = fname.split("_dataset_set")[0]
            else:
                group_id = fname[:19] if len(fname) >= 19 else fname.split(".")[0]
            rows.append({
                "file_path": os.path.join(class_path, fname),
                "label_name": folder,
                "label_id": label_id,
                "group_id": group_id,
            })
    return pd.DataFrame(rows)


def group_split(df, val=0.15, test=0.15, seed=config.SEED):
    """Group-aware 70/15/15 split (no match leaks across splits)."""
    gss1 = GroupShuffleSplit(n_splits=1, test_size=val + test, random_state=seed)
    train_idx, temp_idx = next(gss1.split(df, groups=df["group_id"]))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_temp = df.iloc[temp_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=test / (val + test), random_state=seed)
    val_idx, test_idx = next(gss2.split(df_temp, groups=df_temp["group_id"]))
    df_val = df_temp.iloc[val_idx].reset_index(drop=True)
    df_test = df_temp.iloc[test_idx].reset_index(drop=True)

    # Leakage assertion (proposal requires no group leakage).
    tr, va, te = set(df_train.group_id), set(df_val.group_id), set(df_test.group_id)
    assert not (tr & va) and not (tr & te) and not (va & te), "Group leakage!"
    return df_train, df_val, df_test


def compute_class_weights(df_train):
    """Inverse-frequency weights for weighted cross-entropy."""
    counts = df_train["label_id"].value_counts().sort_index()
    total = len(df_train)
    weights = [total / (config.NUM_CLASSES * counts.get(i, 1)) for i in range(config.NUM_CLASSES)]
    w = torch.tensor(weights, dtype=torch.float32)
    print("Class weights:")
    for i, val in enumerate(w):
        print(f"  {config.ID_TO_NAME[i]}: {val:.4f}")
    return w


# ===========================================================================
# Train / eval loops
# ===========================================================================
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    loss_sum, preds, labels = 0.0, [], []

    with torch.set_grad_enabled(train):
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            if train:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if train:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * inputs.size(0)
            preds.extend(outputs.argmax(1).cpu().numpy())
            labels.extend(targets.cpu().numpy())

    loss = loss_sum / len(loader.dataset)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return loss, acc, f1, preds, labels


def train(model, train_loader, val_loader, criterion, optimizer,
          epochs=config.EPOCHS, patience=4, save_path=config.VIDEO_CKPT):
    best_f1, no_improve, history = 0.0, 0, []
    for epoch in range(epochs):
        t_loss, t_acc, t_f1, _, _ = run_epoch(model, train_loader, criterion, optimizer)
        v_loss, v_acc, v_f1, _, _ = run_epoch(model, val_loader, criterion)
        history.append({"epoch": epoch + 1, "train_f1": t_f1, "val_f1": v_f1,
                        "train_acc": t_acc, "val_acc": v_acc})
        print(f"Epoch {epoch+1}/{epochs} | "
              f"train acc {t_acc:.3f} f1 {t_f1:.3f} | val acc {v_acc:.3f} f1 {v_f1:.3f}")

        if v_f1 > best_f1:
            best_f1, no_improve = v_f1, 0
            torch.save(model.state_dict(), save_path)
            print(f"  saved best (val f1 {v_f1:.3f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    return model, history, best_f1


def save_confusion_matrix(y_true, y_pred, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=config.DISPLAY_NAMES, yticklabels=config.DISPLAY_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Video Model - Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Confusion matrix -> {out_path}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cpu":
        print("WARNING: running on CPU - in Colab set Runtime > Change runtime type > GPU.")

    config.ensure_models_dir()

    df = build_dataset_index(config.DATASET_DIR)
    if df.empty:
        raise SystemExit(
            f"No clips found under {config.DATASET_DIR}. "
            f"Unzip {config.DATASET_ZIP} there (folders: {config.CLASS_FOLDERS})."
        )
    print(f"Total clips: {len(df)} | matches: {df.group_id.nunique()}")
    print(df.label_name.value_counts().to_string())

    df_train, df_val, df_test = group_split(df)
    print(f"Split -> train {len(df_train)} | val {len(df_val)} | test {len(df_test)}")

    train_loader = DataLoader(VideoClipDataset(df_train), batch_size=config.BATCH,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(VideoClipDataset(df_val), batch_size=config.BATCH,
                            shuffle=False, num_workers=2)
    test_loader = DataLoader(VideoClipDataset(df_test), batch_size=config.BATCH,
                             shuffle=False, num_workers=2)

    model = TemporalPoolingResNet(config.NUM_CLASSES, pretrained=True).to(DEVICE)
    weights = compute_class_weights(df_train).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    model, history, best_val_f1 = train(model, train_loader, val_loader, criterion, optimizer)

    # ---- Final test evaluation --------------------------------------------
    _, test_acc, test_f1, preds, labels = run_epoch(model, test_loader, criterion)
    print(f"\nTEST | acc {test_acc:.4f} | macro-F1 {test_f1:.4f}")
    report = classification_report(labels, preds, labels=list(range(config.NUM_CLASSES)),
                                   target_names=config.DISPLAY_NAMES, output_dict=True)
    print(classification_report(labels, preds, labels=list(range(config.NUM_CLASSES)),
                                target_names=config.DISPLAY_NAMES))

    save_confusion_matrix(labels, preds, config.VIDEO_CONFUSION)

    with open(config.LABEL_MAPPING, "w") as f:
        json.dump(config.label_mapping_dict(), f, indent=2)
    with open(config.VIDEO_METRICS, "w") as f:
        json.dump({"test_accuracy": test_acc, "test_macro_f1": test_f1,
                   "best_val_macro_f1": best_val_f1,
                   "per_class": report, "history": history}, f, indent=2)

    print(f"\nSaved: {config.VIDEO_CKPT}\n       {config.LABEL_MAPPING}\n"
          f"       {config.VIDEO_METRICS}\n       {config.VIDEO_CONFUSION}")


if __name__ == "__main__":
    main()
