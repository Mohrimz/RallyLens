"""
src/train_sequence.py - train the next-shot prediction model (GRU).

Model: embedding + 2-layer GRU(128) + linear head.
Loss:  weighted cross-entropy (counters the majority-class collapse where the
       original notebook predicted "Drop" ~92% of the time).
Saves: sequence_model.pth + metrics (accuracy, macro-F1, per-class, top-3).

pipeline.py imports ShotPredictorGRU from here, so importing this module must
not start training - everything runs from main().

Run (from the repo root):
    python src/train_sequence.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import config
from data_sequence import (
    ensure_shuttleset, parse_rallies, build_windows, group_split,
    SequenceDataset, class_distribution,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================================
# Model
# ===========================================================================
class ShotPredictorGRU(nn.Module):
    """Embedding -> 2-layer GRU -> linear head for next-shot prediction."""

    def __init__(self, num_classes=config.NUM_CLASSES, embed_dim=32,
                 hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(num_classes + 1, embed_dim,
                                      padding_idx=config.PAD_TOKEN)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.emb(x)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])   # last timestep


# ===========================================================================
# Metrics
# ===========================================================================
def top_k_accuracy(logits, targets, k=3):
    topk = logits.topk(k, dim=1).indices
    return (topk == targets.unsqueeze(1)).any(dim=1).float().mean().item()


def evaluate(model, loader):
    model.eval()
    preds, labels, top3_hits, n = [], [], 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x).cpu()
            preds.extend(logits.argmax(1).numpy())
            labels.extend(y.numpy())
            top3_hits += top_k_accuracy(logits, y, k=3) * len(y)
            n += len(y)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return acc, f1, top3_hits / n, preds, labels


def compute_class_weights(df_train):
    """Inverse-frequency weights over the training targets."""
    counts = df_train["target"].value_counts().sort_index()
    total = len(df_train)
    weights = [total / (config.NUM_CLASSES * counts.get(i, 1)) for i in range(config.NUM_CLASSES)]
    return torch.tensor(weights, dtype=torch.float32)


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
    plt.title("Sequence Model - Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Confusion matrix -> {out_path}")


# ===========================================================================
# Main
# ===========================================================================
def main(epochs=20, lr=1e-3, batch=64):
    print(f"Device: {DEVICE}")
    config.ensure_models_dir()

    set_folder = ensure_shuttleset()
    rallies = parse_rallies(set_folder)
    windows = build_windows(rallies)

    print("\nTarget-class distribution:")
    for name, (count, frac) in class_distribution(windows).items():
        print(f"  {name}: {count} ({frac*100:.1f}%)")

    df_train, df_val, df_test = group_split(windows)
    print(f"\nSplit -> train {len(df_train)} | val {len(df_val)} | test {len(df_test)}")

    train_loader = DataLoader(SequenceDataset(df_train), batch_size=batch, shuffle=True)
    val_loader = DataLoader(SequenceDataset(df_val), batch_size=batch)
    test_loader = DataLoader(SequenceDataset(df_test), batch_size=batch)

    model = ShotPredictorGRU().to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    weights = compute_class_weights(df_train).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)   # weighted -> avoids collapse
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_f1, history = 0.0, []
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        val_acc, val_f1, val_top3, _, _ = evaluate(model, val_loader)
        history.append({"epoch": epoch + 1, "val_acc": val_acc,
                        "val_macro_f1": val_f1, "val_top3": val_top3})
        marker = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), config.SEQUENCE_CKPT)
            marker = "  saved"
        print(f"Epoch {epoch+1}/{epochs} | val acc {val_acc:.3f} "
              f"f1 {val_f1:.3f} top3 {val_top3:.3f}{marker}")

    model.load_state_dict(torch.load(config.SEQUENCE_CKPT, map_location=DEVICE))

    # ---- Final test evaluation --------------------------------------------
    test_acc, test_f1, test_top3, preds, labels = evaluate(model, test_loader)
    print(f"\nTEST | acc {test_acc:.4f} | macro-F1 {test_f1:.4f} | top-3 {test_top3:.4f}")
    report = classification_report(labels, preds, labels=list(range(config.NUM_CLASSES)),
                                   target_names=config.DISPLAY_NAMES, output_dict=True)
    print(classification_report(labels, preds, labels=list(range(config.NUM_CLASSES)),
                                target_names=config.DISPLAY_NAMES))

    save_confusion_matrix(labels, preds, config.SEQ_CONFUSION)

    with open(config.SEQUENCE_METRICS, "w") as f:
        json.dump({"test_accuracy": test_acc, "test_macro_f1": test_f1,
                   "test_top3_accuracy": test_top3, "best_val_macro_f1": best_f1,
                   "per_class": report, "history": history}, f, indent=2)

    print(f"\nSaved: {config.SEQUENCE_CKPT}\n       {config.SEQUENCE_METRICS}\n"
          f"       {config.SEQ_CONFUSION}")


if __name__ == "__main__":
    main()
