"""
src/data_sequence.py - ShuttleSet22 parsing for the next-shot model.

Responsibilities:
  - download/locate ShuttleSet22 `set/<match>/setN.csv` files
  - map the Chinese `type` column to a 5-class id via config.SHUTTLE_MAP
  - build last-4 -> next windows (FIXED so a serve can be a target)
  - group-split by match (no match leaks across splits)

train_sequence.py imports the parsing + dataset helpers from here.

Run standalone to print dataset statistics:
    python src/data_sequence.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit

import config


# ===========================================================================
# Acquire ShuttleSet22
# ===========================================================================
def ensure_shuttleset(root=config.SHUTTLESET_DIR):
    """Download + extract ShuttleSet22 if the `set/` folder is missing.

    Uses the zip of CoachAI-Projects (no git dependency). Safe to call when
    the data is already present.
    """
    set_folder = os.path.join(root, "set")
    if os.path.isdir(set_folder) and os.listdir(set_folder):
        return set_folder

    import urllib.request
    import zipfile
    import shutil

    url = "https://github.com/wywyWang/CoachAI-Projects/archive/refs/heads/main.zip"
    zip_path = "/content/coachai.zip"
    extract_to = "/content/_coachai_extract"

    print("Downloading ShuttleSet22 ...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

    src = os.path.join(
        extract_to, "CoachAI-Projects-main",
        "CoachAI-Challenge-IJCAI2023", "ShuttleSet22",
    )
    os.makedirs(os.path.dirname(root), exist_ok=True)
    if os.path.exists(root):
        shutil.rmtree(root)
    shutil.move(src, root)

    os.remove(zip_path)
    shutil.rmtree(extract_to, ignore_errors=True)
    print(f"ShuttleSet22 ready at {root}")
    return set_folder


# ===========================================================================
# Parse rallies
# ===========================================================================
def parse_rallies(set_folder=config.SHUTTLESET_SET):
    """Read every setN.csv, map `type` -> class id, group shots per rally.

    Returns a list of dicts: {match_id, set_id, rally_id, shots:[ids]}.
    Rallies with fewer than 2 mapped shots are dropped; unknown shot types
    are skipped and reported.
    """
    rallies, skipped = [], []
    matches = [m for m in os.listdir(set_folder)
               if os.path.isdir(os.path.join(set_folder, m))]

    for match in matches:
        match_path = os.path.join(set_folder, match)
        for set_file in ("set1.csv", "set2.csv", "set3.csv"):
            path = os.path.join(match_path, set_file)
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_csv(path)
            except Exception as e:
                print(f"Error reading {path}: {e}")
                continue
            if "rally" not in df.columns or "type" not in df.columns:
                continue

            for rally_id, rally_df in df.groupby("rally"):
                shots = []
                for shot_type in rally_df["type"].astype(str).str.strip():
                    if shot_type in config.SHUTTLE_MAP:
                        shots.append(config.SHUTTLE_MAP[shot_type])
                    else:
                        skipped.append(shot_type)
                if len(shots) >= 2:
                    rallies.append({
                        "match_id": match,
                        "set_id": set_file,
                        "rally_id": rally_id,
                        "shots": shots,
                    })

    if skipped:
        uniq = sorted(set(skipped))
        print(f"Skipped {len(skipped)} unmapped shots ({len(uniq)} types): {uniq}")
    print(f"Parsed {len(rallies)} rallies")
    return rallies


# ===========================================================================
# Build windows  (FIX: a serve CAN be a target)
# ===========================================================================
def build_windows(rallies, seq_len=config.SEQ_LEN):
    """Build (history -> next-shot) windows.

    For every position i in a rally we predict shots[i] from the up-to-`seq_len`
    shots BEFORE it, left-padded with PAD_TOKEN. Because i starts at 0, the
    first shot of a rally (the serve) is a valid target predicted from an
    all-padding history - this is the windowing fix vs the original notebook,
    which started at shots[1] and so could never predict a serve.
    """
    windows = []
    for r in rallies:
        shots = r["shots"]
        for i in range(len(shots)):                 # i == 0 -> serve as target
            history = shots[max(0, i - seq_len):i]  # shots strictly before i
            if len(history) < seq_len:
                history = [config.PAD_TOKEN] * (seq_len - len(history)) + history
            windows.append({
                "input": history,
                "target": shots[i],
                "match_id": r["match_id"],
            })
    print(f"Built {len(windows)} windows")
    return windows


def group_split(windows, val=0.15, test=0.15, seed=config.SEED):
    """Group-aware split by match_id (no match leaks across splits)."""
    df = pd.DataFrame(windows)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=val + test, random_state=seed)
    train_idx, temp_idx = next(gss1.split(df, groups=df["match_id"]))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_temp = df.iloc[temp_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=test / (val + test), random_state=seed)
    val_idx, test_idx = next(gss2.split(df_temp, groups=df_temp["match_id"]))
    df_val = df_temp.iloc[val_idx].reset_index(drop=True)
    df_test = df_temp.iloc[test_idx].reset_index(drop=True)

    tr, va, te = set(df_train.match_id), set(df_val.match_id), set(df_test.match_id)
    assert not (tr & va) and not (tr & te) and not (va & te), "Match leakage!"
    return df_train, df_val, df_test


# ===========================================================================
# Dataset
# ===========================================================================
class SequenceDataset(Dataset):
    def __init__(self, df):
        self.inputs = [torch.tensor(x, dtype=torch.long) for x in df["input"]]
        self.targets = torch.tensor(df["target"].values, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def class_distribution(windows):
    """Counts of target classes (handy for spotting imbalance)."""
    from collections import Counter
    counts = Counter(w["target"] for w in windows)
    total = sum(counts.values())
    return {config.ID_TO_NAME[i]: (counts.get(i, 0), counts.get(i, 0) / total)
            for i in range(config.NUM_CLASSES)}


def main():
    set_folder = ensure_shuttleset()
    rallies = parse_rallies(set_folder)
    windows = build_windows(rallies)
    print("\nTarget-class distribution:")
    for name, (count, frac) in class_distribution(windows).items():
        print(f"  {name}: {count} ({frac*100:.1f}%)")
    df_train, df_val, df_test = group_split(windows)
    print(f"\nSplit -> train {len(df_train)} | val {len(df_val)} | test {len(df_test)}")


if __name__ == "__main__":
    main()
