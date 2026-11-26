"""
src/pipeline.py - the combined inference pipeline.

Loads both checkpoints and exposes:
  - predict_current(clip)      -> current shot top-3
  - RallyHistory (4-slot)      -> rolling shot memory
  - predict_next(history)      -> next-shot top-3
  - analyze_clip(clip)         -> current + history update + next + coaching
  - evaluate_folder(dir)       -> combined accuracy report over a labelled tree

Model classes and preprocessing are imported from the training modules so the
architectures always match the saved checkpoints (single source of truth).

Run a quick combined evaluation:
    python src/pipeline.py /path/to/labelled/dataset
"""

import os
import sys
import json
from collections import deque, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

import config
from train_video import TemporalPoolingResNet, preprocess_clip
from train_sequence import ShotPredictorGRU

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================================
# Model loading
# ===========================================================================
def _load_state(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def load_models(video_ckpt=config.VIDEO_CKPT, seq_ckpt=config.SEQUENCE_CKPT):
    """Load the video + sequence models. Missing checkpoints are returned as None."""
    video_model = seq_model = None

    if os.path.exists(video_ckpt):
        video_model = _load_state(
            TemporalPoolingResNet(config.NUM_CLASSES, pretrained=False).to(DEVICE), video_ckpt)
        print(f"Loaded video model: {video_ckpt}")
    else:
        print(f"WARNING: video checkpoint not found ({video_ckpt}) - run train_video.py")

    if os.path.exists(seq_ckpt):
        seq_model = _load_state(ShotPredictorGRU().to(DEVICE), seq_ckpt)
        print(f"Loaded sequence model: {seq_ckpt}")
    else:
        print(f"WARNING: sequence checkpoint not found ({seq_ckpt}) - run train_sequence.py")

    return video_model, seq_model


# ===========================================================================
# Prediction
# ===========================================================================
def _topk(probs, k=3):
    k = min(k, probs.numel())
    vals, idx = probs.topk(k)
    return [{"id": int(i), "label": config.ID_TO_NAME[int(i)], "prob": float(v)}
            for v, i in zip(vals, idx)]


def predict_current(video_model, clip_path, k=3):
    """Classify one clip -> top-k {id, label, prob}."""
    x = preprocess_clip(clip_path).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(video_model(x), dim=1)[0].cpu()
    top = _topk(probs, k)
    return {"top": top, "pred_id": top[0]["id"],
            "label": top[0]["label"], "confidence": top[0]["prob"]}


def predict_next(seq_model, history_ids, k=3):
    """Predict the next shot from a list of recent class ids -> top-k."""
    hist = list(history_ids)[-config.SEQ_LEN:]
    if len(hist) < config.SEQ_LEN:
        hist = [config.PAD_TOKEN] * (config.SEQ_LEN - len(hist)) + hist
    x = torch.tensor([hist], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        probs = F.softmax(seq_model(x), dim=1)[0].cpu()
    return {"top": _topk(probs, k)}


# ===========================================================================
# Rally history (4 slots)
# ===========================================================================
class RallyHistory:
    def __init__(self, maxlen=config.SEQ_LEN):
        self.buf = deque(maxlen=maxlen)

    def append(self, class_id):
        self.buf.append(int(class_id))

    def ids(self):
        return list(self.buf)

    def labels(self):
        return [config.ID_TO_NAME[i] for i in self.buf]

    def reset(self):
        self.buf.clear()


# ===========================================================================
# Coaching insight (rule-based "why-card" seed)
# ===========================================================================
def coaching_line(current_label, next_label):
    rules = {
        "Smash": f"Attack played - opponent likely forced into {next_label}.",
        "Drop Shot": f"Net pressure applied - expect {next_label} in return.",
        "Short Serve": f"Rally started - anticipate {next_label}.",
        "Lift": f"Lifted to the back - prepare for {next_label}.",
        "Clear": f"Cleared deep - next likely {next_label}.",
    }
    return rules.get(current_label, f"Next likely shot: {next_label}.")


def analyze_clip(video_model, seq_model, clip_path, history):
    """Full per-clip analysis: current shot, history update, next shot, coaching."""
    current = predict_current(video_model, clip_path)
    history.append(current["pred_id"])

    nxt = coach = None
    if seq_model is not None:
        nxt = predict_next(seq_model, history.ids())["top"]
        coach = coaching_line(current["label"], nxt[0]["label"])

    return {
        "clip": os.path.basename(clip_path),
        "current": current,
        "history": history.labels(),
        "next": nxt,
        "coaching": coach,
    }


# ===========================================================================
# Combined evaluation over a labelled folder tree
# ===========================================================================
def evaluate_folder(dataset_dir, per_class=5):
    """Sample clips per class from a labelled tree and report current-shot accuracy."""
    import glob

    video_model, seq_model = load_models()
    if video_model is None:
        raise SystemExit("No video model - train it first.")

    results, correct = [], 0
    for folder in config.CLASS_FOLDERS:
        true_id = config.FOLDER_TO_ID[folder]
        clips = sorted(glob.glob(os.path.join(dataset_dir, folder, "*.mp4")))[:per_class]
        for clip in clips:
            history = RallyHistory()
            r = analyze_clip(video_model, seq_model, clip, history)
            ok = r["current"]["pred_id"] == true_id
            correct += ok
            results.append({"true": config.ID_TO_NAME[true_id],
                            "pred": r["current"]["label"],
                            "conf": round(r["current"]["confidence"], 3),
                            "ok": ok})

    n = len(results)
    print(f"\nCombined current-shot accuracy: {correct}/{n} ({correct/n*100:.1f}%)\n")
    print(f"{'True':<14}{'Pred':<14}{'Conf':<8}{'OK'}")
    print("-" * 42)
    for r in results:
        print(f"{r['true']:<14}{r['pred']:<14}{r['conf']:<8}{'Y' if r['ok'] else 'N'}")
    print("\nPrediction distribution:", dict(Counter(r["pred"] for r in results)))
    return results


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else config.DATASET_DIR
    evaluate_folder(target)
