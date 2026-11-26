"""
api.py - FastAPI backend for the Badminton Analytics frontend.

Loads the trained model23 models (video classifier + next-shot GRU) and exposes:
  GET  /health   -> service + model status
  POST /analyze  -> upload a shot clip (+ optional rally history) -> JSON with
                    current shot, top-3, confidence, Grad-CAM image (data URI),
                    predicted next shot, coaching line, uncertainty flag.

Run on Colab (GPU) or locally (CPU). See backend/README.md.
Set MODEL_DIR to where video_model.pth + sequence_model.pth live
(default: the Colab Drive path /content/drive/MyDrive/BadmintonFYP/model23).
"""

import os
import io
import json
import base64
import tempfile

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Config (mirrors config.py / model23)
# ---------------------------------------------------------------------------
CLASSES = [
    (0, "Short Serve", "00_Short Serve"),
    (1, "Lift",        "02_Lift"),
    (2, "Clear",       "12_Clear"),
    (3, "Drop Shot",   "05_Drop Shot"),
    (4, "Smash",       "14_Smash"),
]
NUM_CLASSES = len(CLASSES)
ID_TO_NAME = {c[0]: c[1] for c in CLASSES}
NAME_TO_ID = {c[1]: c[0] for c in CLASSES}
PAD_TOKEN = NUM_CLASSES
SEQ_LEN = 4
NUM_FRAMES = 16
IMG = 224
CONF_THRESH = 0.60
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODEL_DIR = os.environ.get("MODEL_DIR", "/content/drive/MyDrive/BadmintonFYP/model23")
VIDEO_CKPT = os.path.join(MODEL_DIR, "video_model.pth")
SEQUENCE_CKPT = os.path.join(MODEL_DIR, "sequence_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WHY = {
    "Short Serve": "low contact near the net, short forward push",
    "Lift": "underarm lift sending the shuttle high to the rear court",
    "Clear": "overhead hit driving the shuttle to the back court",
    "Drop Shot": "soft downward touch landing just over the net",
    "Smash": "steep, powerful overhead downward attack",
}
# Rule-based coaching: one real line per shot + a short cue for the predicted
# next shot (named, no numbers). Matches notebooks/09_rally_pes.ipynb.
COACH_LINE = {
    "Short Serve": "Serve is in - get back to base and watch the net.",
    "Lift":        "You are on the defensive now - get low and ready for the smash.",
    "Clear":       "Good deep clear - recover to centre and stay balanced.",
    "Drop Shot":   "Nice drop - follow it in and pressure the net.",
    "Smash":       "Strong smash - move in and look to finish at the net.",
}
NEXT_HINT = {
    "Short Serve": "watch for a net reply",
    "Lift":        "be ready to defend the smash",
    "Clear":       "expect it deep to the back court",
    "Drop Shot":   "cover the net",
    "Smash":       "brace and prepare to defend",
}


# ---------------------------------------------------------------------------
# Models (must match training)
# ---------------------------------------------------------------------------
class TemporalPoolingResNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        base = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = self.features(x).view(b, t, 512).mean(dim=1)
        return self.fc(x)


class ShotPredictorGRU(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, embed=32, hidden=128, layers=2, drop=0.3):
        super().__init__()
        self.emb = nn.Embedding(num_classes + 1, embed, padding_idx=PAD_TOKEN)
        self.gru = nn.GRU(embed, hidden, layers, batch_first=True, dropout=drop if layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(drop), nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        x = self.emb(x)
        o, _ = self.gru(x)
        return self.fc(o[:, -1, :])


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
_norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)


def load_frames(path, n=NUM_FRAMES):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, max(total - 1, 0), n, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        elif frames:
            frames.append(frames[-1].copy())
    cap.release()
    while len(frames) < n:
        frames.append(frames[-1].copy() if frames else np.zeros((IMG, IMG, 3), np.uint8))
    return frames[:n]


def letterbox(img, size=IMG):
    h, w = img.shape[:2]
    s = min(size / h, size / w)
    nh, nw = int(h * s), int(w * s)
    r = cv2.resize(img, (nw, nh))
    c = np.zeros((size, size, 3), np.uint8)
    y, x = (size - nh) // 2, (size - nw) // 2
    c[y:y + nh, x:x + nw] = r
    return c


# ---------------------------------------------------------------------------
# Load models once at startup
# ---------------------------------------------------------------------------
video_model = None
seq_model = None
TARGET = None


def _load_state(model, path):
    ck = torch.load(path, map_location=DEVICE)
    state = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    model.load_state_dict(state)
    model.eval()
    return model


def load_models():
    global video_model, seq_model, TARGET
    if os.path.exists(VIDEO_CKPT):
        video_model = _load_state(TemporalPoolingResNet().to(DEVICE), VIDEO_CKPT)
        TARGET = video_model.features[7]  # ResNet18 layer4 (for Grad-CAM)
        print("Loaded video model:", VIDEO_CKPT)
    else:
        print("WARNING: video model not found at", VIDEO_CKPT)
    if os.path.exists(SEQUENCE_CKPT):
        seq_model = _load_state(ShotPredictorGRU().to(DEVICE), SEQUENCE_CKPT)
        print("Loaded sequence model:", SEQUENCE_CKPT)
    else:
        print("WARNING: sequence model NOT found at", SEQUENCE_CKPT)
        try:
            print("   MODEL_DIR contents:", os.listdir(MODEL_DIR))
        except OSError as e:
            print("   could not list MODEL_DIR:", e)
    print("Models ready -> video:", video_model is not None,
          "| sequence (next-shot):", seq_model is not None)


# ---------------------------------------------------------------------------
# Inference + Grad-CAM
# ---------------------------------------------------------------------------
def analyze_clip(path):
    """Return (overlay_rgb, top3[(label, prob)]) using the video model + Grad-CAM."""
    lb = [letterbox(f) for f in load_frames(path)]
    x = torch.stack(
        [_norm(torch.from_numpy(f.astype(np.float32) / 255.0).permute(2, 0, 1)) for f in lb]
    ).unsqueeze(0).to(DEVICE)

    store = {}

    def fh(m, i, o):
        store["a"] = o
        o.register_hook(lambda g: store.update(g=g))

    h = TARGET.register_forward_hook(fh)
    video_model.zero_grad()
    logits = video_model(x)
    probs = torch.softmax(logits, dim=1)[0]
    order = [int(j) for j in probs.argsort(descending=True)]
    pred = order[0]
    logits[0, pred].backward()
    h.remove()

    A, G = store["a"], store["g"]
    w = G.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * A).sum(dim=1)).detach().cpu().numpy()
    t = int(cam.reshape(cam.shape[0], -1).sum(1).argmax())
    heat = cv2.resize(cam[t], (IMG, IMG))
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    hc = cv2.cvtColor(cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = (0.55 * lb[t] + 0.45 * hc).astype(np.uint8)

    top3 = [(ID_TO_NAME[i], float(probs[i])) for i in order[:3]]
    return overlay, top3


def predict_next(ids):
    if seq_model is None:
        return None
    h = list(ids)[-SEQ_LEN:]
    if len(h) < SEQ_LEN:
        h = [PAD_TOKEN] * (SEQ_LEN - len(h)) + h
    with torch.no_grad():
        p = torch.softmax(seq_model(torch.tensor([h], dtype=torch.long, device=DEVICE)), dim=1)[0]
    v, idx = p.topk(3)
    return [(ID_TO_NAME[int(i)], float(val)) for val, i in zip(v, idx)]


def to_data_uri(rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
app = FastAPI(title="RallyLens API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    load_models()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "video_model_loaded": video_model is not None,
        "sequence_model_loaded": seq_model is not None,
        "classes": [c[1] for c in CLASSES],
        "model_dir": MODEL_DIR,
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), history: str = Form("[]")):
    """history = JSON list of prior shot ids in the rally (before this clip)."""
    if video_model is None:
        return {"error": "Video model not loaded. Check MODEL_DIR / model23 path."}

    try:
        prior = [int(i) for i in json.loads(history)]
    except Exception:
        prior = []

    # save the upload to a temp file for OpenCV
    suffix = os.path.splitext(file.filename or "clip.mp4")[1] or ".mp4"
    data = await file.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()
    try:
        overlay, top3 = analyze_clip(tmp.name)
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass

    cur_label, cur_conf = top3[0]
    cur_id = NAME_TO_ID[cur_label]
    rally = prior + [cur_id]                       # rally including this shot
    nxt = predict_next(rally)
    coaching = COACH_LINE.get(cur_label, "Shot played.")
    if nxt is not None:
        nshot = nxt[0][0]
        coaching += " Likely next: %s - %s." % (nshot, NEXT_HINT.get(nshot, "stay ready"))

    return {
        "current": {
            "label": cur_label,
            "confidence": cur_conf,
            "uncertain": bool(cur_conf < CONF_THRESH),
            "why": WHY.get(cur_label, ""),
            "top3": [{"label": l, "prob": p} for l, p in top3],
        },
        "gradcam": to_data_uri(overlay),
        "next": ([{"label": l, "prob": p} for l, p in nxt] if nxt else None),
        "coaching": coaching,
        "history": [ID_TO_NAME[i] for i in rally],
        "history_ids": rally,
    }


if __name__ == "__main__":
    import uvicorn
    load_models()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
