"""
config.py - shared configuration for the scripts in src/.

The values mirror the configuration cells of the Colab notebooks
(notebooks/01_train_video.ipynb and notebooks/02_train_sequence.ipynb), so the
scripts and the notebooks train and evaluate the same way. Paths assume Google
Colab with Google Drive mounted at /content/drive; override them with the
environment variables below when running elsewhere.
"""

import os

# ---------------------------------------------------------------------------
# Stroke classes: (id, display name, clip-dataset folder)
# ---------------------------------------------------------------------------
CLASSES = [
    (0, "Short Serve", "00_Short Serve"),
    (1, "Lift", "02_Lift"),
    (2, "Clear", "12_Clear"),
    (3, "Drop Shot", "05_Drop Shot"),
    (4, "Smash", "14_Smash"),
]
NUM_CLASSES = len(CLASSES)
DISPLAY_NAMES = [c[1] for c in CLASSES]
CLASS_FOLDERS = [c[2] for c in CLASSES]
ID_TO_NAME = {c[0]: c[1] for c in CLASSES}
FOLDER_TO_ID = {c[2]: c[0] for c in CLASSES}
PAD_TOKEN = NUM_CLASSES  # id 5 left-pads rally histories shorter than SEQ_LEN

# ---------------------------------------------------------------------------
# Paths (Google Colab + Google Drive by default)
# ---------------------------------------------------------------------------
DRIVE = os.environ.get("RALLYLENS_DRIVE", "/content/drive/MyDrive/BadmintonFYP")
MODELS = os.environ.get("MODEL_DIR", DRIVE + "/model23")
DATASET_ZIP = DRIVE + "/DATASET2.zip"
DATASET_DIR = os.environ.get("RALLYLENS_DATASET_DIR", "/content/Dataset")
SHUTTLESET_DIR = os.environ.get("RALLYLENS_SHUTTLESET_DIR", "/content/ShuttleSet22")
SHUTTLESET_SET = SHUTTLESET_DIR + "/set"

VIDEO_CKPT = MODELS + "/video_model.pth"
SEQUENCE_CKPT = MODELS + "/sequence_model.pth"
LABEL_MAPPING = MODELS + "/label_mapping.json"
VIDEO_METRICS = MODELS + "/video_metrics.json"
VIDEO_CONFUSION = MODELS + "/video_confusion_matrix.png"
SEQUENCE_METRICS = MODELS + "/sequence_metrics.json"
SEQ_CONFUSION = MODELS + "/sequence_confusion_matrix.png"

# ---------------------------------------------------------------------------
# Video stroke classifier
# ---------------------------------------------------------------------------
NUM_FRAMES = 16  # frames sampled uniformly from each clip
IMG = 224        # letterboxed frame size
BATCH = 8
EPOCHS = 10      # the deployed classifier was trained for 10 epochs (notebook 01)
LR = 1e-4
SEED = 42
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Next-shot sequence model
# ---------------------------------------------------------------------------
SEQ_LEN = 4  # the GRU sees the last four shots

# ShuttleSet22 stroke type (Chinese label) -> project class id.
# All clear variants -> Clear; Lift = lobs only; pushes -> Drop Shot; drives -> Clear.
SHUTTLE_MAP = {
    "發短球": 0, "發長球": 0,
    "挑球": 1, "防守回挑": 1,
    "長球": 2, "平球": 2, "小平球": 2, "後場抽平球": 2, "防守回抽": 2,
    "切球": 3, "過度切球": 3, "過渡切球": 3, "放小球": 3, "擋小球": 3, "勾球": 3, "推球": 3,
    "殺球": 4, "點扣": 4, "撲球": 4,
}


def ensure_models_dir():
    """Create the checkpoint/artefact folder if needed and return its path."""
    os.makedirs(MODELS, exist_ok=True)
    return MODELS


def label_mapping_dict():
    """Label metadata saved next to the video checkpoint (label_mapping.json)."""
    return {
        "num_classes": NUM_CLASSES,
        "display_names": DISPLAY_NAMES,
        "class_folders": CLASS_FOLDERS,
        "id_to_name": {str(i): n for i, n in ID_TO_NAME.items()},
        "pad_token": PAD_TOKEN,
    }
