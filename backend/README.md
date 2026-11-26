# RallyLens backend (FastAPI)

Serves the two trained models to the dashboard.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service status: device, whether each model loaded, class names |
| `POST /analyze` | Upload one stroke clip, plus the rally so far as `history` (JSON list of shot ids) |

`/analyze` returns:
- the stroke, its confidence and top-3 probabilities;
- a low-confidence flag (below 0.60);
- a Grad-CAM heatmap (PNG data URI);
- the top-3 predicted next shots;
- a coaching cue;
- the updated rally history.

See [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md#api) for an example response. FastAPI also serves interactive docs at `/docs`.

The service needs `video_model.pth` and `sequence_model.pth`, produced by notebooks 01 and 02. Point it at their folder with the `MODEL_DIR` environment variable (default `/content/drive/MyDrive/BadmintonFYP/model23`).

## Option A: Google Colab (GPU), recommended

1. Open [`api.ipynb`](api.ipynb) in Colab and select a GPU runtime.
2. Run the three cells:
   - **Cell 1** checks the GPU, mounts Drive, installs FastAPI/uvicorn and downloads `cloudflared`.
   - **Cell 2** writes `api.py`.
   - **Cell 3** starts the API and opens a Cloudflare quick tunnel.
3. Copy the printed `https://<random>.trycloudflare.com` URL into the dashboard's **Settings**. Opening `<url>/health` should show `"video_model_loaded": true`.

The tunnel URL changes every time the cell runs, and the API stays up only while the cell is running.

## Option B: run locally

```bash
cd backend
pip install -r requirements.txt
# Windows PowerShell: $env:MODEL_DIR="C:\path\to\model23"
export MODEL_DIR=/path/to/model23
python api.py            # serves on http://localhost:8000 (override with PORT)
```

The service runs the FP32 model, with a Grad-CAM backward pass on every request. On a CPU, expect roughly 140 ms per clip for the model's forward pass (notebook 06 benchmark), plus video decoding and Grad-CAM. The 35.6 ms figure in the results refers to the INT8 model, which this service doesn't load yet.

## Quick test

```bash
curl http://localhost:8000/health
```
