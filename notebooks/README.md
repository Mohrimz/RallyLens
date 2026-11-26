# Notebooks

All experiments run on **Google Colab with a GPU**, with Google Drive mounted.
- Run **01** and **02** first: they train the two models that the other notebooks load.
- Outputs (checkpoints, figures, metrics JSON) are written to `MyDrive/BadmintonFYP/model23`.
- The saved cell outputs in this folder are from the original runs.

| Notebook | What it does | Key result |
|---|---|---|
| [01_train_video](01_train_video.ipynb) | Trains the 5-class stroke classifier (ResNet-18 + temporal pooling) with session-grouped splits | Test accuracy 0.951, macro-F1 0.947 |
| [02_train_sequence](02_train_sequence.ipynb) | Maps ShuttleSet22's 19 stroke labels to 5 classes, builds serve-inclusive windows, trains the next-shot GRU | Top-3 0.971, macro-F1 0.567 |
| [03_pipeline](03_pipeline.ipynb) | Chains both models with a rolling 4-shot rally history and coaching rules; end-to-end check on 25 clips | 25/25 clips processed and classified |
| [04_app](04_app.ipynb) | Streamlit version of the app, run inside Colab and shared through a Cloudflare tunnel | Earlier prototype UI |
| [05_calibration_robustness](05_calibration_robustness.ipynb) | Temperature scaling, ECE, risk–coverage, accuracy per held-out session | ECE 0.024; per-session accuracy 0.883–0.987 |
| [06_quantization_latency](06_quantization_latency.ipynb) | INT8 post-training quantization (PyTorch FX) and CPU benchmark | 3.9× faster, 4× smaller, −0.01 accuracy |
| [07_xai_explainability](07_xai_explainability.ipynb) | Grad-CAM "why-cards" | One card per stroke class |
| [08_label_efficiency](08_label_efficiency.ipynb) | Retrains on 10/25/50/100% of the labels | 50% of labels → 94% of full macro-F1 |
| [09_rally_pes](09_rally_pes.ipynb) | Rally analyzer: motion-energy event spotting, then classification of each detected hit | Timeline, shot mix, tempo, next shot |
| [10_overfitting_check](10_overfitting_check.ipynb) | Train / validation / test gap for the stroke classifier | Macro-F1 gap 0.041 |
| [11_learning_curves](11_learning_curves.ipynb) | Per-epoch curves with the early-stopping point for both models (re-training runs) | Best validation epoch marked |
| [12_underfitting_check](12_underfitting_check.ipynb) | Both models against majority-class and random baselines | 0.951 vs 0.178 / 0.200 (strokes) |
| [13_hyperparameter_tuning](13_hyperparameter_tuning.ipynb) | GRU grid search and dropout sweep; video learning-rate sweep (validation only) | Best video learning rate 1e-4 |

## Data

- **ShuttleSet22** downloads automatically from the [CoachAI-Projects](https://github.com/wywyWang/CoachAI-Projects) repository.
- **Stroke-clip dataset** (not included). Place your copy as `DATASET2.zip` in `MyDrive/BadmintonFYP`. It should contain the class folders `00_Short Serve`, `02_Lift`, `12_Clear`, `05_Drop Shot` and `14_Smash`. Alternatively, set `GDRIVE_FILE_ID` in the setup cell to a Drive file you can access.
