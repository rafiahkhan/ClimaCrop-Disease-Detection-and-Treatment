# ClimaCrop — Crop disease detection and treatment

Farmers used to wait days or months on lab turnaround, travel, or specialist queues just to learn whether a leaf was diseased and what treatment path made sense—while the crop kept deteriorating. This solution shortens that wait: one photo of cotton, maize, rice, or wheat foliage can produce a first disease screening in under three seconds, with clear treatment-style suggestions on the same screen, so growers can respond during the same field visit and still confirm every step with local extension and pesticide label requirements. The underlying system learned from 35,000+ example leaves; the materials published here are the runnable application and trained model bundle, not the full private image archive.

---

## 1. Overview

This repository contains the full stack used to:

- **Train** a single multi-class classifier on leaf imagery for four crops.
- **Serve** inference over HTTP with consistent preprocessing (segmentation, resize, normalization).
- **Present** results in a web UI with optional natural-language recommendations.

The training image corpus (**35,000+** labeled leaves) and on-disk folder trees (`Cottons/`, `Maize/`, `Rice/`, `Wheat/`) are **not** included in the repository; only code, configuration, trained weights (`models/best_model.pt`), and the label map (`models/class_mapping.json`) are distributed here.

---

## 2. Scope

| Item | Value |
|------|--------|
| **Crops** | Cottons, Maize, Rice, Wheat (4) |
| **Classes** | 26 (diseases, healthy, and defined special folders) |
| **Training volume** | 35,000+ images (local dataset) |
| **Classifier** | EfficientNet-B0 (ImageNet-pretrained backbone, fine-tuned classifier head) |
| **Input resolution** | 224×224 RGB |

---

## 3. Features

- **Leaf-centric preprocessing:** HSV-based leaf mask, optional **background blur** outside the mask so the model focuses on foliage (`src/preprocessing/`).
- **Augmented training:** rotations, horizontal flip, brightness/contrast, mild scale/shift (`config.yaml` → `augmentation`).
- **Configurable training loop:** batch size, epochs, learning rate, weight decay, device, early stopping targets (`config.yaml` → `training`).
- **Versioned artifacts:** best checkpoint and `class_mapping.json` under `models/`.
- **REST API:** health check, multipart prediction, optional standalone recommendations endpoint (`src/api.py`).
- **Web client:** drag-and-drop or file picker, preview, analyze action, confidence visualization, top-3 logits, recommendations panel (`frontend/`).
- **CLI inference:** single-image path for scripts and debugging (`src/inference.py`).

---

## 4. Model architecture

- **Backbone:** **EfficientNet-B0** (`torchvision.models.efficientnet_b0`), ImageNet **IMAGENET1K_V1** weights at training start, then fine-tuned end-to-end (`src/train.py`).
- **Classification head:** replaced final linear layer with **26** outputs (one softmax distribution over all labels in `class_mapping.json`).
- **Inference:** checkpoint stores `backbone`, `num_classes`, and `model_state_dict`; the API loads weights and runs evaluation mode (`src/api.py`, `src/inference.py`).

```
Input image
    → load RGB, validate
    → HSV segmentation + optional background blur
    → resize 224×224, ImageNet normalization
    → EfficientNet-B0 forward
    → argmax + softmax probabilities (top-3 exposed in API)
```

---

## 5. Technical stack

| Layer | Technology |
|-------|------------|
| Deep learning | PyTorch, torchvision |
| Vision utilities | OpenCV (segmentation pipeline), Pillow |
| Training utilities | scikit-learn (split), tqdm, matplotlib |
| Server | FastAPI, Uvicorn, python-multipart |
| Configuration | YAML (`config.yaml`) |
| Recommendations | Ollama HTTP API (`requests`; base URL `OLLAMA_BASE_URL` or default `http://127.0.0.1:11434`) |
| Client | HTML5, CSS3, JavaScript (no SPA framework) |

**Python dependencies** are pinned or listed in `requirements.txt` (install with `pip install -r requirements.txt`).

---

## 6. Repository layout

```text
desease_detection/
├── app.py                      # Uvicorn entry (runs src.api:app)
├── config.yaml                 # Data, segmentation, augmentation, model, training
├── requirements.txt
├── LICENSE                     # MIT
├── README.md
├── models/
│   ├── best_model.pt           # Trained EfficientNet-B0 weights + metadata
│   └── class_mapping.json      # Class name → index (26 classes)
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── script.js
└── src/
    ├── api.py                  # FastAPI application
    ├── train.py                # Training script
    ├── inference.py            # CLI + Ollama helper
    ├── dataset.py              # Image discovery, PyTorch Dataset
    ├── preprocessing/          # image_io, segmentation, pipeline
    └── utils/                  # transforms, visualization
```

---

## 7. Dataset layout (for training)

On your machine, organize data as:

```text
<data_root>/
├── Cottons/<class_folder>/*.jpg|png|…
├── Maize/<class_folder>/…
├── Rice/<class_folder>/…
└── Wheat/<class_folder>/…
```

Each immediate subfolder under a crop name becomes part of the label `Crop_FolderName`. Set `data.root_dir` in `config.yaml` to `<data_root>`. Class names must stay consistent with `models/class_mapping.json` if you reuse the shipped checkpoint.

---

## 8. Installation

```bash
git clone https://github.com/rafiahkhan/ClimaCrop-Disease-Detection-and-Treatment.git
cd ClimaCrop-Disease-Detection-and-Treatment
pip install -r requirements.txt
```

**GPU:** install a CUDA-enabled PyTorch build matching your driver if you set `training.device: "cuda"` in `config.yaml`.

---

## 9. Training

```bash
python -m src.train --config config.yaml
```

**Outputs:** `models/best_model.pt`, `models/class_mapping.json`.

**Notable `config.yaml` keys:**

- `data.root_dir`, `data.crops`, `data.img_size`, `data.train_val_split`, `data.num_workers`
- `segmentation.*` (enable, HSV bounds, blur, kernel)
- `augmentation.*`
- `model.backbone` (must remain `efficientnet_b0` for this project’s documented design)
- `training.*` (batch, epochs, lr, weight decay, device, early stopping)

---

## 10. Running the API and web UI

```bash
python app.py
```

Default listen options are defined in `app.py` (port **8001**). Equivalent:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8001
```

Open `http://127.0.0.1:8001` (or your host) in a browser.

**Ollama (recommendations):**

```bash
ollama serve
ollama pull llama3.1:8b
```

Optional remote Ollama host:

```bash
export OLLAMA_BASE_URL=http://127.0.0.1:11434
```

---

## 11. CLI inference

```bash
python -m src.inference --config config.yaml --image_path /path/to/leaf.jpg
python -m src.inference --config config.yaml --image_path /path/to/leaf.jpg --disable_llm
```

Arguments include `--llm_model` (Ollama model tag) when LLM output is enabled.

---

## 12. HTTP API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves the web UI (`frontend/index.html`) |
| `GET` | `/health` | JSON: service health and whether the model loaded |
| `GET` | `/styles.css`, `/script.js` | Static assets for the UI |
| `POST` | `/api/predict` | Multipart form field `file` (image). Query: `include_recommendations` (bool), `llm_model` (string). Returns predicted class, confidence, `top_predictions`, optional `recommendations` |
| `POST` | `/api/recommendations` | JSON body: `disease_name`, optional `llm_model`. Returns structured recommendation text |

CORS is enabled for browser access during development (`src/api.py`).

---

## 13. Web application flow

1. User opens the served URL.
2. User selects an image (drag-and-drop onto the upload card or **Choose file**).
3. Preview appears; user may remove and choose again.
4. User clicks **Analyze disease**; the client posts to `/api/predict`.
5. **Left column:** predicted label, confidence bar, and top alternative classes.
6. **Right column:** formatted treatment-style text when Ollama succeeds; otherwise an error or informational message while the classification result remains valid.

---
## 14. UI screenshots

<p align="center">
  <img src="https://github.com/user-attachments/assets/3f7d18cd-4bfa-431b-b761-042a032944c7" alt="web UI" width="85%" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/ba7c51cb-eae1-43db-9dd6-1a1ad63e2558" alt="web UI" width="85%" />
</p>

---

## 15. Disease classes (26 labels)

The model assigns each image to one of **26** categories that combine **crop** and **condition** in a single label string, for example **cotton** problems such as aphids, army worm, bacterial blight, powdery mildew, or target spot, versus **Cottons_Healthy**; **maize** outcomes such as common rust, gray leaf spot, northern leaf blight, or general blight, versus **Maize_Healthy** and a non-maize leaf bucket; **rice** diseases including bacterial leaf blight, brown spot, false smut, leaf blast, leaf scald, and sheath blight, contrasted with **Rice_Healthy**; and **wheat** classes such as tan spot, mildew, smut, fusarium head blight, brown rust, yellow rust, alongside **Wheat_Wheat___Healthy**. Together these cover the main foliar symptoms and healthy baselines used during training.

Labels are stored as `Crop_FolderName` strings with a stable integer index for training and inference. The **complete, authoritative list** of all 26 names and their indices is **`models/class_mapping.json`**; use that file when auditing API responses, retraining, or building evaluation reports.

---

## 16. Troubleshooting

| Symptom | Check |
|--------|--------|
| `503` / model not loaded | `models/best_model.pt` and `models/class_mapping.json` exist; run server from repo root so `config.yaml` resolves |
| CUDA errors | Set `training.device: "cpu"` or install a matching CUDA PyTorch wheel |
| Empty or failed recommendations | Ollama running (`ollama serve`), model pulled, `OLLAMA_BASE_URL` correct |
| Wrong labels after retraining | Ensure `class_mapping.json` matches folder names; client and server use the same `models/` |

---

## 17. License

**MIT License** — see `LICENSE`. Copyright notice as in that file.

---
