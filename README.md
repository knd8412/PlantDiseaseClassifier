# 🌿 PlantDiseaseClassifier

> **AI-powered plant disease detection from leaf images** — built with PyTorch, tracked with ClearML, and deployed on Hugging Face Spaces.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Interface-Gradio-orange?logo=gradio)](https://gradio.app/)
[![ClearML](https://img.shields.io/badge/Tracking-ClearML-purple)](https://clear.ml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?logo=githubactions)](https://github.com/features/actions)

---

## 🔗 Quick Links

| Resource | Link |
|---|---|
| 📁 GitHub Repository | [knd8412/PlantDiseaseClassifier](https://github.com/knd8412/PlantDiseaseClassifier) |
| 🚀 Live Demo (Hugging Face) | [Vinuit/PlantDiseaseClassifier](https://huggingface.co/spaces/Vinuit/PlantDiseaseCLassifier) |
| 🧪 Baseline CNN Experiment | `c0422871afdd43a4905b6801890f3324` |
| 🧠 ResNet18 Experiment | `d6035906610145b7b2cfeca0fb1fa155` |

---

## 📖 Overview

**PlantDiseaseClassifier** is an end-to-end machine learning system that identifies plant diseases from photographs of leaves. Upload a photo — get a diagnosis.

The model is trained on the [PlantVillage dataset](https://huggingface.co/datasets/), which contains **55,400 images** across **39 plant–disease classes** at 256×256 resolution. Examples of classes include `Tomato_Early_Blight`, `Grape_Black_Rot`, and `Apple_Cedar_Rust`.

The project covers the full ML lifecycle:

```
Raw Data → Preprocessing → Model Training → Experiment Tracking → Evaluation → Deployment
```

---

## ✨ Features

- 🧠 **Custom CNN** trained from scratch on PlantVillage
- 🔄 **Transfer Learning** option with ResNet18
- 📊 **Real-time experiment tracking** via ClearML (metrics, checkpoints, hyperparameters)
- 🖼️ **Data augmentation** and normalization pipeline
- 🌐 **Interactive Gradio web app** — upload any leaf photo and get a prediction
- 📦 **Batch classification** support
- 🚀 **Public deployment** on Hugging Face Spaces
- 🔁 **CI/CD automation** via self-hosted GitHub Actions runner
- 🪝 **Pre-commit hooks** for code quality and style (flake8)
- 🔍 **Architecture auto-detection** — evaluation script figures out your model type automatically
- 🗂️ **Error gallery** — visual analysis of worst misclassification patterns

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Deep Learning Framework | PyTorch |
| Experiment Tracking | ClearML (KCL-hosted server) |
| Web Interface | Gradio |
| Deployment | Hugging Face Spaces |
| CI/CD | GitHub Actions (self-hosted runner) |
| Linting | flake8 |
| Testing | pytest |
| Code Quality | pre-commit hooks |
| Version Control | Git / GitHub |

---

## 🗂️ Project Structure

```
PlantDiseaseClassifier/
├── .github/              # GitHub Actions CI/CD workflows
├── configs/              # YAML configuration files for training/evaluation
├── data/                 # Dataset utilities and preprocessing scripts
├── datasetNotebooks/     # Exploratory data analysis notebooks
├── examples/             # Sample images for testing
├── src/
│   ├── train.py          # Main training script
│   └── evaluate.py       # Evaluation script with error gallery
├── tests/                # pytest test suite
├── ui/
│   └── app.py            # Gradio web application
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Project metadata and tooling config
├── .pre-commit-config.yaml
├── .flake8
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/knd8412/PlantDiseaseClassifier.git
cd PlantDiseaseClassifier
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Set Up ClearML Experiment Tracking

Only needed once. Skip if you don't need experiment tracking.

```bash
clearml-init
```

### 5. Launch the Web App

```bash
python -m ui.app
```

The Gradio app will open in your browser at **http://localhost:7860**.

---

## 🏋️ Training a Model

```bash
# Train using default config (25% subset for fast prototyping)
python src/train.py --config configs/train.yaml
```

The dataset downloads automatically from Hugging Face Hub on first run and is cached at `~/.cache/huggingface/datasets/`. Subsequent runs use the cache — no re-download needed.

> 💡 **Tip:** Adjust `batch_size` in the config to fit your GPU/CPU memory. To force CPU, set `device: cpu` in the config file.

**Outputs after training:**

| File | Description |
|---|---|
| `outputs/best.pt` | Best model weights (by validation accuracy) |
| `outputs/metrics.json` | Summary of the last run's metrics |
| ClearML dashboard | Full task logs, metrics curves, and registered model |

---

## 📊 Evaluating a Model

The evaluation script **auto-detects the model architecture** using a 3-step fallback:

1. **Checkpoint metadata** — reads embedded `model_config` if saved by the updated `train.py`
2. **Auto-inference** — analyzes `state_dict` weight shapes and key patterns
3. **Config file fallback** — uses `--config` or `configs/train.yaml` as last resort

```bash
# Basic evaluation (architecture auto-detected)
python src/evaluate.py --model outputs/best.pt --split val

# Evaluate on the test split
python src/evaluate.py --model outputs/best.pt --split test

# Validate your setup without running full evaluation
python src/evaluate.py --model outputs/best.pt --dry-run

# Skip error gallery for faster runs
python src/evaluate.py --model outputs/best.pt --split val --no-error-gallery

# Override architecture with a specific config (for old checkpoints)
python src/evaluate.py --model outputs/best.pt --config configs/train_quick_test.yaml --split val
```

### Evaluation Output

| Output | Description |
|---|---|
| Overall Accuracy | % of correctly classified samples |
| Top-5 Accuracy | % where correct class appears in top 5 predictions |
| Per-class Metrics | Precision, recall, F1-score for each disease class |
| `confusion_matrix.png` | Visual heatmap of classification patterns |
| `errors/` directory | Error gallery with misclassified samples |
| JSON results file | Full metrics and per-class statistics |

---

## 🔍 Error Gallery

The error gallery visualizes the **worst confusion patterns** your model makes — useful for diagnosing where and why it fails.

```bash
python src/evaluate.py --model outputs/best.pt --split val \
    --error-gallery \
    --gallery-top-pairs 5 \
    --gallery-samples-per-pair 10
```

**Generated output:**

- Image grids of misclassified samples per confusion pair
- Metadata files with sample indices and confusion statistics
- Full analysis report in Markdown format

See [`ERROR_GALLERY_README.md`](ERROR_GALLERY_README.md) for detailed documentation.

---

## 📉 Confusion Matrix Options

The confusion matrix defaults to the **15 most confused classes** for readability.

```bash
# Show top 10 most confused classes
python src/evaluate.py --model outputs/best.pt --split val --cm-classes 10

# Show all 39 classes
python src/evaluate.py --model outputs/best.pt --split val --cm-classes 0
```

---

## 🧪 Running Tests

```bash
pytest tests/
```

Pre-commit hooks run automatically on every `git commit` to enforce code quality. To run them manually:

```bash
pre-commit run --all-files
```

---

## 📈 Experiment Tracking with ClearML

All training and evaluation runs are automatically logged to ClearML:

- 📉 Accuracy and loss curves
- 🖼️ Confusion matrix uploaded as an artifact
- 🗂️ Error gallery images organized by confusion pair
- 📄 Error analysis Markdown as a downloadable artifact
- ⚙️ All hyperparameters captured automatically

Check your ClearML project dashboard after any `train.py` or `evaluate.py` run.

---

## 🌐 Deployment

The app is publicly deployed on **Hugging Face Spaces**:

👉 [https://huggingface.co/spaces/Vinuit/PlantDiseaseCLassifier](https://huggingface.co/spaces/Vinuit/PlantDiseaseCLassifier)

Upload any plant leaf image and receive an instant disease prediction.

---

## 👥 Contributors

| GitHub | Name |
|---|---|
| [@knd8412](https://github.com/knd8412) | Kamyar Nadarkhani |
| [@Vinuitik](https://github.com/Vinuitik) | Vinuitik |
| [@k23099462](https://github.com/k23099462) | Jaroslav Rakoto-Miklas |
| [@SoroushSoroush20041383](https://github.com/Soroush20041383) | Soroush |

---

*Built with 🌱 for plant health and deep learning.*
