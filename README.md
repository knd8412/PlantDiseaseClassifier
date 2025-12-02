## 📘 Overview

The goal of this project is to build an AI model capable of detecting plant diseases from leaf images.
This is an **image classification problem** — the system receives a leaf picture and predicts the corresponding **plant–disease class** (ex *Tomato_Early_Blight* or *Grape_Black_Rot*).

# PlantDiseaseClassifier (Scratch CNN + Early Stopping + ClearML)
A plant disease detector from leaf photos using the PlantVillage dataset (39 classes, 55,400 images at 256×256).

📊 **Project Board:** [View on GitHub Projects](https://github.com/<username>/<repo-name>/projects)

> An AI-powered system that detects plant diseases from leaf images using deep learning, with full experiment tracking, CI/CD automation, and a deployable Gradio interface.

The workflow covers the entire machine learning lifecycle:
1. **Data preparation** (using the [PlantVillage dataset](https://huggingface.co/datasets/plant_village))
2. **Model development and training** (PyTorch CNN + transfer learning baseline)
3. **Experiment tracking** (ClearML)
4. **Evaluation and comparison**
5. **Deployment** (Gradio UI on Hugging Face Spaces)

---

## ⚙️ Tech Stack

| Component | Technology |
|------------|-------------|
| Framework | PyTorch |
| Experiment Tracking | ClearML (KCL-hosted server) |
| Interface | Gradio |
| Deployment | Hugging Face Spaces |
| CI/CD | GitHub Actions (self-hosted runner) |
| Linting & Testing | flake8, pytest, pre-commit hooks |
| Version Control | Git / GitHub Enterprise (KCL) |

---

## 🧠 Features

- Custom CNN model for image classification
- Optional transfer learning with ResNet18
- Data augmentation and normalization
- Real-time experiment tracking and metric comparison with ClearML
- Automatically logged model checkpoints and hyperparameters
- Interactive Gradio web app for leaf image uploads
- Batch image classification support
- Public deployment on Hugging Face Spaces
- Fully automated CI/CD via self-hosted GitHub Actions runner
- Pre-commit hooks enforcing code quality and style


## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Optional: configure ClearML (one-time)
clearml-init  # use KCL ClearML server settings provided by your course
# Train (subset for fast prototyping)
python src/train.py --config configs/train.yaml

# Evaluate the trained model
python src/evaluate.py --model outputs/best.pt --split val
```

The first run will download the dataset from the Hugging Face Hub.
If you are in an offline environment, pre-download the dataset or cache it locally.

## Outputs
- `outputs/best.pt` — best model weights (by validation accuracy).
- `outputs/metrics.json` — last run metrics summary.
- ClearML: check your project dashboard for the task, metrics, and registered model.

### Evaluation Outputs
- `confusion_matrix.png` — visual confusion matrix heatmap
- `errors/` directory — error gallery with misclassified samples (when enabled)
- JSON results file — comprehensive metrics and per-class statistics

## Model Evaluation

The project includes a comprehensive evaluation script that provides detailed metrics and error analysis for trained models.

### Basic Evaluation
```bash
python src/evaluate.py --model outputs/best.pt --split val
```

> **Important:** The `--config` must match the architecture used to train the model. If you trained with `configs/train_quick_test.yaml`, use that same config for evaluation:
> ```bash
> python src/evaluate.py --model outputs/best.pt --config configs/train_quick_test.yaml --split val
> ```

This command generates:
- **Overall accuracy** and **top-5 accuracy** metrics
- **Per-class precision, recall, and F1 scores**
- **Confusion matrix** visualization (`confusion_matrix.png`)
- **JSON results** file with detailed metrics

### Error Gallery Feature
The evaluation script includes an advanced error gallery that visualizes the worst confusion patterns:

```bash
python src/evaluate.py --model outputs/best.pt --split val \
    --error-gallery \
    --gallery-top-pairs 5 \
    --gallery-samples-per-pair 10
```

The error gallery generates:
- **Image grids** showing misclassified samples for each confusion pair
- **Metadata files** with sample indices and confusion statistics
- **Comprehensive analysis** in markdown format

For detailed documentation on the error gallery functionality, see [`ERROR_GALLERY_README.md`](ERROR_GALLERY_README.md).

### Advanced Options
- `--config`: Specify alternative configuration file (default: `configs/train.yaml`)
- `--output`: Custom path for results JSON file
- `--gallery-top-pairs`: Number of worst confusion pairs to analyze
- `--gallery-samples-per-pair`: Number of misclassified samples per confusion pair

### Output Interpretation
Evaluation results include:
- **Overall Accuracy**: Percentage of correctly classified samples
- **Top-5 Accuracy**: Percentage where correct class is in top 5 predictions
- **Per-class Metrics**: Precision, recall, F1-score for each disease class
- **Confusion Matrix**: Visual representation of classification patterns

### ClearML Integration
Evaluation tasks are automatically logged to ClearML with:
- Accuracy metrics tracked over time
- Confusion matrix images uploaded as artifacts
- Error gallery images organized by confusion pair
- Error analysis markdown available as downloadable artifact

## Notes
- For speed during prototyping, the config uses a 25% subset.
- Adjust `batch_size` to fit your GPU/CPU memory.
- To run on CPU, keep `device: cpu` in the config.
