# PlantDiseaseClassifier (Scratch CNN + Early Stopping + ClearML)
A plant disease detector from leaf photos using the PlantVillage dataset (39 classes, 55,400 images at 256×256).

📊 **Project Board:** [View on GitHub Projects](https://github.com/<username>/<repo-name>/projects)

> An AI-powered system that detects plant diseases from leaf images using deep learning, with full experiment tracking, CI/CD automation, and a deployable Gradio interface.

---

## 📘 Overview

The goal of this project is to build an AI model capable of detecting plant diseases from leaf images.
This is an **image classification problem** — the system receives a leaf picture and predicts the corresponding **plant–disease class** (ex *Tomato_Early_Blight* or *Grape_Black_Rot*).

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
