# PlantVillage Baseline (Scratch CNN + Early Stopping + ClearML)

Minimal scaffold for first task:
- Small CNN written from scratch (PyTorch).
- Training loop with early stopping (on validation accuracy).
- ClearML experiment logging (metrics, params, model checkpoint).
- Loads **PlantVillage** via datasets and does a stratified split.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Optional: configure ClearML (one-time)
clearml-init  # use KCL ClearML server settings provided by your course
# Train (subset for fast prototyping)
python src/train.py --config configs/train.yaml
```

The first run will download the dataset from the Hugging Face Hub.
If you are in an offline environment, pre-download the dataset or cache it locally.

## Outputs
- `outputs/best.pt` — best model weights (by validation accuracy).
- `outputs/metrics.json` — last run metrics summary.
- ClearML: check your project dashboard for the task, metrics, and registered model.

## Notes
- For speed during prototyping, the config uses a 25% subset.
- Adjust `batch_size` to fit your GPU/CPU memory.
- To run on CPU, keep `device: cpu` in the config.
