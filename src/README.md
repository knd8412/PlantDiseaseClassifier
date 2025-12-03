# Plant Disease Classifier - Source Code

This folder contains the source code for the Plant Disease Classifier project.

## Structure

- **`train.py`**: The main entry point for training the model. It handles configuration loading, data setup, and initiates the training process.
- **`trainer.py`**: Contains the `Trainer` class which encapsulates the training loop, evaluation, and logging logic.
- **`utils.py`**: Utility functions and classes, including `Config`, `EarlyStopping`, `set_seed`, and `accuracy`.
- **`clearml_utils.py`**: Helper functions for integrating with ClearML for experiment tracking.
- **`models/`**: Directory containing model definitions.
  - `convnet_scratch.py`: Definition of the `SmallCNN` model and `build_model` factory.

## Usage

To train the model, run the `train.py` script from the root of the repository:

```bash
python src/train.py --config configs/train.yaml
```

## Components

### Trainer

The `Trainer` class in `trainer.py` provides a simple interface for training:

```python
trainer = Trainer(model, config, task)
trainer.train(train_loader, val_loader, test_loader)
```

### Configuration

The configuration is managed via the `Config` dataclass in `utils.py`, which is populated from a YAML file.

### Models

Models are defined in the `models/` directory. The `build_model` function in `models/convnet_scratch.py` is used to instantiate the model based on configuration.
