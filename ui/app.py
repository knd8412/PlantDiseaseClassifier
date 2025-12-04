import logging
import sys
import time
from pathlib import Path

import gradio as gr
import torch
from data.labels import get_class_names_for_model
from data.transforms import get_transforms
from PIL import Image
from torchvision import transforms

from data.transforms import get_transforms
from data.labels import get_class_names_for_model
from src.models.convnet_scratch import build_model as build_cnn
from src.models.resnet import ResNet18Classifier
from src.models.ViT import ViT_b_16
from ui.disease_info import disease_info
from ui.styles import styles_css

# Configure logging for Hugging Face Spaces
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
    force=True,
)
logger = logging.getLogger(__name__)


def log_and_print(msg, level="INFO"):
    """Log and print to ensure visibility in Hugging Face Spaces"""
    print(f"[{level}] {msg}", flush=True)
    if level == "INFO":
        logger.info(msg)
    elif level == "ERROR":
        logger.error(msg)
    elif level == "WARNING":
        logger.warning(msg)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_transform = get_transforms(
    image_size=224, train=False, normalize=True, augment=False
)["color"]

resnet_checkpoint = "src/models/checkpoints/resnet18_best.pt"
cnn_checkpoint = "src/models/checkpoints/best_scratch_cnn.pt"
vit_checkpoint = "src/models/checkpoints/best_vit_model.pth"
models = {}

ImageNet_mean = [0.485, 0.456, 0.406]
ImageNet_std = [0.229, 0.224, 0.225]


def load_nn_models():
    log_and_print("=== Starting model loading ===")
    log_and_print(f"Device: {device}")
    models = {}
    try:
        log_and_print(f"Loading CNN checkpoint from: {cnn_checkpoint}")
        checkpoint = torch.load(cnn_checkpoint, map_location=device)
        log_and_print(
            f"CNN checkpoint loaded. Keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dict'}"
        )
        state = checkpoint.get("model_state", checkpoint)

        if "head.2.weight" in state:
            num_classes_cnn = state["head.2.weight"].shape[0]
        else:
            num_classes_cnn = 38

        scratch_cnn_model = build_cnn(
            num_classes=(num_classes_cnn),
            channels=[32, 64, 128],
            regularisation="dropout",
            dropout=0.5,
        )

        scratch_cnn_model.load_state_dict(state)
        scratch_cnn_model.to(device)
        scratch_cnn_model.eval()
        models["Baseline CNN"] = scratch_cnn_model
        log_and_print(
            f"✓ Successfully loaded Baseline CNN with num_classes={num_classes_cnn}"
        )

    except Exception as e:
        log_and_print(f"✗ Failed to load Baseline CNN: {e}", "ERROR")
        logger.error(f"✗ Failed to load Baseline CNN: {e}", exc_info=True)

    try:
        log_and_print(f"Loading ResNet checkpoint from: {resnet_checkpoint}")
        checkpoint = torch.load(resnet_checkpoint, map_location=device)
        log_and_print(
            f"ResNet checkpoint loaded. Keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dict'}"
        )
        state = checkpoint.get("model_state", checkpoint)

        if "model.fc.weight" in state:
            num_classes_resnet = state["model.fc.weight"].shape[0]
        elif "model.fc.1.weight" in state:
            num_classes_resnet = state["model.fc.1.weight"].shape[0]
        else:
            num_classes_resnet = 38

        resnet_model = ResNet18Classifier(
            num_classes=(num_classes_resnet),
            pretrained=False,
            dropout=0.2,
            train_backbone=True,
        )

        resnet_model.load_state_dict(state)

        resnet_model.to(device)
        resnet_model.eval()
        models["ResNet_18"] = resnet_model
        log_and_print(
            f"✓ Successfully loaded ResNet18 with num_classes={num_classes_resnet}"
        )

    except Exception as e:
        log_and_print(f"✗ Failed to load ResNet18: {e}", "ERROR")
        logger.error(f"✗ Failed to load ResNet18: {e}", exc_info=True)

    try:
        log_and_print(f"Loading ViT checkpoint from: {vit_checkpoint}")
        checkpoint = torch.load(vit_checkpoint, map_location=device)
        state = torch.load(vit_checkpoint, map_location=device)
        log_and_print(
            f"ViT checkpoint loaded. Keys: {state.keys() if isinstance(state, dict) else 'Not a dict'}"
        )

        new_state = {}
        for k, v in state.items():
            new_state[f"model.{k}"] = v
        num_classes_vit = new_state["model.heads.3.weight"].shape[0]

        vit_model = ViT_b_16(num_classes=num_classes_vit, dropout=0.3, device=device)
        vit_model.load_state_dict(new_state)
        vit_model.eval()
        models["ViT_b_16"] = vit_model
        log_and_print(
            f"✓ Successfully loaded ViT_b_16 with num_classes={num_classes_vit}"
        )

    except Exception as e:
        log_and_print(f"✗ Failed to load ViT_b_16: {e}", "ERROR")
        logger.error(f"✗ Failed to load ViT_b_16: {e}", exc_info=True)

    log_and_print(
        f"=== Model loading complete. Loaded models: {list(models.keys())} ==="
    )
    return models


models = load_nn_models()

flagged = []


def predict(image, model_name):
    log_and_print(f"\n=== Starting prediction with model: {model_name} ===")
    if not image:
        log_and_print("No image provided", "WARNING")
        return {}

    model = models.get(model_name, None)
    log_and_print(f"Model retrieved: {model is not None}")
    class_names = get_class_names_for_model(model)
    log_and_print(f"Number of classes: {len(class_names)}")
    if model is None:
        log_and_print(
            f"Model '{model_name}' not found in loaded models: {list(models.keys())}",
            "ERROR",
        )
        prob = 1 / len(class_names)
        return {name: prob for name in class_names}

    try:
        log_and_print(
            f"Image type: {type(image)}, mode: {image.mode if hasattr(image, 'mode') else 'N/A'}, size: {image.size if hasattr(image, 'size') else 'N/A'}"
        )
        img_tensor = img_transform(image).unsqueeze(0).to(device)
        log_and_print(
            f"Tensor shape: {img_tensor.shape}, dtype: {img_tensor.dtype}, device: {img_tensor.device}"
        )
        log_and_print(
            f"Tensor stats - min: {img_tensor.min():.3f}, max: {img_tensor.max():.3f}, mean: {img_tensor.mean():.3f}"
        )

        with torch.no_grad():
            log_and_print("Running forward pass...")
            logits = model(img_tensor)[0]
            log_and_print(
                f"Logits shape: {logits.shape}, min: {logits.min():.3f}, max: {logits.max():.3f}"
            )

        probabilities = torch.softmax(logits, dim=0)
        log_and_print(
            f"Probabilities - min: {probabilities.min():.3f}, max: {probabilities.max():.3f}, sum: {probabilities.sum():.3f}"
        )
        top_probs, top_indices = torch.topk(probabilities, k=5)
        result = {}

        for id, p in zip(top_indices.cpu().tolist(), top_probs.cpu().tolist()):
            class_name = class_names[id]
            result[class_name] = float(p)
            log_and_print(f"  Top prediction: {class_name} = {p:.4f}")

        log_and_print(
            f"=== Prediction complete. Top class: {max(result, key=result.get)} ==="
        )
        return result
    except Exception as e:
        log_and_print(f"Error during prediction: {e}", "ERROR")
        logger.error(f"Error during prediction: {e}", exc_info=True)
        return {}


def predict_batch(files, model_name):
    log_and_print(
        f"\n=== Starting batch prediction with {len(files) if files else 0} files ==="
    )
    if not files:
        log_and_print("No files provided for batch prediction", "WARNING")
        return []

    rows = []
    for idx, f in enumerate(files):
        log_and_print(f"Processing file {idx+1}/{len(files)}: {Path(f).name}")
        try:
            img = Image.open(f).convert("RGB")
            log_and_print(f"  Image loaded: {img.size}")
        except Exception as e:
            log_and_print(f"  Failed to load image: {e}", "ERROR")
            continue

        scores = predict(img, model_name)

        if not scores:
            log_and_print(f"  No scores returned for {Path(f).name}", "WARNING")
            continue

        top_class = max(scores, key=scores.get)
        top_prob = scores[top_class]
        rows.append([Path(f).name, top_class, round(float(top_prob), 4)])
        log_and_print(f"  Result: {top_class} ({top_prob:.4f})")

    log_and_print(f"=== Batch prediction complete. Processed {len(rows)} images ===")
    return rows


def predict_with_table(image, model_name):
    scores = predict(image, model_name)

    if not scores:
        return {}, []

    top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    top_scores = {cls: prob for cls, prob in top_items}
    table_rows = [[cls, round(prob, 4)] for cls, prob in top_items]

    return top_scores, table_rows


def flag_prediction(image, scores):
    if image is None or not scores:
        return (
            "<p id='footer-note'> No prediction available to flag. </p>",
            [[]],
        )

    top_class = max(scores, key=scores.get)
    top_prob = scores[top_class]

    flagged.append(
        [time.strftime("%d-%m-%Y %H:%M:%S"), top_class, round(float(top_prob), 4)]
    )
    msg = f"<p id='footer-note'>Flagged prediction {top_class}: (p={top_prob:0.3f})</p>"

    return msg, flagged


def explain_top(scores):
    if not scores:
        return ""

    top_class = max(scores, key=scores.get)
    info = disease_info.get(top_class, "")
    return f"<p id='footer-note'>{top_class}:{info} </p>"


with gr.Blocks(css=styles_css, theme=gr.themes.Soft()) as demo:
    gr.HTML("<h1 id='app-title'> Plant Disease Classifier</h1>")
    gr.HTML(
        "<p id='app-subtitle'>Upload a leaf image to view the top 5 disease output_table probabilities.</p>"
    )

    gr.HTML(
        "<p id='footer-note'>1. Upload an image of a single leaf. <br> 2. Click Analyse Leaf. <br> 3. Inspect the top 5 predicted diseases and probabilities. </p>"
    )

    with gr.Tab("Single Image"):

        selected_model = gr.Radio(
            ["Baseline CNN", "ResNet_18", "ViT_b_16"],
            value="Baseline CNN",
            label="Model",
        )
        with gr.Row():
            with gr.Column(scale=3, elem_classes=["card"]):
                image_input = gr.Image(
                    type="pil",
                    image_mode="RGB",
                    label="Leaf image",
                    height=350,
                )
                predict_btn = gr.Button(
                    "Analyse leaf", variant="primary", elem_id="analyse_btn"
                )
                gr.HTML("<p id='footer-note'>Example Images</p>")
                gr.Examples(
                    examples=[
                        ["examples/tomato_earlyB.jpeg"],
                        ["examples/pepper_bell_healthy.jpeg"],
                        ["examples/Tomato_mold.jpeg"],
                        ["examples/potato_late_blight.jpeg"],
                        ["examples/tomato_spider_mites.jpeg"],
                    ],
                    inputs=image_input,
                )

            with gr.Column(scale=2, elem_classes=["card"]):

                scores_state = gr.State({})

                output_table = gr.Dataframe(
                    headers=["Class", "Probability"],
                    interactive=False,
                    label="Top 5 predictions table",
                )
                gr.HTML(
                    "<p id='footer-note'> Tip: Crop the photo to focus on a single leaf for best results. </p>"
                )
                explanation_html = gr.HTML("")

                gr.HTML("<p id='footer-note'>Incorrect prediction?</p>")
                flag_btn = gr.Button("Flag this prediction", variant="primary")
                flag_msg = gr.HTML("")
                flag_table = gr.Dataframe(
                    headers=["Time", "Top class", "Top probability"],
                    interactive=False,
                    label="Flagged predictions (session)",
                )

    predict_btn.click(
        fn=predict_with_table,
        inputs=[image_input, selected_model],
        outputs=[scores_state, output_table],
    ).then(fn=explain_top, inputs=scores_state, outputs=explanation_html)
    flag_btn.click(
        fn=flag_prediction,
        inputs=[image_input, scores_state],
        outputs=[flag_msg, flag_table],
    )

    with gr.Tab("Batch Mode"):
        gr.HTML(
            "<p id='footer-note'>Upload multiple leaf images to analyse them in one go. "
            "For each file we show the top predicted disease and probability. </p>"
        )

        batch_files = gr.File(
            label="Leaf images",
            file_count="multiple",
            type="filepath",
        )
        batch_btn = gr.Button("Analyse batch")
        batch_table = gr.Dataframe(
            headers=["Filename", "Top class", "Top probability"],
            interactive=False,
            label="Batch results",
        )

        batch_btn.click(
            fn=predict_batch,
            inputs=[batch_files, selected_model],
            outputs=batch_table,
        )

if __name__ == "__main__":
    demo.launch(show_api=False)
