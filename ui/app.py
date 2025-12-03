import time
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

from data.transforms import get_transforms
from src.data.labels import class_names
from src.models.convnet_scratch import build_model as build_cnn
from src.models.resnet import ResNet18Classifier
from ui.disease_info import disease_info
from ui.styles import styles_css

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_transform = get_transforms(
    image_size=224, train=False, normalize=True, augment=False
)["color"]
resnet_checkpoint = "src/models/checkpoints/resnet18_best.pt"
cnn_checkpoint = "src/models/checkpoints/best_scratch_cnn.pt"
models = {}

ImageNet_mean = [0.485, 0.456, 0.406]
ImageNet_std = [0.229, 0.224, 0.225]


def load_nn_models():
    models = {}
    try:
        scratch_cnn_model = build_cnn(
            num_classes=len(class_names),
            channels=[32, 64, 128],
            regularisation="batchnorm",
            dropout=0.3,
        )

        checkpoint = torch.load(cnn_checkpoint, map_location=device)
        state = checkpoint.get("model_state", checkpoint)
        scratch_cnn_model.load_state_dict(state)
        scratch_cnn_model.to(device)
        scratch_cnn_model.eval()
        models["Baseline CNN"] = scratch_cnn_model

    except Exception as e:
        print(f"Exception: {e}")

    try:
        resnet_model = ResNet18Classifier(
            num_classes=len(class_names),
            pretrained=False,
            dropout=0.2,
            train_backbone=True,
        )
        checkpoint = torch.load(resnet_checkpoint, map_location=device)

        state = checkpoint.get("model_state", checkpoint)
        resnet_model.load_state_dict(state)

        resnet_model.to(device)
        resnet_model.eval()
        models["ResNet_18"] = resnet_model

    except Exception as e:
        print(f"Exception: {e}")

    return models


models = load_nn_models()

flagged = []


def predict(image, model_name):
    if not image:
        return {}

    model = models.get(model_name, None)
    if model is None:
        prob = 1 / len(class_names)
        return {name: prob for name in class_names}

    img_tensor = img_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)[0]

    probabilities = torch.softmax(logits, dim=0)
    top_probs, top_indices = torch.topk(probabilities, k=5)
    result = {}

    for id, p in zip(top_indices.cpu().tolist(), top_probs.cpu().tolist()):
        class_name = class_names[id]
        result[class_name] = float(p)

    return result


def predict_batch(files, model_name):

    if not files:
        return []

    rows = []
    for f in files:
        try:
            img = Image.open(f).convert("RGB")
        except Exception:
            continue

        scores = predict(img, model_name)

        if not scores:
            continue

        top_class = max(scores, key=scores.get)
        top_prob = scores[top_class]
        rows.append([Path(f).name, top_class, round(float(top_prob), 4)])

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
            flagged,
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
        "<p id='app-subtitle'>Upload a leaf image. The model resizes it to 256x256 and returns the top 5 disease probabilities.</p>"
    )

    gr.HTML(
        "<p id='footer-note'>1. Upload an image of a single leaf. <br> 2. Click Analyse Leaf. <br> 3. Inspect the top 5 predicted diseases and probabilities. </p>"
    )

    with gr.Tab("Single Image"):

        selected_model = gr.Radio(
            ["Baseline CNN", "ResNet_18"], value="Baseline CNN", label="Model"
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
                        ["examples/tomato_mold.jpeg"],
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
