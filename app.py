import time
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

from data.transforms import get_transforms
from src.models.convnet_scratch import build_model as build_cnn
from src.models.resnet import ResNet18Classifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = [
    "Apple_scab",
    "Apple_black_rot",
    "Apple_cedar_apple_rust",
    "Apple_healthy",
    "Background_without_leaves",
    "Blueberry_healthy",
    "Cherry_powdery_mildew",
    "Cherry_healthy",
    "Corn_gray_leaf_spot",
    "Corn_common_rust",
    "Corn_northern_leaf_blight",
    "Corn_healthy",
    "Grape_black_rot",
    "Grape_black_measles",
    "Grape_leaf_blight",
    "Grape_healthy",
    "Orange_haunglongbing",
    "Peach_bacterial_spot",
    "Peach_healthy",
    "Pepper_bacterial_spot",
    "Pepper_healthy",
    "Potato_early_blight",
    "Potato_late_blight",
    "Potato_healthy",
    "Raspberry_healthy",
    "Soybean_healthy",
    "Squash_powdery_mildew",
    "Strawberry_leaf_scorch",
    "Strawberry_healthy",
    "Tomato_bacterial_spot",
    "Tomato_early_blight",
    "Tomato_late_blight",
    "Tomato_leaf_mold",
    "Tomato_septoria_leaf_spot",
    "Tomato_spider_mites_two-spotted_spider_mite",
    "Tomato_target_spot",
    "Tomato_yellow_leaf_curl_virus",
    "Tomato_mosaic_virus",
    "Tomato_healthy",
]


img_transform = get_transforms(
    image_size=224, train=False, normalize=True, augment=False
)["color"]
resnet_checkpoint = "src/models/checkpoints/resnet18_best.pt"
cnn_checkpoint = "src/models/checkpoints/cnn_scratch_best.pt"
models = {}

ImageNet_mean = [0.485, 0.456, 0.406]
ImageNet_std = [0.229, 0.224, 0.225]


styling_css = """
:root {
    --bg-dark: #020617;
    --bg-light: #f3f4f6;
    --text-dark: #020617;
    --text-light: #e5e7eb;
    --card-dark: rgba(15, 23, 42, 0.92);
    --card-light: #ffffff;
    --border-dark: rgba(148, 163, 184, 0.5);
    --border-light: rgba(209, 213, 219, 1);
    --muted-dark: #9ca3af;
    --muted-light: #6b7280;
}

@media (prefers-color-scheme: dark) {
    body {
        background-color: var(--bg-dark);
        color: var(--text-light);
    }
}

@media (prefers-color-scheme: light) {
    body {
        background-color: var(--bg-light);
        color: var(--text-dark);
    }
}

.gradio-container {
    min-height: 100vh;
    background:
        radial-gradient(circle at 0% 0%, #22c55e33 0, transparent 45%),
        radial-gradient(circle at 100% 100%, #0ea5e933 0, transparent 55%),
        radial-gradient(circle at 50% 10%, #910ee955 0, transparent 45%),
        var(--bg-dark);
    font-family: system-ui, sans-serif;
    margin: 0;
}

#app-title, #app-subtitle {
    text-align: center;
    color: var(--text-light);
}

#app-title {
    font-size: 2rem;
    font-weight: 300;
    margin-bottom: 0.4rem;
}

#app-subtitle {
    font-size: 1.5rem;
    margin-bottom: 1.75rem;
}

.card {
    background: var(--card-dark);
    border-radius: 16px;
    border: 1px solid var(--border-dark);
    padding: 1.25rem;
    box-shadow: 0 16px 35px rgba(15, 23, 42, 0.8);
}

.card .gr-image {
    border-radius: 12px;
    border: 1px solid rgba(200, 163, 184, 0.4);
    overflow: hidden;
}

#analyse_btn {
    background: linear-gradient(90deg, #6366f1, #a855f7);
    color: white;
    font-weight: 600;
    padding: 0.7rem 1.6rem;
    border: none;
    border-radius: 999px;
    box-shadow: 0 14px 30px rgba(88, 80, 236, 0.2);
    transition: transform 0.1s ease, box-shadow 0.1s ease;
}

#analyse_btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 18px 38px rgba(88, 80, 236, 0.3);
    filter: brightness(1.05);
}

#footer-note {
    font-size: 0.95rem;
    color: var(--text-light);
    margin-top: 1rem;
}

.gradio-container table {
    font-size: 0.9rem;
    border-collapse: collapse;
    width: 100%;
}

.gradio-container table thead th {
    background-color: rgba(15, 23, 42, 0.9);
    padding: 0.5rem;
    border-bottom: 1px solid var(--border-dark);
}

.gradio-container table tbody td {
    background-color: transparent;
    padding: 0.5rem;
    border-bottom: 1px solid rgba(30, 41, 59, 0.5);
}
"""

DISEASE_INFO = {
    "Apple_scab": " Fungal disease causing dark, scabby spots on leaves and fruit.",
    "Tomato_healthy": " Leaf appears healthy, with uniform colour and no obvious lesions.",
    "Tomato_early_blight": " Brown lesions with concentric rings, usually starting on older leaves.",
    "Tomato_late_blight": " Irregular water-soaked lesions that can rapidly destroy foliage.",
}


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
        [time.strftime("%Y-%m-%d %H:%M:%S"), top_class, round(float(top_prob), 4)]
    )
    msg = f"<p id='footer-note'>Flagged prediction {top_class}: (p={top_prob:0.3f})</p>"

    return msg, flagged


def explain_top(scores):
    if not scores:
        return ""

    top_class = max(scores, key=scores.get)
    info = DISEASE_INFO.get(top_class, "")
    return f"<p id='footer-note'>{top_class}:{info} </p>"


with gr.Blocks(css=styling_css, theme=gr.themes.Soft()) as demo:
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
