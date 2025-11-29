import time
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/..."

# from src.models.cnn import PlantDiseaseModel
# from src.data.labels import CLASS_NAMES
# model = PlantDiseaseModel(num_classes = len(CLASS_NAMES))
# model.load_state_dict(torch.load(model_path,map_location=device))
# model.to(device)
# model.eval()

model = None
CLASS_NAMES = [
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
    "Potato_healthy",
    "Potato_late_blight",
    "Raspberry_healthy",
    "Soybean_healthy",
    "Squash_powdery_mildew",
    "Strawberry_healthy",
    "Strawberry_leaf_scorch",
    "Tomato_bacterial_spot",
    "Tomato_early_blight",
    "Tomato_healthy",
    "Tomato_late_blight",
    "Tomato_leaf_mold",
    "Tomato_septoria_leaf_spot",
    "Tomato_spider_mites_two-spotted_spider_mite",
    "Tomato_target_spot",
    "Tomato_mosaic_virus",
    "Tomato_yellow_leaf_curl_virus",
]


transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ]
)

CUSTOM_CSS = """

:root {
    color-scheme: light dark;
    --page-bg: #020617;
    --page-text: #e5e7eb;
    --card-bg: rgba(15, 23, 42, 0.92);
    --card-border: rgba(148, 163, 184, 0.5);
    --muted-text: #9ca3af;

}

@media (prefers-color-scheme: light){

    :root {
        --page-bg: #f3f4f6;
        --page-text: #020617;
        --card-bg: #ffffff;
        --card-border: rgba(209, 213, 219, 1);
        --muted-text: #6b7280;
    }

}

.gradio-container {
    min-height: 100vh;
    margin: 0;
    background:
        radial-gradient(circle at 0% 0%, #22c55e33 0, transparent 45%),
        radial-gradient(circle at 100% 100%, #0ea5e933 0, transparent 55%),
        radial-gradient(circle at 50% 10%, #910ee955 0, transparent 45%),
        #020617 !important;

    font-family: Arial, system-ui;
}

#app-title {
    text-align: center;
    font-size: 2rem;
    font-weight: 100;
    margin: 0 0 0.4rem;
    color: white !important;
}

#app-subtitle {
    text-align: center;
    margin-bottom: 1.75rem;
    font-size: 25px;
    font-family: Arial;
    color: white !important;
}

.card {
    background: var(--card-bg);
    border-radius: 18px;
    border: 1px solid var(--card-border);
    box-shadow: 0 16px 35px rgba(15, 23, 42, 0.9);
    padding: 1.2rem 1.3rem;
}

.card .gr-image {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(200, 163, 184, 0.45);
}

#analyse_btn,
#analyse_btn button {
    background: linear-gradient(90deg, #6366f1, #a855f7);
    border: none ;
    color: #e5e7eb;
    font-weight: 600;
    padding: 0.7rem 1.6rem;
    border-radius: 999px;
    box-shadow: 0 14px 30px rgba(88, 80, 236, 0.2);
    transition:
        transform 0.1s ease-out,
        box-shadow 0.1s ease-out,
        filter 0.1s ease-out;
}

#analyse_btn:hover,
#analyse_btn button:hover {
    transform: translateY(-1px);
    box-shadow: 0 18px 38px rgba(88, 80, 236, 0.3);
    filter: brightness(1.05);
}


#prediction_label {
    border-radius: 14px;
    padding: 0.9rem 1rem;
    background-color: transparent;
}

#footer-note {
    font-size: 16px;
    margin-top: 0.75rem;
    color: white !important;
}

.gradio-container table {
    font-size: 0.9rem;
    border-collapse: collapse;
}

.gradio-container table thead th {
    background-color: rgba(15, 23, 42, 0.95);
    padding: 0.4rem 0.6rem;
    border-bottom: 1px solid rgba(148, 163, 184, 0.6);
}

.gradio-container table tbody td {
    background-color: transparent;
    padding: 0.35rem 0.6rem;
    border-bottom: 1px solid rgba(30, 41, 59, 0.6);
}



"""


DISEASE_INFO = {
    "Apple_scab": " Fungal disease causing dark, scabby spots on leaves and fruit.",
    "Tomato_healthy": " Leaf appears healthy, with uniform colour and no obvious lesions.",
    "Tomato_early_blight": " Brown lesions with concentric rings, usually starting on older leaves.",
    "Tomato_late_blight": " Irregular water-soaked lesions that can rapidly destroy foliage.",
}


flagged = []


def predict(image):
    if image is None:
        return {}

    if model is None:
        prob = 1 / len(CLASS_NAMES)
        return {name: prob for name in CLASS_NAMES}

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img)[0]

    probabilities = torch.softmax(logits, dim=0)

    top_probs, top_indices = torch.topk(probabilities, k=5)

    result = {}

    for id, p in zip(top_indices.cpu().tolist(), top_probs.cpu().tolist()):
        class_name = CLASS_NAMES[id]
        result[class_name] = float(p)

    return result


def predict_batch(files):

    if not files:
        return []

    rows = []
    for f in files:
        try:
            img = Image.open(f).convert("RGB")
        except Exception:
            continue

        scores = predict(img)
        if not scores:
            continue

        top_class = max(scores, key=scores.get)
        top_prob = scores[top_class]
        rows.append([Path(f).name, top_class, round(float(top_prob), 4)])

    return rows


def predict_with_table(image, model_name):
    scores = predict(image)

    if not scores:
        return {}, []

    top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    top_scores = {cls: prob for cls, prob in top_items}
    table_rows = [[cls, round(prob, 4)] for cls, prob in top_items]

    return top_scores, table_rows


def flag_prediction(image, scores):
    if image is None or not scores:
        return (
            "<p id='footer-note'> No prediction available to flag yet.</p>",
            flagged,
        )

    top_class = max(scores, key=scores.get)
    top_prob = scores[top_class]

    flagged.append(
        [time.strftime("%Y-%m-%d %H:%M:%S"), top_class, round(float(top_prob), 4)]
    )
    msg = f"<p id='footer-note'>Flagged prediction {top_class}: (p={top_prob:.3f})</p>"

    return msg, flagged


def explain_top(scores):
    if not scores:
        return ""

    top_class = max(scores, key=scores.get)
    info = DISEASE_INFO.get(
        top_class, "No description available for this disease class yet"
    )
    return f"<p id='footer-note'>{top_class}:{info} </p>"


# interface = gr.Interface(
#     fn=predict,
#     inputs=gr.Image(type="pil",label="Upload a leaf image",image_mode="RGB"),
#     outputs=gr.Label(num_top_classes=5,label="Predicted Disease"),
#     title="Plant Disease Classifier",
#     description="Upload image of leaf to predict disease. Resized to 256x256 and returns most likely diseases",
#     examples=["examples/tomato_healthy.jpg"]
# )

with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML("<h1 id='app-title'> Plant Disease Classifier</h1>")
    gr.HTML(
        "<p id='app-subtitle'>Upload a leaf image. The model resizes it to 256x256 "
        "and returns the top 5 disease probabilities.</p>"
    )

    gr.HTML(
        "<p id='footer-note'>1. Upload or drag/drop a clear image of a single leaf. <br> 2. Click Analyse Leaf. <br> 3. Inspect the top 5 predicted diseases and probabilities. </p>"
    )

    with gr.Tab("Single Image"):

        modelSelector = gr.Radio(
            ["Baseline CNN", "ResNet 18"], value="Baseline CNN", label="Model"
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
                # output_label = gr.Label(
                #     num_top_classes=5, label=f"Predicted diseases (top {5})"
                # )

                scores_state = gr.State({})

                output_table = gr.Dataframe(
                    headers=["Class", "Probability"],
                    interactive=False,
                    label="Top 5 predictions table",
                )

                gr.HTML(
                    " <p id='footer-note'> Tip: crop the photo so it focuses on a single leaf for best results. </p>"
                )

                explanation_html = gr.HTML("")

                gr.HTML("<p id='footer-note'>Think this prediction is wrong?</p>")
                flag_btn = gr.Button("Flag this prediction", variant="primary")
                flag_msg = gr.HTML("")
                flag_table = gr.Dataframe(
                    headers=["Time", "Top class", "Top probability"],
                    interactive=False,
                    label="Flagged predictions (session)",
                )

    predict_btn.click(
        fn=predict_with_table,
        inputs=[image_input, modelSelector],
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
            inputs=batch_files,
            outputs=batch_table,
        )

if __name__ == "__main__":
    demo.launch(show_api=False)
