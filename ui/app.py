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
    color-scheme: dark;
}

.gradio-container {
    min-height: 100vh;
    margin: 0;
    background:
        radial-gradient(circle at 0% 0%, #22c55e33 0, transparent 45%),
        radial-gradient(circle at 100% 100%, #0ea5e933 0, transparent 55%),
        radial-gradient(circle at 50% 10%, #910ee955 0, transparent 45%),
        #020617;
    color: #e5e7eb;
    font-family: system-ui;
}

#app-title {
    text-align: center;
    font-size: 2rem;
    font-weight: 100;
    margin: 0 0 0.4rem;
}

#app-subtitle {
    text-align: center;
    color: #9ca3af;
    margin-bottom: 1.75rem;
    font-size: 25px;
    font-family: system-ui;
}

.card {
    background: rgba(15, 23, 42, 0.92);
    border-radius: 18px;
    border: 1px solid rgba(148, 163, 184, 0.5);
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
    background-color: #020617;
}

#footer-note {
    font-size: 0.85rem;
    margin-top: 0.75rem;
}
"""


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
        "and returns the <strong>top 5 disease probabilities</strong>.</p>"
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
        with gr.Column(scale=2, elem_classes=["card"]):
            output_label = gr.Label(
                num_top_classes=5,
                label=f"Predicted diseases (top {5})",
                elem_id="prediction_label",
            )
            gr.HTML(
                " <p id='footer-note'> Tip: crop the photo so it focuses on a **single leaf** for best results. </p>"
            )

    predict_btn.click(fn=predict, inputs=image_input, outputs=output_label)


if __name__ == "__main__":
    demo.launch(show_api=False)
