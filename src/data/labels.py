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

class_names_38 = [
    "Apple_scab",
    "Apple_black_rot",
    "Apple_cedar_apple_rust",
    "Apple_healthy",
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


def get_class_names_for_model(model):
    # Works for CNN and ResNet
    if hasattr(model, "head"):  # CNN
        out_features = model.head[-1].out_features
    else:  # ResNet
        fc = model.model.fc
        if hasattr(fc, "out_features"):
            out_features = fc.out_features
        else:  # dropouts in seq, take last layer
            out_features = fc[-1].out_features

    if out_features == 38:
        return class_names_38
    elif out_features == 39:
        return class_names
    else:
        raise ValueError(f"Unexpected number of output classes: {out_features}")
