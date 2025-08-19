# Copyright 2025 Chaudhry Ahmed Ali Khan
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from gradcam_multi import generate_gradcam_explanations
from config import Config
from ultralytics import YOLO

# ------------------ Image preprocessing ------------------ #
def load_and_preprocess(image_path):
    original_img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(original_img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    processed_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    return original_img, processed_tensor

def shrink_image(image, scale=0.85):
    if isinstance(image, Image.Image):
        image = np.array(image)
    h, w = image.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

# ------------------ Prediction interpretation ------------------ #
def interpret_predictions(preds):
    is_organic_prob = preds['is_organic'][0][0]
    label = "Organic" if is_organic_prob > 0.5 else "Inorganic"
    confidence = is_organic_prob if is_organic_prob > 0.5 else 1 - is_organic_prob
    confidence_pct = confidence * 100

    qg_probs = preds['quality_grade'][0]
    qg_idx = np.argmax(qg_probs)
    quality_map = {0: "Bad", 1: "Medium", 2: "Good"}
    quality_label = quality_map.get(qg_idx, "Unknown")
    quality_conf_pct = qg_probs[qg_idx] * 100

    size_probs = preds['size'][0]
    size_idx = np.argmax(size_probs)
    size_map = {0: "Small", 1: "Medium", 2: "Big"}
    size_label = size_map.get(size_idx, "Unknown")
    size_conf_pct = size_probs[size_idx] * 100

    shiny_prob = preds['shininess'][0][0]
    shiny_label = "Shiny" if shiny_prob > 0.5 else "Dull"
    shiny_conf_pct = shiny_prob * 100 if shiny_label == "Shiny" else (1 - shiny_prob) * 100

    darkspot_prob = preds['darkspots'][0][0]
    darkspot_label = "Yes" if darkspot_prob > 0.5 else "No"
    darkspot_conf_pct = darkspot_prob * 100 if darkspot_label == "Yes" else (1 - darkspot_prob) * 100

    shape_probs = preds['shape_irregularity'][0]
    shape_idx = np.argmax(shape_probs)
    shape_map = {0: "Normal", 1: "Some irregularity", 2: "Lots of irregularity"}
    shape_label = shape_map.get(shape_idx, "Unknown")
    shape_conf_pct = shape_probs[shape_idx] * 100

    pred_text = (
        r"$\mathbf{PREDICTION}$" + "\n"
        + r"$\mathbf{Label:}$" + f" {label} ({confidence_pct:.1f}%)\n"
        + r"$\mathbf{Quality:}$" + f" {quality_label} ({quality_conf_pct:.1f}%)\n"
        + r"$\mathbf{Size:}$" + f" {size_label} ({size_conf_pct:.1f}%)\n"
        + r"$\mathbf{Shine:}$" + f" {shiny_label} ({shiny_conf_pct:.1f}%)\n"
        + r"$\mathbf{Dark\ Spots:}$" + f" {darkspot_label} ({darkspot_conf_pct:.1f}%)\n"
        + r"$\mathbf{Shape:}$" + f" {shape_label} ({shape_conf_pct:.1f}%)"
    )
    return pred_text, (shiny_label, darkspot_label, size_label, shape_label)

def plot_results(original_img, class_overlay, quality_overlay, pred_text, class_text, quality_text):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(shrink_image(original_img, scale=0.85)); axs[0].axis('off'); axs[0].set_title(r"$\mathbf{Original\ Image}$", fontsize=14)
    axs[0].text(0.3, -0.08, pred_text, transform=axs[0].transAxes, fontsize=10, va='top', ha='center', wrap=True)

    axs[1].imshow(shrink_image(class_overlay, scale=0.85)); axs[1].axis('off'); axs[1].set_title(r"$\mathbf{Classification\ Grad\text{-}CAM}$", fontsize=14)
    axs[1].text(0.3, -0.08, class_text, transform=axs[1].transAxes, fontsize=10, va='top', ha='center', wrap=True)

    axs[2].imshow(shrink_image(quality_overlay, scale=0.85)); axs[2].axis('off'); axs[2].set_title(r"$\mathbf{Quality\ Grad\text{-}CAM}$", fontsize=14)
    axs[2].text(0.3, -0.08, quality_text, transform=axs[2].transAxes, fontsize=10, va='top', ha='center', wrap=True)

    plt.tight_layout()
    plt.show()

# ------------------ Main change: add class and quality notes ------------------ #
def predict_and_explain(image_path, classifier_model, yolo_model):
    original_img = cv2.imread(image_path)
    results = yolo_model(original_img)

    object_counts = 0
    shiny_count = 0
    darkspot_count = 0
    irregular_shape_count = 0

    # Initialize size counters
    size_counter = {"Small": 0, "Medium": 0, "Big": 0}

    # Draw YOLO boxes and get classifier stats
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # draw box, no label

            crop = original_img[y1:y2, x1:x2]
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
            crop_resized = cv2.resize(crop, (224, 224))
            crop_resized = np.expand_dims(crop_resized.astype(np.float32)/255.0, axis=0)
            preds = classifier_model(crop_resized)
            preds = {k: v.numpy() for k, v in preds.items()}

            _, (shiny, darkspot, size, shape) = interpret_predictions(preds)
            shiny_count += shiny == "Shiny"
            darkspot_count += darkspot == "Yes"
            irregular_shape_count += shape != "Normal"
            object_counts += 1
            size_counter[size] += 1  # Count size occurrences

    # Grad-CAM on full image
    _, processed_tensor = load_and_preprocess(image_path)
    class_head = 'is_organic'
    quality_head = 'quality_grade'
    class_overlay, quality_overlay, _, _ = generate_gradcam_explanations(classifier_model, processed_tensor, class_head, quality_head)

    preds = classifier_model(processed_tensor)
    preds = {k: v.numpy() for k, v in preds.items()}
    pred_text, _ = interpret_predictions(preds)

    # Add YOLO object counts and stats with size breakdown
    pred_text += (
        f"\n$\mathbf{{Total\ Detections:}}$ {object_counts} objects\n"
        f"- Shiny: {shiny_count}\n"
        f"- Dark Spots: {darkspot_count}\n"
        f"- Size Counts: {size_counter['Big']} Big, {size_counter['Medium']} Medium, {size_counter['Small']} Small\n"
        f"- Irregular Shape: {irregular_shape_count}"
    )

    # Classification & quality notes
    _, class_notes_tuple = interpret_predictions(preds)
    shiny_label, darkspot_label, size_label, shape_label = class_notes_tuple

    class_text = (
        f"$\\mathbf{{CLASSIFICATION}}$\n"
        f"- Shape: {shape_label}\n"
        f"- Shine: {shiny_label}\n"
        f"- Dark Spots: {darkspot_label}"
    )
    quality_text = (
        f"$\\mathbf{{QUALITY}}$\n"
        f"- Size: {size_label}\n"
        f"- Smoothness: {'Good' if size_label=='Big' else 'Average'}"
    )

    return original_img, class_overlay, quality_overlay, pred_text, class_text, quality_text

# ------------------ Main ------------------ #
if __name__ == "__main__":
    test_image_path = os.path.join(Config.TEST_IMAGES_DIR, "test3.jpg")  # Change to your test image path
    classifier_model = tf.keras.models.load_model(os.path.join(Config.MODEL_SAVE_PATH, "model_final.keras"))
    yolo_model = YOLO("runs/detect/train2/weights/best.pt")

    print("Models loaded.")
    print(f"Processing: {test_image_path}")

    results = predict_and_explain(test_image_path, classifier_model, yolo_model)
    plot_results(*results)
