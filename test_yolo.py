# Copyright 2025 Chaudhry Ahmed Ali Khan
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import os
import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from scripts.config import Config
from scripts.gradcam_multi import generate_gradcam_explanations

def load_classifier():
    model_path = os.path.join(Config.MODEL_SAVE_PATH, "model_final.keras")
    model = tf.keras.models.load_model(model_path)
    print("Classifier loaded.")
    return model

def preprocess_crop(crop):
    # Resize to classifier input
    crop_resized = cv2.resize(crop, (224, 224))
    crop_resized = crop_resized.astype(np.float32) / 255.0
    processed = np.expand_dims(crop_resized, axis=0)
    return processed

def interpret_predictions(preds):
    # Same as your previous interpret_predictions function
    is_organic_prob = preds['is_organic'][0][0]
    label = "Organic" if is_organic_prob > 0.5 else "Inorganic"
    confidence = is_organic_prob if is_organic_prob > 0.5 else 1 - is_organic_prob

    qg_probs = preds['quality_grade'][0]
    qg_idx = np.argmax(qg_probs)
    quality_map = {0: "Bad", 1: "Medium", 2: "Good"}
    quality_label = quality_map.get(qg_idx, "Unknown")

    size_probs = preds['size'][0]
    size_idx = np.argmax(size_probs)
    size_map = {0: "Small", 1: "Medium", 2: "Big"}
    size_label = size_map.get(size_idx, "Unknown")

    shiny_prob = preds['shininess'][0][0]
    shiny_label = "Shiny" if shiny_prob > 0.5 else "Dull"

    darkspot_prob = preds['darkspots'][0][0]
    darkspot_label = "Yes" if darkspot_prob > 0.5 else "No"

    shape_probs = preds['shape_irregularity'][0]
    shape_idx = np.argmax(shape_probs)
    shape_map = {0: "Normal", 1: "Some irregularity", 2: "Lots of irregularity"}
    shape_label = shape_map.get(shape_idx, "Unknown")

    return {
        "label": label,
        "quality": quality_label,
        "size": size_label,
        "shine": shiny_label,
        "darkspots": darkspot_label,
        "shape": shape_label
    }

def main():
    # Load models
    yolo_model = YOLO("runs/detect/train2/weights/best.pt")
    classifier_model = load_classifier()

    test_images_dir = Config.TEST_IMAGES_DIR
    images = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.png'))]

    for idx, img_name in enumerate(images):
        img_path = os.path.join(test_images_dir, img_name)
        print(f"\nProcessing: {img_path}")

        frame = cv2.imread(img_path)
        results = yolo_model(frame)

        obj_counts = {}
        shiny_count = 0
        darkspot_count = 0
        irregular_size_count = 0
        irregular_shape_count = 0

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # bounding boxes
            classes = result.boxes.cls.cpu().numpy()  # class indices
            confidences = result.boxes.conf.cpu().numpy()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                cls_idx = int(classes[i])
                cls_name = Config.FRUIT_CLASSES[cls_idx]

                crop = frame[y1:y2, x1:x2]
                processed = preprocess_crop(crop)
                preds = classifier_model(processed)
                preds = {k: v.numpy() for k, v in preds.items()}
                stats = interpret_predictions(preds)

                # Update counts
                obj_counts[cls_name] = obj_counts.get(cls_name, 0) + 1
                if stats["shine"] == "Shiny":
                    shiny_count += 1
                if stats["darkspots"] == "Yes":
                    darkspot_count += 1
                if stats["size"] != "Medium":
                    irregular_size_count += 1
                if stats["shape"] != "Normal":
                    irregular_shape_count += 1

                # Draw bbox and label on image
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{cls_name}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display runtime stats
        print(f"Object counts: {obj_counts}")
        print(f"Shiny objects: {shiny_count}")
        print(f"Darkspots: {darkspot_count}")
        print(f"Irregular sizes: {irregular_size_count}")
        print(f"Irregular shapes: {irregular_shape_count}")

        # Show image
        cv2.imshow("Detections + Stats", frame)
        key = cv2.waitKey(0)  # press any key to continue
        if key == 27:  # ESC to quit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
