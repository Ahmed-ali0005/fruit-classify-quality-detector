# Copyright 2025 Chaudhry Ahmed Ali Khan
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import os

class Config:
    # Root directory of your project
    ROOT_DIR = "D:/plants-app/fruit-classify-quality-detector"

    # Paths
    IMAGE_DIR = os.path.join(ROOT_DIR, "data", "images")  # Where images are stored
    CSV_PATH = os.path.join(ROOT_DIR, "data", "labels.csv")  # Your label file
    MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "outputs", "model_weights")
    GRADCAM_SAVE_PATH = os.path.join(ROOT_DIR, "outputs", "predictions", "gradcam")
    TEST_IMAGES_DIR = os.path.join(ROOT_DIR, "data", "test_images")  # Directory for test images
    LOG_DIR = os.path.join(ROOT_DIR, "outputs", "logs")
    YOLO_MODEL_PATH = os.path.join(ROOT_DIR, "runs", "detect", "train3", "weights") 

    # Input image size for the model
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 8
    input_shape = IMAGE_SIZE + (3,)  # Input shape for the model
    learning_rate = 1e-3
    epochs = 20
    patience = 5

    # Fruit classes (optional if you want to classify fruit type too)
    FRUIT_CLASSES = [
        "mango", "banana", "apple", "orange", "guava",
        "pomegranate", "grapes", "watermelon", "papaya", "strawberries"
    ]

    # Label Mappings
    IS_ORGANIC = {0: "Inorganic", 1: "Organic"}
    QUALITY_GRADE = {0: "Bad", 1: "Mid", 2: "Good"}
    SIZE = {0: "Small", 1: "Mid", 2: "Large"}
    SHININESS = {0: "Dull", 1: "Shiny"}
    DARKSPOTS = {0: "None", 1: "Yes"}
    SHAPE_IRREGULARITY = {0: "None", 1: "Some", 2: "Lots"}
    NOTES = ["batch_single", "batch_double"]

    # Multi-output heads
    HEADS = [
        "is_organic",
        "quality_grade",
        "size",
        "shininess",
        "darkspots",
        "shape_irregularity"
    ]

    # Mapping of head to number of classes
    HEAD_OUTPUT_DIM = {
        "is_organic": 2,
        "quality_grade": 3,
        "size": 3,
        "shininess": 2,
        "darkspots": 2,
        "shape_irregularity": 3
    }

    # For reverse mapping predictions → readable labels
    LABEL_MAPS = {
        "is_organic": IS_ORGANIC,
        "quality_grade": QUALITY_GRADE,
        "size": SIZE,
        "shininess": SHININESS,
        "darkspots": DARKSPOTS,
        "shape_irregularity": SHAPE_IRREGULARITY
    }
