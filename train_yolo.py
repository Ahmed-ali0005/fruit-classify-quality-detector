# Copyright 2025 Chaudhry Ahmed Ali Khan
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

# train_yolo.py

from ultralytics import YOLO

def train_yolo():
    """
    Train a YOLOv8 model on your fruit dataset with stable settings
    suitable for small GPUs and small datasets.
    """

    # Load pretrained nano model
    model = YOLO("yolov8n.pt")

    # Train with careful settings to avoid NaNs and improve accuracy
    model.train(
        data="D:/plants-app/fruit-classify-quality-detector/datasetYolo/data.yaml",  # your dataset
        epochs=100,               # longer training for small dataset
        imgsz=640,                # image size
        batch=8,                 # small batch for MX450 VRAM
        lr0=1e-4,                 # lower LR for stability
        optimizer="AdamW",        # better for small dataset
        patience=30,              # early stopping patience
        amp=False,                # disable mixed precision to avoid NaNs
        workers=0,                # Windows safe
        augment=True,             # enable mosaic/mixup/HSV augmentation
        close_mosaic=0            # do not disable mosaic
    )

    # Save final model (Ultralytics also saves best.pt automatically)
    model.save("D:/plants-app/fruit-classify-quality-detector/outputs/model_weights/yolov8_fruit_final.pt")
    print("Training complete and model saved!")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # required on Windows
    train_yolo()
