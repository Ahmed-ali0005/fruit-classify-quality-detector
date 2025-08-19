# Copyright 2025 Chaudhry Ahmed Ali Khan
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

# scripts/train.py
import os
import sys
import datetime

# allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow as tf
from config import Config
from dataset_loader import load_dataset
from model_builder import build_fruit_classifier
from loss_metrics import get_losses, get_metrics
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

# -------------------------
# GPU memory growth (avoid grabbing all VRAM)
# -------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU(s) available: {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"❌ GPU error: {e}")
else:
    print("⚠️ No GPU found. Training on CPU.")


# -------------------------
# Overall accuracy callback
# -------------------------
class OverallAccuracyCallback(tf.keras.callbacks.Callback):
    """
    Computes an aggregated accuracy across all heads after each epoch.
    For each batch it computes (total_correct_labels / total_labels)
    across all heads, and averages across batches.
    """

    def __init__(self, train_data, val_data):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data

    def _preds_to_dict(self, preds):
        if isinstance(preds, dict):
            return preds
        if isinstance(preds, (list, tuple)):
            return dict(zip(self.model.output_names, preds))
        return {self.model.output_names[0]: preds}

    def _batch_overall_fraction(self, batch_x, batch_y):
        preds = self.model.predict(batch_x, verbose=0)
        preds = self._preds_to_dict(preds)

        correct = 0
        total = 0

        for head_name in self.model.output_names:
            y_pred = preds.get(head_name)
            y_true = batch_y.get(head_name)

            if y_pred is None or y_true is None:
                continue

            if tf.is_tensor(y_true):
                y_true = y_true.numpy()
            if tf.is_tensor(y_pred):
                y_pred = y_pred.numpy()

            if y_pred.ndim > 1 and y_pred.shape[1] > 1:
                y_pred_classes = np.argmax(y_pred, axis=1)
                if y_true.ndim > 1 and y_true.shape[1] > 1:
                    y_true_classes = np.argmax(y_true, axis=1)
                else:
                    y_true_classes = np.ravel(y_true).astype(int)
            else:
                y_pred_classes = (np.ravel(y_pred) > 0.5).astype(int)
                y_true_classes = np.ravel(y_true).astype(int)

            if len(y_pred_classes) != len(y_true_classes):
                m = min(len(y_pred_classes), len(y_true_classes))
                y_pred_classes = y_pred_classes[:m]
                y_true_classes = y_true_classes[:m]

            correct += np.sum(y_pred_classes == y_true_classes)
            total += len(y_true_classes)

        return correct, total

    def _dataset_overall_accuracy(self, dataset):
        fractions = []
        for batch_x, batch_y in dataset:
            try:
                correct, total = self._batch_overall_fraction(batch_x, batch_y)
                if total > 0:
                    fractions.append(correct / total)
            except Exception as e:
                print(f"⚠️ Skipping batch in overall-acc calc due to error: {e}")
                continue
        return float(np.mean(fractions)) if fractions else 0.0

    def on_epoch_end(self, epoch, logs=None):
        overall_train_acc = self._dataset_overall_accuracy(self.train_data)
        overall_val_acc = self._dataset_overall_accuracy(self.val_data)
        print(f"\n📊 Epoch {epoch+1} — Overall Train Accuracy: {overall_train_acc:.4f} | Overall Val Accuracy: {overall_val_acc:.4f}")


# -------------------------
# Main
# -------------------------
def main():
    train_ds, val_ds, test_ds = load_dataset(Config)

    # Build model with frozen base
    model = build_fruit_classifier(Config.input_shape, base_trainable=False)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=Config.learning_rate),
        loss=get_losses(),
        metrics=get_metrics()
    )

    epochs = Config.epochs
    freeze_epochs = epochs // 2
    finetune_epochs = epochs - freeze_epochs

    # Callbacks (reuse yours, example below)
    callbacks = [
        TensorBoard(log_dir=os.path.join(Config.LOG_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))),
        ModelCheckpoint(
            filepath=os.path.join(Config.MODEL_SAVE_PATH, 'model_best.keras'),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=Config.patience,
            restore_best_weights=True,
            verbose=1
        ),
        OverallAccuracyCallback(train_ds, val_ds),
    ]

    print(f"🚀 Training with frozen base for {freeze_epochs} epochs...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=freeze_epochs,
        callbacks=callbacks,
        verbose=1
    )

    print("🚀 Unfreezing base model for fine-tuning...")
    model.set_base_trainable(True)

    # Recompile with smaller LR
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=Config.learning_rate / 10),
        loss=get_losses(),
        metrics=get_metrics()
    )

    print(f"🚀 Fine-tuning for {finetune_epochs} epochs...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=finetune_epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    final_path = os.path.join(Config.MODEL_SAVE_PATH, 'model_final.keras')
    model.save(final_path)
    print(f"✅ Model saved to {final_path}")

if __name__ == "__main__":
    main()
