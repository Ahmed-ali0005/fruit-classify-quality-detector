# Copyright 2025 Chaudhry Ahmed Ali Khan
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import sys
import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from config import Config

# Allow importing from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

AUTOTUNE = tf.data.AUTOTUNE


def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, Config.IMAGE_SIZE)
    image = image / 255.0
    return image

def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
    return image

def preprocess_and_augment(image_path, augment=True):
    image = preprocess_image(image_path)
    if augment:
        image = augment_image(image)
    return image

def make_full_path(path, config):
    """Safely build absolute paths from CSV values."""
    path = str(path).replace("\\", "/").strip()

    # Already absolute path → return as-is
    if os.path.isabs(path):
        return os.path.abspath(path)

    # Normalize IMAGE_DIR
    image_dir_norm = config.IMAGE_DIR.replace("\\", "/").strip()

    # Avoid double-joining if path already contains IMAGE_DIR
    if path.startswith(image_dir_norm) or path.startswith(os.path.basename(image_dir_norm)):
        return os.path.abspath(path)

    return os.path.abspath(os.path.join(config.IMAGE_DIR, path))


def load_dataset(config, test_split=0.2, val_split=0.1, shuffle=True, seed=42):
    df = pd.read_csv(config.CSV_PATH)

    # Drop rows with missing required columns
    df = df.dropna(subset=config.HEADS + ['image_path'])

    # Fix all image paths
    df["image_path"] = df["image_path"].apply(lambda p: make_full_path(p, config))

    # Show debug info
    print(f"🔍 First 5 image paths from CSV after processing:\n{df['image_path'].head().tolist()}")
    print(f"📂 IMAGE_DIR from config: {config.IMAGE_DIR}")

    # Filter missing files
    missing_files = df[~df["image_path"].apply(os.path.exists)]
    if not missing_files.empty:
        print(f"⚠️ Skipping {len(missing_files)} missing images...")
        print("Example missing files:", missing_files["image_path"].head().tolist())
        df = df[df["image_path"].apply(os.path.exists)]

    if df.empty:
        raise ValueError("❌ No valid images found. Check your CSV paths and IMAGE_DIR.")

    # Split into train/val/test
    train_val_df, test_df = train_test_split(df, test_size=test_split, random_state=seed)
    train_df, val_df = train_test_split(train_val_df, test_size=val_split, random_state=seed)

    return (
        df_to_dataset(train_df, shuffle, config.BATCH_SIZE),
        df_to_dataset(val_df, shuffle=False, batch_size=config.BATCH_SIZE),
        df_to_dataset(test_df, shuffle=False, batch_size=config.BATCH_SIZE)
    )


def df_to_dataset(df, shuffle=True, batch_size=8):
    def encode_label(row):
        return {
            'is_organic': tf.cast(row[0], tf.float32),
            'quality_grade': tf.one_hot(tf.cast(row[1], tf.int32), 3),
            'size': tf.one_hot(tf.cast(row[2], tf.int32), 3),
            'shininess': tf.cast(row[3], tf.float32),
            'darkspots': tf.cast(row[4], tf.float32),
            'shape_irregularity': tf.one_hot(tf.cast(row[5], tf.int32), 3),
        }

    image_paths = df['image_path'].values.astype(str)
    label_data = df[['is_organic', 'quality_grade', 'size', 'shininess', 'darkspots', 'shape_irregularity']].values

    image_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(label_data)

    # Zip images and labels together
    ds = tf.data.Dataset.zip((image_ds, label_ds))

    # Shuffle before batching (optional)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))

    # Map preprocessing and encoding
    def process_path_and_label(image_path, label_row):
        image = preprocess_image(image_path)
        if shuffle:  # augment only during training
            image = augment_image(image)
        label = encode_label(label_row)
        return image, label

    ds = ds.map(process_path_and_label, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)

    return ds
