# Copyright 2025 Chaudhry Ahmed Ali Khan
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

# loss_metrics.py

import tensorflow as tf

def get_losses():
    """
    Returns a dictionary mapping each output head to its corresponding loss function.
    """
    return {
        'is_organic': tf.keras.losses.BinaryCrossentropy(),
        'quality_grade': tf.keras.losses.CategoricalCrossentropy(),
        'size': tf.keras.losses.CategoricalCrossentropy(),
        'shininess': tf.keras.losses.BinaryCrossentropy(),
        'darkspots': tf.keras.losses.BinaryCrossentropy(),
        'shape_irregularity': tf.keras.losses.CategoricalCrossentropy()
    }

def get_metrics():
    """
    Returns a dictionary mapping each output head to a list of metrics.
    Includes accuracy and AUC for each head.
    """
    return {
        'is_organic': [
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc')
        ],
        'quality_grade': [
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc', multi_label=True)
        ],
        'size': [
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc', multi_label=True)
        ],
        'shininess': [
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc')
        ],
        'darkspots': [
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc')
        ],
        'shape_irregularity': [
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc', multi_label=True)
        ]
    }
