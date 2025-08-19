# Copyright 2025 Chaudhry Ahmed Ali Khan
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import tensorflow as tf
from tensorflow.keras import layers, Input
from tensorflow.keras.applications import MobileNetV2

def build_fruit_classifier(input_shape=(224,224,3), base_trainable=False):
    inputs = Input(shape=input_shape)

    # Pass your 'inputs' explicitly as input_tensor to MobileNetV2
    base_model = MobileNetV2(include_top=False,
                             input_tensor=inputs,
                             weights='imagenet',
                             pooling=None)  # keep pooling None to access last conv

    base_model.trainable = base_trainable

    # Access last conv output (connected to inputs)
    last_conv_output = base_model.get_layer('Conv_1').output  # This is connected to inputs

    # Add global avg pooling and heads
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(last_conv_output)
    x = layers.Dropout(0.3)(x)

    out_is_organic = layers.Dense(1, activation='sigmoid', name='is_organic')(x)
    out_quality_grade = layers.Dense(3, activation='softmax', name='quality_grade')(x)
    out_size = layers.Dense(3, activation='softmax', name='size')(x)
    out_shininess = layers.Dense(1, activation='sigmoid', name='shininess')(x)
    out_darkspots = layers.Dense(1, activation='sigmoid', name='darkspots')(x)
    out_shape_irregularity = layers.Dense(3, activation='softmax', name='shape_irregularity')(x)

    # Build model with all outputs including last conv layer for Grad-CAM
    model = tf.keras.Model(
        inputs=inputs,
        outputs={
            'last_conv_output': last_conv_output,
            'is_organic': out_is_organic,
            'quality_grade': out_quality_grade,
            'size': out_size,
            'shininess': out_shininess,
            'darkspots': out_darkspots,
            'shape_irregularity': out_shape_irregularity
        }
    )

    def set_base_trainable(trainable):
        base_model.trainable = trainable
        print(f"Base model trainable set to {trainable}")

    model.set_base_trainable = set_base_trainable

    return model
