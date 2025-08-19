# Copyright 2025 Chaudhry Ahmed Ali Khan
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

import tensorflow as tf
import numpy as np
import cv2

def compute_gradcam(model, processed_tensor, head_name):
    model_input = model.input
    last_conv_layer = model.get_layer('Conv_1')
    last_conv_output = last_conv_layer.output
    head_output = model.get_layer(head_name).output

    grad_model = tf.keras.Model(inputs=model_input, outputs=[last_conv_output, head_output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(processed_tensor)
        tape.watch(conv_outputs)

        if len(predictions.shape) == 2:
            class_idx = tf.argmax(predictions[0])
            target = predictions[:, class_idx]
        else:
            target = predictions[:, 0]

    grads = tape.gradient(target, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)

    return heatmap.numpy()

def overlay_heatmap(heatmap, image, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, colormap)
    overlayed_img = heatmap_color * alpha + image
    overlayed_img = np.clip(overlayed_img, 0, 255).astype(np.uint8)
    return overlayed_img

def generate_gradcam_explanations(model, processed_tensor, class_head, quality_head):
    class_heatmap = compute_gradcam(model, processed_tensor, class_head)
    quality_heatmap = compute_gradcam(model, processed_tensor, quality_head)

    # Convert processed tensor (normalized) back to uint8 image for overlay
    img_for_overlay = (processed_tensor[0].numpy() * 255).astype(np.uint8)

    class_overlay = overlay_heatmap(class_heatmap, img_for_overlay)
    quality_overlay = overlay_heatmap(quality_heatmap, img_for_overlay)

    classification_text = f"Grad-CAM for head: {class_head}"
    quality_text = f"Grad-CAM for head: {quality_head}"

    return class_overlay, quality_overlay, classification_text, quality_text


