# Fruit Classification & Quality Grading with Visual Explanations

## Overview
This project detects fruits in images, classifies them as **organic or inorganic**, predicts their **quality**, **size**, **shininess**, **dark spots**, and **shape irregularity**, and provides **visual explanations using Grad-CAM**. It also integrates **YOLO** for object detection to handle multiple fruits in one image.

Key Features:
- Detects multiple fruits in batch images using YOLOv8.
- Classifies each fruit as Organic/Inorganic.
- Predicts quality metrics: quality grade, size, shine, dark spots, and shape irregularity.
- Generates Grad-CAM visualizations for both classification and quality assessment.
- Displays clear summary with object counts and defect statistics.
- Modular design to allow **training** or **inference**.

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Setup & Installation](#setup--installation)
3. [Requirements](#requirements)
4. [Usage](#usage)
   - [Run Inference](#run-inference)
   - [Train Model](#train-model)
   - [YOLO Object Detection](#yolo-object-detection)
5. [Output Format](#output-format)
6. [Dataset](#dataset)
7. [Model Architecture](#model-architecture)
8. [Grad-CAM Explanation](#grad-cam-explanation)
9. [License](#license)
10. [References](#references)

---

## Project Structure
FRUIT-CLASSIFY-QUALITY-DETECTOR/
├── scripts/
│ ├── train.py                  # Train classifier model
│ ├── dataset_loader.py         # Load dataset
│ ├── model_builder.py          # Build MobileNetV2-based classifier
│ ├── loss_metrics.py           # Loss functions and metrics
│ ├── gradcam_multi.py          # Grad-CAM explanation generation
│ ├── predict.py                # Run inference + Grad-CAM + YOLO
│ └── config.py                 # Paths, constants, and configuration
├── data/
│ ├── images/                   # Fruit images
│ └── labels.csv                # Labels and annotations
├──datasetYolo/
│ ├──train/
| │ ├──images/                  # YOLO training images
| │ └──labels/                  # YOLO training labels
│ ├──valid/
| │ ├──images/                  # YOLO validation images
| │ └──labels/                  # YOLO validation labels
│ └──data.yaml                  # YOLO dataset configuration
|──runs/
|  └──detect/
|    └──train3/
|      └──weights/              # Saved YOLO model weights
├── outputs/
│ ├── model_weights/            # Saved trained weights
│ ├── logs/                     # Training logs
│ └── predictions/              # Inference results
├──train_yolo.py                # train Yolo model
├──test_yolo.py                 # test working of Yolo model
├── requirements_minimal.txt    # Minimal requirements (run only)
├── requirements_full.txt       # Full requirements (train + run)
├── LICENSE                     # Licensing
└── README.md                   # Project documentation


---

## Setup & Installation

1. **Clone the repository**
```bash
git clone https://github.com/Ahmed-ali0005/fruit-classify-quality-detector.git
cd fruit-classify-quality-detector
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

3. **Install dependencies**
  - Minimal (for inference)
    ```bash
    pip install -r requirements_minimal.txt
    ```
  - Full (for training + YOLO + Grad-CAM):
    ```bash
    pip install -r requirements_full.txt
    ```
---

## Requirements

### Operating System
    -Windows 10 or higher, macOS 10.15+, or Linux (Ubuntu 20.04)

## Hardware
    -CPU: Intet i7 (minimum)
    -GPU: NVIDIA GPU with CUDA support (minimum)
    -RAM: 16 GB (minimum)
    -Storage: 8 GB free (minimum)
    
### Software and Libraries
  - Python >= 3.10
  - TensorFlow 2.10
  - PyTorch 2.7.1 + CUDA 11.8 (for YOLO)
  - Ultralytics YOLOv8
  - OpenCV, Pillow, Matplotlib
  - Pandas, NumPy, scikit-learn
  - Optional GUI: PyQt5 + labelImg
  - TensorBoard (for training monitoring)
    
---

## Usage

    ### Training
    ```bash
    python scripts/train.py #for training classifier + quality model
    python scripts/train_yolo.py #for training YOLO model
    ```
  - Uses MobileNetV2 as base.
  - Multi-head classifier for organic and quality metrics.
  - Configurable via scripts/config.py:
  - Dataset path
  - Model save path
  - Training parameters (batch size, epochs, learning rate)
  - Logs available in outputs/logs/
  - Trained weights saved in outputs/model_weights/

    ### Testing YOLO Model
    ```bash
    python test_yolo.py #for testing the yolo model
    ```
   - To check if yolo model has been trained correctly and producing correct results.
   - Detects all fruits in an image.
   - Bounding boxes drawn without displaying class names (optional).

   ### Run Inference
   ```bash
   python scripts/predict.py #for running inference on a given image
   ``` 
  - Detect fruits in the image.
  - Runs classifier on each fruit
  - Generates Grad-CAM overlays for classification and quality.
  - Prints summary with object counts and quality statistics.

---

## Output Format

Example Output:
| Original Image (w/ boxes)  | Classification Grad-CAM  | Quality Grad-CAM       |
| -------------------------- | ------------------------ | ---------------------- |
| **PREDICTION**             | **CLASSIFICATION**       | **QUALITY**            |
| Label: Organic (94.3%)     | Shape asymmetry detected | Smooth, glossy surface |
| Quality: Medium (0.68)     | Uneven color tones       | Some wrinkling visible |
| Objects: 6                 | Dull shine               | No dark spots detected |
| Big: 3, Medium: 2, Small:1 |                          |                        |
| Shiny: 4                   |                          |                        |

---

## Dataset

- The dataset for the classifier + quality detector is assumed to be in this format:
  - Images: fruits in various backgrounds.
  - Labels in labels.csv:
     - image_path (1.jpg etc.)
     - fruit_type (apple, banana, etc.)
     - is_organic (0: inorganic, 1: organic)
     - quality_grade (0: bad, 1: medium, 2: good)
     - size (0: small, 1: medium, 2: big)
     - shininess (0: dull, 1: shiny)
     - shape_irregularity (0: normal, 1: some, 2: lots)
     - notes (batch_single, batch_double)

- The dataset for the yolo model is assumed to be in this format:
    - Images: fruits in various backgrounds.
    - Labels: class x_center y_center width height
    - Class: 0: Apple, 1: Banana, 2: Grapes, 3: Guava, 4: Mango, 5: Orange, 6: Papaya, 7: Pomegrenate, 8: Strawberry, 9: Watermelon

---

## Model Architecture

  - This project consists of two main models working together:
    1. YOLOv8 -Fruit Detection
    2. MobileNetV2-based Classifier -Organic/Quality Grading

1. YOLOv8 -Fruit Detection

    Purpose in Project: Detect multiple fruits in an image before passing each fruit crop to the classifier for organic/quality prediction.

    Architecture Overview:
        Backbone: CSPDarknet-inspired feature extractor
            Extracts hierarchical features from input image.
            Captures both low-level (edges, textures) and high-level (fruit shapes) features.
        Neck: PANet-like Path Aggregation Network
            Aggregates features from different scales.
            Improves detection of small, medium, and large fruits.
        Head: YOLO Detection Head
            Predicts bounding boxes, objectness score, and class probabilities.
            Uses anchor-free or anchor-based detection (v8 is mostly anchor-free).

    Input & Output in Your Project:
        Input: Full image of arbitrary size (resized internally by YOLO).
        Output: Bounding boxes with coordinates (x1, y1, x2, y2) for each detected fruit.
        Integration:
            YOLO detects objects and outputs bounding boxes.
            Each box is cropped from the original image.
            Cropped fruit is resized (224, 224) and passed to the MobileNetV2 classifier.
            Classifier outputs organic/quality predictions per fruit.

    Diagram (Simplified):
        Full Image
        │
        ▼
        YOLOv8 Backbone (CSPDarknet)
        │
        ▼
        PANet Neck (Feature Aggregation)
        │
        ▼
        YOLO Head (Bounding Boxes + Objectness + Class)
        │
        ▼
        Crops → MobileNetV2 Classifier → Organic/Quality Predictions

2. MobileNetV2-based Classifier -Organic/Quality Grading
    Purpose: Classify each detected fruit as organic/inorganic and predict quality attributes.

    Architecture Overview:
    Base: MobileNetV2 pretrained on ImageNet.
        Lightweight CNN with depthwise separable convolutions.
        Input size: (224, 224, 3)
    Shared Feature Extraction:
        Extracts rich feature maps for all heads.
    Multiple Output Heads (Multi-task Learning):
    | Head                 | Output Shape | Purpose                                                         |
    | -------------------- | ------------ | --------------------------------------------------------------- |
    | `is_organic`         | (1,)         | Predicts organic (1) vs inorganic (0)                           |
    | `quality_grade`      | (3,)         | Predicts quality: Bad, Medium, Good                             |
    | `size`               | (3,)         | Predicts size: Small, Medium, Big                               |
    | `shininess`          | (1,)         | Predicts shiny (1) vs dull (0)                                  |
    | `darkspots`          | (1,)         | Predicts presence of dark spots (Yes/No)                        |
    | `shape_irregularity` | (3,)         | Predicts shape: Normal, Some irregularity, Lots of irregularity |

    Integration in Project Pipeline:
        Receives YOLO-cropped fruit images.
        Predicts organic/quality attributes for each fruit.
        Grad-CAM visualizations are generated for both classification and quality outputs for interpretability.

    Diagram:
        YOLO Crop (224x224)
                │
                ▼
            MobileNetV2 Base (Feature Extraction)
                │
                ├── Head: is_organic → Organic / Inorganic
                ├── Head: quality_grade → Bad / Medium / Good
                ├── Head: size → Small / Medium / Big
                ├── Head: shininess → Shiny / Dull
                ├── Head: darkspots → Yes / No
                └── Head: shape_irregularity → Normal / Some / Lots
                │
                ▼
            Grad-CAM Visualizations

---

## Grad-CAM Explanation:

3. Grad-CAM – Visual Explanations

Purpose: Provide visual interpretability for the MobileNetV2 classifier outputs, helping understand why the model made a prediction.

Overview:
    Grad-CAM (Gradient-weighted Class Activation Mapping) highlights important regions in the image that influenced the model’s decision.
    In this project, Grad-CAM is applied to two heads of the multi-task classifier:
        Classification Head (is_organic) – Highlights features that determine whether a fruit is organic or inorganic.
        Quality Head (quality_grade) – Highlights features that affect quality predictions such as size, shine, dark spots, and shape irregularity.

Integration in Project Pipeline:
    Take the YOLO-cropped fruit image (224x224) as input.
    Forward pass through the MobileNetV2 multi-head model.
    Compute gradients of the target output (classification or quality) with respect to the final convolutional layer.
    Weight the feature maps by these gradients and combine them to produce a heatmap overlay.
    Overlay the heatmap on the original image to visualize which regions contributed most to the prediction.

Diagram:
    YOLO Crop (224x224)
        │
        ▼
    MobileNetV2 Base (Feature Extraction)
        │
        ▼
    Gradients w.r.t Target Head
        │
        ▼
    Weighted Feature Maps → Heatmap
        │
        ▼
    Overlay on Original Crop → Grad-CAM Visualization


Output in Project:
    Classification Grad-CAM: Shows regions influencing organic vs inorganic prediction.
    Quality Grad-CAM: Shows regions influencing size, shine, dark spots, and shape irregularity prediction.
    Can be visualized side-by-side with the original image and YOLO boxes for full interpretability.

---

## License

This project is licensed under the Apache License 2.0.

You are free to:
    Use the code for personal, academic, or commercial purposes
    Modify, adapt, or improve the code
    Distribute the original or modified code

Conditions:
    You must include a copy of the Apache 2.0 license with any redistribution
    Retain copyright and attribution notices
    Clearly indicate if you modified any files
    The software is provided “as-is,” without any warranty

For the full license, see the LICENSE file.

---

## References

1. **YOLOv8** – Ultralytics, https://github.com/ultralytics/ultralytics  
2. **TensorFlow & Keras** – TensorFlow 2.10, https://www.tensorflow.org  
3. **Grad-CAM** – Selvaraju et al., 2017, "Grad-CAM: Visual Explanations from Deep Networks"  
4. **MobileNetV2** – Sandler et al., 2018, "MobileNetV2: Inverted Residuals and Linear Bottlenecks"  
5. **PIL / OpenCV / Matplotlib** – Image processing and visualization  
6. **Python Libraries** – NumPy, Pandas, Scikit-learn, etc.
