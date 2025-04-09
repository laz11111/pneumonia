# pneumonia
Pneumonia Detection and Localization using YOLO
This project focuses on the detection and localization of pneumonia in chest X-ray images using the YOLO (You Only Look Once) object detection framework. The model is trained and evaluated on the RSNA Pneumonia Detection Challenge dataset, which provides annotated chest radiographs with bounding box information for pneumonia cases.

🔬 Objective
The goal of this project is to:

Accurately detect pneumonia cases in chest X-ray images.

Localize the regions affected by pneumonia using bounding boxes.

Provide a fast and robust solution that can assist radiologists in clinical diagnosis.

🧠 Model Architecture
We used the YOLOv5 (or other version if applicable — please specify) object detection architecture for this task due to its high performance and real-time inference capabilities.

Input: Chest X-ray images (grayscale or converted to 3-channel RGB)

Output: Bounding boxes with confidence scores around regions indicating pneumonia.

📊 Dataset
Dataset Used: RSNA Pneumonia Detection Challenge Dataset

Description: The dataset contains chest radiographs with bounding box annotations for pneumonia-affected areas, along with corresponding labels.

⚙️ Implementation Highlights
Data preprocessing: conversion of DICOM to PNG/JPEG, normalization, and augmentation.

YOLO annotation conversion: bounding box data from RSNA CSVs transformed into YOLO format.

Model training: custom training pipeline using PyTorch-based YOLO framework.

Evaluation metrics: mAP, IoU, precision, recall.

📦 Output
Trained model weights.

Annotated predictions with bounding boxes for pneumonia localization.

Performance metrics and visualization results (bounding boxes overlaid on X-rays).
