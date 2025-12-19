# Real-time Plant Disease Dataset Development and Detection

A high-performance deep learning system for identifying diseases in Maize, Rice, and Wheat leaves. This project implements a full pipeline from web-scale dataset generation to real-time webcam inference using state-of-the-art Convolutional Neural Networks (EfficientNetV2).

## 🚀 Features
- **Automated Dataset Generation:** Scrapes, cleans, and augments plant disease images.
- **High-Capacity Models:** Built on EfficientNetV2 for superior accuracy-to-latency ratio.
- **Two-Phase Training:** Implements warmup and deep fine-tuning for maximum convergence.
- **Robust Evaluation:** Uses Test Time Augmentation (TTA) to ensure reliable metrics.
- **Real-time Detection:** OpenCV-based inference with ROI tracking for live webcam use.

## 📊 Performance Metrics (Verified)
The current model (`EfficientNetV2B0`) achieved the following results using **5-Round TTA**:

| Metric | Result |
| :--- | :--- |
| **Total Test Images** | 3,777 |
| **TTA Accuracy** | **81.20%** |
| **Ensemble Loss** | 0.5147 |

### Detailed Class Performance (Macro Avg)
- **Precision:** 81.65%
- **Recall:** 81.22%
- **F1-Score:** 81.26%

> [!TIP]
> The model shows exceptionally high performance (>90%) on identifying healthy plant tissue, making it a reliable tool for early anomaly detection.

## 🛠️ Installation
```bash
# Clone the repository and install dependencies
pip install tensorflow opencv-python scikit-learn matplotlib pandas numpy bing-image-downloader
```

## 📖 Usage
### 1. Data Preparation
```bash
python main.py
```
### 2. Model Training
```bash
python train.py
```
### 3. Model Evaluation
```bash
python evaluate.py
```
### 4. Real-time Inference (Webcam)
```bash
python detect.py
```

## 📁 Project Structure
- `main.py`: Dataset generation and preprocessing.
- `train.py`: Model architecture and training loops.
- `evaluate.py`: TTA evaluation and visualization generation.
- `detect.py`: Real-time inference script.
- `plant_dataset/`: Generated image repository.

## 📈 Visualizations
Found in the project root after running `evaluate.py`:
- `confusion_matrix.png`: detailed class-to-class leakage analysis.
- `classification_report_heatmap.png`: Precision/Recall/F1 per disease.

---
*Developed for research in Real-time Plant Disease Dataset Development and Detection.*
