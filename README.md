# Real-Time Traffic Sign Recognition 🚦

## 📌 Overview
This project implements a **real-time traffic sign recognition system** using computer vision and deep learning techniques.  
The application detects traffic signs from a live webcam feed and classifies them using a **Convolutional Neural Network (CNN)** trained on the **GTSRB dataset**.

The system is designed to run on a standard laptop without requiring specialized hardware, making it suitable for low-cost real-time applications.

---

## 🎯 Project Goals
- Detect traffic signs in real time from a webcam stream
- Classify detected signs into predefined categories
- Reduce false positives using confidence thresholds
- Stabilize predictions using temporal smoothing
- Achieve real-time performance (FPS monitoring)

---

## 🧠 Technologies Used
- **Python 3.11**
- **OpenCV** – video capture, image processing, sign detection
- **TensorFlow / Keras** – CNN training and inference
- **NumPy** – numerical operations

---

## 🗂 Dataset
The model is trained on the **German Traffic Sign Recognition Benchmark (GTSRB)**:
- Over **50,000 images**
- **43 traffic sign classes**
- Variations in illumination, scale, and orientation

---

## ⚙️ System Architecture
The real-time recognition pipeline consists of the following steps:

1. Capture frame from webcam  
2. Detect traffic sign candidates using OpenCV  
3. Extract region of interest (ROI)  
4. Apply contrast normalization  
5. Resize and normalize input image (32×32)  
6. Predict class probabilities using CNN  
7. Apply confidence threshold  
8. Apply temporal smoothing across frames  
9. Display label, confidence score, FPS, and bounding box  

---

## ✨ Original Contributions
Compared to a basic traffic sign classifier, this project introduces:
- **UNKNOWN class** when prediction confidence is below a threshold
- **Temporal averaging** of predictions to reduce flickering
- **Webcam-adapted preprocessing** for varying lighting conditions
- **Modular design**, separating detection and classification stages

---

## 📊 Experimental Results
- Training accuracy: **> 98%**
- Validation accuracy: **≈ 99%**
- Stable real-time recognition on clear, well-lit signs
- Reduced false positives using confidence thresholding
- Smooth predictions due to temporal filtering
- Real-time FPS displayed during execution

---

## 🖥 Hardware Requirements
- Laptop or PC
- Integrated webcam
- No external sensors or GPU required

---

## ▶️ How to Run
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install opencv-python tensorflow numpy
