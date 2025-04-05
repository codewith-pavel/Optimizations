# 🧠 Multi-Stage Knowledge Distillation with Layer Fusion for Skin Cancer Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

**Repository**: [GitHub Link](https://github.com/codewith-pavel/Optimizations)  
**Authors**: Mahir Afser Pavel, Ramisa Asad, Md Ikramuzzaman, Murad Mustakim, Riasat Khan

---

## 📌 Abstract

This repository presents a **multi-stage knowledge distillation (MSKD)** framework augmented with **layer fusion** and **explainable AI (XAI)** to classify skin cancer using the HAM10000 dataset. A hybrid teacher model (ViT + ConvNeXT) distills knowledge into a lightweight student (CNN + EfficientNet), achieving:

- **Accuracy**: 95.88%  
- **F1-Score**: 95.91%  
- **AUC**: 99.02%

Post-training quantization compresses the model by **91.81%**, making it highly suitable for edge deployment. To ensure interpretability, XAI techniques (Grad-CAM, Score-CAM, LIME) are integrated and validated by dermatology experts.

---

## 🚀 Key Features

- **Hybrid Architectures**  
  - **Teacher**: ViT + ConvNeXT for superior hierarchical feature learning  
  - **Student**: CNN + EfficientNet optimized for speed and compactness  

- **Multi-Stage Knowledge Distillation**  
  - Intermediate exits with dedicated losses  
  - Layer fusion for multi-scale feature aggregation  

- **Model Optimization**  
  - Post-training quantization (FP32 → INT8)  
  - Training speed: **61 ms/step**  
  - Inference speed: **15 ms/step**

- **Explainability**  
  - Visual explanations with Grad-CAM, Score-CAM, LIME  
  - Validated heatmaps for critical diagnostic regions

---

## 📊 Dataset: HAM10000

- **Total Images**: 10,015 dermoscopic images  
- **Classes**: 7 skin lesion types  
- **Input Shape**: 64×64 RGB  
- **Preprocessing**: Normalization, oversampling, and augmentation (brightness shifts, horizontal/vertical flipping, rotations)

### 🔢 Class Distribution

| Class                   | Training | Test |
|-------------------------|----------|------|
| Melanocytic nevi        | 5,364    | 1,341 |
| Melanoma                | 890      | 223   |
| Benign keratosis        | 879      | 220   |
| Basal cell carcinoma    | 411      | 103   |
| Actinic keratoses       | 262      | 65    |
| Vascular lesions        | 114      | 28    |
| Dermatofibroma          | 92       | 23    |

---

## 🛠️ Methodology

### 🔁 System Pipeline

1. **Data Acquisition & Preprocessing**
2. **Teacher Model Training (ViT + ConvNeXT)**
3. **Student Model Training (CNN + EfficientNet)**
4. **Multi-Stage Knowledge Distillation**
   - Intermediate Exit Loss (`α = 0.3`)
   - Logits Distillation Loss (`β = 0.3`)
   - Classification Loss (`γ = 0.4`)
5. **Layer Fusion for Feature Aggregation**
6. **Post-Training Quantization**
7. **Explainability via Grad-CAM, Score-CAM, LIME**
8. **Expert Validation of XAI Outputs**

---

## 📈 Results

### 🔬 Classification Performance

| Model                          | Accuracy | F1-Score | AUC     |
|--------------------------------|----------|----------|---------|
| Baseline CNN                   | 73.82%   | 74.20%   | 95.92%  |
| Single-Stage Distillation      | 90.06%   | 90.02%   | 98.59%  |
| Multi-Stage Distillation       | 92.22%   | 92.29%   | 98.81%  |
| **Multi-Stage + Layer Fusion** | **95.88%** | **95.91%** | **99.02%** |

### ⚡ Efficiency Metrics

| Model                   | Size Reduction | Training Time | Inference Time |
|-------------------------|----------------|---------------|----------------|
| Baseline Student        | —              | 240 ms/step   | 40 ms/step     |
| Quantized Distilled     | 91.81%         | 61 ms/step    | 15 ms/step     |

---

## 🧪 Explainable AI (XAI)

XAI methods integrated into this project include:

- **Grad-CAM**: Highlights spatial class activations in CNN layers  
- **Score-CAM**: Model-agnostic saliency with activation scores  
- **LIME**: Local surrogate models for interpretable pixel regions  

Domain experts validated that the highlighted regions correspond to medically relevant diagnostic zones.

---
