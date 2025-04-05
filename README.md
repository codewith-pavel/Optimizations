# Multi-Stage Knowledge Distillation with Layer Fusion for Skin Cancer Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

**Repository**: [GitHub Link](https://github.com/codewith-pavel/Optimizations)  
**Authors**: Mahir Afser Pavel, Ramisa Asad, Md Ikramuzzaman, Murad Mustakim, Riasat Khan  

---

## 📌 Abstract
This repository implements a **multi-stage knowledge distillation framework** integrated with **layer fusion** and **explainable AI (XAI)** for robust skin cancer classification using the HAM10000 dataset. The hybrid teacher model (ViT + ConvNeXT) transfers knowledge to a lightweight student model (CNN + EfficientNet), achieving state-of-the-art performance:
- **Accuracy**: 95.88%
- **F1-Score**: 95.91%
- **AUC**: 99.02%

Post-training quantization reduces model size by **91.81%**, enabling deployment on resource-constrained devices. XAI methods (Grad-CAM, Score-CAM, LIME) provide interpretable visualizations validated by domain experts.

---

## 🚀 Key Features
- **Hybrid Architectures**:  
  - **Teacher Model**: ViT + ConvNeXT for high-performance feature extraction.  
  - **Student Model**: CNN + EfficientNet for efficiency-accuracy balance.  
- **Multi-Stage Knowledge Distillation**:  
  - Intermediate exits for hierarchical feature learning.  
  - Layer fusion for enhanced feature aggregation.  
- **Optimization**:  
  - Post-training quantization (FP32 → INT8).  
  - Training time: **61 ms/step**, inference: **15 ms/step**.  
- **Explainability**:  
  - Score-CAM, Grad-CAM, and LIME visualizations.  
  - Domain expert validation of highlighted diagnostic regions.  

---

## 📊 Dataset: HAM10000
- **Description**: 10,015 dermatoscopic images across 7 skin lesion classes.  
- **Class Distribution**:  
  | Class                   | Training Images | Test Images |
  |-------------------------|------------------|-------------|
  | Melanocytic nevi        | 5,364           | 1,341       |
  | Melanoma                | 890             | 223         |
  | Benign keratosis        | 879             | 220         |
  | Basal cell carcinoma    | 411             | 103         |
  | Actinic keratoses       | 262             | 65          |
  | Vascular lesions        | 114             | 28          |
  | Dermatofibroma          | 92              | 23          |

- **Preprocessing**: Resizing (64×64), normalization, oversampling, and augmentation (brightness, flipping, etc.).

---

## 🛠️ Methodology
### System Pipeline
1. **Data Acquisition & Preprocessing**  
2. **Hybrid Model Training**  
   - Teacher: ViT + ConvNeXT.  
   - Student: CNN + EfficientNet.  
3. **Multi-Stage Knowledge Distillation**  
   - Loss Components:  
     - Intermediate Loss (`α=0.3`).  
     - Logits Distillation Loss (`β=0.3`).  
     - Classification Loss (`γ=0.4`).  
4. **Post-Training Quantization**  
5. **XAI Visualization & Expert Validation**  

![Pipeline](https://via.placeholder.com/800x400.png?text=System+Architecture+Diagram)

---

## 📈 Results
### Performance Comparison
| Model                          | Accuracy | F1-Score | AUC   |
|--------------------------------|----------|----------|-------|
| Baseline (CNN)                 | 73.82%   | 74.20%   | 95.92%|
| Single-Stage Distillation      | 90.06%   | 90.02%   | 98.59%|
| Multi-Stage Distillation       | 92.22%   | 92.29%   | 98.81%|
| **Multi-Stage + Layer Fusion** | **95.88%** | **95.91%** | **99.02%** |

### Resource Efficiency
| Model                          | Size Reduction | Training Time | Inference Time |
|--------------------------------|----------------|---------------|----------------|
| Baseline Student               | -              | 240 ms/step   | 40 ms/step     |
| Quantized Distilled Model      | 91.81%         | 61 ms/step    | 15 ms/step     |

---

## 🛠️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/codewith-pavel/Optimizations.git
   cd Optimizations
