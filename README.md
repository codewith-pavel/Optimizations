```markdown
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
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ Usage
### Training
```python
python train.py \
  --teacher vit_convnext \
  --student cnn_efficientnet \
  --batch_size 32 \
  --epochs 100 \
  --lr 0.001
```

### Inference with Quantized Model
```python
python infer.py \
  --model_path models/quantized_model_int8.tflite \
  --image_path data/test/melanoma_001.jpg
```

### XAI Visualization
```python
python explain.py \
  --method Score-CAM \
  --image_path data/test/benign_005.jpg \
  --output_dir results/
```

---

## 📂 Repository Structure
```
Optimizations/
├── data/                   # Preprocessed HAM10000 dataset
├── models/                 # Pretrained and quantized models
├── src/
│   ├── train.py            # Training script
│   ├── infer.py            # Inference script
│   ├── distill.py          # Knowledge distillation
│   └── explain.py          # XAI visualization
├── utils/                  # Data preprocessing and augmentation
├── requirements.txt        # Dependencies
└── README.md
```

---

## 🤝 Contributing
Contributions are welcome! Please open an **issue** or **pull request** for:
- Bug fixes
- Performance improvements
- New XAI methods

---

## 📜 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 📝 Citation
If you use this work, please cite:
```bibtex
@article{pavel2024multistage,
  title={Multi-Stage Knowledge Distillation with Layer Fusion-Based Deep Learning Approach for Skin Cancer Classification},
  author={Pavel, Mahir Afser and Asad, Ramisa and Ikramuzzaman, Md and Mustakim, Murad and Khan, Riasat},
  journal={arXiv preprint arXiv:XXXX.XXXX},
  year={2024}
}
```

---

## 📧 Contact
For questions or collaborations, contact:  
- Mahir Afser Pavel: [mahir.pavel@northsouth.edu](mailto:mahir.pavel@northsouth.edu)  
- Ramisa Asad: [ramisa.asad@northsouth.edu](mailto:ramisa.asad@northsouth.edu)
```
