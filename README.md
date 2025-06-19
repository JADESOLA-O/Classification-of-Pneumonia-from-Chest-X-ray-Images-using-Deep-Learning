# Classification of Pneumonia from Chest X-ray Images Using Deep Learning

This project explores the use of **Convolutional Neural Networks (CNNs)** for classifying chest X-ray images into four categories: **COVID-19, Pneumonia, Tuberculosis, and Normal**. I did evaluation on three CNN architectures **ResNet-50**, **VGG-16**, and **VGG-19** to determine their effectiveness in accurately diagnosing these pulmonary conditions.
The models were trained and tested using a publicly available **Kaggle dataset** of 5,788 labeled X-ray images. Experiments were conducted using **PyTorch** in a **GPU-enabled environment** (Google Colab) to ensure optimal performance.

The project highlights the potential of deep learning in medical imaging and provides insights into trade-offs between **model accuracy** and **computational efficiency**.



## 📂 Dataset
The dataset was sourced from Kaggle and contains **5,788 images** across four classes:
- **COVID-19**: 576 images  
- **Pneumonia**: 3,398 images  
- **Normal**: 1,114 images  
- **Tuberculosis**: 700 images  

### 📊 Data Splitting
- **Training Set**: 60%  
- **Validation Set**: 20%  
- **Test Set**: 20%  
Images were resized to **120×120 pixels** and normalized using PyTorch’s `transforms.Normalize()`.

### Augmentation Techniques
To improve model generalization:
- `RandomHorizontalFlip`
- `RandomRotation`

---

## Deep Learning Models

Three popular CNN architectures were evaluated:
- **ResNet-50**  
  - Deep residual network with skip connections  
  - Strong generalization and learning capacity
  - 
- **VGG-16**  
  - 16-layer architecture  
  - Uniform structure and simplicity
  - 
- **VGG-19**  
  - 19-layer version of VGG  
  - Deeper feature extraction  

### Training Configuration
- **Epochs**: Up to 25  
- **Optimizer**: Adam  
- **Learning Rate**: 0.001  
- **Early Stopping**: Enabled to avoid overfitting
- 

### Performance Metrics
I used the following metrics to evaluate model performance:
- **Per-class F1-scores**  
- **Micro-average F1-score**  
- **Macro-average F1-score**  
- **Area Under the Curve (AUC)**  

Macro-averages were prioritized due to class imbalance.
 Model Comparison

| Model         | COVID-19 | Normal | Pneumonia | Tuberculosis | F1 (Micro) | F1 (Macro) | AUC (Macro) |
|---------------|----------|--------|-----------|--------------|------------|-------------|--------------|
| **ResNet-50** | 0.97     | 0.89   | 0.96      | 0.98         | 0.96       | 0.95        | **1.00**     |
| **VGG-19**    | 0.94     | 0.91   | 0.94      | 0.98         | 0.96       | 0.95        | **1.00**     |
| **VGG-16**    | 0.93     | 0.94   | 0.94      | 0.97         | 0.96       | 0.95        | 0.99         |

---

# Results Summary
ResNet-50
  - Achieved highest performance with **macro F1 = 0.95** and **AUC = 1.00**
  - Robust learning and stable convergence
  - Excellent at distinguishing nuanced features — ideal for clinical use

VGG-19
  - Early training fluctuations, but stabilized
  - Good pneumonia classification
  - Slight underperformance in COVID-19 and Normal detection

VGG-16
  - Most consistent performance across training and validation
  - Minimal misclassifications
  - Macro F1 = 0.95, AUC = 0.99 — a reliable and efficient model


# Limitations

- **Colab GPU timeouts** limited epoch runs to 25 max
- GPU disconnections slowed down experimentation
- A paid Colab plan was used for consistent training sessions

## 🧾 Keywords
CNN, Chest X-ray, Pneumonia, COVID-19, Tuberculosis, ResNet50, VGG16, Medical Imaging, Deep Learning, PyTorch


You can access the full training and evaluation pipeline using this notebook:
[🔗 Google Colab Notebook](https://colab.research.google.com/drive/1IlN-WKqiPTpdCk27n3X3reomOA9weXxp?usp=sharing#scrollTo=fxa3wUrj1U7b)]







