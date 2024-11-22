# Classification-of-Pneumonia-from-Chest-X-ray-Images-using-Deep-Learning
This project explores the use of Convolutional Neural Networks (CNNs) for classifying chest X-ray images into four categories: COVID-19, Pneumonia, Tuberculosis, and Normal. We evaluate three CNN architectures—ResNet-50, VGG-16, and VGG-19—to determine their effectiveness in accurately diagnosing these pulmonary conditions.

The models were trained and tested using a publicly available Kaggle dataset of 5,788 labeled X-ray images. Experiments were conducted using PyTorch and a GPU environment to ensure optimal performance. The project highlights the potential of deep learning in medical imaging and provides insights into trade-offs between model accuracy and computational efficiency.

# Dataset
The dataset used in this project was sourced from Kaggle, containing a total of 5,788 images across four classes:

COVID-19: 576 images
Pneumonia: 3,398 images
Normal: 1,114 images
Tuberculosis: 700 images

# Data Splitting
The dataset was split into three parts using a 60:20:20 ratio:

Training Set: 60% of images
Validation Set: 20% of images
Test Set: 20% of images
The images were resized to 120x120 pixels and normalized to standardize pixel values using PyTorch's transforms.Normalize() function. Augmentation techniques such as Random Horizontal Flip and Random Rotation were applied to improve model generalization.

# Deep Learning Models
We tested three popular CNN architectures:

ResNet-50: A residual network with skip connections, known for its strong learning and generalization capabilities.
VGG-16: A 16-layer deep model with a simple and uniform architecture.
VGG-19: A 19-layer variant of VGG, providing additional depth for feature extraction.
All models were trained for 25 epochs using the Adam optimizer with a learning rate of 0.001. Early stopping was implemented to prevent overfitting.

# Performance Metrics
To evaluate model performance, we used the following metrics:

Per-class F1-scores
Micro-average F1-score
Macro-average F1-score
Area Under the Curve (AUC)
Given the class imbalance in the dataset, macro-average metrics were prioritized to ensure equal weight for each class. Below is a comparison of the results:

Model	Per-class F1-scores	Micro-average F1	Macro-average F1	Macro-average AUC
ResNet-50	COVID-19: 0.97, Normal: 0.89, Pneumonia: 0.96, Tuberculosis: 0.98	0.96	0.95	1.00
VGG-19	COVID-19: 0.94, Normal: 0.91, Pneumonia: 0.94, Tuberculosis: 0.98	0.96	0.95	1.00
VGG-16	COVID-19: 0.93, Normal: 0.94, Pneumonia: 0.94, Tuberculosis: 0.97	0.96	0.95	0.99

# Results
ResNet-50
Achieved superior performance with a macro-average F1-score of 0.95 and an AUC of 1.00.
Demonstrated robust learning capabilities with rapid convergence and stability across training and validation.
Outperformed VGG models in distinguishing between nuanced features, making it ideal for clinical applications.
VGG-19
Showed notable fluctuations in the early epochs but achieved a stable performance with a macro-average F1-score of 0.95.
Effective in pneumonia detection but slightly less accurate in classifying COVID-19 and normal cases.
VGG-16
Provided consistent performance, with minimal misclassifications.
Achieved a macro-average F1-score of 0.95 and an AUC of 0.99, making it a reliable option for diagnostic applications.

# Visualization
Below link contain key visualizations for all model including:

Training and validation accuracy/loss graphs
Confusion matrix
ROC-AUC curves for all classes

https://colab.research.google.com/drive/1IlN-WKqiPTpdCk27n3X3reomOA9weXxp?usp=sharing
