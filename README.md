# pritish73-ICH-Detection-EfficientNet
#  Intracranial Hemorrhage Detection using Deep Learning

##  Overview
This project focuses on detecting Intracranial Hemorrhage (ICH) from CT scan images using deep learning.

##  Key Features
- EfficientNet-B0 based classification model
- Class imbalance handling using weighted loss
- Threshold tuning for optimal precision-recall tradeoff
- Data augmentation for better generalization
- Medical evaluation metrics (Recall, Specificity, F1-score)

##  Final Results
- Accuracy: 95.7%
- Precision: 80%
- Recall: 83.3%
- F1-score: 81.6%
- Specificity: 97.3%

##  Model Details
- Backbone: EfficientNet-B0
- Loss: Weighted CrossEntropy
- Threshold: 0.6
- Optimizer: Adam
- Scheduler: ReduceLROnPlateau

##  Dataset
Dataset is not included due to licensing restrictions.

##  Tech Stack
- Python
- PyTorch
- NumPy
- OpenCV
- scikit-learn

##  Highlights
- Balanced precision-recall performance for medical safety
- Avoids false negatives while controlling false positives

##  Author
Pritish Dutta
