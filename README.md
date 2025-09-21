# Medical Image Classification - Pneumonia Detection

## Overview
This project implements a deep learning system for detecting pneumonia in chest X-ray images. The system uses Convolutional Neural Networks (CNN) and transfer learning to classify X-ray images as normal or showing signs of pneumonia.

## Features
- **Multiple Model Architectures**: Custom CNN and transfer learning (VGG16, ResNet50)
- **Data Augmentation**: Robust preprocessing for medical images
- **Model Interpretability**: GradCAM visualizations for decision explanation
- **Comprehensive Evaluation**: Detailed metrics and confusion matrices
- **Single Image Prediction**: Easy interface for new X-ray analysis
- **Medical-grade Accuracy**: Optimized for healthcare applications

## Dataset
The project uses the **Chest X-Ray Images (Pneumonia)** dataset:
- **Training**: 5,216 images
- **Validation**: 16 images  
- **Testing**: 624 images
- **Classes**: Normal, Pneumonia
- **Source**: Mendeley Data / Kaggle

## Requirements
```
tensorflow>=2.13.0
numpy>=1.21.0
pandas>=1.3.0
opencv-python>=4.5.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
pillow>=8.0.0
```

## Installation
```bash
# Clone repository
git clone https://github.com/username/medical-image-classification.git
cd medical-image-classification

# Install dependencies
pip install -r requirements.txt

# Download dataset
mkdir -p data/chest_xray
# Download from Kaggle: chest-xray-pneumonia dataset
```

## Usage

### Basic Usage
```python
from medical_image_classifier import MedicalImageClassifier

# Initialize classifier
classifier = MedicalImageClassifier(img_size=(224, 224))

# Setup data
classifier.create_data_generators(train_dir, val_dir, test_dir)

# Build and train model
classifier.build_transfer_learning_model('vgg16')
classifier.train(epochs=30)

# Evaluate
classifier.evaluate()
```

### Advanced Usage
```python
# Custom CNN architecture
classifier.build_cnn_model()

# Transfer learning with ResNet50
classifier.build_transfer_learning_model('resnet50')

# Single image prediction
result = classifier.predict_single_image('xray.jpg')

# Model interpretability
classifier.gradcam_visualization('xray.jpg')
```

## Model Architectures

### Transfer Learning (Recommended)
- **Base Model**: VGG16 or ResNet50 pre-trained on ImageNet
- **Custom Head**: Dense layers with dropout for medical classification
- **Optimization**: Adam optimizer with adaptive learning rate

### Custom CNN
- **Layers**: 4 convolutional blocks with batch normalization
- **Pooling**: Max pooling for feature reduction
- **Dense Layers**: Fully connected layers with dropout
- **Output**: Single neuron with sigmoid activation

## Data Preprocessing
1. **Image Normalization**: Pixel values scaled to [0,1]
2. **Data Augmentation**: Rotation, shift, flip, zoom for training
3. **Resize**: All images standardized to 224x224 pixels
4. **Class Balance**: Handled through weighted loss or sampling

## Performance Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate for pneumonia detection
- **Recall**: Sensitivity for pneumonia cases
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

## Model Interpretability
The system includes GradCAM (Gradient-weighted Class Activation Mapping) visualization:
- **Heatmaps**: Show which areas the model focuses on
- **Medical Validation**: Helps radiologists verify AI decisions
- **Trust Building**: Increases confidence in automated diagnosis

## File Structure
```
medical_image_classification/
├── medical_image_classifier.py
├── requirements.txt
├── README.md
├── data/
│   └── chest_xray/
│       ├── train/
│       ├── val/
│       └── test/
├── models/
│   ├── best_medical_model.h5
│   └── checkpoints/
├── results/
│   ├── confusion_matrices/
│   ├── gradcam_visualizations/
│   └── training_plots/
└── utils/
    ├── data_loader.py
    └── visualization.py
```

## Clinical Integration
### Workflow Integration
1. **
