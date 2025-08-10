# Malaria Detection using Deep Learning
## Overview
This project implements a deep learning-based approach for detecting malaria-infected cells from microscopic images. 
The goal is to build an accurate and efficient classification model that can distinguish between **Parasitized** and **Uninfected** cell images.

By leveraging convolutional neural networks (CNNs) and advanced image augmentation techniques, the system is designed to assist medical professionals in 
speedy and reliable malaria diagnosis, potentially reducing the workload in resource-constrained regions.
## Dataset
The dataset used contains cell images labeled as **Parasitized** or **Uninfected**.

- **Source**: Publicly available malaria cell image dataset
- **Classes**: 
  - `Parasitized`: Images containing malaria-infected cells.
  - `Uninfected`: Images of healthy cells.
- **Format**: RGB images (converted to a uniform size during preprocessing).
## Data Preprocessing & Augmentation
To improve model generalization and robustness, the dataset undergoes the following preprocessing steps:

1. **Image Resizing** – All images are resized to a fixed input size suitable for the model.
2. **Normalization** – Pixel values are scaled to the range [0, 1].
3. **Data Augmentation** – Using `albumentations` and TensorFlow's augmentation layers:
   - Random flips
   - Random rotations
   - Contrast adjustments
   - Rescaling

These augmentations help the model adapt to varying imaging conditions and reduce overfitting.
## Model Architecture
The model is implemented using **TensorFlow** and **Keras**. Key layers include:

- **Input Layer** – Accepts preprocessed image tensors.
- **Convolutional Layers** – Extract spatial features using filters and ReLU activations.
- **Batch Normalization** – Stabilizes and speeds up training.
- **Pooling Layers** – Reduces spatial dimensions while preserving key features.
- **Dropout Layers** – Regularization to prevent overfitting.
- **Fully Connected Layers** – Combines extracted features for classification.
- **Output Layer** – Single neuron with sigmoid activation for binary classification.
## Training Configuration
- **Loss Function**: Binary Cross-Entropy
- **Optimizer**: Adam
- **Metrics**: Binary Accuracy, Precision, Recall, AUC
- **Callbacks**:
  - EarlyStopping (to prevent overfitting)
  - ReduceLROnPlateau (to adjust learning rate dynamically)
  - ModelCheckpoint (to save the best-performing model)

The training pipeline uses mini-batch gradient descent with shuffling to improve convergence.
## Evaluation & Metrics
The trained model is evaluated using:

- **Accuracy** – Overall correctness of predictions.
- **Precision & Recall** – Measures for handling class imbalance.
- **AUC-ROC** – Area under the ROC curve to evaluate discriminative power.
- **Confusion Matrix** – Visual representation of classification performance.
## Results & Visualizations
Key outputs include:

- Confusion matrix plots
- ROC curves
- Training & validation loss/accuracy graphs

These visualizations help in understanding the model's performance and potential areas of improvement.
## Usage Instructions

1. **Install Dependencies**
```bash
pip install tensorflow albumentations matplotlib seaborn scikit-learn
```

2. **Run Training**
```bash
python malaria_detection.py
```

3. **Evaluate Model**
```bash
python evaluate.py --model best_model.h5
```

4. **Make Predictions**
```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model("best_model.h5")
img = cv2.imread("cell_image.png")
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
img = np.expand_dims(img, axis=0)
prediction = model.predict(img)
print("Parasitized" if prediction[0][0] > 0.5 else "Uninfected")
```
