# Potato Disease Classification

This repository contains a Jupyter Notebook for building a Convolutional Neural Network (CNN) model to classify potato leaf diseases. The model distinguishes between Early Blight, Late Blight, and Healthy potato leaves.

## Table of Contents

* [Introduction](#introduction)
* [Dataset](#dataset)
* [Project Structure](#project-structure)
* [Model Training](#model-training)
* [Results](#results)
* [Requirements](#requirements)
* [Usage](#usage)
* [Acknowledgements](#acknowledgements)

## Introduction

Potato blight is one of the most destructive diseases affecting potato crops. Early detection is crucial for effective disease management.

This project provides an image-based deep learning classifier using TensorFlow/Keras to identify common potato diseases from leaf images.

The model classifies images into three categories:

* Potato\_\_\_Early\_blight
* Potato\_\_\_Late\_blight
* Potato\_\_\_healthy

## Dataset

The dataset contains potato leaf images categorized into the three classes mentioned above.

It is included directly in this repository under the PlantVillage/ directory.

**Dataset Structure:**

```
PlantVillage/
├── Potato___Early_blight/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Potato___Late_blight/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Potato___healthy/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Project Structure

```
.
├── potato-disease-classification.ipynb
├── PlantVillage/
│   ├── Potato___Early_blight/
│   ├── Potato___Late_blight/
│   └── Potato___healthy/
├── README.md
└── .gitignore
```

## Model Training

The notebook potato-disease-classification.ipynb covers the entire workflow:

* Loading and preprocessing images with tf.keras.preprocessing.image\_dataset\_from\_directory
* Defining a CNN model architecture with TensorFlow/Keras
* Training for a configurable number of epochs
* Evaluating accuracy and loss on a test set

**Model Highlights:**

* Input Size: Images resized to 256x256x3
* Architecture: Multiple Conv2D + MaxPooling2D layers for feature extraction → Flatten → Dense layers → softmax output for classification
* Batch Size: Default 32 during training and inference

## Results

After 3 epochs of training, the model achieved \~90% accuracy on the test set. Further training (e.g., 10–20 epochs) may improve results.

Accuracy/loss curves and confusion matrix can be added after further evaluation.

## Requirements

To run the notebook, you’ll need Python 3.8+ and the following libraries:

```bash
pip install tensorflow matplotlib numpy scikit-learn jupyterlab Pillow
```

Or simply install from requirements.txt:

```
tensorflow
matplotlib
numpy
scikit-learn
jupyterlab
Pillow
```

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/Kibs-Neville/potato-disease-classification.git
   cd potato-disease-classification
   ```

2. The dataset is already included in the PlantVillage/ folder.

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the notebook:

   ```bash
   jupyter lab potato-disease-classification.ipynb
   ```

5. Follow the instructions in the notebook to train and evaluate the model. The trained model (potato\_disease\_classification\_model.h5) will be saved in the root directory.

## Acknowledgements

* Dataset sourced from https://www.kaggle.com/datasets/arjuntejaswi/plant-village
* TensorFlow/Keras for deep learning framework
* Inspired by real-world crop disease detection research
