# Pneumonia Detection from Chest X-Rays and FDA Submission

## 2D Medical Image Classification - Udacity AI for Heathcare Course

In this project, I analyzed data from the NIH Chest X-ray Dataset and train a CNN to classify a given chest x-ray for the presence or absence of pneumonia. This project culminate in a model that can predict the presence of pneumonia with human radiologist-level accuracy that can be prepared for submission to the FDA for 510(k) clearance as software as a medical device. As part of the submission preparation, I formally described my model, the data that it was trained on, and a validation plan that meets FDA criteria.

<img src="https://www.linkpicture.com/q/Capture_290.png" width="800px" height="auto">

For this project, I used the medical images with clinical labels for each image that were extracted from their accompanying radiology reports.

I used 112,000 chest x-rays with disease labels acquired from 30,000 patients.

## About the Dataset

The dataset provided to you for this project was curated by the NIH specifically to address the problem of a lack of large x-ray datasets with ground truth labels to be used in the creation of disease detection algorithms. 

The data is mounted in the Udacity Jupyter GPU workspace provided to you, along with code to load the data. Alternatively, you can download the data from the [kaggle website](https://www.kaggle.com/nih-chest-xrays/data) and run it locally. You are STRONGLY recommended to complete the project using the Udacity workspace since the data is huge, and you will need GPU to accelerate the training process.

There are 112,120 X-ray images with disease labels from 30,805 unique patients in this dataset.  The disease labels were created using Natural Language Processing (NLP) to mine the associated radiological reports. The labels include 14 common thoracic pathologies: 
- Atelectasis 
- Consolidation
- Infiltration
- Pneumothorax
- Edema
- Emphysema
- Fibrosis
- Effusion
- Pneumonia
- Pleural thickening
- Cardiomegaly
- Nodule
- Mass
- Hernia 

The biggest limitation of this dataset is that image labels were NLP-extracted so there could be some erroneous labels but the NLP labeling accuracy is estimated to be >90%.

The original radiology reports are not publicly available but you can find more details on the labeling process [here.](https://arxiv.org/abs/1705.02315) 

### Dataset Contents: 

1. 112,120 frontal-view chest X-ray PNG images in 1024*1024 resolution (under images folder)
2. Meta data for all images (Data_Entry_2017.csv): Image Index, Finding Labels, Follow-up #,
Patient ID, Patient Age, Patient Gender, View Position, Original Image Size and Original Image
Pixel Spacing.

## Project Steps
This project has the following steps.

1. Exploratory Data Analysis
2. Building and Training Your Model
3. Clinical Workflow Integration
4. FDA Preparation

## Libraries Used

```
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import sklearn.model_selection as skl
import tensorflow as tf

from glob import glob
from itertools import chain
from random import sample
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, plot_precision_recall_curve, f1_score, confusion_matrix

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.applications.resnet import ResNet50 
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
```


## Files in the Repository
- **EDA.ipynb:** the notebook for exploratory data analysis.
- **Build and train model.ipynb:** the notebook for building and training the model.
- **Inference.ipynb:** the notebook for clinical workflow integration.
- **FDA_Submission_Template.md:** This is the template for creating the FDA submission.
- **FDA  Submission.pdf:** the document for FDA submission.
- **sample_labels.csv:** the .csv file that used to assess images in the pixel-level.
- **.dcm files:**  test files to test the clinical workflow integration. test3.dcm image is above.
