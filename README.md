# Pneumonia-Detection-From-Chest-X-Rays-and-FDA-Submission

## 2D Medical Image Classification - Udacity AI for Heathcare Course

In this project, I analyzed data from the NIH Chest X-ray Dataset and train a CNN to classify a given chest x-ray for the presence or absence of pneumonia. This project culminate in a model that can predict the presence of pneumonia with human radiologist-level accuracy that can be prepared for submission to the FDA for 510(k) clearance as software as a medical device. As part of the submission preparation, I formally described my model, the data that it was trained on, and a validation plan that meets FDA criteria.

<img src="https://www.linkpicture.com/q/Capture_290.png" width="800px" height="auto">

For this project, I used the medical images with clinical labels for each image that were extracted from their accompanying radiology reports.

I used 112,000 chest x-rays with disease labels acquired from 30,000 patients.

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
- **Build and train model.ipynb:** the notebook for train the model.
- **FDA  Submission.pdf:** the document for FDA submission.
- **sample_labels.csv:** the .csv file that includes medical image labels.
- **test.dcm files:** predicted medical images with trained model and their predicted labels. test3.dcm image is above.
