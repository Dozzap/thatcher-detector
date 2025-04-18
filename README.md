# Thatcherized Face Classification using VGG16 Feature Extraction and SVM

This project presents an implementation of a hybrid machine learning pipeline designed to classify facial images as either **normal** or **thatcherized**. The system integrates **deep convolutional feature extraction** using a pre-trained **VGG16 model**, alongside traditional **Support Vector Machine (SVM)** classification augmented with **facial geometry metrics**.

## Overview

The Thatcher effect is a perceptual phenomenon where local feature manipulation (e.g., inversion of eyes or mouth) is more noticeable in upright faces than in inverted ones. This project operationalizes the classification of such manipulated images by:

- Applying facial landmark detection to isolate and alter key facial regions.
- Computing handcrafted geometric features from facial landmarks.
- Leveraging deep visual features extracted from VGG16's convolutional layers.
- Training an SVM classifier on the combined feature set.


More about the data generation on this [link](https://github.com/Dozzap/thatcher-data-set-generator).
