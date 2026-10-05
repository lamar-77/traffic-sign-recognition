# Traffic Sign Recognition

A deep learning project for classifying German traffic signs using CNN.

## Project Idea

I am building a traffic sign recognition model using a Convolutional Neural Network (CNN). The model will take an image of a traffic sign and predict which of the 43 classes it belongs to.

This is a supervised, multiclass image classification project because the training images have labels, and the model needs to choose one class for each image. My goal is to train the model using labeled images and then test how well it recognizes signs it has not seen before.

## Dataset

For this project, I am using the German Traffic Sign Recognition Benchmark (GTSRB), a dataset of German traffic sign images divided into 43 classes.

The dataset includes:

- `Train/` and `Train.csv` for the training images and their labels.
- `Test/` and `Test.csv` for evaluating the model on test images.
- `Meta/` and `Meta.csv` with reference images and information about the traffic sign classes.

## Data Preprocessing

Before training, I preprocessed the images using the following steps:

- Converted the images to RGB format.
- Resized each image to 32 × 32 pixels.
- Normalized pixel values from 0–255 to 0–1.

The 39,209 images listed in `Train.csv` were split into:

- Training set: 31,367 images.
- Validation set: 7,842 images.

The 12,630 images listed in `Test.csv` were processed in the same way and kept separate for the final model evaluation. 
