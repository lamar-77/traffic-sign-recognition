import kagglehub # To download dataset
import os # Helps with files
import pandas as pd # To handels data
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


# Download latest version
path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")

print("Path to dataset files:", path)

# Create paths for the CSV files
train_csv = os.path.join(path , "Train.csv")
test_csv = os.path.join(path, "Test.csv")
meta_csv = os.path.join(path, "Meta.csv")

train_data = pd.read_csv(train_csv)
test_data = pd.read_csv(test_csv)
print(train_data.head())

# 1- resize step

X = [] # for images
y = [] # for labels

# loop through every training row
for row in train_data.itertuples(index= False):
 image_path = os.path.join(path, row.Path) # full path to the image
 image = Image.open(image_path).convert("RGB").resize((32,32))
 X.append(np.array(image)) # convert the images into a numpy array and store it
 y.append(row.ClassId) # store the label for this image

# convert the lists into numpy array
X = np.array(X) 
y = np.array(y)

print(X.shape , y.shape)

# 2- normalization step

X = X.astype(np.float32) / 255.0

print("Max:", X.max())
print("Min:", X.min())

# 3- split the training data into train and validation sets

X_train, X_val , y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=77, stratify=y)

print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)

# 4- test dataset

X_test = []
y_test = []

for row in test_data.itertuples(index= False):
 image_path = os.path.join(path, row.Path)
 image = Image.open(image_path).convert("RGB").resize((32,32))

 X_test.append(np.array(image))
 y_test.append(row.ClassId)

X_test = np.array(X_test, dtype=np.float32) / 255.0
y_test = np.array(y_test)

print("Test:", X_test.shape, y_test.shape)
print("Max :", X_test.max())
print("Min:", X_test.min())