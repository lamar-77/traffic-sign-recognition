import kagglehub # To download dataset
import os # Helps with files
import pandas as pd # To handels data

# Download latest version
path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")

print("Path to dataset files:", path)

# Create paths for the CSV files
train_csv = os.path.join(path , "Train.csv")
test_csv = os.path.join(path, "Test.csv")
meta_csv = os.path.join(path, "Meta.csv")

train_data = pd.read_csv(train_csv)
print(train_data.head())