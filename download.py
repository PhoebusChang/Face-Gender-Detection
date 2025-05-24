import kagglehub
import os

# Set donwload directory
os.environ["KAGGLEHUB_CACHE"] = "./"

# Download dataset
path = kagglehub.dataset_download("cashutosh/gender-classification-dataset")

print("Path to dataset files:", path)