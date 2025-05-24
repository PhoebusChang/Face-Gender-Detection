import kagglehub
import os

# Set donwload directory
os.environ["KAGGLEHUB_CACHE"] = "./"

# Download dataset
path = kagglehub.dataset_download("ashwingupta3012/male-and-female-faces-dataset")

print("Path to dataset files:", path)