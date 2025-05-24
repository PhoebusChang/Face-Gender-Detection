# Face Gender Detection with CNN

A gender classification model based on simple convolutional neural networks. It could predict the genders of the people in images based on details learned from the CNN.

## 🎯 Project Overview

This project uses the PyTorch Lightning implementation to create a convolutional neural network, which could do gender classification from facial images. The model also includes comprehensive visualization tools to help understand what the neural network is doing.

## 🏗️ Architecture

### Model Structure
- **Input**: 512×512 RGB face images
- **Conv Layer 1**: 32 filters, 7×7 kernel, stride=2 (512→256)
- **Conv Layer 2**: 64 filters, 5×5 kernel, stride=2 (256→128) 
- **Conv Layer 3**: 128 filters, 3×3 kernel, stride=2 (128→64)
- **MaxPool**: 2×2, stride=2 (64→32)
- **Fully Connected**: 128×32×32 → 256 → 2 classes
- **Output**: Binary classification (Male/Female)

## 🚀 Getting Started

### Requirements
```bash
pip install torch torchvision pytorch-lightning matplotlib pillow
```

### Dataset Structure
```
datasets/
└── ashwingupta3012/
    └── male-and-female-faces-dataset/
        └── versions/1/
            └── Male and Female face dataset/
                ├── male/
                │   ├── image1.jpg
                │   └── ...
                └── female/
                    ├── image1.jpg
                    └── ...
```

### Training with Visualization
```bash
python main_visualize.py
```

### Inference Only
```bash
python main.py
```

## 📁 File Structure

```
Face-Gender-Detection/
├── main.py                    # Main training and inference script
├── main_visualize.py         # Training with visualization features
├── visuals/                  # Generated visualization images
│   ├── epoch_0.png          # Feature maps per epoch
│   ├── epoch_0_filters.png  # Filter visualizations
│   └── ...
├── model.pth                 # Saved model weights
├── test.png                  # Test image for inference
└── README.md
```

## 🔧 Configuration

### Data Preprocessing
```python
transforms = transforms.Compose([
    transforms.Resize((512, 512)),           # Resize to fixed dimensions
    transforms.ToTensor(),                   # Convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5),   # Normalize to [-1, 1]
                        (0.5, 0.5, 0.5))
])
```

### Training Parameters
- **Batch Size**: 32
- **Learning Rate**: 0.001 (Adam optimizer)
- **Scheduler**: StepLR (decay by 0.1 every 10 epochs)
- **Max Epochs**: 10
- **Train/Val Split**: 80/20

## 📈 Model Performance

The model tracks training and validation loss with PyTorch Lightning's built-in logging:
- Real-time loss visualization in progress bars
- Automatic GPU/CPU device selection
- Mixed precision training support

## 🎨 Visualization Details

### Feature Map Interpretation
- **Bright areas (yellow)**: High activation regions
- **Dark areas (purple)**: Low activation regions
- **Progressive abstraction**: Conv1→Conv2→Conv3 shows increasing feature complexity

### Filter Interpretation
- **Red regions**: Positive weights
- **Blue regions**: Negative weights
- **Filter evolution**: Shows how kernels adapt during training

## 🔍 Usage Examples

### Training
```python
# Training with visualization
python main_visualize.py
```

### Inference
```python
# Place your test image as 'test.png' and run:
python main.py
# Follow prompts to classify images
```

### Expected Output
```
Predicted: female
Confidence: 87.34%
Probabilities: Female: 87.34%, Male: 12.66%
```
