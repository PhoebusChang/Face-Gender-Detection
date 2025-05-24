import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl

from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, random_split

from matplotlib import pyplot as plt

transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.ImageFolder(
    root='datasets/cashutosh/gender-classification-dataset/versions/1/Training',
    transform=transforms
)

val_dataset = datasets.ImageFolder(
    root='datasets/cashutosh/gender-classification-dataset/versions/1/Validation',
    transform=transforms
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class imageCNN(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=7,
            stride=2,
            padding=3
        )
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            stride=2,
            padding=2
        )
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.fc1 = nn.Linear(128 * 32 * 32, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 2)

        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]
    
model = imageCNN()
if input("train model? (y/n): ") == 'y':
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='auto',
        devices='auto'
    )
    trainer.fit(model, train_loader, val_loader)

    torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    # Load the model
    model = imageCNN()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Define class labels (adjust based on your dataset structure)
    class_labels = ['female', 'male']  # or ['male', 'female'] - check your dataset folders

    while (user_input := input("classify? (y/n): ")) == 'y':
        try:
            # Load and transform the image
            from PIL import Image
            image = Image.open('test.png').convert('RGB')
            
            
            # Transform and add batch dimension
            input_tensor = transforms(image).unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                predicted_class = torch.argmax(output, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Display results
            print(f"Predicted: {class_labels[predicted_class]}")
            print(f"Confidence: {confidence:.2%}")
            print(f"Probabilities: Female: {probabilities[0][0]:.2%}, Male: {probabilities[0][1]:.2%}")
            
        except FileNotFoundError:
            print("test.png not found!")
        except Exception as e:
            print(f"Error: {e}")
