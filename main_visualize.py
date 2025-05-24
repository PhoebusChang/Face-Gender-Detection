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

dataset = datasets.ImageFolder(
    root='datasets/ashwingupta3012/male-and-female-faces-dataset/versions/1/Male and Female face dataset',
    transform=transforms
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

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
    
        self.feature_maps = {}
    
    def forward(self, x):
        
        self.feature_maps['input'] = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        self.feature_maps['conv1'] = x 
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        self.feature_maps['conv2'] = x 
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        self.feature_maps['conv3'] = x 
        
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
    
    def on_train_epoch_end(self):
        
        if self.current_epoch % 1 == 0:
            self.visualize_features()
    
    def visualize_features(self):
        
        sample_batch = next(iter(self.trainer.train_dataloader))
        sample_x, sample_y = sample_batch
        sample_x = sample_x[:2].to(self.device)  # 2 samples for visualization
        
        with torch.no_grad():
            _ = self(sample_x)
        
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(f'Epoch {self.current_epoch}: Feature Maps & Filters Visualization', fontsize=16)
        
        for img_idx in range(2):
            row_offset = img_idx * 5
            
            original = sample_x[img_idx].detach().cpu()
            original = original * 0.5 + 0.5 
            ax = plt.subplot(10, 16, row_offset * 16 + 1)
            ax.imshow(original.permute(1, 2, 0))
            ax.set_title(f'Original {img_idx+1}', fontsize=8)
            ax.axis('off')
            
            conv1_features = self.feature_maps['conv1'][img_idx].detach().cpu()
            conv1_weights = self.conv1.weight.detach().cpu()
            
            for i in range(8):
                ax = plt.subplot(10, 16, row_offset * 16 + 2 + i)
                ax.imshow(conv1_features[i], cmap='viridis')
                ax.set_title(f'C1 FM{i}', fontsize=6)
                ax.axis('off')
                
                ax = plt.subplot(10, 16, row_offset * 16 + 10 + i)
                filter_viz = torch.mean(conv1_weights[i], dim=0)
                ax.imshow(filter_viz, cmap='RdBu_r')
                ax.set_title(f'C1 F{i}', fontsize=6)
                ax.axis('off')
            
            conv2_features = self.feature_maps['conv2'][img_idx].detach().cpu()
            conv2_weights = self.conv2.weight.detach().cpu()
            
            for i in range(8):
                ax = plt.subplot(10, 16, (row_offset + 1) * 16 + 2 + i)
                ax.imshow(conv2_features[i], cmap='viridis')
                ax.set_title(f'C2 FM{i}', fontsize=6)
                ax.axis('off')
                
                ax = plt.subplot(10, 16, (row_offset + 1) * 16 + 10 + i)
                ax.imshow(conv2_weights[i, 0], cmap='RdBu_r')
                ax.set_title(f'C2 F{i}', fontsize=6)
                ax.axis('off')
            
            conv3_features = self.feature_maps['conv3'][img_idx].detach().cpu()
            conv3_weights = self.conv3.weight.detach().cpu()
            
            for i in range(8):
                ax = plt.subplot(10, 16, (row_offset + 2) * 16 + 2 + i)
                ax.imshow(conv3_features[i], cmap='viridis')
                ax.set_title(f'C3 FM{i}', fontsize=6)
                ax.axis('off')
                
                ax = plt.subplot(10, 16, (row_offset + 2) * 16 + 10 + i)
                ax.imshow(conv3_weights[i, 0], cmap='RdBu_r')
                ax.set_title(f'C3 F{i}', fontsize=6)
                ax.axis('off')
            
            ax = plt.subplot(10, 16, (row_offset + 3) * 16 + 1)
            conv1_avg = torch.mean(conv1_features[:16], dim=0)
            ax.imshow(conv1_avg, cmap='viridis')
            ax.set_title(f'C1 Avg', fontsize=8)
            ax.axis('off')
            
            ax = plt.subplot(10, 16, (row_offset + 3) * 16 + 2)
            conv2_avg = torch.mean(conv2_features[:16], dim=0)
            ax.imshow(conv2_avg, cmap='viridis')
            ax.set_title(f'C2 Avg', fontsize=8)
            ax.axis('off')
            
            ax = plt.subplot(10, 16, (row_offset + 3) * 16 + 3)
            conv3_avg = torch.mean(conv3_features[:16], dim=0)
            ax.imshow(conv3_avg, cmap='viridis')
            ax.set_title(f'C3 Avg', fontsize=8)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'visuals/epoch_{self.current_epoch}.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        self.visualize_filters_only()
    
    def visualize_filters_only(self):
        """Separate visualization just for filters/kernels"""
        fig, axes = plt.subplots(3, 16, figsize=(20, 6))
        fig.suptitle(f'Epoch {self.current_epoch}: Learned Filters', fontsize=14)
        
        conv1_weights = self.conv1.weight.detach().cpu()
        for i in range(16):
            filter_viz = torch.mean(conv1_weights[i], dim=0)
            axes[0, i].imshow(filter_viz, cmap='RdBu_r')
            axes[0, i].set_title(f'C1-{i}', fontsize=6)
            axes[0, i].axis('off')
        
        conv2_weights = self.conv2.weight.detach().cpu()
        for i in range(16):
            axes[1, i].imshow(conv2_weights[i, 0], cmap='RdBu_r')
            axes[1, i].set_title(f'C2-{i}', fontsize=6)
            axes[1, i].axis('off')
        
        conv3_weights = self.conv3.weight.detach().cpu()
        for i in range(16):
            axes[2, i].imshow(conv3_weights[i, 0], cmap='RdBu_r')
            axes[2, i].set_title(f'C3-{i}', fontsize=6)
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'visuals/epoch_{self.current_epoch}_filters.png', dpi=150, bbox_inches='tight')
        plt.close()
    
model = imageCNN()

trainer = pl.Trainer(
    max_epochs=10,
    accelerator='auto',
    devices='auto'
)
trainer.fit(model, train_loader, val_loader)
