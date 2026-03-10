"""
CNN for Finance - Image-Based Financial Analysis
Fixed version with proper image channel handling
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import os
from PIL import Image
import io

# ============================================================================
# PYTORCH IMPORTS
# ============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.models import resnet18, vgg16, alexnet

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# PART 1: CHART IMAGE GENERATION
# ============================================================================

class FinancialChartGenerator:
    """
    Generate financial chart images for CNN training
    """
    
    def __init__(self, figsize=(224, 224), dpi=100):
        self.figsize = figsize
        self.dpi = dpi
        self.chart_types = ['candlestick', 'line', 'bar', 'point_figure']
        
    def fetch_stock_data(self, symbol, period='1y'):
        """Fetch stock data from Yahoo Finance"""
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        return data
    
    def fig2img(self, fig):
        """Convert matplotlib figure to numpy array (cross-platform)"""
        # Save figure to a temporary buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        
        # Open with PIL and convert to RGB (remove alpha channel)
        img = Image.open(buf).convert('RGB')
        img_array = np.array(img)
        
        # Close figure to free memory
        plt.close(fig)
        buf.close()
        
        return img_array
    
    def generate_candlestick_chart(self, data, title="", save_path=None):
        """Generate candlestick chart"""
        fig, ax = plt.subplots(figsize=(self.figsize[0]/self.dpi, 
                                        self.figsize[1]/self.dpi), dpi=self.dpi)
        
        # Calculate width of candlesticks
        width = 0.6
        
        # Plot candlesticks
        for i, (idx, row) in enumerate(data.iterrows()):
            color = 'green' if row['Close'] >= row['Open'] else 'red'
            
            # Draw the candle body
            rect = Rectangle((i - width/2, min(row['Open'], row['Close'])), 
                           width, abs(row['Close'] - row['Open']),
                           facecolor=color, edgecolor=color)
            ax.add_patch(rect)
            
            # Draw the wicks
            ax.plot([i, i], [row['Low'], max(row['Open'], row['Close'])], 
                   color=color, linewidth=1)
            ax.plot([i, i], [row['High'], min(row['Open'], row['Close'])], 
                   color=color, linewidth=1)
        
        # Formatting
        ax.set_xlim(-1, len(data))
        ax.set_ylim(data['Low'].min() * 0.95, data['High'].max() * 1.05)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
        
        # Remove x-axis labels for cleaner image
        ax.set_xticks([])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            return save_path
        else:
            return self.fig2img(fig)
    
    def generate_line_chart(self, data, title="", save_path=None):
        """Generate line chart"""
        fig, ax = plt.subplots(figsize=(self.figsize[0]/self.dpi, 
                                        self.figsize[1]/self.dpi), dpi=self.dpi)
        
        # Plot closing prices
        ax.plot(data.index, data['Close'], color='blue', linewidth=2)
        
        # Add moving averages
        if len(data) > 20:
            sma20 = data['Close'].rolling(20).mean()
            ax.plot(data.index, sma20, color='orange', linewidth=1.5, alpha=0.7, 
                   label='SMA20')
        
        if len(data) > 50:
            sma50 = data['Close'].rolling(50).mean()
            ax.plot(data.index, sma50, color='red', linewidth=1.5, alpha=0.7, 
                   label='SMA50')
        
        # Formatting
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Remove x-axis labels
        ax.set_xticks([])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            return save_path
        else:
            return self.fig2img(fig)
    
    def generate_technical_chart(self, data, title=""):
        """Generate chart with multiple technical indicators"""
        fig, axes = plt.subplots(3, 1, figsize=(self.figsize[0]/self.dpi * 1.5,
                                                self.figsize[1]/self.dpi), 
                                 dpi=self.dpi, gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Price chart with Bollinger Bands
        ax1 = axes[0]
        ax1.plot(data.index, data['Close'], color='black', linewidth=1.5, label='Close')
        
        # Bollinger Bands
        sma20 = data['Close'].rolling(20).mean()
        std20 = data['Close'].rolling(20).std()
        upper_band = sma20 + 2 * std20
        lower_band = sma20 - 2 * std20
        
        ax1.plot(data.index, sma20, color='blue', linewidth=1, alpha=0.7, label='SMA20')
        ax1.plot(data.index, upper_band, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax1.plot(data.index, lower_band, color='green', linestyle='--', linewidth=1, alpha=0.5)
        ax1.fill_between(data.index, lower_band, upper_band, alpha=0.1)
        
        ax1.set_title(title, fontsize=12, fontweight='bold')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks([])
        
        # RSI
        ax2 = axes[1]
        rsi = self.calculate_rsi(data['Close'])
        ax2.plot(data.index, rsi, color='purple', linewidth=1.5)
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks([])
        
        # Volume
        ax3 = axes[2]
        colors = ['green' if close >= open else 'red' 
                 for close, open in zip(data['Close'], data['Open'])]
        ax3.bar(range(len(data)), data['Volume'], color=colors, alpha=0.7)
        ax3.set_ylabel('Volume')
        ax3.set_xlabel('Time')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks([])
        
        plt.tight_layout()
        
        return self.fig2img(fig)
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

# ============================================================================
# PART 2: CUSTOM DATASET FOR FINANCIAL IMAGES (FIXED)
# ============================================================================

class FinancialImageDataset(Dataset):
    """
    Custom dataset for financial chart images
    Fixed version with proper image channel handling
    """
    
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images: List of numpy arrays (images)
            labels: List of corresponding labels
            transform: Optional transform to be applied
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image (already as numpy array)
        image = self.images[idx]
        
        # Convert to PIL Image and ensure RGB (3 channels)
        if len(image.shape) == 3 and image.shape[2] == 4:
            # RGBA image - convert to RGB
            image = Image.fromarray(image).convert('RGB')
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # Already RGB
            image = Image.fromarray(image)
        elif len(image.shape) == 2:
            # Grayscale - convert to RGB
            image = Image.fromarray(image).convert('RGB')
        else:
            # Fallback
            image = Image.fromarray(image).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label

# ============================================================================
# PART 3: CNN ARCHITECTURES FOR FINANCIAL IMAGES
# ============================================================================

class ChartPatternCNN(nn.Module):
    """
    Custom CNN for chart pattern recognition
    """
    
    def __init__(self, num_classes=3):
        super(ChartPatternCNN, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Conv block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Calculate the size after convolutions
        self._to_linear = None
        self._get_conv_output()
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self._to_linear, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def _get_conv_output(self):
        """Calculate the size after convolutional layers"""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            dummy = self.conv_layers(dummy)
            self._to_linear = dummy.view(1, -1).size(1)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ============================================================================
# PART 4: TRAINING AND EVALUATION
# ============================================================================

class FinanceCNNTrainer:
    """
    Trainer for financial CNN models
    """
    
    def __init__(self, model, device=device):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, dataloader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        batches = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            try:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batches += 1
                
                # Print progress for first batch
                if batch_idx == 0 and batches == 1:
                    print(f"First batch - Loss: {loss.item():.4f}")
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
            
        return total_loss / max(batches, 1)
    
    def validate(self, dataloader, criterion):
        """Validate the model"""
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0
        batches = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                try:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    batches += 1
                    
                except Exception as e:
                    print(f"Error in validation: {e}")
                    continue
        
        accuracy = 100 * correct / total if total > 0 else 0
        return val_loss / max(batches, 1), accuracy
    
    def train(self, train_loader, val_loader, epochs=50, learning_rate=0.001):
        """Full training loop"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        print("\nStarting training...")
        print("-" * 60)
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            self.val_accuracies.append(val_acc)
            
            scheduler.step()
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.2f}%")
        
        print("Training complete!")
        
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Training Loss', color='blue')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# ============================================================================
# PART 5: SYNTHETIC DATASET GENERATION
# ============================================================================

def create_synthetic_dataset(n_samples=200):
    """Create synthetic dataset for demonstration"""
    chart_gen = FinancialChartGenerator()
    images = []
    labels = []
    
    print("Generating synthetic chart images...")
    
    for i in range(n_samples):
        try:
            # Generate random stock data
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            
            # Create realistic price movements
            returns = np.random.randn(30) * 0.02
            price = 100 * np.exp(np.cumsum(returns))
            
            # Create OHLC data
            data = pd.DataFrame({
                'Open': price * (1 + np.random.randn(30) * 0.005),
                'High': price * (1 + np.abs(np.random.randn(30) * 0.01)),
                'Low': price * (1 - np.abs(np.random.randn(30) * 0.01)),
                'Close': price * (1 + np.random.randn(30) * 0.002),
                'Volume': np.random.randint(1000000, 10000000, 30)
            }, index=dates)
            
            # Ensure High is always highest and Low is always lowest
            data['High'] = np.maximum(data['High'], data['Open'], data['Close'])
            data['Low'] = np.minimum(data['Low'], data['Open'], data['Close'])
            
            # Generate chart
            chart_img = chart_gen.generate_technical_chart(data)
            
            # Skip if image generation failed
            if chart_img is None or chart_img.size == 0:
                continue
                
            images.append(chart_img)
            
            # Create synthetic label based on recent trend
            if price[-1] > price[0] * 1.05:
                labels.append(0)  # uptrend
            elif price[-1] < price[0] * 0.95:
                labels.append(1)  # downtrend
            else:
                labels.append(2)  # sideways
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1} images")
                
        except Exception as e:
            print(f"Error generating image {i}: {e}")
            continue
    
    print(f"Successfully generated {len(images)} images")
    return images, labels

# ============================================================================
# PART 6: MAIN EXECUTION
# ============================================================================

def main():
    """Main function to demonstrate CNN for finance"""
    
    print("="*80)
    print("CNN FOR FINANCE - IMAGE-BASED FINANCIAL ANALYSIS")
    print("="*80)
    
    # Part 1: Create dataset
    print("\n1. Creating synthetic financial dataset...")
    images, labels = create_synthetic_dataset(n_samples=200)
    
    if len(images) == 0:
        print("Failed to generate images. Exiting.")
        return
    
    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"Train: {len(X_train)} images")
    print(f"Validation: {len(X_val)} images")
    print(f"Test: {len(X_test)} images")
    
    # Part 2: Define transforms
    print("\n2. Preparing data loaders...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = FinancialImageDataset(X_train, y_train, transform)
    val_dataset = FinancialImageDataset(X_val, y_val, transform)
    test_dataset = FinancialImageDataset(X_test, y_test, transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Part 3: Create and train model
    print("\n3. Training CNN model...")
    
    model = ChartPatternCNN(num_classes=3)
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    trainer = FinanceCNNTrainer(model)
    trainer.train(train_loader, val_loader, epochs=20, learning_rate=0.001)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Part 4: Evaluate on test set
    print("\n4. Evaluating on test set...")
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = trainer.validate(test_loader, criterion)
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)

# ============================================================================
# BONUS: VISUALIZE SAMPLE IMAGES
# ============================================================================

def visualize_samples(images, labels, num_samples=5):
    """Visualize sample images from the dataset"""
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    class_names = ['Uptrend', 'Downtrend', 'Sideways']
    
    for i in range(num_samples):
        idx = np.random.randint(0, len(images))
        axes[i].imshow(images[idx])
        axes[i].set_title(f"{class_names[labels[idx]]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()
    
    # Uncomment to visualize samples
    # print("\nVisualizing sample images...")
    # images, labels = create_synthetic_dataset(n_samples=10)
    # visualize_samples(images, labels)