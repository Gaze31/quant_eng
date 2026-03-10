"""
Comprehensive Neural Network Training Examples in Python
Includes multiple architectures, datasets, and visualization techniques
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PYTORCH IMPORTS
# ============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# EXAMPLE 1: BASIC CLASSIFICATION WITH MOON DATASET
# ============================================================================

class SimpleClassifier(nn.Module):
    """
    Simple neural network for binary classification
    """
    def __init__(self, input_size=2, hidden_size=64, output_size=1):
        super(SimpleClassifier, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size//2, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

def example_binary_classification():
    """
    Binary classification on moon dataset
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: BINARY CLASSIFICATION - MOON DATASET")
    print("="*70)
    
    # Generate dataset
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = SimpleClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    train_losses = []
    test_accuracies = []
    
    print("\nTraining Progress:")
    print("-" * 50)
    
    for epoch in range(100):
        # Training
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor.to(device))
            test_preds = (test_outputs > 0.5).float()
            accuracy = (test_preds.cpu() == y_test_tensor).float().mean()
            test_accuracies.append(accuracy.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/100], Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_outputs = model(X_test_tensor.to(device))
        final_preds = (final_outputs > 0.5).float().cpu().numpy()
    
    final_accuracy = accuracy_score(y_test, final_preds)
    print(f"\nFinal Test Accuracy: {final_accuracy:.4f}")
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Training loss
    axes[0].plot(train_losses, label='Training Loss', color='blue')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Test accuracy
    axes[1].plot(test_accuracies, label='Test Accuracy', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Test Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Decision boundary
    plot_decision_boundary(model, X_test, y_test, scaler, axes[2])
    axes[2].set_title('Decision Boundary')
    
    plt.tight_layout()
    plt.show()
    
    return model

def plot_decision_boundary(model, X, y, scaler, ax):
    """Plot decision boundary for binary classification"""
    # Create mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Predict on mesh points
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_tensor = torch.FloatTensor(mesh_points).to(device)
    
    model.eval()
    with torch.no_grad():
        Z = model(mesh_tensor)
        Z = (Z > 0.5).float().cpu().numpy()
    
    Z = Z.reshape(xx.shape)
    
    # Plot
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

# ============================================================================
# EXAMPLE 2: MULTI-CLASS CLASSIFICATION
# ============================================================================

class MultiClassClassifier(nn.Module):
    """
    Neural network for multi-class classification
    """
    def __init__(self, input_size=4, hidden_size=128, num_classes=3):
        super(MultiClassClassifier, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size//2),
            nn.BatchNorm1d(hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.BatchNorm1d(hidden_size//4),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size//4, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

def example_multiclass_classification():
    """
    Multi-class classification on Iris dataset
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: MULTI-CLASS CLASSIFICATION - IRIS DATASET")
    print("="*70)
    
    # Load Iris dataset from sklearn
    from sklearn.datasets import load_iris
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Initialize model
    model = MultiClassClassifier(input_size=4, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Training loop
    train_losses = []
    test_accuracies = []
    
    print("\nTraining Progress:")
    print("-" * 50)
    
    for epoch in range(100):
        # Training
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor.to(device))
            _, test_preds = torch.max(test_outputs, 1)
            accuracy = (test_preds.cpu() == y_test_tensor).float().mean()
            test_accuracies.append(accuracy.item())
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/100], Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_outputs = model(X_test_tensor.to(device))
        _, final_preds = torch.max(final_outputs, 1)
        final_preds = final_preds.cpu().numpy()
    
    final_accuracy = accuracy_score(y_test, final_preds)
    print(f"\nFinal Test Accuracy: {final_accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, final_preds, target_names=iris.target_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, final_preds)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Training loss
    axes[0, 0].plot(train_losses, color='blue')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Test accuracy
    axes[0, 1].plot(test_accuracies, color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Test Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=iris.target_names, 
                yticklabels=iris.target_names,
                ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    
    # Plot 4: Feature importance (using first two features for visualization)
    plot_decision_regions(model, X_test, y_test, scaler, iris, axes[1, 1])
    
    plt.tight_layout()
    plt.show()
    
    return model

def plot_decision_regions(model, X, y, scaler, iris, ax):
    """Plot decision regions for first two features"""
    # Use only first two features for visualization
    X_2d = X[:, :2]
    
    # Create mesh
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Create full feature set with zeros for other features
    mesh_points_full = np.zeros((xx.ravel().shape[0], X.shape[1]))
    mesh_points_full[:, :2] = np.c_[xx.ravel(), yy.ravel()]
    
    mesh_tensor = torch.FloatTensor(mesh_points_full).to(device)
    
    model.eval()
    with torch.no_grad():
        Z = model(mesh_tensor)
        _, Z = torch.max(Z, 1)
        Z = Z.cpu().numpy()
    
    Z = Z.reshape(xx.shape)
    
    # Plot
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='Set1')
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='Set1', edgecolors='black')
    ax.set_xlabel(iris.feature_names[0])
    ax.set_ylabel(iris.feature_names[1])
    ax.set_title('Decision Regions (First 2 Features)')

# ============================================================================
# EXAMPLE 3: REGRESSION
# ============================================================================

class RegressionNN(nn.Module):
    """
    Neural network for regression
    """
    def __init__(self, input_size=1, hidden_size=128):
        super(RegressionNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def example_regression():
    """
    Regression on synthetic sine wave data
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: REGRESSION - SINE WAVE")
    print("="*70)
    
    # Generate synthetic data
    X = np.linspace(-3, 3, 1000).reshape(-1, 1)
    y = np.sin(X) + 0.1 * np.random.randn(1000, 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    # Create data loader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = RegressionNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    train_losses = []
    test_losses = []
    
    print("\nTraining Progress:")
    print("-" * 50)
    
    for epoch in range(200):
        # Training
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor.to(device))
            test_loss = criterion(test_outputs, y_test_tensor.to(device))
            test_losses.append(test_loss.item())
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/200], Train Loss: {avg_train_loss:.6f}, Test Loss: {test_loss:.6f}")
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        X_all_scaled = scaler_X.transform(X)
        X_all_tensor = torch.FloatTensor(X_all_scaled).to(device)
        y_pred_scaled = model(X_all_tensor).cpu().numpy()
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Training and test loss
    axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(test_losses, label='Test Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Test Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Plot 2: Actual vs Predicted
    axes[0, 1].scatter(X, y, alpha=0.5, s=10, label='Actual', color='blue')
    axes[0, 1].plot(X, y_pred, color='red', linewidth=2, label='Predicted')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].set_title('Sine Wave Regression')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Residuals
    residuals = y - y_pred
    axes[1, 0].scatter(y_pred, residuals, alpha=0.5)
    axes[1, 0].axhline(y=0, color='red', linestyle='--')
    axes[1, 0].set_xlabel('Predicted Values')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residual Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Residual distribution
    axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Residual')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Residual Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate metrics
    mse = np.mean((y - y_pred) ** 2)
    mae = np.mean(np.abs(y - y_pred))
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    
    print(f"\nRegression Metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R² Score: {r2:.6f}")
    
    return model

# ============================================================================
# EXAMPLE 4: CNN FOR IMAGE CLASSIFICATION
# ============================================================================

class SimpleCNN(nn.Module):
    """
    Simple CNN for image classification
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Conv block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            
            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
        )
        
        # Calculate size after convolutions (for 28x28 input)
        self.fc_input_size = 128 * 3 * 3  # After 3 pooling layers
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

def example_cnn_mnist():
    """
    CNN example on MNIST dataset
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: CNN FOR IMAGE CLASSIFICATION - MNIST")
    print("="*70)
    
    # Load MNIST dataset
    from torchvision import datasets, transforms
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load MNIST
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Training loop
    train_losses = []
    test_accuracies = []
    
    print("\nTraining Progress:")
    print("-" * 50)
    
    for epoch in range(5):  # 5 epochs for demonstration
        # Training
        model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}: Test Accuracy: {accuracy:.2f}%')
    
    # Visualize results
    visualize_cnn_results(model, test_loader)
    
    return model

def visualize_cnn_results(model, test_loader):
    """Visualize CNN predictions"""
    model.eval()
    
    # Get a batch of test images
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # Make predictions
    with torch.no_grad():
        images_gpu = images.to(device)
        outputs = model(images_gpu)
        _, predicted = torch.max(outputs, 1)
    
    # Plot images with predictions
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    fig.suptitle('MNIST Predictions (Green = Correct, Red = Incorrect)', fontsize=16)
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            image = images[idx].squeeze().numpy()
            true_label = labels[idx].item()
            pred_label = predicted[idx].item()
            
            ax.imshow(image, cmap='gray')
            color = 'green' if true_label == pred_label else 'red'
            ax.set_title(f'True: {true_label}\nPred: {pred_label}', 
                        color=color, fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# EXAMPLE 5: ADVANCED TECHNIQUES
# ============================================================================

class AdvancedNN(nn.Module):
    """
    Neural network with advanced features
    """
    def __init__(self, input_size, num_classes):
        super(AdvancedNN, self).__init__()
        
        # Multiple branches
        self.branch1 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.branch2 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Tanh(),
            nn.Dropout(0.3)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
        
        # Final layers
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Two parallel branches
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        
        # Concatenate
        combined = torch.cat([out1, out2], dim=1)
        
        # Attention weights
        attention_weights = self.attention(combined)
        
        # Apply attention
        attended = combined * attention_weights[:, 0:1]
        
        # Final layers
        out = self.fc1(attended)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

def example_advanced_techniques():
    """
    Demonstrate advanced neural network techniques
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: ADVANCED TECHNIQUES")
    print("="*70)
    
    # Generate complex dataset
    X, y = make_classification(
        n_samples=2000, 
        n_features=20, 
        n_informative=15, 
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Initialize model with advanced features
    model = AdvancedNN(input_size=20, num_classes=3).to(device)
    
    # Use multiple optimizers for different parts
    optimizer1 = optim.Adam(model.branch1.parameters(), lr=0.001)
    optimizer2 = optim.Adam(model.branch2.parameters(), lr=0.001)
    optimizer3 = optim.Adam(list(model.attention.parameters()) + 
                           list(model.fc1.parameters()) + 
                           list(model.fc2.parameters()), lr=0.001)
    
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate schedulers
    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=20, gamma=0.5)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=20, gamma=0.5)
    scheduler3 = optim.lr_scheduler.StepLR(optimizer3, step_size=20, gamma=0.5)
    
    # Training loop
    train_losses = []
    test_accuracies = []
    attention_weights_history = []
    
    print("\nTraining Progress:")
    print("-" * 50)
    
    for epoch in range(50):
        # Training
        model.train()
        epoch_loss = 0
        batch_attention = []
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Zero all optimizers
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Step all optimizers
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            
            epoch_loss += loss.item()
            
            # Track attention weights
            with torch.no_grad():
                out1 = model.branch1(batch_X)
                out2 = model.branch2(batch_X)
                combined = torch.cat([out1, out2], dim=1)
                attention = model.attention(combined)
                batch_attention.append(attention.mean(dim=0).cpu().numpy())
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Average attention weights for this epoch
        avg_attention = np.mean(batch_attention, axis=0)
        attention_weights_history.append(avg_attention)
        
        # Step schedulers
        scheduler1.step()
        scheduler2.step()
        scheduler3.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor.to(device))
            _, test_preds = torch.max(test_outputs, 1)
            accuracy = (test_preds.cpu() == y_test_tensor).float().mean()
            test_accuracies.append(accuracy.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/50], Loss: {avg_loss:.4f}, "
                  f"Test Accuracy: {accuracy:.4f}, "
                  f"Attention: Branch1={avg_attention[0]:.3f}, Branch2={avg_attention[1]:.3f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Training loss
    axes[0, 0].plot(train_losses, color='blue')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Test accuracy
    axes[0, 1].plot(test_accuracies, color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Test Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Attention weights evolution
    attention_history = np.array(attention_weights_history)
    axes[1, 0].plot(attention_history[:, 0], label='Branch 1', color='blue')
    axes[1, 0].plot(attention_history[:, 1], label='Branch 2', color='red')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Attention Weight')
    axes[1, 0].set_title('Attention Weights Evolution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Final attention distribution
    axes[1, 1].bar(['Branch 1', 'Branch 2'], attention_history[-1], 
                   color=['blue', 'red'])
    axes[1, 1].set_ylabel('Attention Weight')
    axes[1, 1].set_title('Final Attention Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return model

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run all examples
    """
    print("="*80)
    print("COMPREHENSIVE NEURAL NETWORK TRAINING EXAMPLES")
    print("="*80)
    print(f"Device: {device}")
    
    # Run examples
    examples = [
        example_binary_classification,
        example_multiclass_classification,
        example_regression,
        example_cnn_mnist,
        example_advanced_techniques
    ]
    
    for i, example in enumerate(examples, 1):
        try:
            print(f"\n{'#'*80}")
            print(f"Running Example {i}")
            print(f"{'#'*80}")
            example()
        except Exception as e:
            print(f"Error in Example {i}: {e}")
            import traceback
            traceback.print_exc()
        
        # Ask user if they want to continue
        if i < len(examples):
            response = input(f"\nExample {i} complete. Run next example? (y/n): ")
            if response.lower() != 'y':
                break
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()