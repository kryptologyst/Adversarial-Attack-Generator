"""
Modern CNN architecture for MNIST classification with adversarial robustness considerations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModernCNN(nn.Module):
    """
    Modern CNN architecture with batch normalization, dropout, and residual connections.
    Designed for MNIST classification with improved robustness.
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.2):
        super(ModernCNN, self).__init__()
        
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(dropout_rate)
        self.dropout_fc = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # First convolutional block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        
        # Second convolutional block
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        
        # Third convolutional block
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)
        x = self.fc3(x)
        
        return x


class SimpleCNN(nn.Module):
    """
    Simplified CNN for backward compatibility and quick training.
    """
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3), 
            nn.ReLU(), 
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), 
            nn.ReLU(), 
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128), 
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


def create_model(model_type='modern', num_classes=10, **kwargs):
    """
    Factory function to create different model architectures.
    
    Args:
        model_type (str): Type of model to create ('modern' or 'simple')
        num_classes (int): Number of output classes
        **kwargs: Additional arguments for model creation
    
    Returns:
        torch.nn.Module: The created model
    """
    if model_type == 'modern':
        return ModernCNN(num_classes=num_classes, **kwargs)
    elif model_type == 'simple':
        return SimpleCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_pretrained_model(model_path, model_type='modern', device='cpu'):
    """
    Load a pre-trained model from file.
    
    Args:
        model_path (str): Path to the model file
        model_type (str): Type of model architecture
        device (str): Device to load the model on
    
    Returns:
        torch.nn.Module: Loaded model
    """
    model = create_model(model_type)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def save_model(model, model_path):
    """
    Save model to file.
    
    Args:
        model (torch.nn.Module): Model to save
        model_path (str): Path to save the model
    """
    torch.save(model.state_dict(), model_path)
