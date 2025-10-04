"""
Training script for MNIST CNN with adversarial robustness considerations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple

from models.cnn_model import create_model, save_model
from attacks.adversarial_attacks import AdversarialAttacks


class Trainer:
    """
    Trainer class for training CNN models with adversarial robustness evaluation.
    """
    
    def __init__(self, 
                 model_type: str = 'modern',
                 device: str = 'cpu',
                 learning_rate: float = 0.001,
                 batch_size: int = 64,
                 num_epochs: int = 10,
                 save_dir: str = './checkpoints'):
        """
        Initialize the trainer.
        
        Args:
            model_type: Type of model to train ('modern' or 'simple')
            device: Device to train on
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            save_dir: Directory to save model checkpoints
        """
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize model
        self.model = create_model(model_type).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Load datasets
        self._load_datasets()
    
    def _load_datasets(self):
        """Load MNIST datasets with data augmentation."""
        # Data augmentation for training
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])
        
        # No augmentation for validation/test
        val_transform = transforms.ToTensor()
        
        # Load datasets
        self.train_dataset = datasets.MNIST(
            root='./data', train=True, download=True, transform=train_transform
        )
        self.val_dataset = datasets.MNIST(
            root='./data', train=False, download=True, transform=val_transform
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100. * correct / total:.2f}%')
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self) -> Dict:
        """
        Train the model for specified number of epochs.
        
        Returns:
            Dictionary with training history
        """
        print(f"Starting training on {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Epochs: {self.num_epochs}, Batch size: {self.batch_size}")
        print("-" * 50)
        
        best_val_acc = 0.0
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Store history
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = os.path.join(self.save_dir, 'best_model.pth')
                save_model(self.model, model_path)
                print(f"New best model saved! Val Acc: {val_acc:.2f}%")
        
        # Save final model
        final_model_path = os.path.join(self.save_dir, 'final_model.pth')
        save_model(self.model, final_model_path)
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': best_val_acc,
            'final_val_acc': self.val_accuracies[-1]
        }
        
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Final validation accuracy: {self.val_accuracies[-1]:.2f}%")
        
        return history
    
    def plot_training_history(self, save_path: str = None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def evaluate_adversarial_robustness(self, epsilon_values: List[float] = None) -> Dict:
        """
        Evaluate model robustness against adversarial attacks.
        
        Args:
            epsilon_values: List of epsilon values to test
        
        Returns:
            Dictionary with robustness evaluation results
        """
        if epsilon_values is None:
            epsilon_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        print("\nEvaluating adversarial robustness...")
        
        attacks = AdversarialAttacks(self.model, self.device)
        results = {}
        
        attack_methods = ['fgsm', 'pgd', 'bim']
        
        for method in attack_methods:
            print(f"\nTesting {method.upper()} attack...")
            results[method] = {}
            
            for epsilon in epsilon_values:
                success_rate, accuracy_drop = attacks.attack_success_rate(
                    self.val_loader, method, epsilon
                )
                results[method][epsilon] = {
                    'success_rate': success_rate,
                    'accuracy_drop': accuracy_drop
                }
                print(f"  Epsilon {epsilon}: Success Rate {success_rate:.2%}, "
                      f"Accuracy Drop {accuracy_drop:.2%}")
        
        # Save results
        results_path = os.path.join(self.save_dir, 'adversarial_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results


def main():
    """Main training function."""
    # Configuration
    config = {
        'model_type': 'modern',  # 'modern' or 'simple'
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'learning_rate': 0.001,
        'batch_size': 64,
        'num_epochs': 10,
        'save_dir': './checkpoints'
    }
    
    print("Adversarial Attack Generator - Model Training")
    print("=" * 50)
    print(f"Device: {config['device']}")
    print(f"Model type: {config['model_type']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['num_epochs']}")
    print("=" * 50)
    
    # Initialize trainer
    trainer = Trainer(**config)
    
    # Train model
    history = trainer.train()
    
    # Plot training history
    trainer.plot_training_history(os.path.join(config['save_dir'], 'training_plots.png'))
    
    # Evaluate adversarial robustness
    robustness_results = trainer.evaluate_adversarial_robustness()
    
    print("\nTraining and evaluation completed!")
    print(f"Results saved in: {config['save_dir']}")


if __name__ == "__main__":
    main()
