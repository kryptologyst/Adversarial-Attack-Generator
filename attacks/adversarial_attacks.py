"""
Comprehensive adversarial attack implementations including FGSM, PGD, C&W, and DeepFool.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union
import math


class AdversarialAttacks:
    """
    Collection of adversarial attack methods for neural networks.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize the adversarial attack class.
        
        Args:
            model: The target neural network model
            device: Device to run attacks on
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def fgsm_attack(self, 
                   image: torch.Tensor, 
                   label: torch.Tensor, 
                   epsilon: float = 0.25) -> torch.Tensor:
        """
        Fast Gradient Sign Method (FGSM) attack.
        
        Args:
            image: Input image tensor
            label: True label tensor
            epsilon: Perturbation magnitude
        
        Returns:
            Adversarial example tensor
        """
        image = image.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        
        image.requires_grad = True
        
        # Forward pass
        output = self.model(image)
        loss = F.cross_entropy(output, label)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Generate adversarial example
        data_grad = image.grad.data
        perturbed_image = image + epsilon * data_grad.sign()
        
        # Clamp to valid range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        
        return perturbed_image.detach()
    
    def pgd_attack(self, 
                  image: torch.Tensor, 
                  label: torch.Tensor, 
                  epsilon: float = 0.25, 
                  alpha: float = 0.01, 
                  num_iter: int = 40) -> torch.Tensor:
        """
        Projected Gradient Descent (PGD) attack.
        
        Args:
            image: Input image tensor
            label: True label tensor
            epsilon: Maximum perturbation magnitude
            alpha: Step size for each iteration
            num_iter: Number of iterations
        
        Returns:
            Adversarial example tensor
        """
        image = image.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        
        # Initialize adversarial example
        adv_image = image.clone().detach()
        
        for _ in range(num_iter):
            adv_image.requires_grad = True
            
            # Forward pass
            output = self.model(adv_image)
            loss = F.cross_entropy(output, label)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Update adversarial example
            data_grad = adv_image.grad.data
            adv_image = adv_image + alpha * data_grad.sign()
            
            # Project back to epsilon ball
            delta = adv_image - image
            delta = torch.clamp(delta, -epsilon, epsilon)
            adv_image = image + delta
            
            # Clamp to valid range
            adv_image = torch.clamp(adv_image, 0, 1).detach()
        
        return adv_image
    
    def cw_attack(self, 
                 image: torch.Tensor, 
                 label: torch.Tensor, 
                 c: float = 1.0, 
                 kappa: float = 0.0, 
                 max_iter: int = 1000, 
                 lr: float = 0.01) -> torch.Tensor:
        """
        Carlini & Wagner (C&W) attack.
        
        Args:
            image: Input image tensor
            label: True label tensor
            c: Confidence parameter
            kappa: Confidence threshold
            max_iter: Maximum number of iterations
            lr: Learning rate
        
        Returns:
            Adversarial example tensor
        """
        image = image.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        
        # Initialize adversarial example
        w = torch.zeros_like(image, requires_grad=True)
        optimizer = torch.optim.Adam([w], lr=lr)
        
        for _ in range(max_iter):
            # Transform w to image space
            adv_image = torch.tanh(w) * 0.5 + 0.5
            
            # Forward pass
            output = self.model(adv_image)
            
            # Compute loss
            f = torch.max(output - output.gather(1, label.unsqueeze(1)) + kappa, torch.zeros_like(output))
            f = f.sum(dim=1)
            
            loss = torch.norm(adv_image - image, p=2) + c * f
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Final adversarial example
        adv_image = torch.tanh(w) * 0.5 + 0.5
        adv_image = torch.clamp(adv_image, 0, 1)
        
        return adv_image.detach()
    
    def deepfool_attack(self, 
                       image: torch.Tensor, 
                       label: torch.Tensor, 
                       max_iter: int = 50, 
                       overshoot: float = 0.02) -> torch.Tensor:
        """
        DeepFool attack implementation.
        
        Args:
            image: Input image tensor
            label: True label tensor
            max_iter: Maximum number of iterations
            overshoot: Overshoot parameter
        
        Returns:
            Adversarial example tensor
        """
        image = image.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        
        adv_image = image.clone().detach()
        adv_image.requires_grad = True
        
        for _ in range(max_iter):
            # Forward pass
            output = self.model(adv_image)
            
            # Check if already misclassified
            pred = output.argmax(dim=1)
            if pred != label:
                break
            
            # Compute gradients
            loss = F.cross_entropy(output, label)
            self.model.zero_grad()
            loss.backward()
            
            # Compute perturbation
            grad = adv_image.grad.data
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1)
            
            # Update adversarial example
            adv_image = adv_image - (1 + overshoot) * grad / grad_norm.view(-1, 1, 1, 1)
            adv_image = torch.clamp(adv_image, 0, 1).detach()
            adv_image.requires_grad = True
        
        return adv_image.detach()
    
    def bim_attack(self, 
                  image: torch.Tensor, 
                  label: torch.Tensor, 
                  epsilon: float = 0.25, 
                  alpha: float = 0.01, 
                  num_iter: int = 10) -> torch.Tensor:
        """
        Basic Iterative Method (BIM) attack.
        
        Args:
            image: Input image tensor
            label: True label tensor
            epsilon: Maximum perturbation magnitude
            alpha: Step size for each iteration
            num_iter: Number of iterations
        
        Returns:
            Adversarial example tensor
        """
        image = image.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        
        # Initialize adversarial example
        adv_image = image.clone().detach()
        
        for _ in range(num_iter):
            adv_image.requires_grad = True
            
            # Forward pass
            output = self.model(adv_image)
            loss = F.cross_entropy(output, label)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Update adversarial example
            data_grad = adv_image.grad.data
            adv_image = adv_image + alpha * data_grad.sign()
            
            # Project back to epsilon ball
            delta = adv_image - image
            delta = torch.clamp(delta, -epsilon, epsilon)
            adv_image = image + delta
            
            # Clamp to valid range
            adv_image = torch.clamp(adv_image, 0, 1).detach()
        
        return adv_image
    
    def attack_success_rate(self, 
                           test_loader, 
                           attack_method: str = 'fgsm', 
                           epsilon: float = 0.25, 
                           **kwargs) -> Tuple[float, float]:
        """
        Calculate attack success rate on a test dataset.
        
        Args:
            test_loader: DataLoader for test dataset
            attack_method: Name of attack method to use
            epsilon: Perturbation magnitude
            **kwargs: Additional arguments for attack method
        
        Returns:
            Tuple of (success_rate, accuracy_drop)
        """
        correct_original = 0
        correct_adversarial = 0
        total = 0
        
        attack_func = getattr(self, f"{attack_method}_attack")
        
        for data, target in test_loader:
            data, target = data.to(self.device), target.to(self.device)
            
            # Original predictions
            with torch.no_grad():
                original_output = self.model(data)
                original_pred = original_output.argmax(dim=1)
                correct_original += (original_pred == target).sum().item()
            
            # Adversarial predictions
            adv_data = attack_func(data, target, epsilon, **kwargs)
            with torch.no_grad():
                adv_output = self.model(adv_data)
                adv_pred = adv_output.argmax(dim=1)
                correct_adversarial += (adv_pred == target).sum().item()
            
            total += target.size(0)
        
        original_accuracy = correct_original / total
        adversarial_accuracy = correct_adversarial / total
        success_rate = 1 - adversarial_accuracy
        accuracy_drop = original_accuracy - adversarial_accuracy
        
        return success_rate, accuracy_drop


def evaluate_attack_robustness(model: nn.Module, 
                             test_loader, 
                             device: str = 'cpu',
                             epsilon_values: list = [0.1, 0.2, 0.3, 0.4, 0.5]) -> dict:
    """
    Evaluate model robustness against different attack methods and epsilon values.
    
    Args:
        model: The neural network model
        test_loader: DataLoader for test dataset
        device: Device to run evaluation on
        epsilon_values: List of epsilon values to test
    
    Returns:
        Dictionary with evaluation results
    """
    attacks = AdversarialAttacks(model, device)
    results = {}
    
    attack_methods = ['fgsm', 'pgd', 'bim']
    
    for method in attack_methods:
        results[method] = {}
        for epsilon in epsilon_values:
            success_rate, accuracy_drop = attacks.attack_success_rate(
                test_loader, method, epsilon
            )
            results[method][epsilon] = {
                'success_rate': success_rate,
                'accuracy_drop': accuracy_drop
            }
    
    return results
