"""
Basic tests for the adversarial attack generator.
"""

import pytest
import torch
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_model import create_model, SimpleCNN, ModernCNN
from attacks.adversarial_attacks import AdversarialAttacks
from config.config_manager import ConfigManager, AppConfig


class TestModels:
    """Test model architectures."""
    
    def test_simple_cnn_creation(self):
        """Test SimpleCNN model creation."""
        model = create_model('simple')
        assert isinstance(model, SimpleCNN)
        assert model.net is not None
    
    def test_modern_cnn_creation(self):
        """Test ModernCNN model creation."""
        model = create_model('modern')
        assert isinstance(model, ModernCNN)
        assert hasattr(model, 'conv1')
        assert hasattr(model, 'bn1')
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        model = create_model('simple')
        x = torch.randn(1, 1, 28, 28)
        output = model(x)
        assert output.shape == (1, 10)
    
    def test_model_device_placement(self):
        """Test model device placement."""
        device = 'cpu'
        model = create_model('simple').to(device)
        assert next(model.parameters()).device.type == device


class TestAdversarialAttacks:
    """Test adversarial attack methods."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.model = create_model('simple')
        self.attacks = AdversarialAttacks(self.model, 'cpu')
        self.sample_image = torch.randn(1, 1, 28, 28)
        self.sample_label = torch.tensor([5])
    
    def test_fgsm_attack(self):
        """Test FGSM attack functionality."""
        epsilon = 0.25
        adv_image = self.attacks.fgsm_attack(
            self.sample_image, 
            self.sample_label, 
            epsilon
        )
        
        # Check output shape
        assert adv_image.shape == self.sample_image.shape
        
        # Check pixel range
        assert torch.all(adv_image >= 0)
        assert torch.all(adv_image <= 1)
    
    def test_pgd_attack(self):
        """Test PGD attack functionality."""
        epsilon = 0.25
        alpha = 0.01
        num_iter = 5  # Small number for testing
        
        adv_image = self.attacks.pgd_attack(
            self.sample_image,
            self.sample_label,
            epsilon,
            alpha,
            num_iter
        )
        
        # Check output shape
        assert adv_image.shape == self.sample_image.shape
        
        # Check pixel range
        assert torch.all(adv_image >= 0)
        assert torch.all(adv_image <= 1)
    
    def test_bim_attack(self):
        """Test BIM attack functionality."""
        epsilon = 0.25
        alpha = 0.01
        num_iter = 5
        
        adv_image = self.attacks.bim_attack(
            self.sample_image,
            self.sample_label,
            epsilon,
            alpha,
            num_iter
        )
        
        # Check output shape
        assert adv_image.shape == self.sample_image.shape
        
        # Check pixel range
        assert torch.all(adv_image >= 0)
        assert torch.all(adv_image <= 1)
    
    def test_deepfool_attack(self):
        """Test DeepFool attack functionality."""
        adv_image = self.attacks.deepfool_attack(
            self.sample_image,
            self.sample_label
        )
        
        # Check output shape
        assert adv_image.shape == self.sample_image.shape
        
        # Check pixel range
        assert torch.all(adv_image >= 0)
        assert torch.all(adv_image <= 1)
    
    def test_cw_attack(self):
        """Test C&W attack functionality."""
        c = 1.0
        max_iter = 10  # Small number for testing
        
        adv_image = self.attacks.cw_attack(
            self.sample_image,
            self.sample_label,
            c=c,
            max_iter=max_iter
        )
        
        # Check output shape
        assert adv_image.shape == self.sample_image.shape
        
        # Check pixel range
        assert torch.all(adv_image >= 0)
        assert torch.all(adv_image <= 1)


class TestConfiguration:
    """Test configuration management."""
    
    def test_config_manager_creation(self):
        """Test ConfigManager creation."""
        config_manager = ConfigManager()
        assert isinstance(config_manager, ConfigManager)
    
    def test_default_config(self):
        """Test default configuration."""
        config_manager = ConfigManager()
        config = config_manager.get_config()
        assert isinstance(config, AppConfig)
        assert config.model.model_type in ['modern', 'simple']
        assert config.attack.fgsm_epsilon > 0
    
    def test_config_validation(self):
        """Test configuration validation."""
        config_manager = ConfigManager()
        assert config_manager.validate_config() is True
    
    def test_config_update(self):
        """Test configuration update."""
        config_manager = ConfigManager()
        config_manager.update_config(device='cpu')
        config = config_manager.get_config()
        assert config.device == 'cpu'


class TestUtilities:
    """Test utility functions."""
    
    def test_tensor_operations(self):
        """Test basic tensor operations."""
        x = torch.randn(1, 1, 28, 28)
        y = torch.randn(1, 1, 28, 28)
        
        # Test clamping
        clamped = torch.clamp(x, 0, 1)
        assert torch.all(clamped >= 0)
        assert torch.all(clamped <= 1)
        
        # Test sign operation
        sign_x = torch.sign(x)
        assert torch.all(torch.abs(sign_x) <= 1)
    
    def test_numpy_conversion(self):
        """Test tensor to numpy conversion."""
        x = torch.randn(1, 1, 28, 28)
        x_np = x.numpy()
        assert isinstance(x_np, np.ndarray)
        assert x_np.shape == (1, 1, 28, 28)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
