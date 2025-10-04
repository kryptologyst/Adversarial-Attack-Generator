"""
Configuration management for the adversarial attack generator.
"""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import yaml


@dataclass
class ModelConfig:
    """Configuration for model architecture and training."""
    model_type: str = 'modern'  # 'modern' or 'simple'
    num_classes: int = 10
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 64
    num_epochs: int = 10
    weight_decay: float = 1e-4
    scheduler_step_size: int = 7
    scheduler_gamma: float = 0.1


@dataclass
class AttackConfig:
    """Configuration for adversarial attacks."""
    # FGSM
    fgsm_epsilon: float = 0.25
    
    # PGD
    pgd_epsilon: float = 0.25
    pgd_alpha: float = 0.01
    pgd_num_iter: int = 40
    
    # BIM
    bim_epsilon: float = 0.25
    bim_alpha: float = 0.01
    bim_num_iter: int = 10
    
    # C&W
    cw_c: float = 1.0
    cw_kappa: float = 0.0
    cw_max_iter: int = 1000
    cw_lr: float = 0.01
    
    # DeepFool
    deepfool_max_iter: int = 50
    deepfool_overshoot: float = 0.02


@dataclass
class UIConfig:
    """Configuration for the web UI."""
    page_title: str = "Adversarial Attack Generator"
    page_icon: str = "🎯"
    layout: str = "wide"
    theme: str = "light"
    max_upload_size: int = 10  # MB
    supported_formats: list = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['png', 'jpg', 'jpeg']


@dataclass
class DatabaseConfig:
    """Configuration for database settings."""
    db_path: str = "adversarial_attacks.db"
    backup_interval: int = 24  # hours
    max_records: int = 10000
    cleanup_old_records: bool = True


@dataclass
class AppConfig:
    """Main application configuration."""
    model: ModelConfig
    attack: AttackConfig
    ui: UIConfig
    database: DatabaseConfig
    
    # General settings
    device: str = 'auto'  # 'auto', 'cpu', 'cuda'
    data_dir: str = './data'
    checkpoint_dir: str = './checkpoints'
    log_level: str = 'INFO'
    random_seed: int = 42
    
    def __post_init__(self):
        # Auto-detect device
        if self.device == 'auto':
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ConfigManager:
    """Configuration manager for the application."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> AppConfig:
        """Load configuration from file or create default."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
                return self._dict_to_config(config_dict)
            except Exception as e:
                print(f"Error loading config: {e}")
                print("Using default configuration")
        
        # Create default configuration
        default_config = AppConfig(
            model=ModelConfig(),
            attack=AttackConfig(),
            ui=UIConfig(),
            database=DatabaseConfig()
        )
        
        # Save default config
        self.save_config(default_config)
        return default_config
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> AppConfig:
        """Convert dictionary to AppConfig object."""
        return AppConfig(
            model=ModelConfig(**config_dict.get('model', {})),
            attack=AttackConfig(**config_dict.get('attack', {})),
            ui=UIConfig(**config_dict.get('ui', {})),
            database=DatabaseConfig(**config_dict.get('database', {})),
            **{k: v for k, v in config_dict.items() 
               if k not in ['model', 'attack', 'ui', 'database']}
        )
    
    def save_config(self, config: Optional[AppConfig] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save (uses current config if None)
        """
        if config is None:
            config = self.config
        
        config_dict = asdict(config)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def get_config(self) -> AppConfig:
        """Get current configuration."""
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """
        Update configuration with new values.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                print(f"Warning: Unknown configuration parameter: {key}")
        
        self.save_config()
    
    def reset_to_default(self) -> None:
        """Reset configuration to default values."""
        default_config = AppConfig(
            model=ModelConfig(),
            attack=AttackConfig(),
            ui=UIConfig(),
            database=DatabaseConfig()
        )
        self.config = default_config
        self.save_config()
    
    def validate_config(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Validate model config
            assert self.config.model.model_type in ['modern', 'simple']
            assert 0 < self.config.model.dropout_rate < 1
            assert self.config.model.learning_rate > 0
            assert self.config.model.batch_size > 0
            assert self.config.model.num_epochs > 0
            
            # Validate attack config
            assert self.config.attack.fgsm_epsilon > 0
            assert self.config.attack.pgd_epsilon > 0
            assert self.config.attack.pgd_alpha > 0
            assert self.config.attack.pgd_num_iter > 0
            
            # Validate UI config
            assert self.config.ui.max_upload_size > 0
            assert all(fmt in ['png', 'jpg', 'jpeg', 'bmp', 'tiff'] 
                      for fmt in self.config.ui.supported_formats)
            
            # Validate database config
            assert self.config.database.max_records > 0
            assert self.config.database.backup_interval > 0
            
            return True
            
        except AssertionError as e:
            print(f"Configuration validation failed: {e}")
            return False
    
    def export_config(self, export_path: str) -> None:
        """
        Export configuration to a different file.
        
        Args:
            export_path: Path to export configuration to
        """
        config_dict = asdict(self.config)
        
        with open(export_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def import_config(self, import_path: str) -> None:
        """
        Import configuration from a file.
        
        Args:
            import_path: Path to import configuration from
        """
        try:
            with open(import_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            self.config = self._dict_to_config(config_dict)
            self.save_config()
            print(f"Configuration imported from {import_path}")
            
        except Exception as e:
            print(f"Error importing configuration: {e}")


# Global configuration instance
config_manager = ConfigManager()


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config_manager.get_config()


def update_config(**kwargs) -> None:
    """Update the global configuration."""
    config_manager.update_config(**kwargs)


def save_config() -> None:
    """Save the current configuration."""
    config_manager.save_config()


def reset_config() -> None:
    """Reset configuration to default values."""
    config_manager.reset_to_default()


# Example usage and configuration templates
def create_example_configs():
    """Create example configuration files for different use cases."""
    
    # Development configuration
    dev_config = AppConfig(
        model=ModelConfig(
            model_type='simple',
            learning_rate=0.01,
            batch_size=32,
            num_epochs=5
        ),
        attack=AttackConfig(
            fgsm_epsilon=0.1,
            pgd_epsilon=0.1,
            pgd_num_iter=10
        ),
        ui=UIConfig(
            theme='light',
            max_upload_size=5
        ),
        database=DatabaseConfig(
            max_records=1000
        ),
        log_level='DEBUG'
    )
    
    # Production configuration
    prod_config = AppConfig(
        model=ModelConfig(
            model_type='modern',
            learning_rate=0.001,
            batch_size=128,
            num_epochs=20
        ),
        attack=AttackConfig(
            fgsm_epsilon=0.25,
            pgd_epsilon=0.25,
            pgd_num_iter=40
        ),
        ui=UIConfig(
            theme='light',
            max_upload_size=20
        ),
        database=DatabaseConfig(
            max_records=50000,
            backup_interval=12
        ),
        log_level='INFO'
    )
    
    # Save example configurations
    config_manager.export_config('config_dev.yaml')
    config_manager.config = dev_config
    config_manager.save_config()
    
    config_manager.export_config('config_prod.yaml')
    config_manager.config = prod_config
    config_manager.save_config()
    
    print("Example configurations created:")
    print("- config_dev.yaml (Development)")
    print("- config_prod.yaml (Production)")


if __name__ == "__main__":
    # Test configuration manager
    config_manager = ConfigManager()
    
    print("Current configuration:")
    print(f"Model type: {config_manager.get_config().model.model_type}")
    print(f"Device: {config_manager.get_config().device}")
    print(f"FGSM epsilon: {config_manager.get_config().attack.fgsm_epsilon}")
    
    # Validate configuration
    if config_manager.validate_config():
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration validation failed")
    
    # Create example configurations
    create_example_configs()
