"""
Unit tests for ConfigManager class.
"""

import logging
import tempfile
from pathlib import Path

import pytest
import yaml

from src.config_manager import ConfigManager


class TestConfigManager:
    """Test suite for ConfigManager class."""
    
    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        # Create a temporary config file
        config_data = {
            'data': {
                'sampling_rate': 256,
                'epoch_duration': 4.0,
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15
            },
            'model': {
                'patch_size': 32,
                'embedding_dim': 128
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            manager = ConfigManager()
            loaded_config = manager.load_config(temp_path)
            
            assert loaded_config == config_data
            assert manager.config == config_data
        finally:
            Path(temp_path).unlink()
    
    def test_load_nonexistent_config(self):
        """Test loading a non-existent configuration file."""
        manager = ConfigManager()
        
        with pytest.raises(FileNotFoundError):
            manager.load_config('nonexistent_config.yaml')
    
    def test_validate_valid_config(self):
        """Test validation of a valid configuration."""
        config = {
            'data': {
                'sampling_rate': 256,
                'epoch_duration': 4.0,
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15
            },
            'preprocessing': {
                'bandpass_low': 0.5,
                'bandpass_high': 45.0
            },
            'model': {
                'patch_size': 32,
                'embedding_dim': 128,
                'num_heads': 8,
                'num_layers': 6,
                'dropout': 0.1
            },
            'training': {
                'optimizer': 'adamw',
                'learning_rate': 0.0001,
                'batch_size': 32,
                'num_epochs': 100,
                'k_folds': 5
            }
        }
        
        manager = ConfigManager()
        manager.config = config
        
        assert manager.validate_config() is True
    
    def test_validate_invalid_split_ratios(self):
        """Test validation fails for invalid split ratios."""
        config = {
            'data': {
                'train_ratio': 0.5,
                'val_ratio': 0.3,
                'test_ratio': 0.3  # Sum = 1.1, should fail
            }
        }
        
        manager = ConfigManager()
        manager.config = config
        
        assert manager.validate_config() is False
    
    def test_validate_negative_learning_rate(self):
        """Test validation fails for negative learning rate."""
        config = {
            'training': {
                'learning_rate': -0.001
            }
        }
        
        manager = ConfigManager()
        manager.config = config
        
        assert manager.validate_config() is False
    
    def test_validate_invalid_optimizer(self):
        """Test validation fails for invalid optimizer name."""
        config = {
            'training': {
                'optimizer': 'invalid_optimizer'
            }
        }
        
        manager = ConfigManager()
        manager.config = config
        
        assert manager.validate_config() is False
    
    def test_validate_invalid_bandpass_frequencies(self):
        """Test validation fails when bandpass_low >= bandpass_high."""
        config = {
            'preprocessing': {
                'bandpass_low': 50.0,
                'bandpass_high': 45.0
            }
        }
        
        manager = ConfigManager()
        manager.config = config
        
        assert manager.validate_config() is False
    
    def test_get_data_config(self):
        """Test retrieving data configuration section."""
        config = {
            'data': {
                'sampling_rate': 256,
                'epoch_duration': 4.0
            }
        }
        
        manager = ConfigManager()
        manager.config = config
        
        data_config = manager.get_data_config()
        assert data_config == config['data']
    
    def test_get_model_config(self):
        """Test retrieving model configuration section."""
        config = {
            'model': {
                'patch_size': 32,
                'embedding_dim': 128
            }
        }
        
        manager = ConfigManager()
        manager.config = config
        
        model_config = manager.get_model_config()
        assert model_config == config['model']
    
    def test_get_training_config(self):
        """Test retrieving training configuration section."""
        config = {
            'training': {
                'learning_rate': 0.0001,
                'batch_size': 32
            }
        }
        
        manager = ConfigManager()
        manager.config = config
        
        training_config = manager.get_training_config()
        assert training_config == config['training']
    
    def test_set_random_seeds(self):
        """Test setting random seeds for reproducibility."""
        import random
        import numpy as np
        
        # Set seed and generate random numbers
        ConfigManager.set_random_seeds(42)
        random_val_1 = random.random()
        numpy_val_1 = np.random.rand()
        
        # Reset seed and generate again
        ConfigManager.set_random_seeds(42)
        random_val_2 = random.random()
        numpy_val_2 = np.random.rand()
        
        # Values should be identical with same seed
        assert random_val_1 == random_val_2
        assert numpy_val_1 == numpy_val_2
    
    def test_save_config(self):
        """Test saving configuration to file."""
        config = {
            'data': {
                'sampling_rate': 256
            },
            'model': {
                'embedding_dim': 128
            }
        }
        
        manager = ConfigManager()
        manager.config = config
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'test_config.yaml'
            manager.save_config(str(output_path))
            
            # Verify file was created
            assert output_path.exists()
            
            # Verify content is correct
            with open(output_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            assert loaded_config == config
    
    def test_init_with_config_path(self):
        """Test initializing ConfigManager with a config path."""
        config_data = {
            'data': {'sampling_rate': 256},
            'model': {'embedding_dim': 128}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            manager = ConfigManager(config_path=temp_path)
            assert manager.config == config_data
        finally:
            Path(temp_path).unlink()
    
    def test_get_preprocessing_config(self):
        """Test retrieving preprocessing configuration section."""
        config = {
            'preprocessing': {
                'bandpass_low': 0.5,
                'bandpass_high': 45.0
            }
        }
        
        manager = ConfigManager()
        manager.config = config
        
        preproc_config = manager.get_preprocessing_config()
        assert preproc_config == config['preprocessing']
    
    def test_get_evaluation_config(self):
        """Test retrieving evaluation configuration section."""
        config = {
            'evaluation': {
                'metrics': ['accuracy', 'f1']
            }
        }
        
        manager = ConfigManager()
        manager.config = config
        
        eval_config = manager.get_evaluation_config()
        assert eval_config == config['evaluation']
    
    def test_validate_empty_config(self):
        """Test validation fails for empty configuration."""
        manager = ConfigManager()
        manager.config = {}
        
        assert manager.validate_config() is False
    
    def test_validate_negative_sampling_rate(self):
        """Test validation fails for negative sampling rate."""
        config = {
            'data': {
                'sampling_rate': -256
            }
        }
        
        manager = ConfigManager()
        manager.config = config
        
        assert manager.validate_config() is False
    
    def test_validate_negative_epoch_duration(self):
        """Test validation fails for negative epoch duration."""
        config = {
            'data': {
                'epoch_duration': -4.0
            }
        }
        
        manager = ConfigManager()
        manager.config = config
        
        assert manager.validate_config() is False
    
    def test_validate_negative_bandpass_low(self):
        """Test validation fails for negative bandpass_low."""
        config = {
            'preprocessing': {
                'bandpass_low': -0.5
            }
        }
        
        manager = ConfigManager()
        manager.config = config
        
        assert manager.validate_config() is False
    
    def test_validate_negative_bandpass_high(self):
        """Test validation fails for non-positive bandpass_high."""
        config = {
            'preprocessing': {
                'bandpass_high': 0
            }
        }
        
        manager = ConfigManager()
        manager.config = config
        
        assert manager.validate_config() is False
    
    def test_validate_negative_patch_size(self):
        """Test validation fails for non-positive patch_size."""
        config = {
            'model': {
                'patch_size': 0
            }
        }
        
        manager = ConfigManager()
        manager.config = config
        
        assert manager.validate_config() is False
    
    def test_validate_negative_embedding_dim(self):
        """Test validation fails for non-positive embedding_dim."""
        config = {
            'model': {
                'embedding_dim': -128
            }
        }
        
        manager = ConfigManager()
        manager.config = config
        
        assert manager.validate_config() is False
    
    def test_validate_negative_num_heads(self):
        """Test validation fails for non-positive num_heads."""
        config = {
            'model': {
                'num_heads': 0
            }
        }
        
        manager = ConfigManager()
        manager.config = config
        
        assert manager.validate_config() is False
    
    def test_validate_negative_num_layers(self):
        """Test validation fails for non-positive num_layers."""
        config = {
            'model': {
                'num_layers': -1
            }
        }
        
        manager = ConfigManager()
        manager.config = config
        
        assert manager.validate_config() is False
    
    def test_validate_invalid_dropout(self):
        """Test validation fails for dropout outside [0, 1)."""
        config = {
            'model': {
                'dropout': 1.5
            }
        }
        
        manager = ConfigManager()
        manager.config = config
        
        assert manager.validate_config() is False
    
    def test_validate_negative_batch_size(self):
        """Test validation fails for non-positive batch_size."""
        config = {
            'training': {
                'batch_size': 0
            }
        }
        
        manager = ConfigManager()
        manager.config = config
        
        assert manager.validate_config() is False
    
    def test_validate_negative_num_epochs(self):
        """Test validation fails for non-positive num_epochs."""
        config = {
            'training': {
                'num_epochs': -10
            }
        }
        
        manager = ConfigManager()
        manager.config = config
        
        assert manager.validate_config() is False
    
    def test_validate_invalid_k_folds(self):
        """Test validation fails for k_folds < 2."""
        config = {
            'training': {
                'k_folds': 1
            }
        }
        
        manager = ConfigManager()
        manager.config = config
        
        assert manager.validate_config() is False
    
    def test_get_missing_section_returns_empty_dict(self):
        """Test that getting a missing section returns empty dict."""
        manager = ConfigManager()
        manager.config = {}
        
        assert manager.get_data_config() == {}
        assert manager.get_model_config() == {}
        assert manager.get_training_config() == {}
        assert manager.get_preprocessing_config() == {}
        assert manager.get_evaluation_config() == {}

