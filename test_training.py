"""
Unit tests for the Training Pipeline.

Tests cover:
- train_epoch function
- validate_epoch function
- create_optimizer function
- create_lr_scheduler function
- EarlyStopping class
- save_checkpoint and load_checkpoint functions
- create_k_fold_splits and create_stratified_k_fold_splits functions
- TrainingPipeline class
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.transformer import EEGTransformer
from src.training import (
    TrainingMetrics,
    EpochResult,
    EarlyStopping,
    EEGDatasetWrapper,
    train_epoch,
    validate_epoch,
    create_optimizer,
    create_lr_scheduler,
    save_checkpoint,
    load_checkpoint,
    create_k_fold_splits,
    create_stratified_k_fold_splits,
    TrainingPipeline,
)


# Test fixtures
@pytest.fixture
def simple_model():
    """Create a simple EEG Transformer model for testing."""
    return EEGTransformer(
        n_channels=19,
        n_samples=256,
        patch_size=32,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        feedforward_dim=128,
        num_classes=2,
        dropout=0.1
    )


@pytest.fixture
def sample_data():
    """Create sample EEG data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_channels = 19
    n_timepoints = 256
    
    signals = np.random.randn(n_samples, n_channels, n_timepoints).astype(np.float32)
    labels = np.random.randint(0, 2, size=n_samples).astype(np.int64)
    
    return signals, labels


@pytest.fixture
def data_loaders(sample_data):
    """Create data loaders for testing."""
    signals, labels = sample_data
    
    # Split into train/val
    train_size = 80
    train_signals = signals[:train_size]
    train_labels = labels[:train_size]
    val_signals = signals[train_size:]
    val_labels = labels[train_size:]
    
    train_dataset = EEGDatasetWrapper(train_signals, train_labels)
    val_dataset = EEGDatasetWrapper(val_signals, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    return train_loader, val_loader


@pytest.fixture
def device():
    """Get the device for testing."""
    return torch.device('cpu')


@pytest.fixture
def training_config():
    """Create a training configuration for testing."""
    return {
        'training': {
            'optimizer': 'adamw',
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'batch_size': 16,
            'num_epochs': 5,
            'warmup_epochs': 2,
            'k_folds': 3,
            'early_stopping_patience': 3,
            'gradient_clip': 1.0
        },
        'model': {
            'patch_size': 32,
            'embedding_dim': 64,
            'num_heads': 4,
            'num_layers': 2
        }
    }


class TestTrainEpoch:
    """Tests for the train_epoch function."""
    
    def test_returns_training_metrics(self, simple_model, data_loaders, device):
        """Test that train_epoch returns TrainingMetrics."""
        train_loader, _ = data_loaders
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        metrics = train_epoch(
            simple_model, train_loader, optimizer, criterion, device
        )
        
        assert isinstance(metrics, TrainingMetrics)
        assert metrics.loss >= 0
        assert 0 <= metrics.accuracy <= 1
        assert metrics.num_samples == len(train_loader.dataset)
    
    def test_model_in_train_mode(self, simple_model, data_loaders, device):
        """Test that model is in training mode during train_epoch."""
        train_loader, _ = data_loaders
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        simple_model.eval()  # Set to eval mode first
        train_epoch(simple_model, train_loader, optimizer, criterion, device)
        
        # Model should be in train mode after train_epoch
        assert simple_model.training
    
    def test_gradient_clipping(self, simple_model, data_loaders, device):
        """Test that gradient clipping is applied."""
        train_loader, _ = data_loaders
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Should not raise any errors
        metrics = train_epoch(
            simple_model, train_loader, optimizer, criterion, device,
            gradient_clip=1.0
        )
        
        assert metrics.loss >= 0


class TestValidateEpoch:
    """Tests for the validate_epoch function."""
    
    def test_returns_training_metrics(self, simple_model, data_loaders, device):
        """Test that validate_epoch returns TrainingMetrics."""
        _, val_loader = data_loaders
        criterion = nn.CrossEntropyLoss()
        
        metrics = validate_epoch(simple_model, val_loader, criterion, device)
        
        assert isinstance(metrics, TrainingMetrics)
        assert metrics.loss >= 0
        assert 0 <= metrics.accuracy <= 1
        assert metrics.num_samples == len(val_loader.dataset)
    
    def test_model_in_eval_mode(self, simple_model, data_loaders, device):
        """Test that model is in eval mode during validate_epoch."""
        _, val_loader = data_loaders
        criterion = nn.CrossEntropyLoss()
        
        simple_model.train()  # Set to train mode first
        validate_epoch(simple_model, val_loader, criterion, device)
        
        # Model should be in eval mode after validate_epoch
        assert not simple_model.training
    
    def test_no_gradient_computation(self, simple_model, data_loaders, device):
        """Test that no gradients are computed during validation."""
        _, val_loader = data_loaders
        criterion = nn.CrossEntropyLoss()
        
        # Clear any existing gradients
        simple_model.zero_grad()
        
        validate_epoch(simple_model, val_loader, criterion, device)
        
        # Check that no gradients were computed
        for param in simple_model.parameters():
            assert param.grad is None or torch.all(param.grad == 0)


class TestCreateOptimizer:
    """Tests for the create_optimizer function."""
    
    def test_adamw_optimizer(self, simple_model):
        """Test creating AdamW optimizer."""
        optimizer = create_optimizer(
            simple_model, 'adamw', learning_rate=0.001, weight_decay=0.01
        )
        
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults['lr'] == 0.001
        assert optimizer.defaults['weight_decay'] == 0.01
    
    def test_adam_optimizer(self, simple_model):
        """Test creating Adam optimizer."""
        optimizer = create_optimizer(
            simple_model, 'adam', learning_rate=0.001, weight_decay=0.01
        )
        
        assert isinstance(optimizer, torch.optim.Adam)
    
    def test_sgd_optimizer(self, simple_model):
        """Test creating SGD optimizer."""
        optimizer = create_optimizer(
            simple_model, 'sgd', learning_rate=0.01, weight_decay=0.001
        )
        
        assert isinstance(optimizer, torch.optim.SGD)
    
    def test_rmsprop_optimizer(self, simple_model):
        """Test creating RMSprop optimizer."""
        optimizer = create_optimizer(
            simple_model, 'rmsprop', learning_rate=0.001, weight_decay=0.01
        )
        
        assert isinstance(optimizer, torch.optim.RMSprop)
    
    def test_invalid_optimizer(self, simple_model):
        """Test that invalid optimizer raises ValueError."""
        with pytest.raises(ValueError, match="Unknown optimizer"):
            create_optimizer(simple_model, 'invalid_optimizer')


class TestCreateLRScheduler:
    """Tests for the create_lr_scheduler function."""
    
    def test_warmup_phase(self, simple_model):
        """Test that learning rate increases during warmup."""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        scheduler = create_lr_scheduler(optimizer, num_epochs=20, warmup_epochs=5)
        
        lrs = []
        for epoch in range(5):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        
        # Learning rate should increase during warmup
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i-1]
    
    def test_cosine_decay_phase(self, simple_model):
        """Test that learning rate decreases after warmup."""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        scheduler = create_lr_scheduler(optimizer, num_epochs=20, warmup_epochs=5)
        
        # Skip warmup
        for _ in range(5):
            scheduler.step()
        
        # Get LRs during decay phase
        lrs = []
        for epoch in range(15):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        
        # Learning rate should generally decrease during cosine decay
        assert lrs[-1] < lrs[0]


class TestEarlyStopping:
    """Tests for the EarlyStopping class."""
    
    def test_no_stop_when_improving(self):
        """Test that training doesn't stop when loss is improving."""
        early_stopping = EarlyStopping(patience=3)
        
        losses = [1.0, 0.9, 0.8, 0.7, 0.6]
        for loss in losses:
            should_stop = early_stopping(loss)
            assert not should_stop
    
    def test_stop_after_patience(self):
        """Test that training stops after patience epochs without improvement."""
        early_stopping = EarlyStopping(patience=3)
        
        # First, improve
        early_stopping(1.0)
        early_stopping(0.5)  # Best loss
        
        # Then, no improvement for patience epochs
        early_stopping(0.6)
        early_stopping(0.7)
        should_stop = early_stopping(0.8)
        
        assert should_stop
        assert early_stopping.should_stop
    
    def test_reset(self):
        """Test that reset clears the state."""
        early_stopping = EarlyStopping(patience=2)
        
        early_stopping(1.0)
        early_stopping(0.5)
        early_stopping(0.6)
        early_stopping(0.7)  # Should trigger stop
        
        early_stopping.reset()
        
        assert early_stopping.counter == 0
        assert early_stopping.best_loss == float('inf')
        assert not early_stopping.should_stop


class TestCheckpointing:
    """Tests for save_checkpoint and load_checkpoint functions."""
    
    def test_save_and_load_checkpoint(self, simple_model):
        """Test saving and loading a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
            scheduler = create_lr_scheduler(optimizer, num_epochs=10, warmup_epochs=2)
            
            config = {'model': {'embedding_dim': 64}}
            metrics_history = [{'epoch': 1, 'loss': 0.5}]
            
            # Save checkpoint
            checkpoint_path = save_checkpoint(
                tmpdir, epoch=5, model=simple_model, optimizer=optimizer,
                scheduler=scheduler, best_val_loss=0.3, config=config,
                metrics_history=metrics_history
            )
            
            assert os.path.exists(checkpoint_path)
            
            # Create new model with same architecture and load checkpoint
            new_model = EEGTransformer(
                n_channels=19, n_samples=256, patch_size=32,
                embedding_dim=64, num_heads=4, num_layers=2,
                feedforward_dim=128  # Must match simple_model fixture
            )
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
            
            checkpoint_data = load_checkpoint(
                checkpoint_path, new_model, new_optimizer
            )
            
            assert checkpoint_data['epoch'] == 5
            assert checkpoint_data['best_val_loss'] == 0.3
            assert checkpoint_data['config'] == config
    
    def test_save_best_model(self, simple_model):
        """Test that best model is saved separately."""
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
            
            save_checkpoint(
                tmpdir, epoch=5, model=simple_model, optimizer=optimizer,
                scheduler=None, best_val_loss=0.3, config={},
                metrics_history=[], is_best=True
            )
            
            best_model_path = os.path.join(tmpdir, 'best_model.pt')
            assert os.path.exists(best_model_path)
    
    def test_config_saved_alongside(self, simple_model):
        """Test that config is saved alongside checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
            config = {'model': {'embedding_dim': 64}}
            
            save_checkpoint(
                tmpdir, epoch=5, model=simple_model, optimizer=optimizer,
                scheduler=None, best_val_loss=0.3, config=config,
                metrics_history=[]
            )
            
            config_path = os.path.join(tmpdir, 'config_epoch_5.json')
            assert os.path.exists(config_path)
            
            with open(config_path) as f:
                loaded_config = json.load(f)
            assert loaded_config == config


class TestKFoldSplits:
    """Tests for k-fold split functions."""
    
    def test_create_k_fold_splits_coverage(self):
        """Test that all samples are covered exactly once in validation."""
        dataset_size = 100
        k_folds = 5
        
        splits = create_k_fold_splits(dataset_size, k_folds, seed=42)
        
        assert len(splits) == k_folds
        
        # Check that each sample appears in exactly one validation fold
        all_val_indices = []
        for train_indices, val_indices in splits:
            all_val_indices.extend(val_indices)
            
            # Check no overlap between train and val
            assert len(set(train_indices) & set(val_indices)) == 0
        
        # All indices should be covered
        assert sorted(all_val_indices) == list(range(dataset_size))
    
    def test_create_stratified_k_fold_splits(self):
        """Test stratified k-fold splits maintain class distribution."""
        np.random.seed(42)
        labels = np.array([0] * 60 + [1] * 40)  # 60% class 0, 40% class 1
        k_folds = 5
        
        splits = create_stratified_k_fold_splits(labels, k_folds, seed=42)
        
        assert len(splits) == k_folds
        
        # Check class distribution in each fold
        for train_indices, val_indices in splits:
            val_labels = labels[val_indices]
            class_0_ratio = np.sum(val_labels == 0) / len(val_labels)
            
            # Should be approximately 60% (within tolerance)
            assert 0.4 <= class_0_ratio <= 0.8


class TestEEGDatasetWrapper:
    """Tests for the EEGDatasetWrapper class."""
    
    def test_dataset_length(self, sample_data):
        """Test that dataset length is correct."""
        signals, labels = sample_data
        dataset = EEGDatasetWrapper(signals, labels)
        
        assert len(dataset) == len(labels)
    
    def test_getitem(self, sample_data):
        """Test that getitem returns correct data."""
        signals, labels = sample_data
        dataset = EEGDatasetWrapper(signals, labels)
        
        signal, label = dataset[0]
        
        assert isinstance(signal, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert signal.shape == (19, 256)
        assert label.dim() == 0


class TestTrainingPipeline:
    """Tests for the TrainingPipeline class."""
    
    def test_initialization(self, simple_model, training_config):
        """Test pipeline initialization."""
        pipeline = TrainingPipeline(
            simple_model, training_config, checkpoint_dir='test_checkpoints'
        )
        
        assert pipeline.model is simple_model
        assert pipeline.learning_rate == 0.001
        assert pipeline.batch_size == 16
    
    def test_train_single_run(self, simple_model, sample_data, training_config):
        """Test single training run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            signals, labels = sample_data
            
            # Split data
            train_signals = signals[:80]
            train_labels = labels[:80]
            val_signals = signals[80:]
            val_labels = labels[80:]
            
            pipeline = TrainingPipeline(
                simple_model, training_config, checkpoint_dir=tmpdir
            )
            
            results = pipeline.train(
                train_signals, train_labels,
                val_signals, val_labels,
                save_checkpoints=False
            )
            
            assert 'best_val_loss' in results
            assert 'final_train_accuracy' in results
            assert 'metrics_history' in results
            assert len(results['metrics_history']) > 0
    
    def test_train_with_cross_validation(self, simple_model, sample_data, training_config):
        """Test k-fold cross-validation training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            signals, labels = sample_data
            
            # Use smaller config for faster test
            training_config['training']['num_epochs'] = 2
            training_config['training']['k_folds'] = 2
            
            pipeline = TrainingPipeline(
                simple_model, training_config, checkpoint_dir=tmpdir
            )
            
            results = pipeline.train_with_cross_validation(
                signals, labels, stratified=True, seed=42
            )
            
            assert 'fold_results' in results
            assert len(results['fold_results']) == 2
            assert 'mean_val_loss' in results
            assert 'std_val_loss' in results


class TestTrainingMetrics:
    """Tests for TrainingMetrics dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = TrainingMetrics(loss=0.5, accuracy=0.8, num_samples=100)
        
        d = metrics.to_dict()
        
        assert d['loss'] == 0.5
        assert d['accuracy'] == 0.8
        assert d['num_samples'] == 100

