"""
Training pipeline for Alzheimer's EEG Detection system.

This module provides the TrainingPipeline class and related utilities for
training, validation, checkpointing, and cross-validation of EEG models.
"""

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, Subset

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics from a single epoch."""
    loss: float
    accuracy: float
    num_samples: int
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            'loss': self.loss,
            'accuracy': self.accuracy,
            'num_samples': self.num_samples
        }


@dataclass
class EpochResult:
    """Container for results from a complete epoch (train + validation)."""
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    learning_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            'epoch': self.epoch,
            'train_loss': self.train_loss,
            'train_accuracy': self.train_accuracy,
            'val_loss': self.val_accuracy,
            'val_accuracy': self.val_accuracy,
            'learning_rate': self.learning_rate
        }


@dataclass
class CheckpointData:
    """Container for checkpoint data."""
    epoch: int
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    scheduler_state_dict: Optional[Dict[str, Any]]
    best_val_loss: float
    config: Dict[str, Any]
    metrics_history: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class EarlyStopping:
    """
    Early stopping mechanism to halt training when validation loss stops improving.
    
    Attributes:
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change to qualify as an improvement
        counter: Current count of epochs without improvement
        best_loss: Best validation loss observed
        should_stop: Flag indicating if training should stop
    """
    
    def __init__(self, patience: int = 15, min_delta: float = 0.0):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop based on validation loss.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered after {self.counter} epochs "
                    f"without improvement. Best loss: {self.best_loss:.6f}"
                )
        
        return self.should_stop
    
    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False



def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip: Optional[float] = None
) -> TrainingMetrics:
    """
    Execute a single training epoch.
    
    Performs forward pass, loss computation, backward pass, and parameter updates
    for all batches in the training loader.
    
    Args:
        model: The neural network model to train
        train_loader: DataLoader providing training batches
        optimizer: Optimizer for parameter updates
        criterion: Loss function
        device: Device to run training on (CPU/GPU)
        gradient_clip: Optional gradient clipping value
        
    Returns:
        TrainingMetrics containing loss and accuracy for the epoch
        
    Requirements: 4.1, 4.6
    """
    model.train()
    
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Move data to device
        data = data.to(device)
        targets = targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits, _ = model(data, return_attention=False)
        
        # Compute loss
        loss = criterion(logits, targets)
        
        # Backward pass
        loss.backward()
        
        # Optional gradient clipping
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        # Update parameters
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * data.size(0)
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == targets).sum().item()
        total_samples += data.size(0)
    
    # Compute epoch metrics
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    
    logger.debug(
        f"Training epoch complete - Loss: {avg_loss:.6f}, "
        f"Accuracy: {accuracy:.4f} ({correct}/{total_samples})"
    )
    
    return TrainingMetrics(
        loss=avg_loss,
        accuracy=accuracy,
        num_samples=total_samples
    )


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> TrainingMetrics:
    """
    Execute a single validation epoch.
    
    Performs forward pass without gradient computation to evaluate model
    performance on validation data.
    
    Args:
        model: The neural network model to evaluate
        val_loader: DataLoader providing validation batches
        criterion: Loss function
        device: Device to run validation on (CPU/GPU)
        
    Returns:
        TrainingMetrics containing loss and accuracy for the epoch
        
    Requirements: 4.6
    """
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, targets in val_loader:
            # Move data to device
            data = data.to(device)
            targets = targets.to(device)
            
            # Forward pass
            logits, _ = model(data, return_attention=False)
            
            # Compute loss
            loss = criterion(logits, targets)
            
            # Track metrics
            total_loss += loss.item() * data.size(0)
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == targets).sum().item()
            total_samples += data.size(0)
    
    # Compute epoch metrics
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    
    logger.debug(
        f"Validation epoch complete - Loss: {avg_loss:.6f}, "
        f"Accuracy: {accuracy:.4f} ({correct}/{total_samples})"
    )
    
    return TrainingMetrics(
        loss=avg_loss,
        accuracy=accuracy,
        num_samples=total_samples
    )



def create_optimizer(
    model: nn.Module,
    optimizer_name: str = 'adamw',
    learning_rate: float = 0.0001,
    weight_decay: float = 0.01,
    **kwargs
) -> Optimizer:
    """
    Create and configure an optimizer for the model.
    
    Args:
        model: The neural network model
        optimizer_name: Name of the optimizer ('adamw', 'adam', 'sgd', 'rmsprop')
        learning_rate: Initial learning rate
        weight_decay: Weight decay (L2 regularization) coefficient
        **kwargs: Additional optimizer-specific arguments
        
    Returns:
        Configured optimizer instance
        
    Raises:
        ValueError: If optimizer_name is not recognized
        
    Requirements: 4.1
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adamw':
        return AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8)
        )
    elif optimizer_name == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8)
        )
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
            nesterov=kwargs.get('nesterov', True)
        )
    elif optimizer_name == 'rmsprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            alpha=kwargs.get('alpha', 0.99),
            eps=kwargs.get('eps', 1e-8)
        )
    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. "
            f"Supported: 'adamw', 'adam', 'sgd', 'rmsprop'"
        )


def create_lr_scheduler(
    optimizer: Optimizer,
    num_epochs: int,
    warmup_epochs: int = 10,
    min_lr: float = 0.0
) -> LambdaLR:
    """
    Create a learning rate scheduler with linear warmup and cosine decay.
    
    The schedule consists of:
    1. Linear warmup from 0 to base_lr for the first warmup_epochs
    2. Cosine annealing decay from base_lr to min_lr for remaining epochs
    
    Args:
        optimizer: The optimizer to schedule
        num_epochs: Total number of training epochs
        warmup_epochs: Number of warmup epochs
        min_lr: Minimum learning rate at the end of training
        
    Returns:
        Configured LambdaLR scheduler
        
    Requirements: 4.2
    """
    def lr_lambda(current_epoch: int) -> float:
        """Compute learning rate multiplier for the current epoch."""
        if current_epoch < warmup_epochs:
            # Linear warmup: scale from 0 to 1
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        else:
            # Cosine annealing decay
            progress = float(current_epoch - warmup_epochs) / float(
                max(1, num_epochs - warmup_epochs)
            )
            # Cosine decay from 1 to min_lr_ratio
            base_lr = optimizer.defaults['lr']
            min_lr_ratio = min_lr / base_lr if base_lr > 0 else 0.0
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
    
    return LambdaLR(optimizer, lr_lambda)



def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[LambdaLR],
    best_val_loss: float,
    config: Dict[str, Any],
    metrics_history: List[Dict[str, Any]],
    is_best: bool = False,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save a model checkpoint to disk.
    
    Saves model weights, optimizer state, scheduler state, epoch number,
    configuration, and training metrics history.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        epoch: Current epoch number
        model: The neural network model
        optimizer: The optimizer
        scheduler: Optional learning rate scheduler
        best_val_loss: Best validation loss observed
        config: Model and training configuration
        metrics_history: List of metrics from all epochs
        is_best: If True, also save as 'best_model.pt'
        metadata: Optional additional metadata (training date, dataset, etc.)
        
    Returns:
        Path to the saved checkpoint file
        
    Requirements: 4.5, 4.7, 8.5, 10.1, 10.5
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare metadata
    if metadata is None:
        metadata = {}
    
    metadata.update({
        'training_date': datetime.now().isoformat(),
        'epoch': epoch,
        'best_val_loss': best_val_loss
    })
    
    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_loss': best_val_loss,
        'config': config,
        'metrics_history': metrics_history,
        'metadata': metadata
    }
    
    # Save checkpoint
    checkpoint_file = checkpoint_path / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_file)
    logger.info(f"Checkpoint saved to {checkpoint_file}")
    
    # Save configuration alongside checkpoint
    config_file = checkpoint_path / f'config_epoch_{epoch}.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save as best model if applicable
    if is_best:
        best_file = checkpoint_path / 'best_model.pt'
        torch.save(checkpoint, best_file)
        logger.info(f"Best model saved to {best_file}")
        
        # Save best config
        best_config_file = checkpoint_path / 'best_config.json'
        with open(best_config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    return str(checkpoint_file)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """
    Load a model checkpoint from disk.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: The neural network model to load weights into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        device: Device to load the model to
        
    Returns:
        Dictionary containing checkpoint data (epoch, metrics_history, etc.)
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        
    Requirements: 10.2, 10.3
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Model weights loaded from {checkpoint_path}")
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("Optimizer state restored")
    
    # Load scheduler state if provided
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info("Scheduler state restored")
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
        'config': checkpoint.get('config', {}),
        'metrics_history': checkpoint.get('metrics_history', []),
        'metadata': checkpoint.get('metadata', {})
    }



def create_k_fold_splits(
    dataset_size: int,
    k_folds: int,
    seed: int = 42
) -> List[Tuple[List[int], List[int]]]:
    """
    Create k-fold cross-validation splits.
    
    Generates k pairs of (train_indices, val_indices) where each sample
    appears in exactly one validation fold.
    
    Args:
        dataset_size: Total number of samples in the dataset
        k_folds: Number of folds
        seed: Random seed for reproducibility
        
    Returns:
        List of (train_indices, val_indices) tuples, one per fold
        
    Requirements: 4.4
    """
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Shuffle indices
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    
    # Calculate fold sizes
    fold_size = dataset_size // k_folds
    remainder = dataset_size % k_folds
    
    folds = []
    start_idx = 0
    
    for fold in range(k_folds):
        # Add one extra sample to first 'remainder' folds
        current_fold_size = fold_size + (1 if fold < remainder else 0)
        end_idx = start_idx + current_fold_size
        
        # Validation indices for this fold
        val_indices = indices[start_idx:end_idx].tolist()
        
        # Training indices are all other indices
        train_indices = np.concatenate([
            indices[:start_idx],
            indices[end_idx:]
        ]).tolist()
        
        folds.append((train_indices, val_indices))
        start_idx = end_idx
    
    return folds


def create_stratified_k_fold_splits(
    labels: np.ndarray,
    k_folds: int,
    seed: int = 42
) -> List[Tuple[List[int], List[int]]]:
    """
    Create stratified k-fold cross-validation splits.
    
    Ensures each fold has approximately the same class distribution
    as the original dataset.
    
    Args:
        labels: Array of class labels
        k_folds: Number of folds
        seed: Random seed for reproducibility
        
    Returns:
        List of (train_indices, val_indices) tuples, one per fold
        
    Requirements: 4.4
    """
    np.random.seed(seed)
    
    # Get unique classes and their indices
    unique_classes = np.unique(labels)
    class_indices = {c: np.where(labels == c)[0] for c in unique_classes}
    
    # Shuffle indices within each class
    for c in unique_classes:
        np.random.shuffle(class_indices[c])
    
    # Initialize folds
    folds = [[] for _ in range(k_folds)]
    
    # Distribute samples from each class across folds
    for c in unique_classes:
        indices = class_indices[c]
        fold_sizes = [len(indices) // k_folds] * k_folds
        
        # Distribute remainder
        for i in range(len(indices) % k_folds):
            fold_sizes[i] += 1
        
        start = 0
        for fold_idx, size in enumerate(fold_sizes):
            folds[fold_idx].extend(indices[start:start + size].tolist())
            start += size
    
    # Create train/val splits
    splits = []
    all_indices = set(range(len(labels)))
    
    for fold_idx in range(k_folds):
        val_indices = folds[fold_idx]
        train_indices = list(all_indices - set(val_indices))
        splits.append((train_indices, val_indices))
    
    return splits



class EEGDatasetWrapper(Dataset):
    """
    PyTorch Dataset wrapper for EEG data.
    
    Wraps numpy arrays of signals and labels into a PyTorch Dataset
    for use with DataLoader.
    """
    
    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        transform: Optional[Callable] = None
    ):
        """
        Initialize the dataset wrapper.
        
        Args:
            signals: EEG signals with shape (n_samples, n_channels, n_timepoints)
            labels: Labels with shape (n_samples,)
            transform: Optional transform to apply to signals
        """
        self.signals = torch.from_numpy(signals).float()
        self.labels = torch.from_numpy(labels).long()
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        signal = self.signals[idx]
        label = self.labels[idx]
        
        if self.transform is not None:
            signal = self.transform(signal)
        
        return signal, label


class TrainingPipeline:
    """
    Complete training pipeline for EEG models.
    
    Orchestrates model training with support for:
    - Single-run training with train/val split
    - K-fold cross-validation
    - Learning rate scheduling with warmup and cosine decay
    - Early stopping
    - Model checkpointing
    - Comprehensive metrics logging
    
    Attributes:
        model: The neural network model
        config: Training configuration dictionary
        device: Device to run training on
        checkpoint_dir: Directory for saving checkpoints
    
    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 6.5
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        checkpoint_dir: str = 'checkpoints'
    ):
        """
        Initialize the training pipeline.
        
        Args:
            model: The neural network model to train
            config: Training configuration dictionary
            device: Device to run training on (auto-detected if None)
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model.to(self.device)
        
        # Extract training config
        train_config = config.get('training', {})
        self.learning_rate = train_config.get('learning_rate', 0.0001)
        self.weight_decay = train_config.get('weight_decay', 0.01)
        self.batch_size = train_config.get('batch_size', 32)
        self.num_epochs = train_config.get('num_epochs', 100)
        self.warmup_epochs = train_config.get('warmup_epochs', 10)
        self.k_folds = train_config.get('k_folds', 5)
        self.patience = train_config.get('early_stopping_patience', 15)
        self.gradient_clip = train_config.get('gradient_clip', None)
        self.optimizer_name = train_config.get('optimizer', 'adamw')
        
        # Initialize components
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
        self.early_stopping = None
        
        # Metrics tracking
        self.metrics_history: List[Dict[str, Any]] = []
        self.best_val_loss = float('inf')
        
        logger.info(f"Training pipeline initialized on {self.device}")
    
    def _create_data_loaders(
        self,
        train_signals: np.ndarray,
        train_labels: np.ndarray,
        val_signals: np.ndarray,
        val_labels: np.ndarray
    ) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders."""
        train_dataset = EEGDatasetWrapper(train_signals, train_labels)
        val_dataset = EEGDatasetWrapper(val_signals, val_labels)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        return train_loader, val_loader
    
    def _reset_model(self):
        """Reset model weights for new training run."""
        for module in self.model.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
    
    def train(
        self,
        train_signals: np.ndarray,
        train_labels: np.ndarray,
        val_signals: np.ndarray,
        val_labels: np.ndarray,
        save_checkpoints: bool = True
    ) -> Dict[str, Any]:
        """
        Train the model on provided data.
        
        Args:
            train_signals: Training signals (n_samples, n_channels, n_timepoints)
            train_labels: Training labels (n_samples,)
            val_signals: Validation signals
            val_labels: Validation labels
            save_checkpoints: Whether to save checkpoints during training
            
        Returns:
            Dictionary containing training results and metrics history
        """
        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(
            train_signals, train_labels, val_signals, val_labels
        )
        
        # Initialize optimizer and scheduler
        self.optimizer = create_optimizer(
            self.model,
            optimizer_name=self.optimizer_name,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        self.scheduler = create_lr_scheduler(
            self.optimizer,
            num_epochs=self.num_epochs,
            warmup_epochs=self.warmup_epochs
        )
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(patience=self.patience)
        
        # Reset metrics
        self.metrics_history = []
        self.best_val_loss = float('inf')
        
        logger.info(
            f"Starting training for {self.num_epochs} epochs "
            f"(train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)})"
        )
        
        for epoch in range(self.num_epochs):
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Training epoch
            train_metrics = train_epoch(
                self.model,
                train_loader,
                self.optimizer,
                self.criterion,
                self.device,
                self.gradient_clip
            )
            
            # Validation epoch
            val_metrics = validate_epoch(
                self.model,
                val_loader,
                self.criterion,
                self.device
            )
            
            # Update learning rate
            self.scheduler.step()
            
            # Record metrics
            epoch_result = {
                'epoch': epoch + 1,
                'train_loss': train_metrics.loss,
                'train_accuracy': train_metrics.accuracy,
                'val_loss': val_metrics.loss,
                'val_accuracy': val_metrics.accuracy,
                'learning_rate': current_lr
            }
            self.metrics_history.append(epoch_result)
            
            # Log progress
            logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs} - "
                f"Train Loss: {train_metrics.loss:.6f}, "
                f"Train Acc: {train_metrics.accuracy:.4f}, "
                f"Val Loss: {val_metrics.loss:.6f}, "
                f"Val Acc: {val_metrics.accuracy:.4f}, "
                f"LR: {current_lr:.6f}"
            )
            
            # Check for best model
            is_best = val_metrics.loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics.loss
            
            # Save checkpoint
            if save_checkpoints and (is_best or (epoch + 1) % 5 == 0):
                save_checkpoint(
                    self.checkpoint_dir,
                    epoch + 1,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.best_val_loss,
                    self.config,
                    self.metrics_history,
                    is_best=is_best
                )
            
            # Check early stopping
            if self.early_stopping(val_metrics.loss):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        return {
            'best_val_loss': self.best_val_loss,
            'final_train_loss': train_metrics.loss,
            'final_train_accuracy': train_metrics.accuracy,
            'final_val_loss': val_metrics.loss,
            'final_val_accuracy': val_metrics.accuracy,
            'epochs_trained': epoch + 1,
            'metrics_history': self.metrics_history
        }
    
    def train_with_cross_validation(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        stratified: bool = True,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Train the model using k-fold cross-validation.
        
        Args:
            signals: All signals (n_samples, n_channels, n_timepoints)
            labels: All labels (n_samples,)
            stratified: Whether to use stratified splits
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing cross-validation results
            
        Requirements: 4.4, 6.5
        """
        # Create k-fold splits
        if stratified:
            splits = create_stratified_k_fold_splits(labels, self.k_folds, seed)
        else:
            splits = create_k_fold_splits(len(labels), self.k_folds, seed)
        
        fold_results = []
        
        logger.info(f"Starting {self.k_folds}-fold cross-validation")
        
        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            logger.info(f"\n{'='*50}")
            logger.info(f"Fold {fold_idx + 1}/{self.k_folds}")
            logger.info(f"{'='*50}")
            
            # Reset model for new fold
            self._reset_model()
            self.model.to(self.device)
            
            # Get fold data
            train_signals = signals[train_indices]
            train_labels = labels[train_indices]
            val_signals = signals[val_indices]
            val_labels = labels[val_indices]
            
            # Update checkpoint directory for this fold
            fold_checkpoint_dir = os.path.join(
                self.checkpoint_dir, f'fold_{fold_idx + 1}'
            )
            original_checkpoint_dir = self.checkpoint_dir
            self.checkpoint_dir = fold_checkpoint_dir
            
            # Train on this fold
            fold_result = self.train(
                train_signals, train_labels,
                val_signals, val_labels,
                save_checkpoints=True
            )
            
            fold_result['fold'] = fold_idx + 1
            fold_result['train_indices'] = train_indices
            fold_result['val_indices'] = val_indices
            fold_results.append(fold_result)
            
            # Restore checkpoint directory
            self.checkpoint_dir = original_checkpoint_dir
            
            logger.info(
                f"Fold {fold_idx + 1} complete - "
                f"Best Val Loss: {fold_result['best_val_loss']:.6f}, "
                f"Final Val Acc: {fold_result['final_val_accuracy']:.4f}"
            )
        
        # Aggregate results
        val_losses = [r['best_val_loss'] for r in fold_results]
        val_accuracies = [r['final_val_accuracy'] for r in fold_results]
        
        cv_results = {
            'fold_results': fold_results,
            'mean_val_loss': np.mean(val_losses),
            'std_val_loss': np.std(val_losses),
            'mean_val_accuracy': np.mean(val_accuracies),
            'std_val_accuracy': np.std(val_accuracies),
            'k_folds': self.k_folds
        }
        
        logger.info(f"\n{'='*50}")
        logger.info("Cross-Validation Summary")
        logger.info(f"{'='*50}")
        logger.info(
            f"Val Loss: {cv_results['mean_val_loss']:.6f} "
            f"± {cv_results['std_val_loss']:.6f}"
        )
        logger.info(
            f"Val Accuracy: {cv_results['mean_val_accuracy']:.4f} "
            f"± {cv_results['std_val_accuracy']:.4f}"
        )
        
        return cv_results
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get the training metrics history."""
        return self.metrics_history
    
    def get_best_val_loss(self) -> float:
        """Get the best validation loss observed during training."""
        return self.best_val_loss
