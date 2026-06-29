"""
Alzheimer's EEG Detection System

A Transformer-based deep learning system for detecting Alzheimer's disease
from multi-channel EEG recordings.
"""

from .config_manager import ConfigManager
from .logger import setup_logging, get_logger
from .data_models import EEGSample, EEGEpoch, EEGDataset, PreprocessedDataset
from .dataset_loader import DatasetLoader, DataConfig
from .transformer import (
    PatchEmbedding,
    PositionalEncoding,
    MultiHeadSelfAttention,
    FeedForwardNetwork,
    TransformerEncoderBlock,
    EEGTransformer,
)
from .training import (
    TrainingPipeline,
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
)
from .visualizer import (
    AttentionExtractor,
    AttentionVisualizer,
    TemporalAttentionVisualizer,
    ChannelConnectivityVisualizer,
    compute_attention_rollout,
)

__version__ = "0.1.0"

__all__ = [
    "ConfigManager",
    "setup_logging",
    "get_logger",
    "EEGSample",
    "EEGEpoch",
    "EEGDataset",
    "PreprocessedDataset",
    "DatasetLoader",
    "DataConfig",
    "PatchEmbedding",
    "PositionalEncoding",
    "MultiHeadSelfAttention",
    "FeedForwardNetwork",
    "TransformerEncoderBlock",
    "EEGTransformer",
    "TrainingPipeline",
    "TrainingMetrics",
    "EpochResult",
    "EarlyStopping",
    "EEGDatasetWrapper",
    "train_epoch",
    "validate_epoch",
    "create_optimizer",
    "create_lr_scheduler",
    "save_checkpoint",
    "load_checkpoint",
    "create_k_fold_splits",
    "create_stratified_k_fold_splits",
    "AttentionExtractor",
    "AttentionVisualizer",
    "TemporalAttentionVisualizer",
    "ChannelConnectivityVisualizer",
    "compute_attention_rollout",
]

