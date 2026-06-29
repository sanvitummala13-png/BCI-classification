"""
Tests for the attention visualization module.

This module tests the AttentionExtractor, AttentionVisualizer, and related
visualization components for EEG Transformer interpretability.
"""

import os
import tempfile
from pathlib import Path
from typing import List

# Import matplotlib first with non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pytest
import torch

from src.transformer import EEGTransformer
from src.visualizer import (
    AttentionExtractor,
    AttentionVisualizer,
    TemporalAttentionVisualizer,
    ChannelConnectivityVisualizer,
    compute_attention_rollout,
)


# Test fixtures
@pytest.fixture
def model_config():
    """Default model configuration for testing."""
    return {
        'n_channels': 19,
        'n_samples': 1024,
        'patch_size': 32,
        'embedding_dim': 64,
        'num_heads': 4,
        'num_layers': 3,
        'feedforward_dim': 128,
        'dropout': 0.1
    }


@pytest.fixture
def model(model_config):
    """Create a test EEGTransformer model."""
    return EEGTransformer(
        n_channels=model_config['n_channels'],
        n_samples=model_config['n_samples'],
        patch_size=model_config['patch_size'],
        embedding_dim=model_config['embedding_dim'],
        num_heads=model_config['num_heads'],
        num_layers=model_config['num_layers'],
        feedforward_dim=model_config['feedforward_dim'],
        dropout=model_config['dropout']
    )


@pytest.fixture
def sample_input(model_config):
    """Generate sample EEG input data."""
    batch_size = 4
    return np.random.randn(
        batch_size,
        model_config['n_channels'],
        model_config['n_samples']
    ).astype(np.float32)


@pytest.fixture
def channel_names():
    """Standard 10-20 EEG channel names."""
    return [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T3', 'C3', 'Cz', 'C4', 'T4',
        'T5', 'P3', 'Pz', 'P4', 'T6',
        'O1', 'O2'
    ]


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestAttentionExtractor:
    """Tests for AttentionExtractor class."""
    
    def test_init(self, model):
        """Test AttentionExtractor initialization."""
        extractor = AttentionExtractor(model)
        assert extractor.model is model
        assert extractor.device is not None
    
    def test_extract_attention_weights(self, model, sample_input):
        """Test attention weight extraction."""
        extractor = AttentionExtractor(model)
        attention_weights = extractor.extract_attention_weights(sample_input)
        
        # Should return list of attention matrices
        assert isinstance(attention_weights, list)
        assert len(attention_weights) == model.num_layers
        
        # Check shapes
        batch_size = sample_input.shape[0]
        n_patches = model.n_patches
        
        for attn in attention_weights:
            assert attn.shape == (batch_size, n_patches, n_patches)
    
    def test_extract_attention_with_tensor_input(self, model, sample_input):
        """Test extraction with torch tensor input."""
        extractor = AttentionExtractor(model)
        tensor_input = torch.from_numpy(sample_input)
        
        attention_weights = extractor.extract_attention_weights(tensor_input)
        
        assert len(attention_weights) == model.num_layers
    
    def test_extract_attention_with_prediction(self, model, sample_input):
        """Test extraction with predictions."""
        extractor = AttentionExtractor(model)
        
        attention_weights, predictions, probabilities = \
            extractor.extract_attention_with_prediction(sample_input)
        
        batch_size = sample_input.shape[0]
        
        assert len(attention_weights) == model.num_layers
        assert predictions.shape == (batch_size,)
        assert probabilities.shape == (batch_size, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_get_layer_attention(self, model, sample_input):
        """Test getting attention from specific layer."""
        extractor = AttentionExtractor(model)
        
        for layer_idx in range(model.num_layers):
            attention = extractor.get_layer_attention(sample_input, layer_idx)
            assert attention.shape[0] == sample_input.shape[0]
    
    def test_get_layer_attention_invalid_index(self, model, sample_input):
        """Test error handling for invalid layer index."""
        extractor = AttentionExtractor(model)
        
        with pytest.raises(IndexError):
            extractor.get_layer_attention(sample_input, model.num_layers + 1)
    
    def test_get_num_layers(self, model):
        """Test getting number of layers."""
        extractor = AttentionExtractor(model)
        assert extractor.get_num_layers() == model.num_layers
    
    def test_get_num_patches(self, model):
        """Test getting number of patches."""
        extractor = AttentionExtractor(model)
        assert extractor.get_num_patches() == model.n_patches


class TestAttentionRollout:
    """Tests for attention rollout computation."""
    
    def test_compute_attention_rollout(self, model, sample_input):
        """Test attention rollout computation."""
        extractor = AttentionExtractor(model)
        attention_weights = extractor.extract_attention_weights(sample_input)
        
        rollout = compute_attention_rollout(attention_weights)
        
        batch_size = sample_input.shape[0]
        n_patches = model.n_patches
        
        assert rollout.shape == (batch_size, n_patches, n_patches)
    
    def test_rollout_with_discard_ratio(self, model, sample_input):
        """Test rollout with attention value discarding."""
        extractor = AttentionExtractor(model)
        attention_weights = extractor.extract_attention_weights(sample_input)
        
        rollout = compute_attention_rollout(attention_weights, discard_ratio=0.1)
        
        assert rollout.shape[0] == sample_input.shape[0]
    
    def test_rollout_empty_weights(self):
        """Test error handling for empty attention weights."""
        with pytest.raises(ValueError):
            compute_attention_rollout([])


class TestTemporalAttentionVisualizer:
    """Tests for TemporalAttentionVisualizer class."""
    
    def test_init(self):
        """Test visualizer initialization."""
        viz = TemporalAttentionVisualizer()
        assert viz.figsize == (14, 8)
        assert viz.dpi == 150
    
    def test_plot_temporal_attention(self, temp_dir):
        """Test temporal attention plotting."""
        viz = TemporalAttentionVisualizer()
        
        seq_len = 32
        attention = np.random.rand(seq_len, seq_len).astype(np.float32)
        
        save_path = os.path.join(temp_dir, "temporal_attention.png")
        fig = viz.plot_temporal_attention(
            attention=attention,
            save_path=save_path
        )
        
        assert fig is not None
        assert os.path.exists(save_path)
        plt.close(fig)
    
    def test_plot_temporal_attention_with_signal(self, temp_dir):
        """Test temporal attention with EEG signal overlay."""
        viz = TemporalAttentionVisualizer()
        
        seq_len = 32
        n_channels = 19
        n_samples = 1024
        
        attention = np.random.rand(seq_len, seq_len).astype(np.float32)
        signal = np.random.randn(n_channels, n_samples).astype(np.float32)
        
        save_path = os.path.join(temp_dir, "temporal_with_signal.png")
        fig = viz.plot_temporal_attention(
            attention=attention,
            signal=signal,
            save_path=save_path
        )
        
        assert fig is not None
        assert os.path.exists(save_path)
        plt.close(fig)
    
    def test_plot_attention_matrix(self, temp_dir):
        """Test attention matrix plotting."""
        viz = TemporalAttentionVisualizer()
        
        seq_len = 32
        attention = np.random.rand(seq_len, seq_len).astype(np.float32)
        
        save_path = os.path.join(temp_dir, "attention_matrix.png")
        fig = viz.plot_attention_matrix(
            attention=attention,
            save_path=save_path
        )
        
        assert fig is not None
        assert os.path.exists(save_path)
        plt.close(fig)


class TestChannelConnectivityVisualizer:
    """Tests for ChannelConnectivityVisualizer class."""
    
    def test_init(self):
        """Test visualizer initialization."""
        viz = ChannelConnectivityVisualizer()
        assert viz.figsize == (10, 8)
        assert viz.dpi == 150
    
    def test_compute_channel_connectivity(self):
        """Test channel connectivity computation."""
        viz = ChannelConnectivityVisualizer()
        
        seq_len = 32
        n_channels = 19
        attention = np.random.rand(seq_len, seq_len).astype(np.float32)
        
        connectivity = viz.compute_channel_connectivity(
            attention=attention,
            n_channels=n_channels,
            patch_size=32,
            n_samples=1024
        )
        
        assert connectivity.shape == (n_channels, n_channels)
    
    def test_plot_connectivity_matrix(self, temp_dir, channel_names):
        """Test connectivity matrix plotting."""
        viz = ChannelConnectivityVisualizer()
        
        n_channels = len(channel_names)
        connectivity = np.random.rand(n_channels, n_channels).astype(np.float32)
        connectivity = (connectivity + connectivity.T) / 2  # Make symmetric
        
        save_path = os.path.join(temp_dir, "connectivity_matrix.png")
        fig = viz.plot_connectivity_matrix(
            connectivity=connectivity,
            channel_names=channel_names,
            save_path=save_path
        )
        
        assert fig is not None
        assert os.path.exists(save_path)
        plt.close(fig)
    
    def test_plot_connectivity_graph(self, temp_dir, channel_names):
        """Test connectivity graph plotting."""
        viz = ChannelConnectivityVisualizer()
        
        n_channels = len(channel_names)
        connectivity = np.random.rand(n_channels, n_channels).astype(np.float32)
        connectivity = (connectivity + connectivity.T) / 2
        
        save_path = os.path.join(temp_dir, "connectivity_graph.png")
        fig = viz.plot_connectivity_graph(
            connectivity=connectivity,
            channel_names=channel_names,
            save_path=save_path
        )
        
        assert fig is not None
        assert os.path.exists(save_path)
        plt.close(fig)


class TestAttentionVisualizer:
    """Tests for main AttentionVisualizer class."""
    
    def test_init(self, model):
        """Test AttentionVisualizer initialization."""
        viz = AttentionVisualizer(model)
        assert viz.model is model
        assert viz.extractor is not None
        assert viz.temporal_viz is not None
        assert viz.connectivity_viz is not None
    
    def test_extract_attention_weights(self, model, sample_input):
        """Test attention weight extraction through visualizer."""
        viz = AttentionVisualizer(model)
        attention_weights = viz.extract_attention_weights(sample_input)
        
        assert len(attention_weights) == model.num_layers
    
    def test_compute_attention_rollout(self, model, sample_input):
        """Test attention rollout through visualizer."""
        viz = AttentionVisualizer(model)
        rollout = viz.compute_attention_rollout(sample_input)
        
        assert rollout.shape[0] == sample_input.shape[0]
    
    def test_plot_temporal_attention(self, model, sample_input, temp_dir):
        """Test temporal attention plotting through visualizer."""
        viz = AttentionVisualizer(model)
        
        save_path = os.path.join(temp_dir, "viz_temporal.png")
        fig = viz.plot_temporal_attention(
            x=sample_input,
            sample_idx=0,
            save_path=save_path
        )
        
        assert fig is not None
        assert os.path.exists(save_path)
        plt.close(fig)
    
    def test_plot_channel_connectivity(self, model, sample_input, temp_dir, channel_names):
        """Test channel connectivity plotting through visualizer."""
        viz = AttentionVisualizer(model)
        
        save_path = os.path.join(temp_dir, "viz_connectivity.png")
        fig = viz.plot_channel_connectivity(
            x=sample_input,
            sample_idx=0,
            channel_names=channel_names,
            save_path=save_path
        )
        
        assert fig is not None
        assert os.path.exists(save_path)
        plt.close(fig)
    
    def test_plot_attention_rollout(self, model, sample_input, temp_dir):
        """Test attention rollout plotting through visualizer."""
        viz = AttentionVisualizer(model)
        
        save_path = os.path.join(temp_dir, "viz_rollout.png")
        fig = viz.plot_attention_rollout(
            x=sample_input,
            sample_idx=0,
            save_path=save_path
        )
        
        assert fig is not None
        assert os.path.exists(save_path)
        plt.close(fig)
    
    def test_generate_batch_visualizations(self, model, sample_input, temp_dir, channel_names):
        """Test batch visualization generation."""
        viz = AttentionVisualizer(model)
        
        labels = np.array([0, 1, 0, 1])
        sample_ids = ['test_001', 'test_002', 'test_003', 'test_004']
        
        saved_files = viz.generate_batch_visualizations(
            x=sample_input,
            labels=labels,
            channel_names=channel_names,
            output_dir=temp_dir,
            sample_ids=sample_ids
        )
        
        # Check that files were created
        assert len(saved_files['temporal']) == 4
        assert len(saved_files['connectivity']) == 4
        assert len(saved_files['rollout']) == 4
        
        for file_list in saved_files.values():
            for filepath in file_list:
                assert os.path.exists(filepath)
    
    def test_generate_layer_comparison(self, model, sample_input, temp_dir):
        """Test layer comparison visualization."""
        viz = AttentionVisualizer(model)
        
        save_path = os.path.join(temp_dir, "layer_comparison.png")
        fig = viz.generate_layer_comparison(
            x=sample_input,
            sample_idx=0,
            save_path=save_path
        )
        
        assert fig is not None
        assert os.path.exists(save_path)
        plt.close(fig)
    
    def test_get_attention_statistics(self, model, sample_input):
        """Test attention statistics computation."""
        viz = AttentionVisualizer(model)
        
        stats = viz.get_attention_statistics(sample_input)
        
        assert stats['n_layers'] == model.num_layers
        assert len(stats['layers']) == model.num_layers
        
        for layer_stats in stats['layers']:
            assert 'mean' in layer_stats
            assert 'std' in layer_stats
            assert 'max' in layer_stats
            assert 'min' in layer_stats
            assert 'entropy' in layer_stats

