"""
Unit tests for the Transformer model architecture.

Tests cover:
- PatchEmbedding layer
- PositionalEncoding layer
- MultiHeadSelfAttention layer
- FeedForwardNetwork layer
- TransformerEncoderBlock
- Full EEGTransformer model
"""

import pytest
import torch
import numpy as np

from src.transformer import (
    PatchEmbedding,
    PositionalEncoding,
    MultiHeadSelfAttention,
    FeedForwardNetwork,
    TransformerEncoderBlock,
    EEGTransformer,
)


class TestPatchEmbedding:
    """Tests for the PatchEmbedding layer."""
    
    def test_output_shape(self):
        """Test that output shape is correct."""
        batch_size = 4
        n_channels = 19
        n_samples = 1024
        patch_size = 32
        embedding_dim = 128
        
        layer = PatchEmbedding(
            n_channels=n_channels,
            patch_size=patch_size,
            embedding_dim=embedding_dim
        )
        
        x = torch.randn(batch_size, n_channels, n_samples)
        output = layer(x)
        
        expected_n_patches = n_samples // patch_size  # 1024 // 32 = 32
        assert output.shape == (batch_size, expected_n_patches, embedding_dim)
    
    def test_get_num_patches(self):
        """Test patch count calculation."""
        layer = PatchEmbedding(n_channels=19, patch_size=32, embedding_dim=128)
        
        assert layer.get_num_patches(1024) == 32
        assert layer.get_num_patches(512) == 16
        assert layer.get_num_patches(256) == 8
    
    def test_different_patch_sizes(self):
        """Test with different patch sizes."""
        batch_size = 2
        n_channels = 19
        n_samples = 1024
        embedding_dim = 64
        
        for patch_size in [16, 32, 64, 128]:
            layer = PatchEmbedding(
                n_channels=n_channels,
                patch_size=patch_size,
                embedding_dim=embedding_dim
            )
            
            x = torch.randn(batch_size, n_channels, n_samples)
            output = layer(x)
            
            expected_n_patches = n_samples // patch_size
            assert output.shape == (batch_size, expected_n_patches, embedding_dim)


class TestPositionalEncoding:
    """Tests for the PositionalEncoding layer."""
    
    def test_output_shape(self):
        """Test that output shape matches input shape."""
        batch_size = 4
        seq_len = 32
        embedding_dim = 128
        
        layer = PositionalEncoding(embedding_dim=embedding_dim, max_len=100)
        
        x = torch.randn(batch_size, seq_len, embedding_dim)
        output = layer(x)
        
        assert output.shape == x.shape
    
    def test_adds_positional_info(self):
        """Test that positional encoding modifies the input."""
        layer = PositionalEncoding(embedding_dim=64, max_len=100, dropout=0.0)
        
        x = torch.zeros(2, 10, 64)
        output = layer(x)
        
        # Output should not be all zeros (positional encoding added)
        assert not torch.allclose(output, x)
    
    def test_exceeds_max_length(self):
        """Test that exceeding max_len raises an error."""
        layer = PositionalEncoding(embedding_dim=64, max_len=50)
        
        x = torch.randn(2, 100, 64)  # seq_len > max_len
        
        with pytest.raises(ValueError, match="exceeds maximum length"):
            layer(x)


class TestMultiHeadSelfAttention:
    """Tests for the MultiHeadSelfAttention layer."""
    
    def test_output_shape(self):
        """Test that output shape matches input shape."""
        batch_size = 4
        seq_len = 32
        embedding_dim = 128
        num_heads = 8
        
        layer = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads
        )
        
        x = torch.randn(batch_size, seq_len, embedding_dim)
        output, attn_weights = layer(x)
        
        assert output.shape == x.shape
        assert attn_weights.shape == (batch_size, seq_len, seq_len)
    
    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1 along the last dimension."""
        layer = MultiHeadSelfAttention(embedding_dim=64, num_heads=4, dropout=0.0)
        layer.eval()  # Disable dropout for deterministic behavior
        
        x = torch.randn(2, 10, 64)
        with torch.no_grad():
            _, attn_weights = layer(x)
        
        # Attention weights should sum to 1 along the last dimension
        sums = attn_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    
    def test_invalid_embedding_dim(self):
        """Test that invalid embedding_dim raises an error."""
        with pytest.raises(ValueError, match="must be divisible"):
            MultiHeadSelfAttention(embedding_dim=100, num_heads=8)
    
    def test_no_weights_returned(self):
        """Test that weights can be disabled."""
        layer = MultiHeadSelfAttention(embedding_dim=64, num_heads=4)
        
        x = torch.randn(2, 10, 64)
        output, attn_weights = layer(x, need_weights=False)
        
        assert output.shape == x.shape
        assert attn_weights is None


class TestFeedForwardNetwork:
    """Tests for the FeedForwardNetwork layer."""
    
    def test_output_shape(self):
        """Test that output shape matches input shape."""
        batch_size = 4
        seq_len = 32
        embedding_dim = 128
        feedforward_dim = 512
        
        layer = FeedForwardNetwork(
            embedding_dim=embedding_dim,
            feedforward_dim=feedforward_dim
        )
        
        x = torch.randn(batch_size, seq_len, embedding_dim)
        output = layer(x)
        
        assert output.shape == x.shape
    
    def test_different_dimensions(self):
        """Test with different dimension configurations."""
        for embedding_dim, feedforward_dim in [(64, 256), (128, 512), (256, 1024)]:
            layer = FeedForwardNetwork(
                embedding_dim=embedding_dim,
                feedforward_dim=feedforward_dim
            )
            
            x = torch.randn(2, 10, embedding_dim)
            output = layer(x)
            
            assert output.shape == x.shape


class TestTransformerEncoderBlock:
    """Tests for the TransformerEncoderBlock."""
    
    def test_output_shape(self):
        """Test that output shape matches input shape."""
        batch_size = 4
        seq_len = 32
        embedding_dim = 128
        
        block = TransformerEncoderBlock(
            embedding_dim=embedding_dim,
            num_heads=8,
            feedforward_dim=512
        )
        
        x = torch.randn(batch_size, seq_len, embedding_dim)
        output, attn_weights = block(x)
        
        assert output.shape == x.shape
        assert attn_weights.shape == (batch_size, seq_len, seq_len)
    
    def test_residual_connection(self):
        """Test that residual connections are working."""
        block = TransformerEncoderBlock(
            embedding_dim=64,
            num_heads=4,
            feedforward_dim=256,
            dropout=0.0
        )
        
        # With zero input, output should not be zero due to layer norm
        x = torch.zeros(2, 10, 64)
        output, _ = block(x)
        
        # Output should be different from input
        assert not torch.allclose(output, x)


class TestEEGTransformer:
    """Tests for the full EEGTransformer model."""
    
    def test_output_shape(self):
        """Test that output shape is correct for binary classification."""
        batch_size = 4
        n_channels = 19
        n_samples = 1024
        
        model = EEGTransformer(
            n_channels=n_channels,
            n_samples=n_samples,
            patch_size=32,
            embedding_dim=128,
            num_heads=8,
            num_layers=6,
            feedforward_dim=512,
            num_classes=2
        )
        
        x = torch.randn(batch_size, n_channels, n_samples)
        logits, _ = model(x)
        
        assert logits.shape == (batch_size, 2)
    
    def test_attention_weights_extraction(self):
        """Test that attention weights can be extracted."""
        model = EEGTransformer(
            n_channels=19,
            n_samples=512,
            patch_size=32,
            num_layers=4
        )
        
        x = torch.randn(2, 19, 512)
        logits, attention_weights = model(x, return_attention=True)
        
        assert len(attention_weights) == 4  # One per layer
        
        n_patches = 512 // 32  # 16
        for attn in attention_weights:
            assert attn.shape == (2, n_patches, n_patches)
    
    def test_predict(self):
        """Test the predict method."""
        model = EEGTransformer(
            n_channels=19,
            n_samples=512,
            num_classes=2
        )
        model.eval()
        
        x = torch.randn(4, 19, 512)
        predictions = model.predict(x)
        
        assert predictions.shape == (4,)
        assert all(p in [0, 1] for p in predictions.tolist())
    
    def test_predict_proba(self):
        """Test the predict_proba method."""
        model = EEGTransformer(
            n_channels=19,
            n_samples=512,
            num_classes=2
        )
        model.eval()
        
        x = torch.randn(4, 19, 512)
        probs = model.predict_proba(x)
        
        assert probs.shape == (4, 2)
        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)
    
    def test_from_config(self):
        """Test creating model from config dictionary."""
        config = {
            'patch_size': 64,
            'embedding_dim': 256,
            'num_heads': 8,
            'num_layers': 4,
            'feedforward_dim': 1024,
            'dropout': 0.2
        }
        
        model = EEGTransformer.from_config(config, n_channels=19, n_samples=1024)
        
        assert model.patch_size == 64
        assert model.embedding_dim == 256
        assert model.num_heads == 8
        assert model.num_layers == 4
        assert model.feedforward_dim == 1024
    
    def test_get_config(self):
        """Test getting model configuration."""
        model = EEGTransformer(
            n_channels=19,
            n_samples=512,
            patch_size=32,
            embedding_dim=128,
            num_heads=8,
            num_layers=6
        )
        
        config = model.get_config()
        
        assert config['n_channels'] == 19
        assert config['n_samples'] == 512
        assert config['patch_size'] == 32
        assert config['embedding_dim'] == 128
        assert config['num_heads'] == 8
        assert config['num_layers'] == 6
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = EEGTransformer(
            n_channels=19,
            n_samples=512,
            patch_size=32,
            embedding_dim=64,
            num_heads=4,
            num_layers=2,
            feedforward_dim=256
        )
        
        param_count = model.count_parameters()
        
        # Should have a reasonable number of parameters
        assert param_count > 0
        assert param_count < 10_000_000  # Less than 10M for this small config
    
    def test_configurable_depth(self):
        """Test that model depth is configurable (Property 11)."""
        for num_layers in [2, 4, 6, 8]:
            model = EEGTransformer(
                n_channels=19,
                n_samples=512,
                num_layers=num_layers
            )
            
            assert len(model.encoder_blocks) == num_layers
            
            # Verify forward pass works
            x = torch.randn(2, 19, 512)
            logits, attn = model(x, return_attention=True)
            
            assert logits.shape == (2, 2)
            assert len(attn) == num_layers
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = EEGTransformer(
            n_channels=19,
            n_samples=512,
            num_layers=2
        )
        
        x = torch.randn(2, 19, 512, requires_grad=True)
        logits, _ = model(x)
        
        loss = logits.sum()
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestModelIntegration:
    """Integration tests for the Transformer model."""
    
    def test_training_step(self):
        """Test a single training step."""
        model = EEGTransformer(
            n_channels=19,
            n_samples=512,
            num_layers=2
        )
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Simulate a batch
        x = torch.randn(4, 19, 512)
        y = torch.tensor([0, 1, 0, 1])
        
        # Forward pass
        logits, _ = model(x)
        loss = criterion(logits, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Loss should be a scalar
        assert loss.dim() == 0
        assert loss.item() > 0
    
    def test_eval_mode(self):
        """Test model in evaluation mode."""
        model = EEGTransformer(
            n_channels=19,
            n_samples=512,
            num_layers=2,
            dropout=0.5  # High dropout to see difference
        )
        
        x = torch.randn(2, 19, 512)
        
        # Get outputs in train mode
        model.train()
        train_out1, _ = model(x)
        train_out2, _ = model(x)
        
        # Get outputs in eval mode
        model.eval()
        with torch.no_grad():
            eval_out1, _ = model(x)
            eval_out2, _ = model(x)
        
        # In eval mode, outputs should be deterministic
        assert torch.allclose(eval_out1, eval_out2)

