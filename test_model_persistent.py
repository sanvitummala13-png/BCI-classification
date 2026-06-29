"""
Unit tests for Model Persistence and Inference API.

Tests cover:
- save_model function
- load_model function
- verify_model_compatibility function
- ModelMetadata dataclass
- ModelPrediction dataclass
- InferenceAPI class
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.transformer import EEGTransformer
from src.model_persistence import (
    ModelMetadata,
    ModelPrediction,
    save_model,
    load_model,
    verify_model_compatibility,
    InferenceAPI,
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
def sample_config():
    """Create a sample configuration for testing."""
    return {
        'model': {
            'patch_size': 32,
            'embedding_dim': 64,
            'num_heads': 4,
            'num_layers': 2,
            'feedforward_dim': 128
        },
        'preprocessing': {
            'bandpass_low': 0.5,
            'bandpass_high': 45.0,
            'normalization': 'zscore'
        },
        'training': {
            'learning_rate': 0.001,
            'batch_size': 32
        }
    }


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return ModelMetadata(
        training_date='2024-01-15T10:30:00',
        dataset_name='PhysioNet',
        accuracy=0.85,
        f1_score=0.83,
        roc_auc=0.90,
        epochs_trained=50,
        best_val_loss=0.35
    )


@pytest.fixture
def sample_eeg_data():
    """Create sample EEG data for testing."""
    np.random.seed(42)
    # Single sample: (channels, samples)
    single = np.random.randn(19, 256).astype(np.float32)
    # Batch: (batch, channels, samples)
    batch = np.random.randn(4, 19, 256).astype(np.float32)
    return single, batch


class TestModelMetadata:
    """Tests for ModelMetadata dataclass."""
    
    def test_to_dict(self, sample_metadata):
        """Test conversion to dictionary."""
        d = sample_metadata.to_dict()
        
        assert d['training_date'] == '2024-01-15T10:30:00'
        assert d['dataset_name'] == 'PhysioNet'
        assert d['accuracy'] == 0.85
        assert d['f1_score'] == 0.83
        assert d['roc_auc'] == 0.90
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            'training_date': '2024-01-15T10:30:00',
            'dataset_name': 'TUH',
            'accuracy': 0.88,
            'f1_score': 0.86,
            'roc_auc': 0.92,
            'epochs_trained': 100,
            'best_val_loss': 0.25
        }
        
        metadata = ModelMetadata.from_dict(data)
        
        assert metadata.training_date == '2024-01-15T10:30:00'
        assert metadata.dataset_name == 'TUH'
        assert metadata.accuracy == 0.88
    
    def test_default_values(self):
        """Test default values are set correctly."""
        metadata = ModelMetadata(training_date='2024-01-01')
        
        assert metadata.dataset_name == 'unknown'
        assert metadata.accuracy == 0.0
        assert metadata.additional_info == {}


class TestModelPrediction:
    """Tests for ModelPrediction dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        prediction = ModelPrediction(
            logits=np.array([0.3, 0.7]),
            probabilities=np.array([0.4, 0.6]),
            predicted_class=1,
            confidence=0.6
        )
        
        d = prediction.to_dict()
        
        assert d['predicted_class'] == 1
        assert d['confidence'] == 0.6
        assert d['class_label'] == 'AD'
        assert len(d['probabilities']) == 2
    
    def test_class_label_control(self):
        """Test class label for Control prediction."""
        prediction = ModelPrediction(
            logits=np.array([0.7, 0.3]),
            probabilities=np.array([0.6, 0.4]),
            predicted_class=0,
            confidence=0.6
        )
        
        d = prediction.to_dict()
        assert d['class_label'] == 'Control'


class TestSaveModel:
    """Tests for save_model function."""
    
    def test_save_model_creates_files(self, simple_model, sample_config, sample_metadata):
        """Test that save_model creates .pt and .json files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test_model')
            
            result_path = save_model(
                simple_model, save_path, sample_config, sample_metadata
            )
            
            # Check .pt file exists
            assert os.path.exists(result_path)
            assert result_path.endswith('.pt')
            
            # Check .json config file exists
            json_path = result_path.replace('.pt', '.json')
            assert os.path.exists(json_path)
    
    def test_save_model_with_metrics(self, simple_model, sample_config):
        """Test saving model with metrics updates metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test_model')
            metrics = {'accuracy': 0.92, 'f1_score': 0.90, 'roc_auc': 0.95}
            
            result_path = save_model(
                simple_model, save_path, sample_config, metrics=metrics
            )
            
            # Load and verify metadata
            checkpoint = torch.load(result_path, weights_only=False)
            assert checkpoint['metadata']['accuracy'] == 0.92
            assert checkpoint['metadata']['f1_score'] == 0.90
    
    def test_save_model_includes_model_config(self, simple_model, sample_config):
        """Test that model configuration is saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test_model')
            
            result_path = save_model(simple_model, save_path, sample_config)
            
            checkpoint = torch.load(result_path, weights_only=False)
            model_config = checkpoint['model_config']
            
            assert model_config['n_channels'] == 19
            assert model_config['n_samples'] == 256
            assert model_config['embedding_dim'] == 64
    
    def test_save_model_json_contains_metadata(self, simple_model, sample_config, sample_metadata):
        """Test that JSON file contains all metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test_model')
            
            result_path = save_model(
                simple_model, save_path, sample_config, sample_metadata
            )
            
            json_path = result_path.replace('.pt', '.json')
            with open(json_path) as f:
                data = json.load(f)
            
            assert 'config' in data
            assert 'model_config' in data
            assert 'metadata' in data
            assert data['metadata']['dataset_name'] == 'PhysioNet'


class TestVerifyModelCompatibility:
    """Tests for verify_model_compatibility function."""
    
    def test_compatible_model(self, simple_model):
        """Test verification passes for compatible model."""
        checkpoint_config = {
            'n_channels': 19,
            'n_samples': 256,
            'patch_size': 32,
            'embedding_dim': 64,
            'num_heads': 4,
            'num_layers': 2,
            'feedforward_dim': 128,
            'num_classes': 2
        }
        
        is_compatible, msg = verify_model_compatibility(checkpoint_config, simple_model)
        
        assert is_compatible
        assert 'compatible' in msg.lower()
    
    def test_incompatible_embedding_dim(self, simple_model):
        """Test verification fails for mismatched embedding_dim."""
        checkpoint_config = {
            'n_channels': 19,
            'n_samples': 256,
            'embedding_dim': 128,  # Different from model's 64
            'num_heads': 4,
            'num_layers': 2
        }
        
        is_compatible, msg = verify_model_compatibility(checkpoint_config, simple_model)
        
        assert not is_compatible
        assert 'embedding_dim' in msg
    
    def test_incompatible_num_layers(self, simple_model):
        """Test verification fails for mismatched num_layers."""
        checkpoint_config = {
            'n_channels': 19,
            'n_samples': 256,
            'embedding_dim': 64,
            'num_heads': 4,
            'num_layers': 6  # Different from model's 2
        }
        
        is_compatible, msg = verify_model_compatibility(checkpoint_config, simple_model)
        
        assert not is_compatible
        assert 'num_layers' in msg


class TestLoadModel:
    """Tests for load_model function."""
    
    def test_load_model_restores_weights(self, simple_model, sample_config):
        """Test that load_model restores model weights correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test_model')
            
            # Save original model
            save_model(simple_model, save_path, sample_config)
            
            # Create new model with same architecture
            new_model = EEGTransformer(
                n_channels=19,
                n_samples=256,
                patch_size=32,
                embedding_dim=64,
                num_heads=4,
                num_layers=2,
                feedforward_dim=128
            )
            
            # Load weights
            loaded_model, config, metadata = load_model(
                save_path + '.pt', model=new_model
            )
            
            # Verify weights match
            for (name1, param1), (name2, param2) in zip(
                simple_model.named_parameters(),
                loaded_model.named_parameters()
            ):
                assert torch.allclose(param1, param2), f"Mismatch in {name1}"
    
    def test_load_model_creates_model_from_config(self, simple_model, sample_config):
        """Test that load_model can create model from checkpoint config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test_model')
            
            # Save model
            save_model(simple_model, save_path, sample_config)
            
            # Load without providing model
            loaded_model, config, metadata = load_model(save_path + '.pt')
            
            assert isinstance(loaded_model, EEGTransformer)
            assert loaded_model.n_channels == 19
            assert loaded_model.embedding_dim == 64
    
    def test_load_model_returns_metadata(self, simple_model, sample_config, sample_metadata):
        """Test that load_model returns correct metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test_model')
            
            save_model(simple_model, save_path, sample_config, sample_metadata)
            
            _, _, metadata = load_model(save_path + '.pt')
            
            assert metadata.dataset_name == 'PhysioNet'
            assert metadata.accuracy == 0.85
    
    def test_load_model_file_not_found(self):
        """Test that load_model raises error for missing file."""
        with pytest.raises(FileNotFoundError):
            load_model('/nonexistent/path/model.pt')
    
    def test_load_model_incompatible_strict(self, simple_model, sample_config):
        """Test that load_model raises error for incompatible model in strict mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test_model')
            
            # Save model
            save_model(simple_model, save_path, sample_config)
            
            # Create incompatible model
            incompatible_model = EEGTransformer(
                n_channels=19,
                n_samples=256,
                patch_size=32,
                embedding_dim=128,  # Different
                num_heads=4,
                num_layers=2,
                feedforward_dim=128
            )
            
            with pytest.raises(ValueError, match="Architecture mismatch"):
                load_model(save_path + '.pt', model=incompatible_model, strict=True)


class TestInferenceAPI:
    """Tests for InferenceAPI class."""
    
    def test_initialization(self, simple_model, sample_config):
        """Test InferenceAPI initialization."""
        api = InferenceAPI(simple_model, sample_config)
        
        assert api.model is simple_model
        assert api.expected_channels == 19
        assert api.expected_samples == 256
    
    def test_from_checkpoint(self, simple_model, sample_config):
        """Test creating InferenceAPI from checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test_model')
            save_model(simple_model, save_path, sample_config)
            
            api = InferenceAPI.from_checkpoint(save_path + '.pt')
            
            assert isinstance(api.model, EEGTransformer)
    
    def test_predict_single_sample(self, simple_model, sample_config, sample_eeg_data):
        """Test prediction on single sample."""
        single, _ = sample_eeg_data
        api = InferenceAPI(simple_model, sample_config)
        
        prediction = api.predict(single, preprocess=False)
        
        assert isinstance(prediction, ModelPrediction)
        assert prediction.predicted_class in [0, 1]
        assert 0 <= prediction.confidence <= 1
        assert len(prediction.probabilities) == 2
    
    def test_predict_batch(self, simple_model, sample_config, sample_eeg_data):
        """Test prediction on batch of samples."""
        _, batch = sample_eeg_data
        api = InferenceAPI(simple_model, sample_config)
        
        predictions = api.predict(batch, preprocess=False)
        
        assert isinstance(predictions, list)
        assert len(predictions) == 4
        for pred in predictions:
            assert isinstance(pred, ModelPrediction)
    
    def test_predict_with_attention(self, simple_model, sample_config, sample_eeg_data):
        """Test prediction with attention weights."""
        single, _ = sample_eeg_data
        api = InferenceAPI(simple_model, sample_config)
        
        prediction = api.predict(single, preprocess=False, return_attention=True)
        
        assert prediction.attention_weights is not None
        assert len(prediction.attention_weights) == 2  # num_layers
    
    def test_predict_proba(self, simple_model, sample_config, sample_eeg_data):
        """Test predict_proba returns probabilities."""
        single, _ = sample_eeg_data
        api = InferenceAPI(simple_model, sample_config)
        
        probs = api.predict_proba(single, preprocess=False)
        
        assert probs.shape == (2,)
        assert np.isclose(np.sum(probs), 1.0)
    
    def test_predict_class(self, simple_model, sample_config, sample_eeg_data):
        """Test predict_class returns class labels."""
        single, batch = sample_eeg_data
        api = InferenceAPI(simple_model, sample_config)
        
        # Single sample
        pred_class = api.predict_class(single, preprocess=False)
        assert pred_class in [0, 1]
        
        # Batch
        pred_classes = api.predict_class(batch, preprocess=False)
        assert pred_classes.shape == (4,)
    
    def test_validate_input_wrong_channels(self, simple_model, sample_config):
        """Test validation fails for wrong number of channels."""
        api = InferenceAPI(simple_model, sample_config)
        wrong_data = np.random.randn(10, 256).astype(np.float32)  # 10 channels instead of 19
        
        with pytest.raises(ValueError, match="Expected 19 channels"):
            api.predict(wrong_data, preprocess=False)
    
    def test_validate_input_wrong_samples(self, simple_model, sample_config):
        """Test validation fails for wrong number of samples."""
        api = InferenceAPI(simple_model, sample_config)
        wrong_data = np.random.randn(19, 128).astype(np.float32)  # 128 samples instead of 256
        
        with pytest.raises(ValueError, match="Expected 256 samples"):
            api.predict(wrong_data, preprocess=False)
    
    def test_predict_with_preprocessing(self, simple_model, sample_config, sample_eeg_data):
        """Test prediction with preprocessing enabled."""
        single, _ = sample_eeg_data
        api = InferenceAPI(simple_model, sample_config)
        
        # This should not raise any errors
        prediction = api.predict(single, sampling_rate=256.0, preprocess=True)
        
        assert isinstance(prediction, ModelPrediction)
        assert prediction.predicted_class in [0, 1]


class TestSaveLoadRoundTrip:
    """Integration tests for save/load round-trip."""
    
    def test_model_produces_same_output_after_load(self, simple_model, sample_config, sample_eeg_data):
        """Test that loaded model produces same output as original."""
        single, _ = sample_eeg_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test_model')
            
            # Get original prediction
            simple_model.eval()
            with torch.no_grad():
                input_tensor = torch.from_numpy(single[np.newaxis, ...])
                original_output, _ = simple_model(input_tensor)
            
            # Save and load
            save_model(simple_model, save_path, sample_config)
            loaded_model, _, _ = load_model(save_path + '.pt')
            
            # Get loaded prediction
            loaded_model.eval()
            with torch.no_grad():
                loaded_output, _ = loaded_model(input_tensor)
            
            # Compare outputs
            assert torch.allclose(original_output, loaded_output, atol=1e-6)

