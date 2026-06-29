"""
Unit tests for the Preprocessor class.

Tests cover bandpass filtering, epoch segmentation, normalization,
artifact detection, and the full preprocessing pipeline.
"""

import pytest
import numpy as np
from src.preprocessor import Preprocessor
from src.data_models import EEGDataset, EEGEpoch, PreprocessedDataset


class TestBandpassFilter:
    """Test cases for bandpass filtering."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a preprocessor with default config."""
        config = {
            'bandpass_low': 0.5,
            'bandpass_high': 45.0,
            'normalization': 'zscore',
            'artifact_threshold': 5.0
        }
        return Preprocessor(config)
    
    def test_bandpass_filter_basic(self, preprocessor):
        """Test bandpass filter on synthetic signal."""
        # Create synthetic signal with 5 channels, 1024 samples
        fs = 256.0
        n_channels = 5
        n_samples = 1024
        signal = np.random.randn(n_channels, n_samples)
        
        filtered = preprocessor.bandpass_filter(signal, 0.5, 45.0, fs)
        
        assert filtered.shape == signal.shape
        assert not np.allclose(filtered, signal)  # Should be different
    
    def test_bandpass_filter_preserves_shape(self, preprocessor):
        """Test that bandpass filter preserves signal shape."""
        fs = 256.0
        signal = np.random.randn(19, 2048)
        
        filtered = preprocessor.bandpass_filter(signal, 1.0, 40.0, fs)
        
        assert filtered.shape == signal.shape
    
    def test_bandpass_filter_invalid_low_freq(self, preprocessor):
        """Test that invalid low frequency raises ValueError."""
        signal = np.random.randn(5, 1024)
        
        with pytest.raises(ValueError, match="Low cutoff frequency must be positive"):
            preprocessor.bandpass_filter(signal, -1.0, 45.0, 256.0)
    
    def test_bandpass_filter_invalid_high_freq(self, preprocessor):
        """Test that high freq >= Nyquist raises ValueError."""
        signal = np.random.randn(5, 1024)
        
        with pytest.raises(ValueError, match="High cutoff.*must be less than Nyquist"):
            preprocessor.bandpass_filter(signal, 0.5, 130.0, 256.0)  # Nyquist = 128
    
    def test_bandpass_filter_low_greater_than_high(self, preprocessor):
        """Test that low >= high raises ValueError."""
        signal = np.random.randn(5, 1024)
        
        with pytest.raises(ValueError, match="High cutoff must be greater than low"):
            preprocessor.bandpass_filter(signal, 50.0, 45.0, 256.0)


class TestEpochSegmentation:
    """Test cases for epoch segmentation."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a preprocessor with default config."""
        config = {'normalization': 'zscore'}
        return Preprocessor(config)
    
    def test_create_epochs_basic(self, preprocessor):
        """Test basic epoch creation."""
        fs = 256.0
        epoch_duration = 2.0  # 2 seconds
        n_channels = 5
        n_samples = int(fs * 10)  # 10 seconds of data
        
        signal = np.random.randn(n_channels, n_samples)
        epochs = preprocessor.create_epochs(signal, epoch_duration, fs)
        
        # Should create 5 epochs of 2 seconds each
        expected_n_epochs = 5
        expected_epoch_samples = int(epoch_duration * fs)
        
        assert epochs.shape == (expected_n_epochs, n_channels, expected_epoch_samples)
    
    def test_create_epochs_preserves_order(self, preprocessor):
        """Test that epochs preserve temporal order."""
        fs = 256.0
        epoch_duration = 1.0
        n_channels = 2
        n_samples = int(fs * 3)  # 3 seconds
        
        # Create signal with increasing values
        signal = np.arange(n_channels * n_samples).reshape(n_channels, n_samples).astype(float)
        epochs = preprocessor.create_epochs(signal, epoch_duration, fs)
        
        # First sample of epoch 1 should be greater than first sample of epoch 0
        assert epochs[1, 0, 0] > epochs[0, 0, 0]
        assert epochs[2, 0, 0] > epochs[1, 0, 0]
    
    def test_create_epochs_invalid_duration(self, preprocessor):
        """Test that invalid epoch duration raises ValueError."""
        signal = np.random.randn(5, 1024)
        
        with pytest.raises(ValueError, match="Epoch duration must be positive"):
            preprocessor.create_epochs(signal, -1.0, 256.0)
    
    def test_create_epochs_duration_exceeds_signal(self, preprocessor):
        """Test that epoch duration > signal length raises ValueError."""
        signal = np.random.randn(5, 256)  # 1 second at 256 Hz
        
        with pytest.raises(ValueError, match="Epoch duration.*exceeds signal length"):
            preprocessor.create_epochs(signal, 2.0, 256.0)  # 2 seconds


class TestNormalization:
    """Test cases for z-score normalization."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a preprocessor with default config."""
        config = {'normalization': 'zscore'}
        return Preprocessor(config)
    
    def test_zscore_normalization_basic(self, preprocessor):
        """Test z-score normalization produces mean~0, std~1."""
        n_epochs = 10
        n_channels = 5
        epoch_samples = 256
        
        epochs = np.random.randn(n_epochs, n_channels, epoch_samples) * 10 + 5
        normalized = preprocessor.normalize(epochs, method='zscore')
        
        # Check mean and std for each channel in each epoch
        for e in range(n_epochs):
            for c in range(n_channels):
                mean = np.mean(normalized[e, c, :])
                std = np.std(normalized[e, c, :])
                assert abs(mean) < 0.01, f"Mean {mean} not close to 0"
                assert abs(std - 1.0) < 0.01, f"Std {std} not close to 1"
    
    def test_zscore_handles_zero_variance(self, preprocessor):
        """Test z-score handles zero variance channels gracefully."""
        epochs = np.ones((2, 3, 256))  # Constant values
        
        # Should not raise, should set to 0
        normalized = preprocessor.normalize(epochs, method='zscore')
        
        assert np.allclose(normalized, 0.0)
    
    def test_minmax_normalization(self, preprocessor):
        """Test min-max normalization produces values in [0, 1]."""
        epochs = np.random.randn(5, 3, 256) * 100
        normalized = preprocessor.normalize(epochs, method='minmax')
        
        assert np.all(normalized >= 0.0)
        assert np.all(normalized <= 1.0)
    
    def test_invalid_normalization_method(self, preprocessor):
        """Test that invalid method raises ValueError."""
        epochs = np.random.randn(5, 3, 256)
        
        with pytest.raises(ValueError, match="Unsupported normalization method"):
            preprocessor.normalize(epochs, method='invalid')


class TestArtifactDetection:
    """Test cases for artifact detection."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a preprocessor with default config."""
        config = {'artifact_threshold': 5.0}
        return Preprocessor(config)
    
    def test_detect_artifacts_clean_signal(self, preprocessor):
        """Test that clean signal has no artifacts flagged."""
        # Create clean signal with normal distribution
        epochs = np.random.randn(10, 5, 256)
        
        artifact_flags = preprocessor.detect_artifacts(epochs, threshold=5.0)
        
        # Most epochs should be clean (z-scores rarely exceed 5)
        assert np.sum(artifact_flags) < len(artifact_flags)
    
    def test_detect_artifacts_with_spike(self, preprocessor):
        """Test that epochs with extreme values are flagged."""
        epochs = np.random.randn(5, 3, 256)
        
        # Add extreme spike to epoch 2
        epochs[2, 0, 100] = 1000.0  # Extreme outlier
        
        artifact_flags = preprocessor.detect_artifacts(epochs, threshold=5.0)
        
        assert artifact_flags[2] == True
    
    def test_detect_flatline_channel(self, preprocessor):
        """Test that flat-line channels are detected."""
        epochs = np.random.randn(5, 3, 256)
        
        # Make epoch 3, channel 1 flat
        epochs[3, 1, :] = 0.0
        
        artifact_flags = preprocessor.detect_artifacts(epochs, threshold=5.0)
        
        assert artifact_flags[3] == True


class TestPreprocessDataset:
    """Test cases for full preprocessing pipeline."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a preprocessor with default config."""
        config = {
            'bandpass_low': 0.5,
            'bandpass_high': 45.0,
            'normalization': 'zscore',
            'artifact_threshold': 5.0,
            'epoch_duration': 2.0,
            'remove_artifacts': False  # Skip ICA for faster tests
        }
        return Preprocessor(config)
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample EEG dataset for testing."""
        n_subjects = 5
        n_channels = 5
        n_samples = 2560  # 10 seconds at 256 Hz
        
        signals = np.random.randn(n_subjects, n_channels, n_samples) * 1e-5
        labels = np.array([0, 1, 0, 1, 0])
        channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'Fz']
        subject_ids = [f'S{i:03d}' for i in range(n_subjects)]
        
        return EEGDataset(
            signals=signals,
            labels=labels,
            channel_names=channel_names,
            sampling_rate=256.0,
            subject_ids=subject_ids
        )
    
    def test_preprocess_dataset_basic(self, preprocessor, sample_dataset):
        """Test full preprocessing pipeline."""
        result = preprocessor.preprocess_dataset(
            sample_dataset,
            epoch_duration=2.0,
            apply_ica=False
        )
        
        assert isinstance(result, PreprocessedDataset)
        assert len(result.epochs) > 0
        
        # Each subject should produce multiple epochs
        # 10 seconds / 2 seconds per epoch = 5 epochs per subject
        # 5 subjects * 5 epochs = 25 total epochs
        assert len(result.epochs) == 25
    
    def test_preprocess_dataset_preserves_labels(self, preprocessor, sample_dataset):
        """Test that preprocessing preserves subject labels."""
        result = preprocessor.preprocess_dataset(
            sample_dataset,
            epoch_duration=2.0,
            apply_ica=False
        )
        
        # Check that epochs have correct labels
        for epoch in result.epochs:
            assert epoch.label in [0, 1]
    
    def test_preprocess_dataset_stores_params(self, preprocessor, sample_dataset):
        """Test that preprocessing parameters are stored."""
        result = preprocessor.preprocess_dataset(
            sample_dataset,
            epoch_duration=2.0,
            apply_ica=False
        )
        
        assert 'bandpass_low' in result.preprocessing_params
        assert 'bandpass_high' in result.preprocessing_params
        assert 'normalization' in result.preprocessing_params
        assert result.preprocessing_params['epoch_duration'] == 2.0

