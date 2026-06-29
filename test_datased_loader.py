"""
Unit tests for dataset loader.
"""

import pytest
import numpy as np
import mne
from pathlib import Path
import tempfile
import shutil

from src.dataset_loader import DatasetLoader, DataConfig
from src.data_models import EEGDataset


class TestDatasetLoader:
    """Test cases for DatasetLoader class."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config(self, temp_data_dir):
        """Create a test configuration."""
        return DataConfig(
            datasets=['physionet'],
            sampling_rate=256.0,
            data_dir=str(temp_data_dir)
        )
    
    @pytest.fixture
    def synthetic_edf_files(self, temp_data_dir):
        """Create synthetic EDF files for testing."""
        # Create directory structure
        control_dir = temp_data_dir / 'physionet' / 'control'
        ad_dir = temp_data_dir / 'physionet' / 'ad'
        control_dir.mkdir(parents=True)
        ad_dir.mkdir(parents=True)
        
        # Create synthetic EEG data
        n_channels = 5
        sfreq = 256.0
        duration = 10  # seconds
        n_samples = int(sfreq * duration)
        
        ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'Fz']
        ch_types = ['eeg'] * n_channels
        
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        
        # Create control subjects
        for i in range(3):
            data = np.random.randn(n_channels, n_samples) * 1e-5  # Scale to realistic EEG amplitude
            raw = mne.io.RawArray(data, info)
            filename = control_dir / f'S{i:03d}.edf'
            raw.export(str(filename), fmt='edf', overwrite=True)
        
        # Create AD subjects
        for i in range(3, 6):
            data = np.random.randn(n_channels, n_samples) * 1e-5
            raw = mne.io.RawArray(data, info)
            filename = ad_dir / f'S{i:03d}.edf'
            raw.export(str(filename), fmt='edf', overwrite=True)
        
        return temp_data_dir / 'physionet'
    
    def test_load_physionet_with_directory_structure(self, config, synthetic_edf_files):
        """Test loading PhysioNet data with automatic label inference."""
        loader = DatasetLoader(config)
        
        dataset = loader.load_physionet(dataset_path=synthetic_edf_files)
        
        assert isinstance(dataset, EEGDataset)
        assert len(dataset) == 6  # 3 control + 3 AD
        assert dataset.signals.shape[0] == 6
        assert dataset.signals.shape[1] == 5  # 5 channels
        assert dataset.sampling_rate == 256.0
        
        # Check labels
        assert np.sum(dataset.labels == 0) == 3  # 3 control
        assert np.sum(dataset.labels == 1) == 3  # 3 AD
        
        # Validate dataset
        assert dataset.validate() is True
    
    def test_load_physionet_with_label_mapping(self, config, synthetic_edf_files):
        """Test loading PhysioNet data with explicit label mapping."""
        loader = DatasetLoader(config)
        
        # Provide explicit label mapping (with directory prefix)
        label_mapping = {
            'control_S000': 0, 'control_S001': 0, 'control_S002': 0,
            'ad_S003': 1, 'ad_S004': 1, 'ad_S005': 1
        }
        
        dataset = loader.load_physionet(
            dataset_path=synthetic_edf_files,
            label_mapping=label_mapping
        )
        
        assert len(dataset) == 6
        assert dataset.validate() is True
    
    def test_load_physionet_nonexistent_path(self, config):
        """Test that loading from nonexistent path raises FileNotFoundError."""
        loader = DatasetLoader(config)
        
        with pytest.raises(FileNotFoundError):
            loader.load_physionet(dataset_path='/nonexistent/path')
    
    def test_load_physionet_no_edf_files(self, config, temp_data_dir):
        """Test that loading from directory with no EDF files raises ValueError."""
        empty_dir = temp_data_dir / 'empty'
        empty_dir.mkdir()
        
        loader = DatasetLoader(config)
        
        with pytest.raises(ValueError, match="No EDF files found"):
            loader.load_physionet(dataset_path=empty_dir)
    
    def test_extract_subject_id(self, config):
        """Test subject ID extraction from various filename formats."""
        loader = DatasetLoader(config)
        
        # Without category directory
        assert loader._extract_subject_id(Path('data/S001.edf')) == 'S001'
        assert loader._extract_subject_id(Path('data/subject_123.edf')) == '123'
        
        # With category directory
        assert loader._extract_subject_id(Path('data/control/S001.edf')) == 'control_S001'
        assert loader._extract_subject_id(Path('data/ad/S002.edf')) == 'ad_S002'
        assert loader._extract_subject_id(Path('data/healthy/subject_ABC.edf')) == 'healthy_ABC'
    
    def test_infer_labels_from_structure(self, config):
        """Test label inference from directory structure."""
        loader = DatasetLoader(config)
        
        edf_files = [
            Path('/data/control/S001.edf'),
            Path('/data/control/S002.edf'),
            Path('/data/ad/S003.edf'),
            Path('/data/alzheimer/S004.edf'),
            Path('/data/healthy/S005.edf'),
        ]
        
        label_mapping = loader._infer_labels_from_structure(edf_files)
        
        assert label_mapping['control_S001'] == 0  # control
        assert label_mapping['control_S002'] == 0  # control
        assert label_mapping['ad_S003'] == 1  # ad
        assert label_mapping['alzheimer_S004'] == 1  # alzheimer
        assert label_mapping['healthy_S005'] == 0  # healthy



class TestDataSplitting:
    """Test cases for data splitting functionality."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing splits."""
        n_subjects = 100
        n_channels = 5
        n_samples = 256
        
        signals = np.random.randn(n_subjects, n_channels, n_samples)
        # Create balanced dataset: 50 control, 50 AD
        labels = np.array([0] * 50 + [1] * 50)
        channel_names = [f'Ch{i}' for i in range(n_channels)]
        subject_ids = [f'S{i:03d}' for i in range(n_subjects)]
        
        return EEGDataset(
            signals=signals,
            labels=labels,
            channel_names=channel_names,
            sampling_rate=256.0,
            subject_ids=subject_ids
        )
    
    def test_create_splits_basic(self, sample_dataset):
        """Test basic train/val/test split creation."""
        config = DataConfig(
            datasets=['test'],
            sampling_rate=256.0,
            data_dir='.'
        )
        loader = DatasetLoader(config)
        
        train_idx, val_idx, test_idx = loader.create_splits(
            sample_dataset,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )
        
        # Check sizes
        assert len(train_idx) == 70
        assert len(val_idx) == 15
        assert len(test_idx) == 15
        
        # Check no overlap
        assert len(set(train_idx) & set(val_idx)) == 0
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(set(val_idx) & set(test_idx)) == 0
        
        # Check all indices covered
        all_indices = set(train_idx + val_idx + test_idx)
        assert all_indices == set(range(100))
    
    def test_create_splits_stratified(self, sample_dataset):
        """Test stratified splitting maintains class balance."""
        config = DataConfig(
            datasets=['test'],
            sampling_rate=256.0,
            data_dir='.'
        )
        loader = DatasetLoader(config)
        
        train_idx, val_idx, test_idx = loader.create_splits(
            sample_dataset,
            stratify=True,
            random_seed=42
        )
        
        # Check class balance in each split
        train_labels = sample_dataset.labels[train_idx]
        val_labels = sample_dataset.labels[val_idx]
        test_labels = sample_dataset.labels[test_idx]
        
        # Should be roughly balanced (within 1-2 samples due to rounding)
        assert abs(np.sum(train_labels == 0) - np.sum(train_labels == 1)) <= 2
        assert abs(np.sum(val_labels == 0) - np.sum(val_labels == 1)) <= 2
        assert abs(np.sum(test_labels == 0) - np.sum(test_labels == 1)) <= 2
    
    def test_create_splits_invalid_ratios(self, sample_dataset):
        """Test that invalid ratios raise ValueError."""
        config = DataConfig(
            datasets=['test'],
            sampling_rate=256.0,
            data_dir='.'
        )
        loader = DatasetLoader(config)
        
        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            loader.create_splits(
                sample_dataset,
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3  # Sum > 1.0
            )
    
    def test_create_k_fold_splits(self, sample_dataset):
        """Test k-fold cross-validation split creation."""
        config = DataConfig(
            datasets=['test'],
            sampling_rate=256.0,
            data_dir='.'
        )
        loader = DatasetLoader(config)
        
        k = 5
        fold_splits = loader.create_k_fold_splits(
            sample_dataset,
            k=k,
            stratify=True,
            random_seed=42
        )
        
        assert len(fold_splits) == k
        
        # Check each fold
        for train_idx, val_idx in fold_splits:
            # Check no overlap
            assert len(set(train_idx) & set(val_idx)) == 0
            
            # Check sizes (approximately 80/20 split for 5-fold)
            assert len(train_idx) == 80
            assert len(val_idx) == 20
            
            # Check class balance in validation fold
            val_labels = sample_dataset.labels[val_idx]
            assert abs(np.sum(val_labels == 0) - np.sum(val_labels == 1)) <= 2
        
        # Check that each sample appears in exactly one validation fold
        all_val_indices = []
        for _, val_idx in fold_splits:
            all_val_indices.extend(val_idx)
        
        assert len(all_val_indices) == 100
        assert len(set(all_val_indices)) == 100  # No duplicates
    
    def test_create_k_fold_invalid_k(self, sample_dataset):
        """Test that invalid k raises ValueError."""
        config = DataConfig(
            datasets=['test'],
            sampling_rate=256.0,
            data_dir='.'
        )
        loader = DatasetLoader(config)
        
        with pytest.raises(ValueError, match="k must be at least 2"):
            loader.create_k_fold_splits(sample_dataset, k=1)



class TestDataQualityValidation:
    """Test cases for data quality validation."""
    
    @pytest.fixture
    def valid_dataset(self):
        """Create a valid dataset for testing."""
        n_subjects = 20
        n_channels = 5
        n_samples = 256
        
        signals = np.random.randn(n_subjects, n_channels, n_samples) * 1e-5
        labels = np.array([0] * 10 + [1] * 10)
        channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'Fz']
        subject_ids = [f'S{i:03d}' for i in range(n_subjects)]
        
        return EEGDataset(
            signals=signals,
            labels=labels,
            channel_names=channel_names,
            sampling_rate=256.0,
            subject_ids=subject_ids
        )
    
    def test_validate_quality_pass(self, valid_dataset):
        """Test quality validation on valid dataset."""
        config = DataConfig(
            datasets=['test'],
            sampling_rate=256.0,
            data_dir='.'
        )
        loader = DatasetLoader(config)
        
        report = loader.validate_data_quality(
            valid_dataset,
            expected_sampling_rate=256.0,
            expected_n_channels=5
        )
        
        assert report['validation_passed'] is True
        assert len(report['errors']) == 0
        assert report['n_subjects'] == 20
        assert report['n_channels'] == 5
        assert report['sampling_rate'] == 256.0
        assert 'signal_statistics' in report
        assert 'channel_statistics' in report
        assert 'class_distribution' in report
    
    def test_validate_quality_sampling_rate_mismatch(self, valid_dataset):
        """Test quality validation fails on sampling rate mismatch."""
        config = DataConfig(
            datasets=['test'],
            sampling_rate=256.0,
            data_dir='.'
        )
        loader = DatasetLoader(config)
        
        report = loader.validate_data_quality(
            valid_dataset,
            expected_sampling_rate=512.0  # Mismatch
        )
        
        assert report['validation_passed'] is False
        assert len(report['errors']) > 0
        assert any('Sampling rate mismatch' in err for err in report['errors'])
    
    def test_validate_quality_channel_count_mismatch(self, valid_dataset):
        """Test quality validation fails on channel count mismatch."""
        config = DataConfig(
            datasets=['test'],
            sampling_rate=256.0,
            data_dir='.'
        )
        loader = DatasetLoader(config)
        
        report = loader.validate_data_quality(
            valid_dataset,
            expected_n_channels=10  # Mismatch
        )
        
        assert report['validation_passed'] is False
        assert len(report['errors']) > 0
        assert any('Channel count mismatch' in err for err in report['errors'])
    
    def test_validate_quality_nan_values(self):
        """Test quality validation detects NaN values."""
        signals = np.random.randn(10, 5, 256)
        signals[0, 0, 0] = np.nan  # Introduce NaN
        
        dataset = EEGDataset(
            signals=signals,
            labels=np.array([0] * 5 + [1] * 5),
            channel_names=['Ch0', 'Ch1', 'Ch2', 'Ch3', 'Ch4'],
            sampling_rate=256.0,
            subject_ids=[f'S{i:03d}' for i in range(10)]
        )
        
        config = DataConfig(
            datasets=['test'],
            sampling_rate=256.0,
            data_dir='.'
        )
        loader = DatasetLoader(config)
        
        report = loader.validate_data_quality(dataset)
        
        assert len(report['warnings']) > 0
        assert any('NaN' in warn for warn in report['warnings'])
        assert report['n_nan_values'] == 1
    
    def test_validate_quality_class_imbalance(self):
        """Test quality validation detects class imbalance."""
        signals = np.random.randn(100, 5, 256)
        labels = np.array([0] * 90 + [1] * 10)  # 9:1 imbalance
        
        dataset = EEGDataset(
            signals=signals,
            labels=labels,
            channel_names=['Ch0', 'Ch1', 'Ch2', 'Ch3', 'Ch4'],
            sampling_rate=256.0,
            subject_ids=[f'S{i:03d}' for i in range(100)]
        )
        
        config = DataConfig(
            datasets=['test'],
            sampling_rate=256.0,
            data_dir='.'
        )
        loader = DatasetLoader(config)
        
        report = loader.validate_data_quality(dataset)
        
        assert len(report['warnings']) > 0
        assert any('class imbalance' in warn.lower() for warn in report['warnings'])

