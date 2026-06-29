"""
Baseline models for EEG-based Alzheimer's disease detection.

This module implements baseline architectures for comparison with the
Transformer model:
- 1D CNN: Temporal feature extraction
- 2D CNN: Spatial-temporal feature extraction
- LSTM: Sequential processing
- Classical ML: Hand-crafted features with SVM, Random Forest, XGBoost
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1D(nn.Module):
    """
    1D Convolutional Neural Network for temporal EEG classification.
    
    Architecture: Conv1D → ReLU → MaxPool → Conv1D → ReLU → MaxPool → FC → Softmax
    
    Processes temporal EEG features by applying 1D convolutions along the
    time dimension, treating each channel independently then combining.
    
    Attributes:
        n_channels: Number of EEG channels
        n_samples: Number of samples per epoch
        num_classes: Number of output classes (default: 2)
    """
    
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        num_classes: int = 2,
        conv1_filters: int = 32,
        conv2_filters: int = 64,
        kernel_size: int = 7,
        pool_size: int = 2,
        fc_dim: int = 128,
        dropout: float = 0.5
    ):
        """
        Initialize the 1D CNN model.
        
        Args:
            n_channels: Number of input EEG channels
            n_samples: Number of samples per EEG epoch
            num_classes: Number of output classes
            conv1_filters: Number of filters in first conv layer
            conv2_filters: Number of filters in second conv layer
            kernel_size: Kernel size for convolutions
            pool_size: Pool size for max pooling
            fc_dim: Dimension of fully connected layer
            dropout: Dropout probability
        """
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.num_classes = num_classes

        # First convolutional block
        self.conv1 = nn.Conv1d(
            in_channels=n_channels,
            out_channels=conv1_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.pool1 = nn.MaxPool1d(kernel_size=pool_size)
        
        # Second convolutional block
        self.conv2 = nn.Conv1d(
            in_channels=conv1_filters,
            out_channels=conv2_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.pool2 = nn.MaxPool1d(kernel_size=pool_size)
        
        # Calculate flattened size after convolutions and pooling
        self._flat_size = self._calculate_flat_size(n_samples, pool_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv2_filters * self._flat_size, fc_dim)
        self.fc2 = nn.Linear(fc_dim, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _calculate_flat_size(self, n_samples: int, pool_size: int) -> int:
        """Calculate the flattened size after conv and pooling layers."""
        # After first pool
        size = n_samples // pool_size
        # After second pool
        size = size // pool_size
        return size
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> tuple:
        """
        Forward pass through the 1D CNN.
        
        Args:
            x: Input tensor of shape (batch, n_channels, n_samples)
            return_attention: Ignored (for API compatibility with Transformer)
            
        Returns:
            Tuple of (logits, None) where logits has shape (batch, num_classes)
        """
        # First conv block: Conv1D → ReLU → MaxPool
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Second conv block: Conv1D → ReLU → MaxPool
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x, None
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        logits, _ = self.forward(x)
        return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        logits, _ = self.forward(x)
        return F.softmax(logits, dim=1)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], n_channels: int, n_samples: int) -> 'CNN1D':
        """Create a CNN1D from configuration dictionary."""
        return cls(
            n_channels=n_channels,
            n_samples=n_samples,
            num_classes=config.get('num_classes', 2),
            conv1_filters=config.get('conv1_filters', 32),
            conv2_filters=config.get('conv2_filters', 64),
            kernel_size=config.get('kernel_size', 7),
            pool_size=config.get('pool_size', 2),
            fc_dim=config.get('fc_dim', 128),
            dropout=config.get('dropout', 0.5)
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration as dictionary."""
        return {
            'n_channels': self.n_channels,
            'n_samples': self.n_samples,
            'num_classes': self.num_classes,
            'model_type': 'CNN1D'
        }



class CNN2D(nn.Module):
    """
    2D Convolutional Neural Network for spatial-temporal EEG classification.
    
    Architecture: Conv2D → ReLU → MaxPool → Conv2D → ReLU → MaxPool → FC → Softmax
    
    Treats EEG channels as a spatial dimension, allowing the model to learn
    both spatial (cross-channel) and temporal patterns simultaneously.
    
    Input is reshaped from (batch, n_channels, n_samples) to 
    (batch, 1, n_channels, n_samples) to apply 2D convolutions.
    
    Attributes:
        n_channels: Number of EEG channels (spatial dimension)
        n_samples: Number of samples per epoch (temporal dimension)
        num_classes: Number of output classes (default: 2)
    """
    
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        num_classes: int = 2,
        conv1_filters: int = 32,
        conv2_filters: int = 64,
        kernel_size: Tuple[int, int] = (3, 7),
        pool_size: Tuple[int, int] = (1, 2),
        fc_dim: int = 128,
        dropout: float = 0.5
    ):
        """
        Initialize the 2D CNN model.
        
        Args:
            n_channels: Number of input EEG channels (spatial dimension)
            n_samples: Number of samples per EEG epoch (temporal dimension)
            num_classes: Number of output classes
            conv1_filters: Number of filters in first conv layer
            conv2_filters: Number of filters in second conv layer
            kernel_size: Kernel size (height, width) for convolutions
            pool_size: Pool size (height, width) for max pooling
            fc_dim: Dimension of fully connected layer
            dropout: Dropout probability
        """
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.num_classes = num_classes
        
        # First convolutional block
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=conv1_filters,
            kernel_size=kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=pool_size)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(
            in_channels=conv1_filters,
            out_channels=conv2_filters,
            kernel_size=kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=pool_size)
        
        # Calculate flattened size after convolutions and pooling
        self._flat_h, self._flat_w = self._calculate_flat_size(
            n_channels, n_samples, pool_size
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv2_filters * self._flat_h * self._flat_w, fc_dim)
        self.fc2 = nn.Linear(fc_dim, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _calculate_flat_size(
        self, 
        n_channels: int, 
        n_samples: int, 
        pool_size: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Calculate the flattened size after conv and pooling layers."""
        h, w = n_channels, n_samples
        # After first pool
        h = h // pool_size[0] if pool_size[0] > 0 else h
        w = w // pool_size[1] if pool_size[1] > 0 else w
        # After second pool
        h = h // pool_size[0] if pool_size[0] > 0 else h
        w = w // pool_size[1] if pool_size[1] > 0 else w
        return max(1, h), max(1, w)
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> tuple:
        """
        Forward pass through the 2D CNN.
        
        Args:
            x: Input tensor of shape (batch, n_channels, n_samples)
            return_attention: Ignored (for API compatibility with Transformer)
            
        Returns:
            Tuple of (logits, None) where logits has shape (batch, num_classes)
        """
        # Add channel dimension: (batch, n_channels, n_samples) -> (batch, 1, n_channels, n_samples)
        x = x.unsqueeze(1)
        
        # First conv block: Conv2D → ReLU → MaxPool
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Second conv block: Conv2D → ReLU → MaxPool
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x, None
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        logits, _ = self.forward(x)
        return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        logits, _ = self.forward(x)
        return F.softmax(logits, dim=1)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], n_channels: int, n_samples: int) -> 'CNN2D':
        """Create a CNN2D from configuration dictionary."""
        return cls(
            n_channels=n_channels,
            n_samples=n_samples,
            num_classes=config.get('num_classes', 2),
            conv1_filters=config.get('conv1_filters', 32),
            conv2_filters=config.get('conv2_filters', 64),
            kernel_size=tuple(config.get('kernel_size', (3, 7))),
            pool_size=tuple(config.get('pool_size', (1, 2))),
            fc_dim=config.get('fc_dim', 128),
            dropout=config.get('dropout', 0.5)
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration as dictionary."""
        return {
            'n_channels': self.n_channels,
            'n_samples': self.n_samples,
            'num_classes': self.num_classes,
            'model_type': 'CNN2D'
        }



class LSTMModel(nn.Module):
    """
    Bidirectional LSTM for sequential EEG classification.
    
    Architecture: BiLSTM → BiLSTM → FC → Softmax
    
    Processes EEG data as a sequence, capturing temporal dependencies
    in both forward and backward directions.
    
    Input shape: (batch, n_channels, n_samples)
    The model transposes to (batch, n_samples, n_channels) to treat
    time steps as the sequence dimension.
    
    Attributes:
        n_channels: Number of EEG channels (input features per time step)
        n_samples: Number of samples per epoch (sequence length)
        num_classes: Number of output classes (default: 2)
    """
    
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        num_classes: int = 2,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        fc_dim: int = 128,
        dropout: float = 0.5
    ):
        """
        Initialize the LSTM model.
        
        Args:
            n_channels: Number of input EEG channels (features per time step)
            n_samples: Number of samples per EEG epoch (sequence length)
            num_classes: Number of output classes
            hidden_size: Hidden size of LSTM layers
            num_layers: Number of stacked LSTM layers
            bidirectional: Whether to use bidirectional LSTM
            fc_dim: Dimension of fully connected layer
            dropout: Dropout probability
        """
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Calculate LSTM output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_size, fc_dim)
        self.fc2 = nn.Linear(fc_dim, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        for module in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> tuple:
        """
        Forward pass through the LSTM.
        
        Args:
            x: Input tensor of shape (batch, n_channels, n_samples)
            return_attention: Ignored (for API compatibility with Transformer)
            
        Returns:
            Tuple of (logits, None) where logits has shape (batch, num_classes)
        """
        # Transpose: (batch, n_channels, n_samples) -> (batch, n_samples, n_channels)
        x = x.transpose(1, 2)
        
        # LSTM forward pass
        # lstm_out: (batch, n_samples, hidden_size * num_directions)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last time step output
        # For bidirectional, concatenate forward and backward final states
        if self.bidirectional:
            # h_n shape: (num_layers * 2, batch, hidden_size)
            # Get last layer's forward and backward hidden states
            forward_hidden = h_n[-2, :, :]  # (batch, hidden_size)
            backward_hidden = h_n[-1, :, :]  # (batch, hidden_size)
            hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            hidden = h_n[-1, :, :]  # (batch, hidden_size)
        
        # Fully connected layers
        x = self.fc1(hidden)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x, None
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        logits, _ = self.forward(x)
        return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        logits, _ = self.forward(x)
        return F.softmax(logits, dim=1)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], n_channels: int, n_samples: int) -> 'LSTMModel':
        """Create an LSTMModel from configuration dictionary."""
        return cls(
            n_channels=n_channels,
            n_samples=n_samples,
            num_classes=config.get('num_classes', 2),
            hidden_size=config.get('hidden_size', 128),
            num_layers=config.get('num_layers', 2),
            bidirectional=config.get('bidirectional', True),
            fc_dim=config.get('fc_dim', 128),
            dropout=config.get('dropout', 0.5)
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration as dictionary."""
        return {
            'n_channels': self.n_channels,
            'n_samples': self.n_samples,
            'num_classes': self.num_classes,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional,
            'model_type': 'LSTM'
        }



class FeatureExtractor:
    """
    Extract hand-crafted features from EEG signals for classical ML models.
    
    Features extracted:
    - Power Spectral Density (PSD) in standard frequency bands
    - Statistical moments (mean, std, skewness, kurtosis)
    - Entropy measures (sample entropy approximation)
    
    Attributes:
        sampling_rate: Sampling frequency of the EEG signals
        freq_bands: Dictionary of frequency band ranges
    """
    
    def __init__(
        self,
        sampling_rate: float = 256.0,
        freq_bands: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        """
        Initialize the feature extractor.
        
        Args:
            sampling_rate: Sampling frequency in Hz
            freq_bands: Dictionary mapping band names to (low, high) frequency ranges
        """
        self.sampling_rate = sampling_rate
        self.freq_bands = freq_bands or {
            'delta': (0.5, 4.0),
            'theta': (4.0, 8.0),
            'alpha': (8.0, 13.0),
            'beta': (13.0, 30.0),
            'gamma': (30.0, 45.0)
        }
    
    def compute_psd_features(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute power spectral density features for each frequency band.
        
        Args:
            signal: EEG signal of shape (n_channels, n_samples)
            
        Returns:
            PSD features of shape (n_channels * n_bands,)
        """
        from scipy import signal as scipy_signal
        
        n_channels, n_samples = signal.shape
        n_bands = len(self.freq_bands)
        psd_features = np.zeros((n_channels, n_bands))
        
        # Compute PSD using Welch's method
        for ch_idx in range(n_channels):
            freqs, psd = scipy_signal.welch(
                signal[ch_idx],
                fs=self.sampling_rate,
                nperseg=min(256, n_samples)
            )
            
            # Extract power in each frequency band
            for band_idx, (band_name, (low, high)) in enumerate(self.freq_bands.items()):
                band_mask = (freqs >= low) & (freqs <= high)
                if np.any(band_mask):
                    psd_features[ch_idx, band_idx] = np.mean(psd[band_mask])
        
        return psd_features.flatten()
    
    def compute_statistical_features(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute statistical moment features.
        
        Args:
            signal: EEG signal of shape (n_channels, n_samples)
            
        Returns:
            Statistical features of shape (n_channels * 4,) for mean, std, skew, kurtosis
        """
        from scipy import stats
        
        n_channels = signal.shape[0]
        stat_features = np.zeros((n_channels, 4))
        
        for ch_idx in range(n_channels):
            ch_signal = signal[ch_idx]
            stat_features[ch_idx, 0] = np.mean(ch_signal)
            stat_features[ch_idx, 1] = np.std(ch_signal)
            stat_features[ch_idx, 2] = stats.skew(ch_signal)
            stat_features[ch_idx, 3] = stats.kurtosis(ch_signal)
        
        return stat_features.flatten()
    
    def compute_entropy_features(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute entropy-based features (approximate entropy).
        
        Uses a simplified sample entropy approximation for efficiency.
        
        Args:
            signal: EEG signal of shape (n_channels, n_samples)
            
        Returns:
            Entropy features of shape (n_channels,)
        """
        n_channels = signal.shape[0]
        entropy_features = np.zeros(n_channels)
        
        for ch_idx in range(n_channels):
            ch_signal = signal[ch_idx]
            # Simplified entropy: use histogram-based entropy
            hist, _ = np.histogram(ch_signal, bins=50, density=True)
            hist = hist[hist > 0]  # Remove zero bins
            entropy_features[ch_idx] = -np.sum(hist * np.log(hist + 1e-10))
        
        return entropy_features
    
    def extract_features(self, signal: np.ndarray) -> np.ndarray:
        """
        Extract all features from an EEG signal.
        
        Args:
            signal: EEG signal of shape (n_channels, n_samples)
            
        Returns:
            Feature vector combining PSD, statistical, and entropy features
        """
        psd_features = self.compute_psd_features(signal)
        stat_features = self.compute_statistical_features(signal)
        entropy_features = self.compute_entropy_features(signal)
        
        return np.concatenate([psd_features, stat_features, entropy_features])
    
    def extract_features_batch(self, signals: np.ndarray) -> np.ndarray:
        """
        Extract features from a batch of EEG signals.
        
        Args:
            signals: EEG signals of shape (n_samples, n_channels, n_timepoints)
            
        Returns:
            Feature matrix of shape (n_samples, n_features)
        """
        n_samples = signals.shape[0]
        
        # Extract features for first sample to get feature dimension
        first_features = self.extract_features(signals[0])
        n_features = len(first_features)
        
        # Extract features for all samples
        features = np.zeros((n_samples, n_features))
        features[0] = first_features
        
        for i in range(1, n_samples):
            features[i] = self.extract_features(signals[i])
        
        return features



class ClassicalMLBaseline:
    """
    Classical machine learning baselines using hand-crafted features.
    
    Implements SVM, Random Forest, and XGBoost classifiers with
    automatic feature extraction from raw EEG signals.
    
    Attributes:
        classifier_type: Type of classifier ('svm', 'rf', 'xgboost')
        feature_extractor: FeatureExtractor instance
        classifier: Trained classifier instance
    """
    
    def __init__(
        self,
        classifier_type: str = 'svm',
        sampling_rate: float = 256.0,
        **classifier_params
    ):
        """
        Initialize the classical ML baseline.
        
        Args:
            classifier_type: Type of classifier ('svm', 'rf', 'xgboost')
            sampling_rate: Sampling frequency for feature extraction
            **classifier_params: Additional parameters for the classifier
        """
        self.classifier_type = classifier_type.lower()
        self.feature_extractor = FeatureExtractor(sampling_rate=sampling_rate)
        self.classifier = None
        self.classifier_params = classifier_params
        self._is_fitted = False
        
        # Initialize classifier
        self._init_classifier()
    
    def _init_classifier(self):
        """Initialize the classifier based on type."""
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        
        if self.classifier_type == 'svm':
            self.classifier = SVC(
                kernel=self.classifier_params.get('kernel', 'rbf'),
                C=self.classifier_params.get('C', 1.0),
                gamma=self.classifier_params.get('gamma', 'scale'),
                probability=True,
                random_state=self.classifier_params.get('random_state', 42)
            )
        elif self.classifier_type == 'rf':
            self.classifier = RandomForestClassifier(
                n_estimators=self.classifier_params.get('n_estimators', 100),
                max_depth=self.classifier_params.get('max_depth', None),
                min_samples_split=self.classifier_params.get('min_samples_split', 2),
                random_state=self.classifier_params.get('random_state', 42)
            )
        elif self.classifier_type == 'xgboost':
            try:
                from xgboost import XGBClassifier
                self.classifier = XGBClassifier(
                    n_estimators=self.classifier_params.get('n_estimators', 100),
                    max_depth=self.classifier_params.get('max_depth', 6),
                    learning_rate=self.classifier_params.get('learning_rate', 0.1),
                    random_state=self.classifier_params.get('random_state', 42),
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
            except ImportError:
                raise ImportError(
                    "XGBoost is not installed. Install it with: pip install xgboost"
                )
        else:
            raise ValueError(
                f"Unknown classifier type: {self.classifier_type}. "
                f"Supported types: 'svm', 'rf', 'xgboost'"
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ClassicalMLBaseline':
        """
        Fit the classifier on training data.
        
        Args:
            X: EEG signals of shape (n_samples, n_channels, n_timepoints)
            y: Labels of shape (n_samples,)
            
        Returns:
            Self for method chaining
        """
        # Extract features
        features = self.feature_extractor.extract_features_batch(X)
        
        # Handle NaN/Inf values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Fit classifier
        self.classifier.fit(features, y)
        self._is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input data.
        
        Args:
            X: EEG signals of shape (n_samples, n_channels, n_timepoints)
            
        Returns:
            Predicted labels of shape (n_samples,)
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction")
        
        features = self.feature_extractor.extract_features_batch(X)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return self.classifier.predict(features)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input data.
        
        Args:
            X: EEG signals of shape (n_samples, n_channels, n_timepoints)
            
        Returns:
            Class probabilities of shape (n_samples, n_classes)
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction")
        
        features = self.feature_extractor.extract_features_batch(X)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return self.classifier.predict_proba(features)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score on test data.
        
        Args:
            X: EEG signals of shape (n_samples, n_channels, n_timepoints)
            y: True labels of shape (n_samples,)
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    @classmethod
    def from_config(
        cls, 
        config: Dict[str, Any], 
        sampling_rate: float = 256.0
    ) -> 'ClassicalMLBaseline':
        """Create a ClassicalMLBaseline from configuration dictionary."""
        return cls(
            classifier_type=config.get('classifier_type', 'svm'),
            sampling_rate=sampling_rate,
            **config.get('classifier_params', {})
        )
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration as dictionary."""
        return {
            'classifier_type': self.classifier_type,
            'classifier_params': self.classifier_params,
            'model_type': 'ClassicalML'
        }


def create_baseline_model(
    model_type: str,
    n_channels: int,
    n_samples: int,
    config: Optional[Dict[str, Any]] = None,
    sampling_rate: float = 256.0
):
    """
    Factory function to create baseline models.
    
    Args:
        model_type: Type of model ('cnn1d', 'cnn2d', 'lstm', 'svm', 'rf', 'xgboost')
        n_channels: Number of EEG channels
        n_samples: Number of samples per epoch
        config: Optional configuration dictionary
        sampling_rate: Sampling rate for classical ML feature extraction
        
    Returns:
        Instantiated baseline model
    """
    config = config or {}
    model_type = model_type.lower()
    
    if model_type == 'cnn1d':
        return CNN1D.from_config(config, n_channels, n_samples)
    elif model_type == 'cnn2d':
        return CNN2D.from_config(config, n_channels, n_samples)
    elif model_type == 'lstm':
        return LSTMModel.from_config(config, n_channels, n_samples)
    elif model_type in ['svm', 'rf', 'xgboost']:
        config['classifier_type'] = model_type
        return ClassicalMLBaseline.from_config(config, sampling_rate)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported types: 'cnn1d', 'cnn2d', 'lstm', 'svm', 'rf', 'xgboost'"
        )

