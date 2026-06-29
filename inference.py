#!/usr/bin/env python
"""
Inference script for Alzheimer's Disease Detection from EEG.

This script loads a pre-trained model and makes predictions on new EEG data:
1. Load pre-trained model from checkpoint
2. Accept new EEG data (from file or directory)
3. Generate predictions with confidence scores

Usage:
    python inference.py --model checkpoints/best_model.pt --input data/new_subject.edf
    python inference.py --model checkpoints/best_model.pt --input data/test_subjects/
    python inference.py --model checkpoints/best_model.pt --input data/test.npy

Requirements: 10.2, 10.4
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model_persistence import load_model, InferenceAPI, ModelPrediction
from src.transformer import EEGTransformer
from src.preprocessor import Preprocessor
from src.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run inference on EEG data for Alzheimer\'s disease detection'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to pre-trained model checkpoint (.pt file)'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input EEG data (file or directory)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to save predictions (JSON file)'
    )
    parser.add_argument(
        '--sampling_rate',
        type=float,
        default=256.0,
        help='Sampling rate of input data in Hz (default: 256)'
    )
    parser.add_argument(
        '--no_preprocess',
        action='store_true',
        help='Skip preprocessing (use if data is already preprocessed)'
    )
    parser.add_argument(
        '--attention',
        action='store_true',
        help='Return attention weights for interpretability'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for inference (default: 32)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda'],
        help='Device for inference (default: auto-detect)'
    )
    return parser.parse_args()


def load_edf_file(filepath: str, sampling_rate: float = 256.0) -> Tuple[np.ndarray, List[str]]:
    """
    Load EEG data from an EDF file.
    
    Args:
        filepath: Path to EDF file
        sampling_rate: Target sampling rate
        
    Returns:
        Tuple of (signal_data, channel_names)
    """
    try:
        import mne
        
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        
        # Resample if necessary
        if raw.info['sfreq'] != sampling_rate:
            logger.info(f"Resampling from {raw.info['sfreq']} Hz to {sampling_rate} Hz")
            raw.resample(sampling_rate)
        
        signal_data = raw.get_data()
        channel_names = raw.ch_names
        
        return signal_data, channel_names
        
    except ImportError:
        raise ImportError("MNE library required for EDF file loading. Install with: pip install mne")


def load_numpy_file(filepath: str) -> np.ndarray:
    """
    Load EEG data from a NumPy file.
    
    Args:
        filepath: Path to .npy or .npz file
        
    Returns:
        Signal data array
    """
    if filepath.endswith('.npz'):
        data = np.load(filepath)
        # Try common key names
        for key in ['signals', 'data', 'eeg', 'X']:
            if key in data:
                return data[key]
        # Return first array if no common key found
        return data[list(data.keys())[0]]
    else:
        return np.load(filepath)


def load_input_data(
    input_path: str,
    sampling_rate: float = 256.0
) -> Tuple[np.ndarray, Optional[List[str]], List[str]]:
    """
    Load EEG data from file or directory.
    
    Args:
        input_path: Path to input file or directory
        sampling_rate: Target sampling rate
        
    Returns:
        Tuple of (signals, channel_names, file_identifiers)
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    
    signals_list = []
    channel_names = None
    file_ids = []
    
    if input_path.is_file():
        # Single file
        files = [input_path]
    else:
        # Directory - find all supported files
        files = list(input_path.glob('*.edf')) + \
                list(input_path.glob('*.EDF')) + \
                list(input_path.glob('*.npy')) + \
                list(input_path.glob('*.npz'))
        
        if not files:
            raise ValueError(f"No supported files found in {input_path}")
    
    logger.info(f"Found {len(files)} file(s) to process")
    
    for filepath in files:
        try:
            filepath_str = str(filepath)
            
            if filepath_str.lower().endswith('.edf'):
                signal, ch_names = load_edf_file(filepath_str, sampling_rate)
                if channel_names is None:
                    channel_names = ch_names
            elif filepath_str.endswith(('.npy', '.npz')):
                signal = load_numpy_file(filepath_str)
            else:
                logger.warning(f"Unsupported file format: {filepath}")
                continue
            
            # Ensure 2D shape (n_channels, n_samples)
            if signal.ndim == 1:
                signal = signal.reshape(1, -1)
            elif signal.ndim == 3:
                # Multiple samples in one file
                for i in range(signal.shape[0]):
                    signals_list.append(signal[i])
                    file_ids.append(f"{filepath.stem}_{i}")
                continue
            
            signals_list.append(signal)
            file_ids.append(filepath.stem)
            
            logger.info(f"Loaded {filepath.name}: shape {signal.shape}")
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            continue
    
    if not signals_list:
        raise ValueError("No valid EEG data could be loaded")
    
    # Standardize signal lengths (truncate to minimum)
    min_samples = min(s.shape[1] for s in signals_list)
    signals = np.array([s[:, :min_samples] for s in signals_list], dtype=np.float32)
    
    logger.info(f"Loaded {len(signals)} samples with shape {signals.shape}")
    
    return signals, channel_names, file_ids


def run_inference(
    model: EEGTransformer,
    signals: np.ndarray,
    config: Dict[str, Any],
    device: torch.device,
    preprocess: bool = True,
    return_attention: bool = False,
    sampling_rate: float = 256.0,
    batch_size: int = 32
) -> List[ModelPrediction]:
    """
    Run inference on EEG signals.
    
    Args:
        model: Loaded EEGTransformer model
        signals: EEG signals (n_samples, n_channels, n_samples)
        config: Model configuration
        device: Device for inference
        preprocess: Whether to preprocess signals
        return_attention: Whether to return attention weights
        sampling_rate: Sampling rate for preprocessing
        batch_size: Batch size for inference
        
    Returns:
        List of ModelPrediction objects
        
    Requirements: 10.4
    """
    # Create inference API
    inference_api = InferenceAPI(model, config, device)
    
    predictions = []
    n_samples = len(signals)
    
    logger.info(f"Running inference on {n_samples} samples...")
    
    for i in range(0, n_samples, batch_size):
        batch_signals = signals[i:i+batch_size]
        
        batch_predictions = inference_api.predict(
            eeg_data=batch_signals,
            sampling_rate=sampling_rate,
            preprocess=preprocess,
            return_attention=return_attention
        )
        
        if isinstance(batch_predictions, list):
            predictions.extend(batch_predictions)
        else:
            predictions.append(batch_predictions)
        
        logger.info(f"Processed {min(i+batch_size, n_samples)}/{n_samples} samples")
    
    return predictions


def format_predictions(
    predictions: List[ModelPrediction],
    file_ids: List[str]
) -> Dict[str, Any]:
    """
    Format predictions for output.
    
    Args:
        predictions: List of ModelPrediction objects
        file_ids: List of file identifiers
        
    Returns:
        Dictionary with formatted predictions
    """
    results = {
        'predictions': [],
        'summary': {
            'total_samples': len(predictions),
            'predicted_ad': 0,
            'predicted_control': 0,
            'average_confidence': 0.0
        }
    }
    
    total_confidence = 0.0
    
    for pred, file_id in zip(predictions, file_ids):
        pred_dict = {
            'file_id': file_id,
            'predicted_class': pred.predicted_class,
            'class_label': 'AD' if pred.predicted_class == 1 else 'Control',
            'confidence': float(pred.confidence),
            'probabilities': {
                'Control': float(pred.probabilities[0]),
                'AD': float(pred.probabilities[1])
            }
        }
        results['predictions'].append(pred_dict)
        
        if pred.predicted_class == 1:
            results['summary']['predicted_ad'] += 1
        else:
            results['summary']['predicted_control'] += 1
        
        total_confidence += pred.confidence
    
    results['summary']['average_confidence'] = total_confidence / len(predictions) if predictions else 0.0
    
    return results


def print_predictions(results: Dict[str, Any]) -> None:
    """Print predictions in a readable format."""
    print("\n" + "="*60)
    print("INFERENCE RESULTS")
    print("="*60)
    
    print(f"\nTotal samples: {results['summary']['total_samples']}")
    print(f"Predicted AD: {results['summary']['predicted_ad']}")
    print(f"Predicted Control: {results['summary']['predicted_control']}")
    print(f"Average confidence: {results['summary']['average_confidence']:.4f}")
    
    print("\n" + "-"*60)
    print("Individual Predictions:")
    print("-"*60)
    
    for pred in results['predictions']:
        confidence_bar = "█" * int(pred['confidence'] * 20)
        print(f"\n{pred['file_id']}:")
        print(f"  Prediction: {pred['class_label']}")
        print(f"  Confidence: {pred['confidence']:.4f} [{confidence_bar}]")
        print(f"  P(Control): {pred['probabilities']['Control']:.4f}")
        print(f"  P(AD):      {pred['probabilities']['AD']:.4f}")


def save_predictions(results: Dict[str, Any], output_path: str) -> None:
    """Save predictions to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Predictions saved to {output_path}")


def main():
    """Main entry point for inference."""
    args = parse_args()
    
    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    
    # Check model file exists
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        sys.exit(1)
    
    try:
        # Load model
        logger.info(f"Loading model from {args.model}")
        model, config, metadata = load_model(
            checkpoint_path=args.model,
            device=device
        )
        
        logger.info(f"Model loaded successfully")
        logger.info(f"  Training date: {metadata.training_date}")
        logger.info(f"  Dataset: {metadata.dataset_name}")
        logger.info(f"  Accuracy: {metadata.accuracy:.4f}")
        logger.info(f"  F1 Score: {metadata.f1_score:.4f}")
        
        # Load input data
        logger.info(f"Loading input data from {args.input}")
        signals, channel_names, file_ids = load_input_data(
            args.input,
            sampling_rate=args.sampling_rate
        )
        
        # Verify input shape matches model expectations
        model_config = model.get_config()
        expected_channels = model_config.get('n_channels')
        expected_samples = model_config.get('n_samples')
        
        if expected_channels and signals.shape[1] != expected_channels:
            logger.warning(
                f"Channel count mismatch: input has {signals.shape[1]}, "
                f"model expects {expected_channels}"
            )
        
        if expected_samples and signals.shape[2] != expected_samples:
            logger.warning(
                f"Sample count mismatch: input has {signals.shape[2]}, "
                f"model expects {expected_samples}. Truncating/padding..."
            )
            # Truncate or pad to expected length
            if signals.shape[2] > expected_samples:
                signals = signals[:, :, :expected_samples]
            else:
                padded = np.zeros((signals.shape[0], signals.shape[1], expected_samples), dtype=np.float32)
                padded[:, :, :signals.shape[2]] = signals
                signals = padded
        
        # Run inference
        predictions = run_inference(
            model=model,
            signals=signals,
            config=config,
            device=device,
            preprocess=not args.no_preprocess,
            return_attention=args.attention,
            sampling_rate=args.sampling_rate,
            batch_size=args.batch_size
        )
        
        # Format results
        results = format_predictions(predictions, file_ids)
        
        # Print results
        print_predictions(results)
        
        # Save results if output path specified
        if args.output:
            save_predictions(results, args.output)
        
        logger.info("\nInference complete!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Inference failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

