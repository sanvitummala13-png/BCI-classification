"""
Download PhysioNet EEG data and run training pipeline.

This script downloads a subset of the PhysioNet EEG Motor Movement/Imagery Dataset
and runs the full training pipeline for demonstration purposes.
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import mne
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_physionet_full(data_dir: str, n_subjects: int = 109, runs_per_subject: list = None):
    """
    Download the FULL PhysioNet EEG Motor Movement/Imagery Dataset using MNE.
    
    The dataset contains 64-channel EEG recordings from 109 subjects performing 
    motor/imagery tasks across 14 runs each.
    
    For AD detection demo, we use different task conditions as proxy classes:
    - Class 0: Baseline/resting state (runs 1, 2)
    - Class 1: Motor imagery tasks (runs 3-14)
    """
    # Set MNE to not ask for confirmation
    mne.set_config('MNE_DATASETS_EEGBCI_PATH', str(Path(data_dir).absolute() / 'physionet'))
    
    data_path = Path(data_dir) / 'physionet'
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Use multiple runs for richer dataset
    if runs_per_subject is None:
        # Run 1: Baseline, eyes open
        # Run 2: Baseline, eyes closed  
        # Run 3: Task 1 (open and close left or right fist)
        # Run 4: Task 2 (imagine opening and closing left or right fist)
        # Run 5: Task 3 (open and close both fists or both feet)
        # Run 6: Task 4 (imagine opening and closing both fists or both feet)
        runs_per_subject = [1, 2, 3, 4, 5, 6]
    
    logger.info(f"Downloading FULL PhysioNet EEG data for {n_subjects} subjects...")
    logger.info(f"Runs per subject: {runs_per_subject} ({len(runs_per_subject)} runs each)")
    logger.info(f"Expected total recordings: {n_subjects * len(runs_per_subject)}")
    
    subjects = list(range(1, n_subjects + 1))
    
    all_signals = []
    all_labels = []
    subject_ids = []
    channel_names = None
    
    for subject_id in subjects:
        try:
            logger.info(f"Processing subject {subject_id}/{n_subjects}...")
            
            for run in runs_per_subject:
                try:
                    raw = mne.io.read_raw_edf(
                        mne.datasets.eegbci.load_data(subject_id, runs=[run], path=str(data_path))[0],
                        preload=True,
                        verbose=False
                    )
                    
                    # Get channel names from first recording
                    if channel_names is None:
                        channel_names = raw.ch_names
                        logger.info(f"Channels: {len(channel_names)}, Sampling rate: {raw.info['sfreq']} Hz")
                    
                    # Get signal data (channels x samples)
                    signal = raw.get_data()
                    
                    # Truncate to fixed length (10 seconds at 160 Hz = 1600 samples)
                    target_samples = 1600
                    if signal.shape[1] >= target_samples:
                        signal = signal[:, :target_samples]
                    else:
                        # Pad if too short
                        pad_width = target_samples - signal.shape[1]
                        signal = np.pad(signal, ((0, 0), (0, pad_width)), mode='constant')
                    
                    all_signals.append(signal)
                    # Label: 0 = baseline (runs 1,2), 1 = motor imagery (runs 3+)
                    label = 0 if run <= 2 else 1
                    all_labels.append(label)
                    subject_ids.append(f"S{subject_id:03d}_R{run}")
                    
                except Exception as e:
                    logger.warning(f"Failed to load subject {subject_id} run {run}: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Failed to process subject {subject_id}: {e}")
            continue
    
    if not all_signals:
        raise ValueError("No data was successfully loaded!")
    
    # Convert to arrays
    signals = np.array(all_signals)
    labels = np.array(all_labels)
    
    logger.info(f"Loaded {len(signals)} recordings: {signals.shape}")
    logger.info(f"Class distribution: Class 0 (baseline)={np.sum(labels==0)}, Class 1 (motor imagery)={np.sum(labels==1)}")
    
    # Save as numpy files for faster loading
    np.save(data_path / 'signals_full.npy', signals)
    np.save(data_path / 'labels_full.npy', labels)
    np.save(data_path / 'channel_names.npy', np.array(channel_names))
    np.save(data_path / 'subject_ids_full.npy', np.array(subject_ids))
    
    logger.info(f"Full dataset saved to {data_path}")
    
    return signals, labels, channel_names, subject_ids


def run_training():
    """Run the full training pipeline with the complete PhysioNet dataset."""
    from src.config_manager import ConfigManager
    from src.data_models import EEGDataset
    from src.preprocessor import Preprocessor
    from src.transformer import EEGTransformer, HybridCNNTransformer
    from src.training import TrainingPipeline
    from src.evaluator import Evaluator
    from src.baselines import create_baseline_model
    
    # Load or download FULL dataset
    data_path = Path('data/physionet')
    
    # Check for full dataset first, then fall back to subset
    if (data_path / 'signals_full.npy').exists():
        logger.info("Loading cached FULL dataset...")
        signals = np.load(data_path / 'signals_full.npy')
        labels = np.load(data_path / 'labels_full.npy')
        channel_names = list(np.load(data_path / 'channel_names.npy'))
        subject_ids = list(np.load(data_path / 'subject_ids_full.npy'))
    else:
        logger.info("Downloading FULL PhysioNet dataset (109 subjects, 6 runs each)...")
        signals, labels, channel_names, subject_ids = download_physionet_full(
            'data', 
            n_subjects=109,  # All 109 subjects
            runs_per_subject=[1, 2, 3, 4, 5, 6]  # 6 runs per subject
        )
    
    logger.info(f"Data shape: {signals.shape}, Labels: {labels.shape}")
    logger.info(f"Total recordings: {len(signals)}")
    
    # Create dataset
    dataset = EEGDataset(
        signals=signals,
        labels=labels,
        channel_names=channel_names,
        sampling_rate=160.0,
        subject_ids=subject_ids
    )
    
    # Enhanced config for FULL dataset
    config = {
        'data': {
            'sampling_rate': 160,
            'epoch_duration': 2.0,
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15
        },
        'preprocessing': {
            'bandpass_low': 0.5,
            'bandpass_high': 45.0,
            'normalization': 'zscore',
            'artifact_threshold': 5.0
        },
        'model': {
            'patch_size': 32,
            'embedding_dim': 128,      # Increased for larger dataset
            'num_heads': 8,            # More attention heads
            'num_layers': 6,           # Deeper model
            'dropout': 0.2,            # Higher dropout for regularization
            'feedforward_dim': 256     # Larger feedforward dimension
        },
        'training': {
            'optimizer': 'adamw',
            'learning_rate': 0.0005,   # Lower LR for stability
            'weight_decay': 0.01,
            'batch_size': 32,          # Larger batch size
            'num_epochs': 50,          # More epochs
            'warmup_epochs': 5,        # More warmup
            'k_folds': 5,
            'early_stopping_patience': 15  # More patience
        }
    }
    
    # Set random seed
    ConfigManager.set_random_seeds(42)
    
    # Preprocess data
    logger.info("Preprocessing data...")
    preprocessor = Preprocessor(config['preprocessing'])
    preprocessed = preprocessor.preprocess_dataset(
        dataset,
        epoch_duration=config['data']['epoch_duration'],
        apply_ica=False  # Skip ICA for speed
    )
    
    logger.info(f"Preprocessed: {len(preprocessed.epochs)} epochs")
    
    # Create train/val/test splits
    n_epochs = len(preprocessed.epochs)
    indices = np.random.permutation(n_epochs)
    
    train_size = int(0.7 * n_epochs)
    val_size = int(0.15 * n_epochs)
    
    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:train_size + val_size].tolist()
    test_indices = indices[train_size + val_size:].tolist()
    
    preprocessed.train_indices = train_indices
    preprocessed.val_indices = val_indices
    preprocessed.test_indices = test_indices
    
    logger.info(f"Splits: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    
    # Prepare data for training
    train_epochs = [preprocessed.epochs[i] for i in train_indices]
    val_epochs = [preprocessed.epochs[i] for i in val_indices]
    test_epochs = [preprocessed.epochs[i] for i in test_indices]
    
    X_train = np.array([e.signal for e in train_epochs])
    y_train = np.array([e.label for e in train_epochs])
    X_val = np.array([e.signal for e in val_epochs])
    y_val = np.array([e.label for e in val_epochs])
    X_test = np.array([e.signal for e in test_epochs])
    y_test = np.array([e.label for e in test_epochs])
    
    logger.info(f"Training data: X={X_train.shape}, y={y_train.shape}")
    
    # Create and train Transformer model
    logger.info("Training Transformer model...")
    
    n_channels = X_train.shape[1]
    n_samples = X_train.shape[2]
    
    model = EEGTransformer(
        n_channels=n_channels,
        n_samples=n_samples,
        patch_size=config['model']['patch_size'],
        embedding_dim=config['model']['embedding_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        num_classes=2,
        dropout=config['model']['dropout'],
        feedforward_dim=config['model']['feedforward_dim']
    )
    
    logger.info(f"Model parameters: {model.count_parameters():,}")
    
    # Train
    device = torch.device('cpu')
    pipeline = TrainingPipeline(
        model=model,
        config=config,
        device=device
    )
    
    history = pipeline.train(
        X_train, y_train,
        X_val, y_val,
        save_checkpoints=True
    )
    
    # Evaluate
    logger.info("Evaluating model...")
    evaluator = Evaluator()
    
    # Convert to tensor for prediction
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test).float()
        y_pred = model.predict(X_test_tensor).numpy()
        y_prob = model.predict_proba(X_test_tensor).numpy()
    
    results = evaluator.evaluate(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob[:, 1],
        model_name='EEGTransformer'
    )
    
    logger.info("\n" + "="*50)
    logger.info("TRANSFORMER RESULTS")
    logger.info("="*50)
    logger.info(f"Accuracy:  {results.accuracy:.4f}")
    logger.info(f"Precision: {results.precision:.4f}")
    logger.info(f"Recall:    {results.recall:.4f}")
    logger.info(f"F1-Score:  {results.f1_score:.4f}")
    logger.info(f"ROC-AUC:   {results.roc_auc:.4f}")
    
    # Train and evaluate Hybrid CNN-Transformer
    logger.info("\n" + "="*50)
    logger.info("Training HYBRID CNN-TRANSFORMER...")
    logger.info("="*50)
    
    hybrid_model = HybridCNNTransformer(
        n_channels=n_channels,
        n_samples=n_samples,
        embedding_dim=config['model']['embedding_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=4,  # Fewer layers since CNN does feature extraction
        feedforward_dim=config['model']['feedforward_dim'],
        num_classes=2,
        dropout=config['model']['dropout']
    )
    
    logger.info(f"Hybrid Model parameters: {hybrid_model.count_parameters():,}")
    
    hybrid_pipeline = TrainingPipeline(
        model=hybrid_model,
        config=config,
        device=device
    )
    
    hybrid_history = hybrid_pipeline.train(
        X_train, y_train,
        X_val, y_val,
        save_checkpoints=False
    )
    
    # Evaluate Hybrid model
    hybrid_model.eval()
    with torch.no_grad():
        hybrid_pred = hybrid_model.predict(X_test_tensor).numpy()
        hybrid_prob = hybrid_model.predict_proba(X_test_tensor).numpy()
    
    hybrid_results = evaluator.evaluate(
        y_true=y_test,
        y_pred=hybrid_pred,
        y_prob=hybrid_prob[:, 1],
        model_name='HybridCNNTransformer'
    )
    
    logger.info("\n" + "="*50)
    logger.info("HYBRID CNN-TRANSFORMER RESULTS")
    logger.info("="*50)
    logger.info(f"Accuracy:  {hybrid_results.accuracy:.4f}")
    logger.info(f"Precision: {hybrid_results.precision:.4f}")
    logger.info(f"Recall:    {hybrid_results.recall:.4f}")
    logger.info(f"F1-Score:  {hybrid_results.f1_score:.4f}")
    logger.info(f"ROC-AUC:   {hybrid_results.roc_auc:.4f}")
    
    # Train and evaluate baselines
    logger.info("\nTraining baseline models...")
    
    baseline_results = {}
    baseline_results['HybridCNNTransformer'] = hybrid_results
    
    # CNN1D baseline
    try:
        logger.info("\n--- Training CNN1D baseline ---")
        cnn1d = create_baseline_model('cnn1d', n_channels=n_channels, n_samples=n_samples)
        cnn1d_pipeline = TrainingPipeline(model=cnn1d, config=config, device=device)
        cnn1d_pipeline.train(X_train, y_train, X_val, y_val, save_checkpoints=False)
        
        cnn1d.eval()
        with torch.no_grad():
            cnn1d_pred = cnn1d.predict(X_test_tensor).numpy()
            cnn1d_prob = cnn1d.predict_proba(X_test_tensor).numpy()
        cnn1d_results = evaluator.evaluate(y_test, cnn1d_pred, cnn1d_prob[:, 1], 'CNN1D')
        baseline_results['CNN1D'] = cnn1d_results
        logger.info(f"CNN1D - Accuracy: {cnn1d_results.accuracy:.4f}, F1: {cnn1d_results.f1_score:.4f}")
    except Exception as e:
        logger.warning(f"CNN1D failed: {e}")
    
    # CNN2D baseline
    try:
        logger.info("\n--- Training CNN2D baseline ---")
        cnn2d = create_baseline_model('cnn2d', n_channels=n_channels, n_samples=n_samples)
        cnn2d_pipeline = TrainingPipeline(model=cnn2d, config=config, device=device)
        cnn2d_pipeline.train(X_train, y_train, X_val, y_val, save_checkpoints=False)
        
        cnn2d.eval()
        with torch.no_grad():
            cnn2d_pred = cnn2d.predict(X_test_tensor).numpy()
            cnn2d_prob = cnn2d.predict_proba(X_test_tensor).numpy()
        cnn2d_results = evaluator.evaluate(y_test, cnn2d_pred, cnn2d_prob[:, 1], 'CNN2D')
        baseline_results['CNN2D'] = cnn2d_results
        logger.info(f"CNN2D - Accuracy: {cnn2d_results.accuracy:.4f}, F1: {cnn2d_results.f1_score:.4f}")
    except Exception as e:
        logger.warning(f"CNN2D failed: {e}")
    
    # LSTM baseline
    try:
        logger.info("\n--- Training LSTM baseline ---")
        lstm = create_baseline_model('lstm', n_channels=n_channels, n_samples=n_samples)
        lstm_pipeline = TrainingPipeline(model=lstm, config=config, device=device)
        lstm_pipeline.train(X_train, y_train, X_val, y_val, save_checkpoints=False)
        
        lstm.eval()
        with torch.no_grad():
            lstm_pred = lstm.predict(X_test_tensor).numpy()
            lstm_prob = lstm.predict_proba(X_test_tensor).numpy()
        lstm_results = evaluator.evaluate(y_test, lstm_pred, lstm_prob[:, 1], 'LSTM')
        baseline_results['LSTM'] = lstm_results
        logger.info(f"LSTM - Accuracy: {lstm_results.accuracy:.4f}, F1: {lstm_results.f1_score:.4f}")
    except Exception as e:
        logger.warning(f"LSTM failed: {e}")
    
    # Save results
    results_path = Path('results')
    results_path.mkdir(exist_ok=True)
    
    evaluator.save_results(filename_prefix='full_experiment_results')
    
    # Print comprehensive comparison
    logger.info("\n" + "="*70)
    logger.info("FULL DATASET EXPERIMENT - MODEL COMPARISON")
    logger.info("="*70)
    logger.info(f"Dataset: {len(signals)} recordings from 109 subjects")
    logger.info(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)} epochs")
    logger.info("-"*70)
    logger.info(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'ROC-AUC':<12}")
    logger.info("-"*70)
    logger.info(f"{'EEGTransformer':<20} {results.accuracy:<12.4f} {results.precision:<12.4f} {results.recall:<12.4f} {results.f1_score:<12.4f} {results.roc_auc:<12.4f}")
    
    for name, res in baseline_results.items():
        logger.info(f"{name:<20} {res.accuracy:<12.4f} {res.precision:<12.4f} {res.recall:<12.4f} {res.f1_score:<12.4f} {res.roc_auc:<12.4f}")
    
    logger.info("-"*70)
    logger.info(f"\nResults saved to {results_path}")
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*70)
    
    return results, baseline_results


if __name__ == '__main__':
    run_training()

