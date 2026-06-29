"""
Unit tests for the Evaluator module.

Tests cover:
- compute_confusion_matrix function
- compute_accuracy, compute_precision, compute_recall, compute_f1_score functions
- compute_specificity function
- compute_per_class_metrics function
- compute_roc_curve and compute_roc_auc functions
- compute_cross_validation_statistics function
- compare_models function
- save_results_json and save_results_csv functions
- Evaluator class
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.evaluator import (
    EvaluationResults,
    CrossValidationResults,
    compute_confusion_matrix,
    compute_accuracy,
    compute_precision,
    compute_recall,
    compute_f1_score,
    compute_specificity,
    compute_per_class_metrics,
    compute_metrics,
    compute_roc_curve,
    compute_roc_auc,
    compute_cross_validation_statistics,
    generate_cross_validation_summary,
    compare_models,
    format_comparison_table,
    save_results_json,
    save_results_csv,
    Evaluator,
)


# Test fixtures
@pytest.fixture
def binary_predictions():
    """Create sample binary predictions for testing."""
    # 10 samples: 6 correct, 4 incorrect
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 0, 1, 1, 1, 1])
    # TN=3, FP=1, FN=2, TP=4
    return y_true, y_pred


@pytest.fixture
def predictions_with_probs():
    """Create predictions with probabilities for ROC testing."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.5, 0.7, 0.8, 0.9, 0.95])
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob


@pytest.fixture
def perfect_predictions():
    """Create perfect predictions for testing."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])
    return y_true, y_pred, y_prob


@pytest.fixture
def sample_evaluation_results():
    """Create sample EvaluationResults for testing."""
    return EvaluationResults(
        accuracy=0.8,
        precision=0.75,
        recall=0.85,
        f1_score=0.7978,
        roc_auc=0.9,
        specificity=0.7,
        confusion_matrix=np.array([[35, 15], [10, 40]]),
        per_class_metrics={
            'Control': {'precision': 0.78, 'recall': 0.70, 'f1': 0.74, 'support': 50},
            'AD': {'precision': 0.73, 'recall': 0.80, 'f1': 0.76, 'support': 50}
        }
    )



class TestConfusionMatrix:
    """Tests for compute_confusion_matrix function."""
    
    def test_basic_confusion_matrix(self, binary_predictions):
        """Test basic confusion matrix computation."""
        y_true, y_pred = binary_predictions
        cm = compute_confusion_matrix(y_true, y_pred)
        
        # Expected: TN=3, FP=1, FN=2, TP=4
        assert cm.shape == (2, 2)
        assert cm[0, 0] == 3  # TN
        assert cm[0, 1] == 1  # FP
        assert cm[1, 0] == 2  # FN
        assert cm[1, 1] == 4  # TP
    
    def test_perfect_predictions(self, perfect_predictions):
        """Test confusion matrix with perfect predictions."""
        y_true, y_pred, _ = perfect_predictions
        cm = compute_confusion_matrix(y_true, y_pred)
        
        # No errors
        assert cm[0, 1] == 0  # FP
        assert cm[1, 0] == 0  # FN
        assert cm[0, 0] + cm[1, 1] == len(y_true)  # All correct
    
    def test_all_positive_predictions(self):
        """Test when all predictions are positive."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 1])
        cm = compute_confusion_matrix(y_true, y_pred)
        
        assert cm[0, 0] == 0  # TN
        assert cm[0, 1] == 2  # FP
        assert cm[1, 0] == 0  # FN
        assert cm[1, 1] == 2  # TP
    
    def test_confusion_matrix_sum(self, binary_predictions):
        """Test that confusion matrix entries sum to total samples."""
        y_true, y_pred = binary_predictions
        cm = compute_confusion_matrix(y_true, y_pred)
        
        assert cm.sum() == len(y_true)


class TestAccuracy:
    """Tests for compute_accuracy function."""
    
    def test_basic_accuracy(self, binary_predictions):
        """Test basic accuracy computation."""
        y_true, y_pred = binary_predictions
        accuracy = compute_accuracy(y_true, y_pred)
        
        # 7 correct out of 10
        assert accuracy == pytest.approx(0.7, abs=0.01)
    
    def test_perfect_accuracy(self, perfect_predictions):
        """Test accuracy with perfect predictions."""
        y_true, y_pred, _ = perfect_predictions
        accuracy = compute_accuracy(y_true, y_pred)
        
        assert accuracy == 1.0
    
    def test_zero_accuracy(self):
        """Test accuracy when all predictions are wrong."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        accuracy = compute_accuracy(y_true, y_pred)
        
        assert accuracy == 0.0
    
    def test_empty_arrays(self):
        """Test accuracy with empty arrays."""
        y_true = np.array([])
        y_pred = np.array([])
        accuracy = compute_accuracy(y_true, y_pred)
        
        assert accuracy == 0.0


class TestPrecision:
    """Tests for compute_precision function."""
    
    def test_basic_precision(self, binary_predictions):
        """Test basic precision computation."""
        y_true, y_pred = binary_predictions
        precision = compute_precision(y_true, y_pred)
        
        # TP=4, FP=1, Precision = 4/5 = 0.8
        assert precision == pytest.approx(0.8, abs=0.01)
    
    def test_perfect_precision(self, perfect_predictions):
        """Test precision with perfect predictions."""
        y_true, y_pred, _ = perfect_predictions
        precision = compute_precision(y_true, y_pred)
        
        assert precision == 1.0
    
    def test_no_positive_predictions(self):
        """Test precision when no positive predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        precision = compute_precision(y_true, y_pred)
        
        # No positive predictions, precision is 0
        assert precision == 0.0


class TestRecall:
    """Tests for compute_recall function."""
    
    def test_basic_recall(self, binary_predictions):
        """Test basic recall computation."""
        y_true, y_pred = binary_predictions
        recall = compute_recall(y_true, y_pred)
        
        # TP=4, FN=2, Recall = 4/6 ≈ 0.667
        assert recall == pytest.approx(0.667, abs=0.01)
    
    def test_perfect_recall(self, perfect_predictions):
        """Test recall with perfect predictions."""
        y_true, y_pred, _ = perfect_predictions
        recall = compute_recall(y_true, y_pred)
        
        assert recall == 1.0
    
    def test_no_true_positives(self):
        """Test recall when no true positives."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        recall = compute_recall(y_true, y_pred)
        
        # No true positives, recall is 0
        assert recall == 0.0


class TestF1Score:
    """Tests for compute_f1_score function."""
    
    def test_basic_f1(self, binary_predictions):
        """Test basic F1 score computation."""
        y_true, y_pred = binary_predictions
        f1 = compute_f1_score(y_true, y_pred)
        
        # Precision=0.8, Recall=0.667, F1 = 2*0.8*0.667/(0.8+0.667) ≈ 0.727
        assert f1 == pytest.approx(0.727, abs=0.01)
    
    def test_perfect_f1(self, perfect_predictions):
        """Test F1 with perfect predictions."""
        y_true, y_pred, _ = perfect_predictions
        f1 = compute_f1_score(y_true, y_pred)
        
        assert f1 == 1.0
    
    def test_f1_formula(self, binary_predictions):
        """Test that F1 follows the harmonic mean formula."""
        y_true, y_pred = binary_predictions
        precision = compute_precision(y_true, y_pred)
        recall = compute_recall(y_true, y_pred)
        f1 = compute_f1_score(y_true, y_pred)
        
        expected_f1 = 2 * precision * recall / (precision + recall)
        assert f1 == pytest.approx(expected_f1, abs=0.001)


class TestSpecificity:
    """Tests for compute_specificity function."""
    
    def test_basic_specificity(self, binary_predictions):
        """Test basic specificity computation."""
        y_true, y_pred = binary_predictions
        specificity = compute_specificity(y_true, y_pred)
        
        # TN=3, FP=1, Specificity = 3/4 = 0.75
        assert specificity == pytest.approx(0.75, abs=0.01)
    
    def test_perfect_specificity(self, perfect_predictions):
        """Test specificity with perfect predictions."""
        y_true, y_pred, _ = perfect_predictions
        specificity = compute_specificity(y_true, y_pred)
        
        assert specificity == 1.0


class TestPerClassMetrics:
    """Tests for compute_per_class_metrics function."""
    
    def test_returns_both_classes(self, binary_predictions):
        """Test that metrics are returned for both classes."""
        y_true, y_pred = binary_predictions
        per_class = compute_per_class_metrics(y_true, y_pred)
        
        assert 'Control' in per_class
        assert 'AD' in per_class
    
    def test_metrics_structure(self, binary_predictions):
        """Test that each class has required metrics."""
        y_true, y_pred = binary_predictions
        per_class = compute_per_class_metrics(y_true, y_pred)
        
        for class_name in ['Control', 'AD']:
            assert 'precision' in per_class[class_name]
            assert 'recall' in per_class[class_name]
            assert 'f1' in per_class[class_name]
            assert 'support' in per_class[class_name]
    
    def test_support_counts(self, binary_predictions):
        """Test that support counts match actual class counts."""
        y_true, y_pred = binary_predictions
        per_class = compute_per_class_metrics(y_true, y_pred)
        
        assert per_class['Control']['support'] == np.sum(y_true == 0)
        assert per_class['AD']['support'] == np.sum(y_true == 1)



class TestROCCurve:
    """Tests for compute_roc_curve function."""
    
    def test_roc_curve_shape(self, predictions_with_probs):
        """Test that ROC curve returns correct shapes."""
        y_true, _, y_prob = predictions_with_probs
        fpr, tpr, thresholds = compute_roc_curve(y_true, y_prob)
        
        assert len(fpr) == len(tpr) == len(thresholds)
        assert len(fpr) > 0
    
    def test_roc_curve_bounds(self, predictions_with_probs):
        """Test that FPR and TPR are in valid range."""
        y_true, _, y_prob = predictions_with_probs
        fpr, tpr, _ = compute_roc_curve(y_true, y_prob)
        
        assert np.all(fpr >= 0) and np.all(fpr <= 1)
        assert np.all(tpr >= 0) and np.all(tpr <= 1)
    
    def test_roc_curve_endpoints(self, predictions_with_probs):
        """Test that ROC curve starts at (0,0) and ends at (1,1)."""
        y_true, _, y_prob = predictions_with_probs
        fpr, tpr, _ = compute_roc_curve(y_true, y_prob)
        
        assert fpr[0] == 0.0
        assert tpr[0] == 0.0
        assert fpr[-1] == 1.0
        assert tpr[-1] == 1.0


class TestROCAUC:
    """Tests for compute_roc_auc function."""
    
    def test_auc_range(self, predictions_with_probs):
        """Test that AUC is in valid range [0, 1]."""
        y_true, _, y_prob = predictions_with_probs
        auc = compute_roc_auc(y_true, y_prob)
        
        assert 0.0 <= auc <= 1.0
    
    def test_perfect_auc(self, perfect_predictions):
        """Test AUC with perfect separation."""
        y_true, _, y_prob = perfect_predictions
        auc = compute_roc_auc(y_true, y_prob)
        
        # Perfect separation should give AUC close to 1
        assert auc >= 0.95
    
    def test_random_auc(self):
        """Test AUC with random predictions."""
        np.random.seed(42)
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.random.rand(10)
        auc = compute_roc_auc(y_true, y_prob)
        
        # Random should be around 0.5
        assert 0.0 <= auc <= 1.0
    
    def test_single_class_auc(self):
        """Test AUC when only one class is present."""
        y_true = np.array([1, 1, 1, 1, 1])
        y_prob = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        auc = compute_roc_auc(y_true, y_prob)
        
        # Should return 0.5 (undefined case)
        assert auc == 0.5


class TestCrossValidationStatistics:
    """Tests for cross-validation statistics functions."""
    
    def test_mean_computation(self, sample_evaluation_results):
        """Test that mean is computed correctly."""
        results = [sample_evaluation_results] * 3
        mean_metrics, _ = compute_cross_validation_statistics(results)
        
        # Mean of identical values should equal the value
        assert mean_metrics['accuracy'] == pytest.approx(0.8, abs=0.001)
        assert mean_metrics['f1_score'] == pytest.approx(0.7978, abs=0.001)
    
    def test_std_computation(self):
        """Test that std is computed correctly."""
        results = [
            EvaluationResults(
                accuracy=0.8, precision=0.75, recall=0.85,
                f1_score=0.8, roc_auc=0.9, specificity=0.7,
                confusion_matrix=np.array([[35, 15], [10, 40]]),
                per_class_metrics={}
            ),
            EvaluationResults(
                accuracy=0.9, precision=0.85, recall=0.95,
                f1_score=0.9, roc_auc=0.95, specificity=0.8,
                confusion_matrix=np.array([[40, 10], [5, 45]]),
                per_class_metrics={}
            )
        ]
        
        mean_metrics, std_metrics = compute_cross_validation_statistics(results)
        
        # Mean of 0.8 and 0.9 is 0.85
        assert mean_metrics['accuracy'] == pytest.approx(0.85, abs=0.001)
        # Std of [0.8, 0.9] with ddof=1 is ~0.0707
        assert std_metrics['accuracy'] == pytest.approx(0.0707, abs=0.01)
    
    def test_empty_results(self):
        """Test with empty results list."""
        mean_metrics, std_metrics = compute_cross_validation_statistics([])
        
        assert mean_metrics == {}
        assert std_metrics == {}
    
    def test_generate_summary(self, sample_evaluation_results):
        """Test cross-validation summary generation."""
        results = [sample_evaluation_results] * 3
        summary = generate_cross_validation_summary(results)
        
        assert isinstance(summary, CrossValidationResults)
        assert len(summary.fold_results) == 3
        assert 'accuracy' in summary.mean_metrics
        assert 'accuracy' in summary.std_metrics
        assert 0 <= summary.best_fold < 3


class TestModelComparison:
    """Tests for model comparison functions."""
    
    def test_compare_models_basic(self, sample_evaluation_results):
        """Test basic model comparison."""
        model_results = {
            'model_a': sample_evaluation_results,
            'model_b': EvaluationResults(
                accuracy=0.85, precision=0.8, recall=0.9,
                f1_score=0.85, roc_auc=0.92, specificity=0.75,
                confusion_matrix=np.array([[38, 12], [8, 42]]),
                per_class_metrics={}
            )
        }
        
        comparison = compare_models(model_results)
        
        assert 'comparison_table' in comparison
        assert 'best_model' in comparison
        assert 'rankings' in comparison
        assert len(comparison['comparison_table']) == 2
    
    def test_best_model_selection(self, sample_evaluation_results):
        """Test that best model is selected by F1 score."""
        model_results = {
            'model_a': sample_evaluation_results,  # F1 = 0.7978
            'model_b': EvaluationResults(
                accuracy=0.85, precision=0.8, recall=0.9,
                f1_score=0.85, roc_auc=0.92, specificity=0.75,
                confusion_matrix=np.array([[38, 12], [8, 42]]),
                per_class_metrics={}
            )
        }
        
        comparison = compare_models(model_results)
        
        # model_b has higher F1
        assert comparison['best_model'] == 'model_b'
    
    def test_empty_comparison(self):
        """Test comparison with no models."""
        comparison = compare_models({})
        
        assert comparison['comparison_table'] == []
        assert comparison['best_model'] is None
    
    def test_format_comparison_table(self, sample_evaluation_results):
        """Test table formatting."""
        model_results = {'model_a': sample_evaluation_results}
        table = format_comparison_table(model_results)
        
        assert 'model_a' in table
        assert 'Accuracy' in table
        assert 'F1' in table


class TestResultSaving:
    """Tests for result saving functions."""
    
    def test_save_results_json(self, sample_evaluation_results):
        """Test saving results to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'results.json')
            save_results_json(sample_evaluation_results, filepath)
            
            assert os.path.exists(filepath)
            
            with open(filepath) as f:
                data = json.load(f)
            
            assert data['accuracy'] == 0.8
            assert 'saved_at' in data
    
    def test_save_results_csv(self, sample_evaluation_results):
        """Test saving results to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'results.csv')
            save_results_csv(sample_evaluation_results, filepath)
            
            assert os.path.exists(filepath)
            
            with open(filepath) as f:
                content = f.read()
            
            assert 'accuracy' in content
            assert '0.8' in content
    
    def test_save_multiple_models_csv(self, sample_evaluation_results):
        """Test saving multiple model results to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_results = {
                'model_a': sample_evaluation_results,
                'model_b': sample_evaluation_results
            }
            
            filepath = os.path.join(tmpdir, 'comparison.csv')
            save_results_csv(model_results, filepath)
            
            assert os.path.exists(filepath)
            
            with open(filepath) as f:
                lines = f.readlines()
            
            # Header + 2 data rows
            assert len(lines) == 3


class TestEvaluator:
    """Tests for the Evaluator class."""
    
    def test_initialization(self):
        """Test evaluator initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = Evaluator(output_dir=tmpdir, save_visualizations=False)
            
            assert evaluator.output_dir == tmpdir
            assert evaluator.model_results == {}
    
    def test_evaluate_basic(self, binary_predictions):
        """Test basic evaluation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = Evaluator(output_dir=tmpdir, save_visualizations=False)
            y_true, y_pred = binary_predictions
            
            results = evaluator.evaluate(y_true, y_pred, model_name='test_model')
            
            assert isinstance(results, EvaluationResults)
            assert results.accuracy == pytest.approx(0.7, abs=0.01)
            assert 'test_model' in evaluator.model_results
    
    def test_evaluate_with_probs(self, predictions_with_probs):
        """Test evaluation with probabilities."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = Evaluator(output_dir=tmpdir, save_visualizations=False)
            y_true, y_pred, y_prob = predictions_with_probs
            
            results = evaluator.evaluate(
                y_true, y_pred, y_prob, model_name='test_model'
            )
            
            assert results.roc_auc > 0.0
    
    def test_compare_models(self, binary_predictions, predictions_with_probs):
        """Test model comparison through evaluator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = Evaluator(output_dir=tmpdir, save_visualizations=False)
            
            y_true1, y_pred1 = binary_predictions
            y_true2, y_pred2, y_prob2 = predictions_with_probs
            
            evaluator.evaluate(y_true1, y_pred1, model_name='model_a')
            evaluator.evaluate(y_true2, y_pred2, y_prob2, model_name='model_b')
            
            comparison = evaluator.compare_models()
            
            assert comparison['best_model'] is not None
            assert len(comparison['comparison_table']) == 2
    
    def test_save_results(self, binary_predictions):
        """Test saving all results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = Evaluator(output_dir=tmpdir, save_visualizations=False)
            y_true, y_pred = binary_predictions
            
            evaluator.evaluate(y_true, y_pred, model_name='test_model')
            saved_files = evaluator.save_results()
            
            assert len(saved_files) > 0
            for filepath in saved_files.values():
                assert os.path.exists(filepath)
    
    def test_generate_report(self, binary_predictions):
        """Test report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = Evaluator(output_dir=tmpdir, save_visualizations=False)
            y_true, y_pred = binary_predictions
            
            evaluator.evaluate(y_true, y_pred, model_name='test_model')
            report = evaluator.generate_report()
            
            assert 'EVALUATION REPORT' in report
            assert 'test_model' in report
            assert 'Accuracy' in report
    
    def test_get_results(self, binary_predictions):
        """Test getting results for specific model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = Evaluator(output_dir=tmpdir, save_visualizations=False)
            y_true, y_pred = binary_predictions
            
            evaluator.evaluate(y_true, y_pred, model_name='test_model')
            
            results = evaluator.get_results('test_model')
            assert results is not None
            
            results = evaluator.get_results('nonexistent')
            assert results is None
    
    def test_clear_results(self, binary_predictions):
        """Test clearing results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = Evaluator(output_dir=tmpdir, save_visualizations=False)
            y_true, y_pred = binary_predictions
            
            evaluator.evaluate(y_true, y_pred, model_name='test_model')
            assert len(evaluator.model_results) == 1
            
            evaluator.clear_results()
            assert len(evaluator.model_results) == 0


class TestEvaluationResultsDataclass:
    """Tests for EvaluationResults dataclass."""
    
    def test_to_dict(self, sample_evaluation_results):
        """Test conversion to dictionary."""
        d = sample_evaluation_results.to_dict()
        
        assert d['accuracy'] == 0.8
        assert d['precision'] == 0.75
        assert d['recall'] == 0.85
        assert 'confusion_matrix' in d
        assert 'per_class_metrics' in d


class TestCrossValidationResultsDataclass:
    """Tests for CrossValidationResults dataclass."""
    
    def test_to_dict(self, sample_evaluation_results):
        """Test conversion to dictionary."""
        cv_results = CrossValidationResults(
            fold_results=[sample_evaluation_results],
            mean_metrics={'accuracy': 0.8},
            std_metrics={'accuracy': 0.05},
            best_fold=0
        )
        
        d = cv_results.to_dict()
        
        assert 'fold_results' in d
        assert 'mean_metrics' in d
        assert 'std_metrics' in d
        assert d['best_fold'] == 0

