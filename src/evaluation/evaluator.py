from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, precision_recall_curve,
    average_precision_score
)
import logging
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from ..models.base_models import BaseModel

class ModelEvaluator:
    """Model Evaluator"""
    def __init__(self, config=None):
        self.config = config
        self.confusion_matrices = {}
        self.feature_importance = {}
        self.predictions = {}
        self.metrics = None
        self.results_path = Path('results/evaluation_results.pkl') if config is None else config.RESULTS_DIR / 'evaluation_results.pkl'
        
    def evaluate_model(self, model: BaseModel, X: np.ndarray, y: np.ndarray, 
                      model_name: str = None) -> Dict[str, float]:
        """Evaluate the performance of a single model
        
        Args:
            model: Model to evaluate
            X: Feature data
            y: Label data
            model_name: Model name
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            # Get prediction results
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred),
                'recall': recall_score(y, y_pred),
                'f1': f1_score(y, y_pred),
                'roc_auc': roc_auc_score(y, y_prob)
            }
            
            # Save confusion matrix
            self.confusion_matrices[model_name] = confusion_matrix(y, y_pred)
            
            # Save prediction results
            self.predictions[model_name] = {
                'y_true': y,
                'y_pred': y_pred,
                'y_prob': y_prob
            }
            
            # Save feature importance if model supports it
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[model_name] = {
                    'importance': model.feature_importances_,
                    'features': model.feature_names if hasattr(model, 'feature_names') else None
                }
                
            return metrics
            
        except Exception as e:
            logging.error(f"Error evaluating model {model_name}: {str(e)}")
            raise
            
    def evaluate_all_models(self, models: Dict[str, BaseModel], 
                          X: np.ndarray, y: np.ndarray) -> Dict[str, Dict]:
        """Evaluate the performance of all models"""
        results = {}
        for name, model in models.items():
            logging.info(f"Evaluating {name}...")
            try:
                results[name] = self.evaluate_model(model, X, y, name)
            except Exception as e:
                logging.error(f"Error evaluating {name}: {str(e)}")
                continue
        return results
        
    def get_feature_importance(self, model_name):
        """Get feature importance"""
        return self.feature_importance.get(model_name)
        
    def get_best_model(self, metric='f1'):
        """Get the best performing model"""
        if self.metrics is None:
            raise ValueError("No evaluation results available")
        return self.metrics.loc[self.metrics[metric].idxmax(), 'model']
        
    def plot_pr_curves(self, save_path=None):
        """Plot PR curves"""
        plt.figure(figsize=(10, 8))
        
        for name, pred_data in self.predictions.items():
            if pred_data['y_proba'] is not None:
                precision, recall, _ = precision_recall_curve(
                    pred_data['y_true'],
                    pred_data['y_proba']
                )
                avg_precision = average_precision_score(
                    pred_data['y_true'],
                    pred_data['y_proba']
                )
                plt.plot(recall, precision, label=f'{name} (AP={avg_precision:.2f})')
                
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def plot_confusion_matrices(self, save_dir=None):
        """Plot confusion matrices"""
        for name, cm in self.confusion_matrices.items():
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            if save_dir:
                save_path = Path(save_dir) / f'confusion_matrix_{name}.png'
                plt.savefig(save_path)
            plt.close()
            
    def save_results(self, results: Dict):
        """Save evaluation results"""
        try:
            # Create results directory
            self.results_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save results
            with open(self.results_path, 'wb') as f:
                joblib.dump(results, f)
            
            # Save as CSV for easy viewing
            csv_path = self.results_path.parent / 'evaluation_results.csv'
            df_results = pd.DataFrame(results).T
            df_results.to_csv(csv_path)
            
            logging.info(f"Evaluation results saved to {self.results_path}")
            logging.info(f"CSV results saved to {csv_path}")
            
        except Exception as e:
            logging.error(f"Error saving evaluation results: {str(e)}")
            raise
            
    def load_results(self):
        """Load evaluation results"""
        try:
            if not self.results_path.exists():
                logging.info("No cached evaluation results found")
                return None
                
            # Load results
            results_data = joblib.load(self.results_path)
            
            # Restore saved data
            self.metrics = results_data['metrics']
            self.confusion_matrices = results_data['confusion_matrices']
            self.feature_importance = results_data['feature_importance']
            self.predictions = results_data['predictions']
            
            logging.info(f"Loaded evaluation results from {self.results_path}")
            logging.info(f"Results timestamp: {results_data['timestamp']}")
            
            return self.metrics
            
        except Exception as e:
            logging.error(f"Error loading evaluation results: {str(e)}")
            return None 