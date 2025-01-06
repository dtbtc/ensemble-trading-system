import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from sklearn.metrics import roc_curve, precision_recall_curve
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List

class ResultVisualizer:
    """Result Visualizer"""
    def __init__(self, config=None):
        self.config = config
        self.figures_dir = Path('results/figures') if config is None else config.FIGURES_DIR
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_results(self, results: Dict):
        """Plot evaluation results"""
        try:
            # Convert results to DataFrame
            df_results = pd.DataFrame(results).T
            
            # Create figure
            plt.figure(figsize=(15, 8))
            
            # Plot bar chart
            x = np.arange(len(df_results.index))
            width = 0.15
            
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            colors = ['blue', 'green', 'red', 'purple', 'orange']
            
            for i, (metric, color) in enumerate(zip(metrics, colors)):
                if metric in df_results.columns:
                    plt.bar(x + i * width, df_results[metric], 
                           width, label=metric, color=color, alpha=0.7)
            
            # Set figure properties
            plt.xlabel('Models')
            plt.ylabel('Score')
            plt.title('Model Performance Comparison')
            plt.xticks(x + width * 2, df_results.index, rotation=45, ha='right')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            save_path = self.figures_dir / 'model_comparison.png'
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            logging.info(f"Results visualization saved to {save_path}")
            
        except Exception as e:
            logging.error(f"Error plotting results: {str(e)}")
            raise
            
    def plot_confusion_matrices(self, confusion_matrices: Dict):
        """Plot confusion matrices"""
        try:
            for name, cm in confusion_matrices.items():
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix - {name}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                
                # Save figure
                save_path = self.figures_dir / f'confusion_matrix_{name}.png'
                plt.savefig(save_path)
                plt.close()
                
            logging.info("Confusion matrices plots saved")
            
        except Exception as e:
            logging.error(f"Error plotting confusion matrices: {str(e)}")
            raise

    def plot_metrics_comparison(self, results: Dict):
        """Plot model performance metrics comparison"""
        try:
            # Convert results to DataFrame
            df_results = pd.DataFrame(results).T
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Model Performance Metrics Comparison', size=16)
            
            # Plot bar charts for each metric
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            for ax, metric in zip(axes.flat, metrics):
                if metric in df_results.columns:
                    df_results[metric].plot(kind='bar', ax=ax)
                    ax.set_title(f'{metric.capitalize()} Score')
                    ax.set_ylim([0, 1])
                    ax.grid(True, alpha=0.3)
                    
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            save_path = self.figures_dir / 'metrics_comparison.png'
            plt.savefig(save_path)
            plt.close()
            
            logging.info(f"Metrics comparison plot saved to {save_path}")
            
        except Exception as e:
            logging.error(f"Error plotting metrics comparison: {str(e)}")
            raise

    def create_interactive_dashboard(self, evaluator):
        """Create interactive dashboard"""
        try:
            # Create new HTML file
            dashboard_path = self.figures_dir / 'dashboard.html'
            
            # Prepare data
            metrics_df = pd.DataFrame(evaluator.metrics) if evaluator.metrics is not None else pd.DataFrame()
            
            # Create performance metrics chart
            fig = go.Figure()
            
            if not metrics_df.empty:
                for metric in ['accuracy', 'precision', 'recall', 'f1']:
                    if metric in metrics_df.columns:
                        fig.add_trace(go.Bar(
                            name=metric,
                            x=metrics_df.index,
                            y=metrics_df[metric],
                            text=metrics_df[metric].round(3),
                            textposition='auto',
                        ))
                
                fig.update_layout(
                    title='Model Performance Metrics',
                    barmode='group',
                    xaxis_title='Models',
                    yaxis_title='Score',
                    yaxis_range=[0, 1]
                )
                
                # Save interactive chart
                fig.write_html(str(dashboard_path))
                logging.info(f"Interactive dashboard saved to {dashboard_path}")
                
        except Exception as e:
            logging.error(f"Error creating interactive dashboard: {str(e)}")
            raise

    def plot_roc_curves(self, predictions: Dict):
        """Plot ROC curves"""
        try:
            plt.figure(figsize=(10, 8))
            
            for name, pred_data in predictions.items():
                if 'y_prob' in pred_data:
                    fpr, tpr, _ = roc_curve(
                        pred_data['y_true'],
                        pred_data['y_prob']
                    )
                    plt.plot(fpr, tpr, label=f'{name}')
            
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save figure
            save_path = self.figures_dir / 'roc_curves.png'
            plt.savefig(save_path)
            plt.close()
            
            logging.info(f"ROC curves saved to {save_path}")
            
        except Exception as e:
            logging.error(f"Error plotting ROC curves: {str(e)}")
            raise

    def plot_feature_importance(self, importance: np.ndarray, 
                              feature_names: List[str], model_name: str):
        """Plot feature importance"""
        try:
            plt.figure(figsize=(12, 6))
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            })
            
            # Sort and select top features
            importance_df = importance_df.sort_values('importance', ascending=True).tail(20)
            
            # Plot horizontal bar chart
            plt.barh(importance_df['feature'], importance_df['importance'])
            plt.title(f'Feature Importance - {model_name}')
            plt.xlabel('Importance')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            save_path = self.figures_dir / f'feature_importance_{model_name}.png'
            plt.savefig(save_path)
            plt.close()
            
            logging.info(f"Feature importance plot saved to {save_path}")
            
        except Exception as e:
            logging.error(f"Error plotting feature importance: {str(e)}")
            raise

    def plot_feature_analysis(self, feature_analysis: Dict):
        """Plot feature analysis results"""
        try:
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': list(feature_analysis['importance'].keys()),
                'importance': list(feature_analysis['importance'].values())
            })
            
            # Sort and select all features
            importance_df = importance_df.sort_values('importance', ascending=True)
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot horizontal bar chart
            plt.barh(importance_df['feature'], importance_df['importance'])
            plt.title('Feature Importance Analysis')
            plt.xlabel('Importance Score')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            save_path = self.figures_dir / 'feature_analysis.png'
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            logging.info(f"Feature analysis plot saved to {save_path}")
            
            # Create heatmap for feature correlation
            if 'correlation' in feature_analysis:
                plt.figure(figsize=(15, 12))
                sns.heatmap(
                    pd.DataFrame(feature_analysis['correlation']),
                    annot=True,
                    cmap='coolwarm',
                    center=0
                )
                plt.title('Feature Correlation Matrix')
                
                # Save heatmap
                save_path = self.figures_dir / 'feature_correlation.png'
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close()
                
                logging.info(f"Feature correlation plot saved to {save_path}")
                
        except Exception as e:
            logging.error(f"Error plotting feature analysis: {str(e)}")
            raise 