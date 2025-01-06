import logging
from pathlib import Path
from src.utils.config import Config
from src.utils.logger import setup_logger
from src.data_processor.processor import DataProcessor
from src.models.model_factory import ModelFactory
from src.models.base_models import BaseModel
from src.evaluation.evaluator import ModelEvaluator
from src.visualization.visualizer import ResultVisualizer
from src.backtest.backtest import BacktestEngine
from typing import Dict, Any
import os
import numpy as np

class ModelingPipeline:
    """Model training and evaluation pipeline"""
    def __init__(self, config_path: str = None):
        # Initialize configuration and logging
        self.config = Config(config_path)
        setup_logger(self.config.LOGS_DIR)
        
        # Initialize components
        self.data_processor = DataProcessor(self.config)
        self.model_factory = ModelFactory()
        self.evaluator = ModelEvaluator(self.config)
        self.visualizer = ResultVisualizer(self.config)
        self.backtest = BacktestEngine(self.config)
        
    def run(self, retrain: bool = False, reevaluate: bool = False, regenerate: bool = False):
        """Run entire modeling process"""
        try:
            # Data processing
            processed_data = self._process_data(regenerate)
            
            # Train or load base models
            base_models = self._train_models(processed_data, retrain)
            
            # Prepare datasets for ensemble models
            datasets = self.data_processor.prepare_datasets(
                features=processed_data['features'],
                labels=processed_data['labels'],
                test_size=0.2,
                val_size=0.1
            )
            
            # Create ensemble models
            ensemble_models = self._create_ensembles(
                base_models=base_models,
                data=(
                    datasets['train'][0], datasets['val'][0], datasets['test'][0],
                    datasets['train'][1], datasets['val'][1], datasets['test'][1]
                )
            )
            
            # Merge all models
            all_models = {**base_models, **ensemble_models}
            
            # Evaluate all models
            results = None
            if reevaluate:
                results = self._evaluate_models(
                    models=all_models,
                    datasets=datasets,
                    reevaluate=reevaluate
                )
                
                # Save evaluation results
                self.evaluator.save_results(results)
                
                # Perform comprehensive visualization analysis
                self._visualize_results(results)
                
                # Print evaluation results
                self._print_evaluation_results(results)
                
            return all_models, results
            
        except Exception as e:
            logging.error(f"Pipeline failed: {str(e)}")
            raise
            
    def _process_data(self, regenerate: bool = False):
        """Data processing step"""
        try:
            if regenerate:
                logging.info("Processing raw data...")
                processed_data = self.data_processor.process_raw_data()
                self.data_processor.save_processed_data(processed_data)
                
                # Visualize feature analysis results
                if 'feature_analysis' in processed_data:
                    self.visualizer.plot_feature_analysis(processed_data['feature_analysis'])
                    
                return processed_data
            else:
                logging.info("Loading pre-processed data...")
                processed_data = self.data_processor.load_processed_data()
                if processed_data is None:
                    logging.info("No processed data found, processing raw data...")
                    processed_data = self.data_processor.process_raw_data()
                    self.data_processor.save_processed_data(processed_data)
                    
                    # Visualize feature analysis results
                    if 'feature_analysis' in processed_data:
                        self.visualizer.plot_feature_analysis(processed_data['feature_analysis'])
                        
                return processed_data
                
        except Exception as e:
            logging.error(f"Error in data processing: {str(e)}")
            raise
            
    def _train_models(self, processed_data, retrain: bool = False) -> Dict[str, Any]:
        """Model training step"""
        try:
            logging.info("Creating and training models...")
            
            # Prepare datasets
            datasets = self.data_processor.prepare_datasets(
                features=processed_data['features'],
                labels=processed_data['labels'],
                test_size=0.2,
                val_size=0.1
            )
            
            # Get training and validation data
            X_train, y_train = datasets['train']
            X_val, y_val = datasets['val']
            
            # Create base model dictionary
            base_models = self.model_factory.create_base_models()
            
            if retrain:
                # If retraining is needed, train all models directly
                logging.info("Retraining all models...")
                base_models = self.model_factory.train_base_models(
                    base_models,
                    X_train, y_train,
                    X_val, y_val,
                    optimize=True,
                    model_dir=self.config.MODELS_DIR
                )
            else:
                # If retraining is not needed, train only missing models
                logging.info("Training only missing models...")
                base_models = self.model_factory.train_base_models(
                    base_models,
                    X_train, y_train,
                    X_val, y_val,
                    optimize=True,
                    model_dir=self.config.MODELS_DIR
                )
                
            logging.info("Model training completed")
            return base_models
            
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise
            
    def _create_ensembles(self, base_models, data):
        """Create ensemble models"""
        try:
            logging.info("Building ensemble models...")
            X_train, X_val, X_test, y_train, y_val, y_test = data
            
            # First try to load pre-trained ensemble models
            ensemble_models = self.model_factory.load_ensemble_models(
                self.config.MODELS_DIR,
                base_models
            )
            
            if ensemble_models:
                logging.info("Successfully loaded pre-trained ensemble models")
                return ensemble_models
            
            # If no pre-trained models are found, create new ensemble models
            ensemble_models = {}
            
            # Create and train Stacking model
            logging.info("Training Stacking model...")
            stacking = self.model_factory.create_ensemble('stacking', base_models)
            if stacking is not None:
                stacking.train(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    model_dir=self.config.MODELS_DIR / 'ensemble' / 'stacking'
                )
                ensemble_models['stacking'] = stacking
            
            # Create and train Blending model
            logging.info("Training Blending model...")
            blending = self.model_factory.create_ensemble('blending', base_models)
            if blending is not None:
                X_full = np.vstack([X_train, X_val])
                y_full = np.concatenate([y_train, y_val])
                blending.train(
                    X=X_full,
                    y=y_full,
                    model_dir=self.config.MODELS_DIR / 'ensemble' / 'blending'
                )
                ensemble_models['blending'] = blending
            
            logging.info("Ensemble models created successfully")
            return ensemble_models
            
        except Exception as e:
            logging.error(f"Error creating ensemble models: {str(e)}")
            raise
            
    def _evaluate_models(self, models: Dict[str, BaseModel], datasets: Dict, reevaluate: bool = False):
        """Evaluate model performance"""
        try:
            if not reevaluate:
                logging.info("Skipping model evaluation")
                return None
            
            logging.info("Evaluating models...")
            results = {}
            
            # Get datasets
            X_test, y_test = datasets['test']
            
            # Evaluate each model
            for name, model in models.items():
                if model is None:
                    logging.warning(f"Skipping evaluation of {name} - model is None")
                    continue
                    
                # Check if the model is correctly initialized
                if hasattr(model, 'meta_model') and model.meta_model is None:
                    logging.warning(f"Skipping evaluation of {name} - meta_model is None")
                    continue
                    
                logging.info(f"Evaluating {name}...")
                try:
                    model_results = self.evaluator.evaluate_model(
                        model=model,
                        X=X_test,
                        y=y_test,
                        model_name=name
                    )
                    results[name] = model_results
                except Exception as e:
                    logging.error(f"Error evaluating {name}: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logging.error(f"Error evaluating models: {str(e)}")
            raise
            
    def _visualize_results(self, results):
        """Visualization analysis step"""
        try:
            # 1. Plot base performance comparison
            self.visualizer.plot_results(results)
            
            # 2. Plot detailed metric comparison
            self.visualizer.plot_metrics_comparison(results)
            
            # 3. Plot confusion matrices
            if hasattr(self.evaluator, 'confusion_matrices'):
                self.visualizer.plot_confusion_matrices(self.evaluator.confusion_matrices)
            
            # 4. Plot ROC curves
            if hasattr(self.evaluator, 'predictions'):
                self.visualizer.plot_roc_curves(self.evaluator.predictions)
            
            # 5. Plot feature importance
            if hasattr(self.evaluator, 'feature_importance'):
                for model_name, importance_data in self.evaluator.feature_importance.items():
                    if importance_data.get('importance') is not None:
                        self.visualizer.plot_feature_importance(
                            importance=importance_data['importance'],
                            feature_names=importance_data.get('features', [f'Feature_{i}' for i in range(len(importance_data['importance']))]),
                            model_name=model_name
                        )
            
            # 6. Create interactive dashboard
            self.visualizer.create_interactive_dashboard(self.evaluator)
            
            logging.info("All visualizations have been generated successfully")
            
        except Exception as e:
            logging.error(f"Error in visualization: {str(e)}")
            raise
            
    def _run_backtest(self, models, data):
        """Backtest analysis step"""
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = data
            
            # Select the best model for backtesting
            best_model_name = self.evaluator.get_best_model()
            best_model = models[best_model_name]
            
            # Run backtest
            performance = self.backtest.run_backtest(
                model=best_model,
                data=X_test,
                initial_capital=10000,
                commission=0.001
            )
            
            # Plot equity curve
            self.backtest.plot_equity_curve()
            
            # Output backtest results
            logging.info(f"\nBacktest Results for {best_model_name}:")
            for metric, value in performance.items():
                logging.info(f"{metric}: {value:.4f}")
                
        except Exception as e:
            logging.error(f"Error in backtesting: {str(e)}")
            raise

    def _load_trained_models(self):
        """Load trained models"""
        try:
            # Use ModelFactory to load trained models
            models = self.model_factory.load_trained_models(self.config.MODELS_DIR)
            if models:
                logging.info("Successfully loaded pre-trained models")
                return models
            else:
                logging.info("No pre-trained models found")
                return None
                
        except Exception as e:
            logging.error(f"Error loading trained models: {str(e)}")
            return None

    def retrain_specific_model(self, model_type: str, data):
        """Retrain specific model"""
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = data
            
            # Create new SVM model
            svm_model = self.model_factory.create_model(model_type)
            
            # Train model
            logging.info(f"Retraining {model_type} model...")
            svm_model.train(
                X_train, y_train,
                X_val, y_val,
                optimize=True,  # Enable hyperparameter optimization
                model_dir=self.config.MODELS_DIR
            )
            
            return svm_model
            
        except Exception as e:
            logging.error(f"Error retraining {model_type} model: {str(e)}")
            raise

    def _print_evaluation_results(self, results: Dict):
        """Print evaluation results"""
        logging.info("\nModel Evaluation Results:")
        logging.info("-" * 50)
        
        for model_name, metrics in results.items():
            logging.info(f"\n{model_name.upper()}:")
            for metric_name, value in metrics.items():
                logging.info(f"{metric_name}: {value:.4f}")

def create_project_structure():
    """Create project directory structure"""
    try:
        # Get project root directory
        project_root = Path(__file__).parent
        
        # Create necessary directories
        directories = [
            'data',
            'data/raw',
            'data/processed',
            'models',
            'results',
            'results/figures',
            'results/reports',
            'results/backtest',
            'logs',
            'config'
        ]
        
        for directory in directories:
            (project_root / directory).mkdir(parents=True, exist_ok=True)
            
        logging.info("Project directory structure created successfully")
        
    except Exception as e:
        logging.error(f"Error creating project structure: {str(e)}")
        raise

def check_permissions():
    """Check permissions for required directories"""
    try:
        dirs_to_check = [
            Path('data'),
            Path('data/processed'),
            Path('data/raw'),
            Path('models'),
            Path('logs')
        ]
        
        for dir_path in dirs_to_check:
            if not dir_path.exists():
                continue
            if not os.access(dir_path, os.W_OK):
                logging.error(f"No write permission for directory: {dir_path}")
                return False
        return True
        
    except Exception as e:
        logging.error(f"Error checking permissions: {str(e)}")
        return False

def fix_permissions():
    """Fix project directory permissions"""
    try:
        dirs_to_fix = [
            Path('data'),
            Path('data/processed'),
            Path('data/raw'),
            Path('models'),
            Path('logs')
        ]
        
        for dir_path in dirs_to_fix:
            if dir_path.exists():
                try:
                    os.chmod(dir_path, 0o777)
                    logging.info(f"Fixed permissions for {dir_path}")
                except Exception as e:
                    logging.warning(f"Could not fix permissions for {dir_path}: {e}")
                    
        return True
    except Exception as e:
        logging.error(f"Error fixing permissions: {e}")
        return False

def main():
    """Main function"""
    try:
        # Get user input
        regenerate = input("Regenerate data? (y/n): ").lower() == 'y'
        retrain = input("Retrain models? (y/n): ").lower() == 'y'
        reevaluate = input("Re-evaluate models? (y/n): ").lower() == 'y'
        
        # Create and run pipeline
        pipeline = ModelingPipeline()
        
        # Run pipeline
        models, results = pipeline.run(
            retrain=retrain,
            reevaluate=reevaluate,
            regenerate=regenerate
        )
        
        # If evaluation was performed, display results
        if reevaluate and results is not None:
            logging.info("\nBest performing models by metric:")
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                if metric in next(iter(results.values())):  # Check if metric exists
                    best_model = max(results.items(), key=lambda x: x[1][metric])
                    logging.info(f"Best {metric}: {best_model[0]} ({best_model[1][metric]:.4f})")
            
    except Exception as e:
        logging.error(f"Main process failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 