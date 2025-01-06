from typing import Dict, List
import numpy as np
from .base_models import (
    BaseModel, LightGBMModel, XGBoostModel, RandomForestModel,
    DecisionTreeModel, SVMModel, KNNModel, LogisticRegressionModel,
    NaiveBayesModel
)
from .ensemble_models import StackingClassifier, BlendingEnsemble
import logging
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from pathlib import Path

class ModelFactory:
    """Model Factory Class"""
    
    def __init__(self):
        # Define all supported model types
        self.model_types = [
            'lightgbm',
            'xgboost',
            'random_forest',
            'decision_tree',
            'svm',
            'knn',
            'logistic_regression',
            'naive_bayes'
        ]

    @staticmethod
    def create_base_models() -> Dict[str, BaseModel]:
        """Create base models"""
        try:
            base_models = {}
            
            # 1. LightGBM
            lgb_params = {
                'n_estimators': 200,
                'learning_rate': 0.05,
                'max_depth': 5,
                'num_leaves': 31,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
            base_models['lightgbm'] = LightGBMModel(params=lgb_params)
            
            # 2. XGBoost
            xgb_params = {
                'n_estimators': 200,
                'learning_rate': 0.05,
                'max_depth': 5,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
            base_models['xgboost'] = XGBoostModel(params=xgb_params)
            
            # 3. Random Forest
            rf_params = {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }
            base_models['random_forest'] = RandomForestModel(params=rf_params)
            
            # 4. Decision Tree
            dt_params = {
                'max_depth': 5,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }
            base_models['decision_tree'] = DecisionTreeModel(params=dt_params)
            
            # 5. SVM
            import warnings
            warnings.filterwarnings('ignore', category=UserWarning, 
                                  message='X has feature names, but SVC was fitted without feature names')
            svm_params = {
                'C': 1.0,
                'kernel': 'rbf',
                'probability': True,
                'random_state': 42,
                'gamma': 'scale'
            }
            base_models['svm'] = SVMModel(params=svm_params)
            
            # 6. KNN
            knn_params = {
                'n_neighbors': 5,
                'weights': 'distance',
                'algorithm': 'auto'
            }
            base_models['knn'] = KNNModel(params=knn_params)
            
            # 7. Logistic Regression
            lr_params = {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42
            }
            base_models['logistic_regression'] = LogisticRegressionModel(params=lr_params)
            
            # 8. Naive Bayes
            nb_params = {}
            base_models['naive_bayes'] = NaiveBayesModel(params=nb_params)
            
            logging.info(f"Created {len(base_models)} base models")
            return base_models
            
        except Exception as e:
            logging.error(f"Error creating base models: {str(e)}")
            raise
            
    def train_base_models(self, base_models: Dict, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray = None, y_val: np.ndarray = None,
                         optimize: bool = True, model_dir: Path = None) -> Dict:
        """Train base models, only train missing models"""
        trained_models = {}
        loaded_models = set()  # Track loaded models
        
        try:
            for name, model in base_models.items():
                # Skip if model already loaded
                if name in loaded_models:
                    continue
                    
                # First try to load existing model
                if model_dir and model.load_model(model_dir):
                    logging.info(f"Loaded pre-trained {name} model")
                    trained_models[name] = model
                    loaded_models.add(name)  # Record loaded model
                    continue
                    
                # Train new model if not exists
                logging.info(f"Training new {name} model...")
                model.train(
                    X_train, y_train,
                    X_val, y_val,
                    optimize=optimize,
                    model_dir=model_dir
                )
                trained_models[name] = model
                loaded_models.add(name)  # Record trained model
                logging.info(f"{name} model training completed")
                
            return trained_models
            
        except Exception as e:
            logging.error(f"Error training base models: {str(e)}")
            raise
            
    def create_ensemble(self, ensemble_type: str, base_models: Dict[str, BaseModel]):
        """Create ensemble model"""
        try:
            if ensemble_type == 'stacking':
                return StackingClassifier(base_models=base_models)
            elif ensemble_type == 'blending':
                return BlendingEnsemble(base_models=base_models)
            else:
                raise ValueError(f"Unknown ensemble type: {ensemble_type}")
                
        except Exception as e:
            logging.error(f"Error creating {ensemble_type} model: {str(e)}")
            raise
            
    def train_ensemble(self, ensemble: BaseModel, X_train: np.ndarray, 
                      y_train: np.ndarray, X_val: np.ndarray = None, 
                      y_val: np.ndarray = None) -> BaseModel:
        """Train ensemble model"""
        try:
            logging.info(f"Training {ensemble.__class__.__name__}...")
            
            # Train ensemble model
            ensemble.train(X_train, y_train, X_val, y_val)
            
            # Evaluate performance
            if X_val is not None and y_val is not None:
                val_pred = ensemble.predict(X_val)
                scores = {
                    'accuracy': accuracy_score(y_val, val_pred),
                    'precision': precision_score(y_val, val_pred),
                    'recall': recall_score(y_val, val_pred),
                    'f1': f1_score(y_val, val_pred)
                }
                
                logging.info("Ensemble validation scores:")
                for metric, score in scores.items():
                    logging.info(f"- {metric}: {score:.4f}")
            
            return ensemble
            
        except Exception as e:
            logging.error(f"Error training ensemble model: {str(e)}")
            raise 

    def load_trained_models(self, model_dir: Path, X_train=None, y_train=None, X_val=None, y_val=None) -> Dict[str, BaseModel]:
        """Load trained models, train new ones if not exist"""
        trained_models = {}
        try:
            # Load base models
            for model_type in self.model_types:
                model = self.create_model(model_type)
                if model.load_model(model_dir):
                    trained_models[model_type] = model
                    logging.info(f"Loaded pre-trained {model_type} model")
                else:
                    logging.info(f"No pre-trained model found for {model_type}, training new model...")
                    if X_train is not None and y_train is not None:
                        # Train missing model
                        model.train(
                            X_train, y_train,
                            X_val, y_val,
                            optimize=True,
                            model_dir=model_dir
                        )
                        trained_models[model_type] = model
                        logging.info(f"Successfully trained new {model_type} model")
                    else:
                        logging.warning(f"Cannot train {model_type} model: training data not provided")
                        
            if trained_models:
                logging.info("Successfully loaded/trained all models")
                return trained_models
            logging.info("No models found or trained")
            return None
            
        except Exception as e:
            logging.error(f"Error loading/training models: {str(e)}")
            return None 

    def create_model(self, model_type: str) -> BaseModel:
        """Create single model"""
        model_type = model_type.lower().strip()
        if model_type == 'lightgbm':
            return LightGBMModel()
        elif model_type == 'xgboost':
            return XGBoostModel()
        elif model_type == 'random_forest':
            return RandomForestModel()
        elif model_type == 'decision_tree':
            return DecisionTreeModel()
        elif model_type == 'svm':
            return SVMModel()
        elif model_type == 'knn':
            return KNNModel()
        elif model_type == 'logistic_regression':
            return LogisticRegressionModel()
        elif model_type == 'naive_bayes':
            return NaiveBayesModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}") 

    def load_ensemble_models(self, model_dir: Path, base_models: Dict[str, BaseModel]) -> Dict[str, BaseModel]:
        """Load ensemble models"""
        try:
            ensemble_models = {}
            ensemble_dir = model_dir / 'ensemble'
            
            if ensemble_dir.exists():
                # Load Stacking model
                stacking_dir = ensemble_dir / 'stacking'
                if stacking_dir.exists():
                    stacking = StackingClassifier(base_models)
                    if stacking.load_model(stacking_dir):
                        # Verify model loaded correctly
                        if stacking.meta_model is not None:
                            ensemble_models['stacking'] = stacking
                        else:
                            logging.error("Stacking meta_model is None after loading")
                
                # Load Blending model
                blending_dir = ensemble_dir / 'blending'
                if blending_dir.exists():
                    blending = BlendingEnsemble(base_models)
                    if blending.load_model(blending_dir):
                        # Verify model loaded correctly
                        if blending.meta_model is not None:
                            ensemble_models['blending'] = blending
                        else:
                            logging.error("Blending meta_model is None after loading")
            
            return ensemble_models
            
        except Exception as e:
            logging.error(f"Error loading ensemble models: {str(e)}")
            raise 