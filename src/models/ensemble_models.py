from typing import Dict, List, Any
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
try:
    from src.models.base_models import BaseModel, SVMModel
except ImportError:
    from base_models import BaseModel, SVMModel
import logging
from pathlib import Path
import joblib
from scipy.stats import uniform
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from sklearn.metrics import f1_score
import copy
import pickle
from io import BytesIO
import pandas as pd
import json

class StackingClassifier(BaseModel):
    """Stacked Classifier"""
    def __init__(self, base_models: Dict[str, BaseModel]):
        super().__init__(model_type='stacking')
        self.base_models = base_models
        self.meta_model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced'
        )
        self.n_folds = 5
        self.feature_names = None  # Store feature names
        
    def load_model_copy(self, model_name: str, fold: int) -> BaseModel:
        """Load a copy of the base model
        Args:
            model_name: Model name
            fold: Current fold number
        Returns:
            Model copy
        """
        try:
            # Get original model
            original_model = self.base_models[model_name]
            
            # Use deep copy
            if hasattr(original_model, 'model'):
                # Use pickle for deep copy
                buffer = BytesIO()
                pickle.dump(original_model.model, buffer)
                buffer.seek(0)
                model_copy = type(original_model)()
                model_copy.model = pickle.load(buffer)
            else:
                # If no internal model, use copy module
                model_copy = copy.deepcopy(original_model)
            
            return model_copy
            
        except Exception as e:
            logging.error(f"Error creating model copy for {model_name}, fold {fold}: {str(e)}")
            raise
            
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              model_dir: Path = None):
        """Train Stacking model using K-fold cross validation"""
        try:
            # Save feature names
            if isinstance(X_train, pd.DataFrame):
                self.feature_names = X_train.columns.tolist()
            else:
                self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
                X_train = pd.DataFrame(X_train, columns=self.feature_names)
            
            if X_val is not None:
                if not isinstance(X_val, pd.DataFrame):
                    X_val = pd.DataFrame(X_val, columns=self.feature_names)
            
            X_train = np.asarray(X_train)
            y_train = np.asarray(y_train)
            
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            meta_features = np.zeros((X_train.shape[0], len(self.base_models)))
            
            # For each base model
            for i, (model_name, model) in enumerate(self.base_models.items()):
                logging.info(f"Processing base model: {model_name}")
                fold_predictions = np.zeros(X_train.shape[0])
                
                # Load independent model copy for each fold
                for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                    logging.info(f"Processing fold {fold+1}/{self.n_folds}")
                    model_copy = self.load_model_copy(model_name, fold)
                    # Convert to DataFrame for prediction
                    X_fold = pd.DataFrame(X_train[val_idx], columns=self.feature_names)
                    fold_predictions[val_idx] = model_copy.predict_proba(X_fold)[:, 1]
                
                meta_features[:, i] = fold_predictions
            
            # Train meta model
            logging.info("Training meta model...")
            self.meta_model.fit(meta_features, y_train)
            
            if model_dir:
                model_dir.mkdir(parents=True, exist_ok=True)
                self.save_model(model_dir)
                logging.info("Stacking model saved successfully")
                
        except Exception as e:
            logging.error(f"Error training stacking model: {str(e)}")
            raise
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict"""
        try:
            if self.meta_model is None:
                raise ValueError(f"{self.model_type} meta_model is not initialized")
            
            proba = self.predict_proba(X)
            return (proba[:, 1] >= 0.5).astype(int)
        except Exception as e:
            logging.error(f"Error in {self.model_type} predict: {str(e)}")
            raise

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability"""
        try:
            if self.meta_model is None:
                raise ValueError(f"{self.model_type} meta_model is not initialized")
            
            meta_features = np.zeros((len(X), len(self.base_models)))
            for i, (name, model) in enumerate(self.base_models.items()):
                # Special handling for SVM models
                if isinstance(model, SVMModel):
                    if isinstance(X, pd.DataFrame):
                        X_np = X.values
                    else:
                        X_np = X
                    meta_features[:, i] = model.predict_proba(X_np)[:, 1]
                else:
                    meta_features[:, i] = model.predict_proba(X)[:, 1]
                
            return self.meta_model.predict_proba(meta_features)
            
        except Exception as e:
            logging.error(f"Error in {self.model_type} predict_proba: {str(e)}")
            raise

    def save_model(self, model_dir: Path):
        """Save model"""
        try:
            model_path = model_dir / f"{self.model_type}_model"
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Save meta model
            meta_model_path = model_path / "meta_model.pkl"
            with open(meta_model_path, 'wb') as f:
                pickle.dump(self.meta_model, f)
            
            # Save base models
            base_models_dir = model_path / "base_models"
            base_models_dir.mkdir(exist_ok=True)
            for name, model in self.base_models.items():
                model_file = base_models_dir / f"{name}.pkl"
                model.save_model(model_file)
            
            # Save feature names
            if self.feature_names:
                feature_names_path = model_path / "feature_names.json"
                with open(feature_names_path, 'w') as f:
                    json.dump(self.feature_names, f)
                
            logging.info(f"Saved {self.model_type} model successfully")
            
        except Exception as e:
            logging.error(f"Error saving {self.model_type} model: {str(e)}")
            raise
            
    def load_model(self, model_dir: Path) -> bool:
        """Load model"""
        try:
            model_path = model_dir / f"{self.model_type}_model"
            
            # Load meta model
            meta_model_path = model_path / "meta_model.pkl"
            if not meta_model_path.exists():
                logging.warning(f"Meta model file not found at {meta_model_path}")
                return False
                
            with open(meta_model_path, 'rb') as f:
                self.meta_model = pickle.load(f)
            
            # Load feature names
            feature_names_path = model_path / "feature_names.json"
            if feature_names_path.exists():
                with open(feature_names_path, 'r') as f:
                    self.feature_names = json.load(f)
            
            logging.info(f"Loaded {self.model_type} model successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error loading {self.model_type} model: {str(e)}")
            return False

class BlendingEnsemble(BaseModel):
    """Blending Ensemble"""
    def __init__(self, base_models: Dict[str, BaseModel]):
        super().__init__(model_type='blending')
        self.base_models = base_models
        self.meta_model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced'
        )
        self.train_ratio = 0.6  # Training set ratio
        self.blend_ratio = 0.2  # Blending set ratio
        self.feature_names = None  # Store feature names
        
    def train(self, X: np.ndarray, y: np.ndarray, model_dir: Path = None):
        """Train ensemble model using Blending method"""
        try:
            logging.info("Starting Blending ensemble training...")
            
            # Save feature names
            if isinstance(X, pd.DataFrame):
                self.feature_names = X.columns.tolist()
                X_array = X.values  # Convert to numpy array but retain original feature names
            else:
                if hasattr(X, 'feature_names_in_'):  # scikit-learn style
                    self.feature_names = X.feature_names_in_.tolist()
                else:
                    self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
                X_array = X
            
            # 1. Split data by time order
            n = len(X_array)
            n_train = int(n * self.train_ratio)
            n_blend = int(n * self.blend_ratio)
            
            # Split data
            train_idx = np.arange(n_train)
            blend_idx = np.arange(n_train, n_train + n_blend)
            test_idx = np.arange(n_train + n_blend, n)
            
            # Split data
            X_blend = X_array[blend_idx]
            y_blend = y[blend_idx]
            X_test = X_array[test_idx]
            y_test = y[test_idx]
            
            logging.info(f"Data split - Train: {n_train}, Blend: {len(X_blend)}, Test: {len(X_test)}")
            
            # 2. Use trained base models to predict on blending set
            blend_predictions = np.zeros((len(X_blend), len(self.base_models)))
            for i, (name, model) in enumerate(self.base_models.items()):
                logging.info(f"Getting predictions from {name} on blending set")
                # Create DataFrame with original feature names
                if isinstance(X, pd.DataFrame):
                    X_blend_df = pd.DataFrame(X_blend, columns=self.feature_names)
                    blend_predictions[:, i] = model.predict_proba(X_blend_df)[:, 1]
                else:
                    blend_predictions[:, i] = model.predict_proba(X_blend)[:, 1]
            
            # 3. Train meta model
            logging.info("Training meta model...")
            self.meta_model.fit(blend_predictions, y_blend)
            
            # 4. Save model
            if model_dir:
                model_dir.mkdir(parents=True, exist_ok=True)
                self.save_model(model_dir)
                logging.info("Blending model saved successfully")
            
            # 5. Evaluate performance (optional)
            if len(X_test) > 0:
                test_predictions = np.zeros((len(X_test), len(self.base_models)))
                for i, (name, model) in enumerate(self.base_models.items()):
                    if isinstance(X, pd.DataFrame):
                        X_test_df = pd.DataFrame(X_test, columns=self.feature_names)
                        test_predictions[:, i] = model.predict_proba(X_test_df)[:, 1]
                    else:
                        test_predictions[:, i] = model.predict_proba(X_test)[:, 1]
                meta_predictions = self.meta_model.predict(test_predictions)
                score = f1_score(y_test, meta_predictions)
                logging.info(f"Blending ensemble test F1 score: {score:.4f}")
                
        except Exception as e:
            logging.error(f"Error training blending model: {str(e)}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict"""
        try:
            if self.meta_model is None:
                raise ValueError(f"{self.model_type} meta_model is not initialized")
            
            proba = self.predict_proba(X)
            return (proba[:, 1] >= 0.5).astype(int)
        except Exception as e:
            logging.error(f"Error in {self.model_type} predict: {str(e)}")
            raise

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability"""
        try:
            if self.meta_model is None:
                raise ValueError(f"{self.model_type} meta_model is not initialized")
            
            meta_features = np.zeros((len(X), len(self.base_models)))
            for i, (name, model) in enumerate(self.base_models.items()):
                # Special handling for SVM models
                if isinstance(model, SVMModel):
                    if isinstance(X, pd.DataFrame):
                        X_np = X.values
                    else:
                        X_np = X
                    meta_features[:, i] = model.predict_proba(X_np)[:, 1]
                else:
                    meta_features[:, i] = model.predict_proba(X)[:, 1]
                
            return self.meta_model.predict_proba(meta_features)
            
        except Exception as e:
            logging.error(f"Error in {self.model_type} predict_proba: {str(e)}")
            raise

    def save_model(self, model_dir: Path):
        """Save model"""
        try:
            model_path = model_dir / f"{self.model_type}_model"
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Save meta model
            meta_model_path = model_path / "meta_model.pkl"
            with open(meta_model_path, 'wb') as f:
                pickle.dump(self.meta_model, f)
            
            # Save base models
            base_models_dir = model_path / "base_models"
            base_models_dir.mkdir(exist_ok=True)
            for name, model in self.base_models.items():
                model_file = base_models_dir / f"{name}.pkl"
                model.save_model(model_file)
                
            # Save feature names
            if self.feature_names:
                feature_names_path = model_path / "feature_names.json"
                with open(feature_names_path, 'w') as f:
                    json.dump(self.feature_names, f)
                
            logging.info(f"Saved {self.model_type} model successfully")
            
        except Exception as e:
            logging.error(f"Error saving {self.model_type} model: {str(e)}")
            raise
            
    def load_model(self, model_dir: Path) -> bool:
        """Load model"""
        try:
            model_path = model_dir / f"{self.model_type}_model"
            
            # Load meta model
            meta_model_path = model_path / "meta_model.pkl"
            if not meta_model_path.exists():
                logging.warning(f"Meta model file not found at {meta_model_path}")
                return False
                
            with open(meta_model_path, 'rb') as f:
                self.meta_model = pickle.load(f)
            
            # Load feature names
            feature_names_path = model_path / "feature_names.json"
            if feature_names_path.exists():
                with open(feature_names_path, 'r') as f:
                    self.feature_names = json.load(f)
            
            logging.info(f"Loaded {self.model_type} model successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error loading {self.model_type} model: {str(e)}")
            return False