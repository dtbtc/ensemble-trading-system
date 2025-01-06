from abc import ABC, abstractmethod
import joblib
import logging
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import numpy as np
from typing import Dict
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import json
from pathlib import Path
from scipy.stats import uniform
from sklearn.preprocessing import StandardScaler

class BaseModel(ABC, BaseEstimator):
    """Base Model Class"""
    def __init__(self, model_type: str = None, params: Dict = None):
        self.model_type = model_type
        self.model = None
        self.params = params or {}
        self.best_params = None
        self.feature_names = None  # Store feature names
        
    def fit_with_names(self, X: np.ndarray, y: np.ndarray):
        """Train model with feature names"""
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            self.model.fit(X, y)
        else:
            if self.feature_names is None:
                self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=self.feature_names)
            self.model.fit(X_df, y)
            
    def predict_with_names(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using feature names"""
        if not isinstance(X, pd.DataFrame) and self.feature_names is not None:
            X = pd.DataFrame(X, columns=self.feature_names)
        return self.model.predict(X)
        
    def predict_proba_with_names(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions using feature names"""
        if not isinstance(X, pd.DataFrame) and self.feature_names is not None:
            X = pd.DataFrame(X, columns=self.feature_names)
        return self.model.predict_proba(X)
        
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray = None, y_val: np.ndarray = None,
                               n_iter: int = 50, cv: int = 5) -> Dict:
        """Base method for hyperparameter optimization"""
        try:
            # Get parameter grid
            param_grid = self.get_param_grid()
            
            # Use random search
            random_search = RandomizedSearchCV(
                estimator=self.model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring='f1',
                n_jobs=-1,
                random_state=42
            )
            
            # Execute search
            random_search.fit(X_train, y_train)
            
            # Save best parameters
            self.best_params = random_search.best_params_
            
            # Log best parameters and score
            logging.info(f"Best parameters found: {self.best_params}")
            logging.info(f"Best score: {random_search.best_score_:.4f}")
            
            return self.best_params
            
        except Exception as e:
            logging.error(f"Error in hyperparameter optimization: {str(e)}")
            raise
            
    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, 
              y_val: np.ndarray = None, optimize: bool = True, model_dir: Path = None):
        """Train model"""
        try:
            # Save feature names
            if isinstance(X, pd.DataFrame):
                self.feature_names = X.columns.tolist()
            else:
                self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
                
            # If model exists, try to load
            if model_dir and not optimize:
                if self.load_model(model_dir):
                    return
                    
            # Execute training
            if optimize:
                best_params = self.optimize_hyperparameters(
                    X, y,
                    X_val, y_val,
                    n_iter=50
                )
                self.model.set_params(**best_params)
                self.best_params = best_params  # Ensure best parameters are saved
            
            # Train model
            self.fit_with_names(X, y)
            logging.info(f"{self.model_type} training completed")
            
            # Save model
            if model_dir:
                self.save_model(model_dir)
                
        except Exception as e:
            logging.error(f"Error training {self.model_type} model: {str(e)}")
            raise
            
    def save_model(self, model_dir: Path):
        """Save model and feature names"""
        try:
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_path = model_dir / f"{self.model_type}_model.pkl"
            joblib.dump(self.model, model_path)
            
            # Save feature names
            if self.feature_names:
                feature_names_path = model_dir / f"{self.model_type}_feature_names.json"
                with open(feature_names_path, 'w') as f:
                    json.dump(self.feature_names, f)
                    
            # Save optimal parameters (even if default)
            params_to_save = self.best_params if self.best_params else self.params
            params_path = model_dir / f"{self.model_type}_params.json"
            with open(params_path, 'w') as f:
                json.dump(params_to_save, f)
                
            logging.info(f"Saved {self.model_type} model, parameters and feature names")
            
        except Exception as e:
            logging.error(f"Error saving {self.model_type} model: {str(e)}")
            raise
            
    def load_model(self, model_dir: Path) -> bool:
        """Load model and feature names"""
        try:
            model_path = model_dir / f"{self.model_type}_model.pkl"
            feature_names_path = model_dir / f"{self.model_type}_feature_names.json"
            params_path = model_dir / f"{self.model_type}_params.json"
            
            if not model_path.exists():
                return False
                
            # Load model
            self.model = joblib.load(model_path)
            
            # Load feature names
            if feature_names_path.exists():
                with open(feature_names_path, 'r') as f:
                    self.feature_names = json.load(f)
                    
            # Load parameters
            if params_path.exists():
                with open(params_path, 'r') as f:
                    self.best_params = json.load(f)
                    
            logging.info(f"Loaded pre-trained {self.model_type} model")
            return True
            
        except Exception as e:
            logging.error(f"Error loading {self.model_type} model: {str(e)}")
            return False

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.predict_with_names(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions"""
        return self.predict_proba_with_names(X)

class LightGBMModel(BaseModel):
    def __init__(self, params: Dict = None):
        super().__init__(model_type='lightgbm', params=params)
        
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'objective': 'binary',
            'metric': 'binary_logloss',
            'is_unbalance': True,
            'verbose': -1
        }
        
        # Update parameters
        self.params = default_params.copy()
        if params:
            self.params.update(params)
            
        self.model = lgb.LGBMClassifier(**self.params)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.predict_with_names(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions"""
        return self.predict_proba_with_names(X)
        
    def get_param_grid(self) -> Dict:
        """Get parameter grid for grid search"""
        return {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'num_leaves': [31, 63, 127],
            'min_child_samples': [20, 50, 100],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, 
              y_val: np.ndarray = None, optimize: bool = True, model_dir: Path = None):
        try:
            # Try to load existing model
            if model_dir and not optimize:
                if self.load_model(model_dir):
                    return
                    
            if optimize:
                param_grid = self.get_param_grid()
                best_params = self.optimize_hyperparameters(X, y, param_grid)
                self.model.set_params(**best_params)
            
            # Prepare callbacks
            callbacks = []
            if X_val is not None and y_val is not None:
                early_stopping = lgb.early_stopping(
                    stopping_rounds=50,
                    first_metric_only=True,
                    verbose=True
                )
                callbacks.append(early_stopping)
            
            # Train model
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)] if X_val is not None else None,
                eval_metric=['auc', 'binary_logloss'],
                callbacks=callbacks
            )
            
            logging.info("LightGBM training completed")
            
            # Save model
            if model_dir:
                self.save_model(model_dir)
                
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            raise

class XGBoostModel(BaseModel):
    def __init__(self, params: Dict = None):
        super().__init__('xgboost', params)
        
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'scale_pos_weight': 1,  # Will be updated based on class proportions during training
            'eval_metric': ['auc', 'logloss']
        }
        
        # Update parameters
        self.params = default_params.copy()
        if params:
            self.params.update(params)
            
        self.model = xgb.XGBClassifier(**self.params)
        
    def get_param_grid(self) -> Dict:
        """Get parameter grid for grid search"""
        return {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2]  # Add gamma parameter
        }
        
    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, 
              y_val: np.ndarray = None, optimize: bool = True, model_dir: Path = None):
        """Train model"""
        try:
            # Calculate class weights
            neg_pos_ratio = np.sum(y == 0) / np.sum(y == 1)
            self.params['scale_pos_weight'] = neg_pos_ratio
            self.model.set_params(**{'scale_pos_weight': neg_pos_ratio})
            
            # If model exists, try to load
            if model_dir and not optimize:  # Only try to load in non-optimization mode
                if self.load_model(model_dir):
                    return
                    
            # Execute training
            if optimize:
                param_grid = self.get_param_grid()
                best_params = self.optimize_hyperparameters(X, y, param_grid)
                self.model.set_params(**best_params)
            
            self.model.fit(X, y)
            logging.info(f"{self.model_type} training completed")
            
            # Save model
            if model_dir:
                self.save_model(model_dir)
                
        except Exception as e:
            logging.error(f"Error training {self.model_type} model: {str(e)}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_with_names(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba_with_names(X)

class RandomForestModel(BaseModel):
    def __init__(self, params: Dict = None):
        super().__init__('random_forest', params)
        
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'min_samples_split': 2,
            'class_weight': 'balanced',  # Add class weights
            'random_state': 42
        }
        
        # Update parameters
        self.params = default_params.copy()
        if params:
            self.params.update(params)
            
        self.model = RandomForestClassifier(**self.params)
        
    def get_param_grid(self) -> Dict:
        """Get parameter grid for grid search"""
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, 
              y_val: np.ndarray = None, optimize: bool = True, model_dir: Path = None):
        """Train model"""
        try:
            # If model exists, try to load
            if model_dir and not optimize:  # Only try to load in non-optimization mode
                if self.load_model(model_dir):
                    return
                    
            # Execute training
            if optimize:
                best_params = self.optimize_hyperparameters(X, y, self.get_param_grid())
                self.model.set_params(**best_params)
            
            self.model.fit(X, y)
            logging.info(f"{self.model_type} training completed")
            
            # Save model
            if model_dir:
                self.save_model(model_dir)
                
        except Exception as e:
            logging.error(f"Error training {self.model_type} model: {str(e)}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_with_names(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba_with_names(X)

class SVMModel(BaseModel):
    def __init__(self, params: Dict = None):
        super().__init__(model_type='svm', params=params)
        
        # Default parameters
        default_params = {
            'C': 1.0,
            'kernel': 'rbf',
            'probability': True,
            'class_weight': 'balanced',
            'random_state': 42,
            'gamma': 'scale',
            'cache_size': 3000,
            'tol': 1e-3,
            'max_iter': 5000  # Increase maximum iterations
        }
        
        # Update parameters
        self.params = default_params.copy()
        if params:
            self.params.update(params)
            
        self.model = SVC(**self.params)
        self.scaler = StandardScaler()  # Add scaler
        
    def get_param_grid(self) -> Dict:
        """Get parameter grid for grid search"""
        return {
            'C': [0.1, 1.0, 10.0, 100.0],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'kernel': ['rbf', 'linear']
        }
        
    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, 
              y_val: np.ndarray = None, optimize: bool = True, model_dir: Path = None):
        """Train model"""
        try:
            # If model exists, try to load
            if model_dir and not optimize:
                if self.load_model(model_dir):
                    return
                    
            # Standardize training data
            X_scaled = self.scaler.fit_transform(X)
            
            # Execute training
            if optimize:
                param_grid = self.get_param_grid()
                best_params = self.optimize_hyperparameters(
                    X_scaled, y,  # Use standardized data
                    param_grid,
                    cv=3
                )
                self.model.set_params(**best_params)
            
            # Train model
            self.model.fit(X_scaled, y)
            
            # If validation set exists, evaluate model
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                val_score = self.model.score(X_val_scaled, y_val)
                logging.info(f"SVM validation score: {val_score:.4f}")
            
            logging.info(f"{self.model_type} training completed")
            
            # Save model
            if model_dir:
                self.save_model(model_dir)
                
        except Exception as e:
            logging.error(f"Error training {self.model_type} model: {str(e)}")
            raise
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
        
    def save_model(self, model_dir: Path):
        """Save model and scaler"""
        try:
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save SVM model
            model_path = model_dir / f"{self.model_type}_model.pkl"
            joblib.dump(self.model, model_path)
            
            # Save scaler
            scaler_path = model_dir / f"{self.model_type}_scaler.pkl"
            joblib.dump(self.scaler, scaler_path)
            
            # Save feature names
            if self.feature_names:
                feature_names_path = model_dir / f"{self.model_type}_feature_names.json"
                with open(feature_names_path, 'w') as f:
                    json.dump(self.feature_names, f)
                    
            logging.info(f"Saved {self.model_type} model and scaler")
            
        except Exception as e:
            logging.error(f"Error saving {self.model_type} model: {str(e)}")
            raise
            
    def load_model(self, model_dir: Path) -> bool:
        """Load model and scaler"""
        try:
            model_path = model_dir / f"{self.model_type}_model.pkl"
            scaler_path = model_dir / f"{self.model_type}_scaler.pkl"
            
            if not model_path.exists() or not scaler_path.exists():
                return False
                
            # Load model and scaler
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            # Load feature names
            feature_names_path = model_dir / f"{self.model_type}_feature_names.json"
            if feature_names_path.exists():
                with open(feature_names_path, 'r') as f:
                    self.feature_names = json.load(f)
                    
            logging.info(f"Loaded {self.model_type} model and scaler")
            return True
            
        except Exception as e:
            logging.error(f"Error loading {self.model_type} model: {str(e)}")
            return False

class KNNModel(BaseModel):
    def __init__(self, params: Dict = None):
        super().__init__(model_type='knn', params=params)
        
        # Default parameters
        default_params = {
            'n_neighbors': 5,
            'weights': 'distance',
            'metric': 'minkowski',
            'p': 2,
            'n_jobs': -1  # Restore parallel processing
        }
        
        # Update parameters
        self.params = default_params.copy()
        if params:
            self.params.update(params)
            
        self.model = KNeighborsClassifier(**self.params)
        
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray = None, y_val: np.ndarray = None,
                               n_iter: int = 15, cv: int = 3) -> Dict:
        """Optimize KNN hyperparameters"""
        try:
            # Simplified parameter space
            param_space = {
                'n_neighbors': [3, 5],  # Reduce options
                'weights': ['distance'],  # Fixed distance weights
                'p': [2]  # Fixed Euclidean distance
            }
            
            # Use random search
            random_search = RandomizedSearchCV(
                estimator=KNeighborsClassifier(n_jobs=-1),
                param_distributions=param_space,
                n_iter=n_iter,
                cv=cv,
                scoring='f1',
                n_jobs=-1,  # Use parallel
                verbose=1
            )
            
            # Execute search
            random_search.fit(X_train, y_train)
            
            # Log best parameters and score
            self.best_params = random_search.best_params_
            self.best_params.update({
                'n_jobs': -1  # Ensure parallel processing
            })
            
            logging.info(f"Best parameters found: {self.best_params}")
            logging.info(f"Best score: {random_search.best_score_:.4f}")
            
            return self.best_params
            
        except Exception as e:
            logging.error(f"Error optimizing KNN hyperparameters: {str(e)}")
            raise
            
    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, 
              y_val: np.ndarray = None, optimize: bool = True, model_dir: Path = None):
        """Train model"""
        try:
            # If model exists, try to load
            if model_dir and not optimize:
                if self.load_model(model_dir):
                    return
                    
            # Execute training
            if optimize:
                best_params = self.optimize_hyperparameters(
                    X, y,
                    X_val, y_val,
                    n_iter=15,
                    cv=3
                )
                self.model.set_params(**best_params)
            
            self.fit_with_names(X, y)
            logging.info(f"{self.model_type} training completed")
            
            # Save model
            if model_dir:
                self.save_model(model_dir)
                
        except Exception as e:
            logging.error(f"Error training {self.model_type} model: {str(e)}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_with_names(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba_with_names(X)

class LogisticRegressionModel(BaseModel):
    def __init__(self, params: Dict = None):
        super().__init__(model_type='logistic', params=params)
        
        # Default parameters
        default_params = {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'class_weight': 'balanced',
            'random_state': 42,
            'max_iter': 5000,  # Increase maximum iterations
            'tol': 1e-4       # Adjust convergence tolerance
        }
        
        # Update parameters
        self.params = default_params.copy()
        if params:
            self.params.update(params)
            
        self.model = LogisticRegression(**self.params)
        
    def get_param_grid(self) -> Dict:
        """Get parameter grid for grid search"""
        return {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        
    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, 
              y_val: np.ndarray = None, optimize: bool = True, model_dir: Path = None):
        """Train model"""
        try:
            # If model exists, try to load
            if model_dir and not optimize:
                if self.load_model(model_dir):
                    return
                    
            # Execute training
            if optimize:
                param_grid = self.get_param_grid()
                best_params = self.optimize_hyperparameters(
                    X, y, 
                    param_grid,
                    cv=3
                )
                # Ensure necessary parameters are maintained
                best_params.update({
                    'class_weight': 'balanced',
                    'random_state': 42,
                    'max_iter': 5000,  # Increase iterations
                    'tol': 1e-4        # Adjust convergence tolerance
                })
                self.model.set_params(**best_params)
            
            self.model.fit(X, y)
            logging.info(f"{self.model_type} training completed")
            
            # Save model
            if model_dir:
                self.save_model(model_dir)
                
        except Exception as e:
            logging.error(f"Error training {self.model_type} model: {str(e)}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_with_names(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba_with_names(X)

class DecisionTreeModel(BaseModel):
    def __init__(self, params: Dict = None):
        super().__init__(model_type='decision_tree', params=params)
        
        # Default parameters
        default_params = {
            'max_depth': 5,
            'min_samples_split': 2,
            'class_weight': 'balanced',  # Add class weights
            'random_state': 42
        }
        
        # Update parameters
        self.params = default_params.copy()
        if params:
            self.params.update(params)
            
        self.model = DecisionTreeClassifier(**self.params)
        
    def get_param_grid(self) -> Dict:
        """Get parameter grid for grid search"""
        return {
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
        
    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, 
              y_val: np.ndarray = None, optimize: bool = True, model_dir: Path = None):
        """Train model"""
        try:
            # If model exists, try to load
            if model_dir and not optimize:  # Only try to load in non-optimization mode
                if self.load_model(model_dir):
                    return
                    
            # Execute training
            if optimize:
                param_grid = self.get_param_grid()  # Use get_param_grid method
                best_params = self.optimize_hyperparameters(X, y, param_grid)
                self.model.set_params(**best_params)
            
            self.model.fit(X, y)
            logging.info(f"{self.model_type} training completed")
            
            # Save model
            if model_dir:
                self.save_model(model_dir)
                
        except Exception as e:
            logging.error(f"Error training {self.model_type} model: {str(e)}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_with_names(X)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba_with_names(X)

class NaiveBayesModel(BaseModel):
    def __init__(self, params: Dict = None):
        super().__init__(model_type='naive_bayes', params=params)
        
        # Default parameters
        default_params = {
            'var_smoothing': 1e-9  # Smoothing parameter
        }
        
        # Update parameters
        self.params = default_params.copy()
        if params:
            self.params.update(params)
            
        self.model = GaussianNB(**self.params)
        
    def get_param_grid(self) -> Dict:
        """Get parameter grid for grid search"""
        return {
            'var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7]
        }
        
    def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, 
              y_val: np.ndarray = None, optimize: bool = True, model_dir: Path = None):
        """Train model"""
        try:
            # If model exists, try to load
            if model_dir and not optimize:
                if self.load_model(model_dir):
                    return
                    
            # Execute training
            if optimize:
                param_grid = self.get_param_grid()  # Use get_param_grid method
                best_params = self.optimize_hyperparameters(
                    X, y, 
                    param_grid,
                    cv=3  # Reduce number of cross-validation folds
                )
                self.model.set_params(**best_params)
            
            self.model.fit(X, y)
            logging.info(f"{self.model_type} training completed")
            
            # Save model
            if model_dir:
                self.save_model(model_dir)
                
        except Exception as e:
            logging.error(f"Error training {self.model_type} model: {str(e)}")
            raise 