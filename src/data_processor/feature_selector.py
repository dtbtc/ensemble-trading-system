import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
import logging
from joblib import Parallel, delayed
from multiprocessing import cpu_count

class FeatureSelector:
    """Feature Selection Class"""
    
    @staticmethod
    def select_by_correlation(df: pd.DataFrame, 
                            threshold: float = 0.95) -> List[str]:
        """
        Feature selection based on correlation
        
        Args:
            df: Feature DataFrame
            threshold: Correlation threshold
            
        Returns:
            List[str]: List of selected feature names
        """
        try:
            # Calculate correlation matrix
            corr_matrix = df.corr().abs()
            
            # Get upper triangular matrix
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Find highly correlated features
            to_drop = [
                column for column in upper.columns 
                if any(upper[column] > threshold)
            ]
            
            # Keep features
            selected_features = [col for col in df.columns if col not in to_drop]
            
            logging.info(f"Selected {len(selected_features)} features based on correlation")
            return selected_features
            
        except Exception as e:
            logging.error(f"Error in correlation-based selection: {str(e)}")
            raise
            
    @staticmethod
    def select_by_importance(X: pd.DataFrame,
                           y: pd.Series,
                           n_features: int = 20,
                           method: str = 'f_score') -> List[str]:
        """
        Feature selection based on importance
        
        Args:
            X: Feature DataFrame
            y: Label Series
            n_features: Number of features to select
            method: Selection method ('f_score', 'mutual_info', 'rf')
            
        Returns:
            List[str]: List of selected feature names
        """
        try:
            # Parallel data preprocessing
            def process_column(col):
                series = X[col]
                # Handle infinity and NaN values
                series = series.replace([np.inf, -np.inf], np.nan)
                return series.fillna(method='ffill').fillna(method='bfill')
                
            # Process all columns in parallel
            n_jobs = cpu_count()
            processed_cols = Parallel(n_jobs=n_jobs)(
                delayed(process_column)(col) for col in X.columns
            )
            X_processed = pd.DataFrame(
                dict(zip(X.columns, processed_cols)),
                index=X.index
            )
            
            if method == 'f_score':
                selector = SelectKBest(
                    score_func=f_classif,
                    k=min(n_features, X.shape[1])
                )
            elif method == 'mutual_info':
                selector = SelectKBest(
                    score_func=mutual_info_classif,
                    k=min(n_features, X.shape[1])
                )
            elif method == 'rf':
                selector = SelectFromModel(
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=42,
                        n_jobs=n_jobs  # Use parallel processing
                    ),
                    max_features=min(n_features, X.shape[1])
                )
                
            # Feature selection
            selector.fit(X_processed, y)
            
            # Get selected features
            selected = X.columns[selector.get_support()].tolist()
            
            logging.info(f"Selected {len(selected)} features using {method}")
            return selected
            
        except Exception as e:
            logging.error(f"Error in importance-based selection: {str(e)}")
            raise
            
    @staticmethod
    def select_by_rfe(X: pd.DataFrame,
                      y: pd.Series,
                      n_features: int = 20) -> List[str]:
        """Using Recursive Feature Elimination"""
        try:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(estimator=estimator, n_features_to_select=n_features)
            
            # Feature selection
            selector.fit(X, y)
            
            # Get selected features
            selected = X.columns[selector.support_]
            
            logging.info(f"Selected {len(selected)} features using RFE")
            return selected.tolist()
            
        except Exception as e:
            logging.error(f"Error in RFE selection: {str(e)}")
            raise
            
    def get_feature_importance_scores(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Get feature importance scores (parallel processing)"""
        try:
            n_jobs = cpu_count()
            
            # Parallel data preprocessing
            def process_column(col):
                series = X[col]
                return series.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
                
            processed_cols = Parallel(n_jobs=n_jobs)(
                delayed(process_column)(col) for col in X.columns
            )
            X_processed = pd.DataFrame(
                dict(zip(X.columns, processed_cols)),
                index=X.index
            )
            
            # Use random forest to calculate feature importance (enable parallel processing)
            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=n_jobs
            )
            rf.fit(X_processed, y)
            
            # Convert feature importance to Series
            importance_scores = pd.Series(
                rf.feature_importances_,
                index=X.columns,
                name='importance'
            ).sort_values(ascending=False)
            
            return importance_scores
            
        except Exception as e:
            logging.error(f"Error calculating feature importance: {str(e)}")
            raise 

    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series, methods: List[str]) -> Dict[str, pd.Series]:
        """
        Calculate feature importance
        
        Args:
            X: Feature DataFrame
            y: Target variable Series
            methods: List of calculation methods ['f_score', 'mutual_info', 'rf']
            
        Returns:
            Dict[str, pd.Series]: Feature importance scores for different methods
        """
        try:
            importance_scores = {}
            
            # 1. F-score
            if 'f_score' in methods:
                selector = SelectKBest(score_func=f_classif, k='all')
                selector.fit(X, y)
                scores = pd.Series(selector.scores_, index=X.columns)
                importance_scores['f_score'] = scores.sort_values(ascending=False)
            
            # 2. Mutual Information
            if 'mutual_info' in methods:
                selector = SelectKBest(score_func=mutual_info_classif, k='all')
                selector.fit(X, y)
                scores = pd.Series(selector.scores_, index=X.columns)
                importance_scores['mutual_info'] = scores.sort_values(ascending=False)
            
            # 3. Random Forest (enable parallel processing)
            if 'rf' in methods:
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X, y)
                scores = pd.Series(rf.feature_importances_, index=X.columns)
                importance_scores['rf'] = scores.sort_values(ascending=False)
            
            return importance_scores
            
        except Exception as e:
            logging.error(f"Error calculating feature importance: {str(e)}")
            raise 