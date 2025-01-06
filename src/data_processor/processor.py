import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from .feature_engineering import FeatureEngineer
from .feature_selector import FeatureSelector
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
from ..utils.file_utils import ensure_dir
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from sklearn.ensemble import RandomForestClassifier

class DataProcessor:
    """Data Processor"""
    def __init__(self, config=None):
        self.config = config
        self.data_dir = Path('data') if config is None else config.DATA_DIR
        self.raw_data = None
        self.processed_data = None
        self.feature_names = None
        self.target = None
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer()
        self.feature_selector = FeatureSelector()
        
        # Add date attributes
        self.train_end_date = None
        self.val_end_date = None
        self.test_end_date = None
        
        # Get parameters from config
        if config:
            self.params = config.get_data_processing_params()
        else:
            self.params = {}
        
        # Ensure necessary directories exist
        self.processed_dir = self.data_dir / 'processed'
        ensure_dir(self.processed_dir)
        
    def process_data(self, data_path: Path = None, output_path: Path = None) -> Dict:
        """
        Process data and return a dictionary containing processed data
        
        Args:
            data_path: Path to raw data
            output_path: Path to save processed data
            
        Returns:
            Dict: Dictionary containing processed features and labels
        """
        try:
            # 1. Load raw data
            df = self.load_raw_data(data_path)
            
            # 2. Data validation and quality check
            if not self.validate_data(df):
                raise ValueError("Data validation failed")
                
            quality_report = self.check_data_quality(df)
            logging.info("Data quality report:")
            for key, value in quality_report.items():
                logging.info(f"{key}: {value}")
            
            # 3. Generate features
            df = self.generate_features()
            
            # 4. Generate labels
            self.generate_labels(
                horizon=self.params.get('label_horizon', 4),
                threshold=self.params.get('label_threshold', 0.001)
            )
            
            # 5. Feature analysis
            feature_analysis = self.analyze_features()
            
            # 6. Feature selection
            if self.feature_selector:
                selected_features = self.select_features(
                    method=self.params.get('feature_selection_method', 'f_score'),
                    n_features=self.params.get('n_features', 30)
                )
                logging.info(f"Selected {len(selected_features)} features")
            
            # Prepare data to save and return
            processed_data = {
                'features': self.processed_data,
                'labels': self.target,
                'feature_names': self.feature_names,
                'scaler': None  # No standardization yet
            }
            
            # Save processed data
            if output_path:
                self.save_processed_data(processed_data)
                
            return processed_data
            
        except Exception as e:
            logging.error(f"Error in data processing: {str(e)}")
            raise
            
    def load_raw_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load raw data"""
        try:
            if file_path is None:
                file_path = self.config.RAW_DATA_PATH
                
            if not Path(file_path).exists():
                # Check other possible locations
                alt_locations = [
                    Path('btcusdt_1h.csv'),
                    Path('data/btcusdt_1h.csv'),
                    Path('data/raw/btcusdt_1h.csv')
                ]
                
                for loc in alt_locations:
                    if loc.exists():
                        file_path = loc
                        logging.info(f"Found data file at alternative location: {loc}")
                        break
                else:
                    raise FileNotFoundError(
                        f"Data file not found in any of these locations: {[str(p) for p in alt_locations]}"
                    )
                    
            logging.info(f"Loading data from {file_path}")
            
            # First read data to check column names
            df = pd.read_csv(file_path)
            logging.info(f"Columns in data: {df.columns.tolist()}")
            
            # Find timestamp column
            time_columns = [col for col in df.columns if any(x in col.lower() for x in ['time', 'date'])]
            if time_columns:
                time_col = time_columns[0]
                # Re-read data with date parsing
                df = pd.read_csv(file_path, parse_dates=[time_col])
                df = df.rename(columns={time_col: 'timestamp'})
                
            # Set time index
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
                logging.info(f"Index type after setting timestamp: {df.index.dtype}")
                
            # Keep only necessary numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            if 'turnover' in df.columns:
                numeric_columns.append('turnover')
                
            df = df[numeric_columns]
            
            # Check data quality
            for col in numeric_columns:
                if df[col].nunique() <= 1:
                    logging.warning(f"Column {col} has only {df[col].nunique()} unique values")
                    
            self.raw_data = df
            logging.info(f"Raw data loaded, shape: {df.shape}")
            
            return df
            
        except Exception as e:
            logging.error(f"Error loading raw data: {str(e)}")
            raise
            
    def save_processed_data(self, data: Dict) -> None:
        """Save processed data to CSV format"""
        try:
            # Ensure directory exists
            ensure_dir(self.processed_dir)
            
            # Save feature data
            features_path = self.processed_dir / 'features.csv'
            data['features'].to_csv(features_path)
            
            # Save label data
            labels_path = self.processed_dir / 'labels.csv'
            pd.Series(data['labels'], name='target').to_csv(labels_path)
            
            logging.info(f"Processed data saved to {self.processed_dir}")
            
        except Exception as e:
            logging.error(f"Error saving processed data: {str(e)}")
            raise

    def load_processed_data(self) -> Dict:
        """Load processed data from CSV files"""
        try:
            features_path = self.processed_dir / 'features.csv'
            labels_path = self.processed_dir / 'labels.csv'
            
            if not features_path.exists():
                logging.info(f"Features file not found at: {features_path}")
                raise FileNotFoundError("Features file not found")
            
            if not labels_path.exists():
                logging.info(f"Labels file not found at: {labels_path}")
                raise FileNotFoundError("Labels file not found")
            
            # Load features and labels
            try:
                features = pd.read_csv(features_path, index_col=0)
                labels = pd.read_csv(labels_path, index_col=0).iloc[:, 0]  # Get first column as labels
            except Exception as e:
                logging.error(f"Error reading CSV files: {e}")
                raise
            
            data = {
                'features': features,
                'labels': labels
            }
            
            logging.info(f"Successfully loaded processed data:")
            logging.info(f"- Features shape: {features.shape}")
            logging.info(f"- Labels shape: {labels.shape}")
            return data
            
        except FileNotFoundError as e:
            logging.warning("No processed data found. Need to process raw data first.")
            raise
        except Exception as e:
            logging.error(f"Error loading processed data: {str(e)}")
            raise
            
    def prepare_datasets(self, features: pd.DataFrame, labels: pd.Series, 
                        test_size: float = 0.2, val_size: float = 0.1) -> Dict:
        """Prepare training, validation and test datasets, including standardization"""
        try:
            # First split out test set
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                features, labels,
                test_size=test_size,
                shuffle=False  # Don't shuffle time series data
            )
            
            # Split validation set from remaining data
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=val_size_adjusted,
                shuffle=False
            )
            
            # Standardize training set and transform validation and test sets
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # Convert back to DataFrame to keep index and column names
            X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
            X_val_scaled = pd.DataFrame(X_val_scaled, index=X_val.index, columns=X_val.columns)
            X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
            
            # Record split dates
            self.train_end_date = X_train.index[-1]
            self.val_end_date = X_val.index[-1]
            self.test_end_date = X_test.index[-1]
            
            logging.info(f"Training set: {X_train_scaled.shape} (end: {self.train_end_date})")
            logging.info(f"Validation set: {X_val_scaled.shape} (end: {self.val_end_date})")
            logging.info(f"Test set: {X_test_scaled.shape} (end: {self.test_end_date})")
            
            return {
                'train': (X_train_scaled, y_train),
                'val': (X_val_scaled, y_val),
                'test': (X_test_scaled, y_test),
                'scaler': scaler  # Save scaler for future use
            }
            
        except Exception as e:
            logging.error(f"Error preparing datasets: {str(e)}")
            raise

    def generate_features(self) -> pd.DataFrame:
        """Generate features"""
        try:
            if self.raw_data is None:
                raise ValueError("Raw data not loaded")
                
            df = self.raw_data.copy()
            
            # Use FeatureEngineer to generate features
            df = self.feature_engineer.add_technical_indicators(df)
            df = self.feature_engineer.add_time_features(df)
            
            # Add other features
            price_cols = ['close', 'high', 'low', 'volume']
            df = self.feature_engineer.add_lagged_features(df, price_cols, [1, 2, 3])
            df = self.feature_engineer.add_rolling_features(df, price_cols, [5, 10, 20])
            
            # Handle missing values
            df = self.handle_missing_values(df)
            
            # Save processed data and feature names
            self.processed_data = df
            self.feature_names = df.columns.tolist()
            
            logging.info(f"Features generated, shape: {df.shape}")
            return df
            
        except Exception as e:
            logging.error(f"Error generating features: {str(e)}")
            raise
            
    def generate_labels(self, horizon: int = 1, threshold: float = 0.0) -> pd.Series:
        """
        Generate labels
        
        Args:
            horizon: Prediction period (hours)
            threshold: Upside/downside threshold
            
        Returns:
            pd.Series: Label series
        """
        try:
            if self.processed_data is None:
                raise ValueError("Processed data not available")
                
            # Calculate future returns
            future_returns = self.processed_data['close'].pct_change(horizon).shift(-horizon)
            
            # Generate labels
            labels = (future_returns > threshold).astype(int)
            
            # Remove NaN values caused by future data
            valid_index = labels.notna()
            labels = labels[valid_index]
            self.processed_data = self.processed_data[valid_index]  # Synchronize feature data
            
            # Save labels
            self.target = labels
            
            # Record label distribution
            label_dist = labels.value_counts(normalize=True)
            logging.info(f"Labels generated:")
            logging.info(f"- Positive ratio: {labels.mean():.3f}")
            logging.info(f"- Label distribution:\n{label_dist}")
            logging.info(f"- Features shape after label alignment: {self.processed_data.shape}")
            logging.info(f"- Labels shape: {labels.shape}")
            
            return labels
            
        except Exception as e:
            logging.error(f"Error generating labels: {str(e)}")
            raise
            
    def analyze_features(self) -> Dict:
        """Analyze features (simplified version)"""
        try:
            if self.processed_data is None:
                raise ValueError("Processed data not available")
            
            analysis = {}
            
            # 1. Calculate basic statistics
            stats_dict = {}
            for col in self.processed_data.columns:
                series = self.processed_data[col]
                stats_dict[col] = {
                    'mean': series.mean(),
                    'std': series.std(),
                    'skew': series.skew(),
                    'kurt': series.kurtosis(),
                    'missing': series.isnull().sum(),
                    'missing_pct': series.isnull().mean() * 100
                }
            analysis['column_stats'] = stats_dict
            
            # 2. Calculate correlation
            correlation = self.processed_data.corr()
            analysis['correlation'] = correlation
            
            # 3. Calculate feature importance (if target variable exists)
            if self.target is not None:
                rf = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1  # Only used in parallel within random forest
                )
                
                # Process data
                X = self.processed_data.copy()
                X = X.replace([np.inf, -np.inf], np.nan)
                X = X.fillna(method='ffill').fillna(method='bfill')
                
                # Calculate feature importance
                rf.fit(X, self.target)
                importance_scores = pd.Series(
                    rf.feature_importances_,
                    index=X.columns,
                    name='importance'
                ).sort_values(ascending=False)
                
                analysis['feature_importance'] = importance_scores
                
                # Record top 10 most important features
                top_features = importance_scores.head(10)
                logging.info("Top 10 important features:")
                for feat, imp in top_features.items():
                    logging.info(f"{feat}: {imp:.4f}")
            
            logging.info("Feature analysis completed")
            return analysis
            
        except Exception as e:
            logging.error(f"Error analyzing features: {str(e)}")
            raise

    def _generate_feature_analysis_report(self, analysis: Dict) -> None:
        """Generate feature analysis report"""
        try:
            report_path = self.config.REPORTS_DIR / 'feature_analysis_report.txt'
            with open(report_path, 'w') as f:
                # 1. Basic statistics
                f.write("=== Basic Statistics ===\n")
                f.write(str(analysis['basic_stats']))
                f.write("\n\n")
                
                # 2. Missing values information
                f.write("=== Missing Values ===\n")
                for feature, count in analysis['missing_values']['count'].items():
                    f.write(f"{feature}: {count} ({analysis['missing_values']['percentage'][feature]:.2f}%)\n")
                f.write("\n")
                
                # 3. Highly correlated features
                f.write("=== Highly Correlated Features ===\n")
                for pair in analysis['correlation']['high_correlation_pairs']:
                    f.write(f"{pair['features'][0]} - {pair['features'][1]}: {pair['correlation']:.3f}\n")
                f.write("\n")
                
                # 4. Target variable correlation
                if 'target_correlation' in analysis:
                    f.write("=== Target Correlation (Top 10) ===\n")
                    for feature, corr in analysis['target_correlation'].head(10).items():
                        f.write(f"{feature}: {corr:.3f}\n")
                    f.write("\n")
                
                # 5. Feature importance
                if 'feature_importance' in analysis:
                    f.write("=== Feature Importance (Top 10 by F-score) ===\n")
                    for feature, score in analysis['feature_importance']['f_score'].head(10).items():
                        f.write(f"{feature}: {score:.3f}\n")
                    f.write("\n")
            
            logging.info(f"Feature analysis report generated at {report_path}")
            
        except Exception as e:
            logging.error(f"Error generating feature analysis report: {str(e)}")
            raise
            
    def select_features(self, method: str = 'f_score', n_features: int = 20) -> List[str]:
        """
        Feature selection
        
        Args:
            method: Feature selection method ('correlation', 'f_score', 'mutual_info', 'rf', 'rfe')
            n_features: Number of features to select
            
        Returns:
            List[str]: List of selected feature names
        """
        try:
            if self.processed_data is None or self.target is None:
                raise ValueError("Data or labels not available")
                
            if method == 'correlation':
                selected_features = self.feature_selector.select_by_correlation(
                    self.processed_data
                )
            elif method in ['f_score', 'mutual_info', 'rf']:
                selected_features = self.feature_selector.select_by_importance(
                    self.processed_data,
                    self.target,
                    n_features,
                    method
                )
            elif method == 'rfe':
                selected_features = self.feature_selector.select_by_rfe(
                    self.processed_data,
                    self.target,
                    n_features
                )
            else:
                raise ValueError(f"Unknown feature selection method: {method}")
                
            # Update data
            self.processed_data = self.processed_data[selected_features]
            self.feature_names = selected_features
            
            logging.info(f"Selected {len(selected_features)} features using {method}")
            logging.info(f"Top features: {selected_features[:5]}")
            
            return selected_features
            
        except Exception as e:
            logging.error(f"Error selecting features: {str(e)}")
            raise
            
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data quality"""
        quality_report = {}
        
        try:
            # Check missing values
            missing = df.isnull().sum()
            quality_report['missing_values'] = missing[missing > 0].to_dict()
            
            # Check outliers (using 3 standard deviation rule)
            outliers = {}
            for col in df.select_dtypes(include=[np.number]).columns:
                mean, std = df[col].mean(), df[col].std()
                outlier_mask = (df[col] < mean - 3*std) | (df[col] > mean + 3*std)
                outlier_count = outlier_mask.sum()
                if outlier_count > 0:
                    outliers[col] = outlier_count
            quality_report['outliers'] = outliers
            
            # Check data types
            quality_report['dtypes'] = df.dtypes.to_dict()
            
            # Check duplicate rows
            duplicates = df.duplicated().sum()
            quality_report['duplicate_rows'] = duplicates
            
            # Check price anomalies (e.g., negative values)
            price_cols = ['open', 'high', 'low', 'close']
            invalid_prices = {col: (df[col] <= 0).sum() for col in price_cols}
            quality_report['invalid_prices'] = invalid_prices
            
            logging.info("Data quality check completed")
            return quality_report
            
        except Exception as e:
            logging.error(f"Error in data quality check: {str(e)}")
            raise 

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data validity"""
        try:
            # Validate required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                missing_cols = [col for col in required_columns if col not in df.columns]
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Validate price relationship
            price_errors = (df['high'] < df['low']).sum()
            if price_errors > 0:
                logging.warning(f"Found {price_errors} rows where high < low")
                
            # Validate time series continuity
            time_diff = df.index.to_series().diff()
            expected_diff = pd.Timedelta(hours=1)
            gaps = (time_diff != expected_diff).sum()
            if gaps > 1:  # Allow one difference (considering first difference is NaN)
                logging.warning(f"Found {gaps-1} gaps in time series")
                
            return True
            
        except Exception as e:
            logging.error(f"Data validation failed: {str(e)}")
            return False 

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values - directly remove rows containing NaN"""
        try:
            # 1. Record original data shape
            original_shape = df.shape
            
            # 2. Replace infinity with NaN
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # 3. Record missing values
            missing_stats = df.isnull().sum()
            missing_cols = missing_stats[missing_stats > 0]
            if len(missing_cols) > 0:
                logging.info("Missing values before deletion:")
                for col, count in missing_cols.items():
                    logging.info(f"{col}: {count} ({count/len(df):.2%})")
            
            # 4. Directly remove rows containing NaN
            df_cleaned = df.dropna()
            
            # 5. Record removed rows
            rows_removed = len(df) - len(df_cleaned)
            if rows_removed > 0:
                logging.warning(f"Removed {rows_removed} rows ({rows_removed/len(df):.2%}) containing NaN values")
                logging.info(f"Data shape changed from {original_shape} to {df_cleaned.shape}")
            else:
                logging.info("No rows were removed - no missing values found")
            
            return df_cleaned
            
        except Exception as e:
            logging.error(f"Error handling missing values: {str(e)}")
            raise 

    def process_raw_data(self) -> Dict:
        """Process raw data"""
        try:
            logging.info("Loading and processing raw data...")
            
            # Load raw data
            raw_data = pd.read_csv(self.config.RAW_DATA_PATH)
            logging.info(f"Loaded raw data with shape: {raw_data.shape}")
            
            # Process data
            processed_features = self.preprocess_features(raw_data)
            processed_labels = self.preprocess_labels(raw_data)
            
            # Feature analysis
            feature_analysis = self.analyze_features(processed_features)
            
            processed_data = {
                'features': processed_features,
                'labels': processed_labels,
                'feature_analysis': feature_analysis
            }
            
            logging.info(f"Successfully processed data:")
            logging.info(f"- Features shape: {processed_features.shape}")
            logging.info(f"- Labels shape: {processed_labels.shape}")
            
            return processed_data
            
        except Exception as e:
            logging.error(f"Error processing raw data: {str(e)}")
            raise 