import yaml
from pathlib import Path
import logging

class Config:
    """Configuration Class"""
    def __init__(self, config_path: str = None):
        # Project root directory
        self.ROOT_DIR = Path(__file__).parent.parent.parent
        
        # Data directories
        self.DATA_DIR = self.ROOT_DIR / 'data'
        self.RAW_DATA_PATH = self.DATA_DIR / 'raw' / 'btcusdt_1h.csv'
        self.PROCESSED_DATA_PATH = self.DATA_DIR / 'processed' / 'processed_data.pkl'
        
        # Model directory
        self.MODELS_DIR = self.ROOT_DIR / 'models'
        
        # Results directory
        self.RESULTS_DIR = self.ROOT_DIR / 'results'
        self.FIGURES_DIR = self.RESULTS_DIR / 'figures'
        self.REPORTS_DIR = self.RESULTS_DIR / 'reports'
        
        # Logs directory
        self.LOGS_DIR = self.ROOT_DIR / 'logs'
        
        # Create necessary directories
        self._create_directories()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logging.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logging.error(f"Error loading configuration: {str(e)}")
            raise
            
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.DATA_DIR / 'raw',
            self.DATA_DIR / 'processed',
            self.MODELS_DIR,
            self.FIGURES_DIR,
            self.REPORTS_DIR,
            self.LOGS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logging.info("Directory structure created successfully")
            
    def get_model_params(self):
        return {
            'LightGBM': {
                'num_leaves': 31,
                'learning_rate': 0.05,
                'n_estimators': 100
            },
            'XGBoost': {
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 100
            },
            'RandomForest': {
                'n_estimators': 100,
                'max_depth': 6
            }
        }
        
    def get_ensemble_params(self, ensemble_type):
        if ensemble_type == 'blending':
            return {
                'weights': None  # 使用等权重
            }
        return {}
        
    def get_backtest_params(self) -> dict:
        """获取回测参数"""
        return self.config['backtest']
        
    def get_visualization_params(self) -> dict:
        """获取可视化参数"""
        return self.config['visualization']
        
    def get_data_processing_params(self) -> dict:
        """获取数据处理参数"""
        return {
            # 数据清洗参数
            'remove_outliers': True,
            'outlier_std_threshold': 3,
            'fill_missing_method': 'ffill',
            
            # 特征工程参数
            'technical_indicators': {
                'basic': True,      # SMA, EMA, RSI 等基础指标
                'advanced': True,   # MACDEXT, STOCHRSI 等高级指标
                'experimental': True # RSRS, CMF 等实验性指标
            },
            'volatility_features': True,
            'time_features': True,
            'lagged_features': True,
            'lag_periods': [1, 2, 3],
            'rolling_windows': [5, 10, 20],
            
            # 标签生成参数
            'label_horizon': 4,
            'label_threshold': 0.001,
            
            # 特征选择参数
            'feature_selection_method': 'f_score',
            'n_features': 30,
            
            # 数据集划分参数
            'test_size': 0.2,
            'validation_size': 0.1
        }
        
    def get_feature_selection_params(self) -> dict:
        """获取特征选择参数"""
        return self.config['data_processing']['feature_selection']
        
    def get_feature_engineering_params(self) -> dict:
        """获取特征工程参数"""
        return self.config['data_processing']['feature_engineering']
        
    def get_sampling_params(self) -> dict:
        """获取采样参数"""
        return self.config['data_processing']['sampling']
        
class DataProcessingConfig:
    """Data Processing Configuration"""
    def __init__(self):
        # Data cleaning configuration
        self.REMOVE_OUTLIERS = True
        self.OUTLIER_STD_THRESHOLD = 3
        self.FILL_MISSING_METHOD = 'ffill'
        
        # Feature engineering configuration
        self.TECHNICAL_INDICATORS = {
            'basic': True,      # Basic indicators: SMA, EMA, RSI
            'advanced': True,   # Advanced indicators: MACDEXT, STOCHRSI
            'experimental': True # Experimental indicators: RSRS, CMF
        }
        self.VOLATILITY_FEATURES = True
        self.TIME_FEATURES = True
        self.LAGGED_FEATURES = True
        self.LAG_PERIODS = [1, 2, 3]
        self.ROLLING_WINDOWS = [5, 10, 20]
        
        # Label generation configuration
        self.LABEL_HORIZON = 4
        self.LABEL_THRESHOLD = 0.01
        
        # Feature selection configuration
        self.FEATURE_SELECTION_METHOD = 'f_score'
        self.N_FEATURES = 30
        
        # Dataset split configuration
        self.TEST_SIZE = 0.2
        self.VALIDATION_SIZE = 0.1
        
        # Missing value handling configuration
        self.MISSING_VALUE_HANDLING = {
            'method': 'drop',  # Only use deletion method
            'log_details': True  # Whether to log detailed deletion information
        } 