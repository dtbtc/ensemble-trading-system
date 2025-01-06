"""
Blending Ensemble Project
------------------------
Research on cryptocurrency trading strategies based on ensemble learning.
"""

from .utils.config import Config
from .utils.logger import setup_logger
from .data_processor import DataProcessor
from .models.base_models import BaseModel
from .models.ensemble_models import StackingClassifier, BlendingEnsemble
from .models.model_factory import ModelFactory
from .evaluation.evaluator import ModelEvaluator
from .visualization.visualizer import ResultVisualizer
from .backtest.backtest import BacktestEngine
from .strategy.risk_manager import RiskManager

__all__ = [
    # Utility classes
    'Config',
    'setup_logger',
    
    # Data processing
    'DataProcessor',
    
    # Base models
    'BaseModel',
    'DecisionTreeModel',
    'RandomForestModel',
    'XGBoostModel',
    'LightGBMModel',
    'LogisticRegressionModel',
    'KNNModel',
    'NaiveBayesModel',
    
    # Ensemble models
    'StackingClassifier',
    'BlendingEnsemble',
    
    # Evaluation and visualization
    'ModelEvaluator',
    'ResultVisualizer',
    
    # Backtesting and risk management
    'BacktestEngine',
    'RiskManager',
    
    # Model factory
    'ModelFactory'
]

__version__ = '0.1.0'