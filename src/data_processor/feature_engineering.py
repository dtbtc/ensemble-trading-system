import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import talib as ta
import logging
from sklearn import linear_model
from concurrent.futures import ThreadPoolExecutor
from functools import partial

class FeatureEngineer:
    """Feature Engineering Class"""
    
    def __init__(self, config=None):
        self.config = config
        self.params = config.get_feature_engineering_params() if config else {}
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        try:
            # Get base data
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            
            # Create feature dictionary
            features = {}
            
            # Helper function: Add feature and handle NaN
            def process_feature(values: np.ndarray, fill_method: str = 'ffill') -> pd.Series:
                series = pd.Series(values, index=df.index)
                if fill_method == 'ffill':
                    series = series.fillna(method='ffill').fillna(method='bfill')
                elif fill_method == 'bfill':
                    series = series.fillna(method='bfill').fillna(method='ffill')
                elif fill_method == 'zero':
                    series = series.fillna(0)
                return series
            
            # 1. Moving average line family
            for period in [5, 10, 20, 30, 50, 100]:
                features[f'SMA_{period}'] = process_feature(ta.SMA(close.values, timeperiod=period))
                features[f'EMA_{period}'] = process_feature(ta.EMA(close.values, timeperiod=period))
                features[f'WMA_{period}'] = process_feature(ta.WMA(close.values, timeperiod=period))
                features[f'DEMA_{period}'] = process_feature(ta.DEMA(close.values, timeperiod=period))
                features[f'TEMA_{period}'] = process_feature(ta.TEMA(close.values, timeperiod=period))
                features[f'TRIMA_{period}'] = process_feature(ta.TRIMA(close.values, timeperiod=period))
                features[f'KAMA_{period}'] = process_feature(ta.KAMA(close.values, timeperiod=period))
            
            # 2. MACD family
            macd, macdsignal, macdhist = ta.MACD(close.values)
            features.update({
                'MACD': process_feature(macd),
                'MACD_Signal': process_feature(macdsignal),
                'MACD_Hist': process_feature(macdhist)
            })
            
            # MACD extension
            macd, macdsignal, macdhist = ta.MACDEXT(close.values)
            features['MACDEXT'] = process_feature(macd)
            features['MACDEXT_Signal'] = process_feature(macdsignal)
            features['MACDEXT_Hist'] = process_feature(macdhist)
            
            # MACD Fix
            macd, macdsignal, macdhist = ta.MACDFIX(close.values)
            features['MACDFIX'] = process_feature(macd)
            features['MACDFIX_Signal'] = process_feature(macdsignal)
            features['MACDFIX_Hist'] = process_feature(macdhist)
            
            # 3. RSI family
            features['RSI_14'] = process_feature(ta.RSI(close.values, timeperiod=14))
            
            # StochRSI
            fastk, fastd = ta.STOCHRSI(close.values)
            features['STOCHRSI'] = process_feature(fastk)
            features['STOCHRSI_Signal'] = process_feature(fastd)
            
            # 4. Momentum indicator family
            features['MOM'] = process_feature(ta.MOM(close.values, timeperiod=10))
            features['ROC_10'] = process_feature(ta.ROC(close.values, timeperiod=10))
            features['ROC_20'] = process_feature(ta.ROC(close.values, timeperiod=20))
            features['ROCP'] = process_feature(ta.ROCP(close.values))
            features['ROCR'] = process_feature(ta.ROCR(close.values))
            features['ROCR100'] = process_feature(ta.ROCR100(close.values))
            
            # 5. Bollinger Bands and Channels
            upperband, middleband, lowerband = ta.BBANDS(close.values)
            features['BB_upper'] = process_feature(upperband)
            features['BB_middle'] = process_feature(middleband)
            features['BB_lower'] = process_feature(lowerband)
            
            # 6. Trend indicators
            features['ADX'] = process_feature(ta.ADX(high.values, low.values, close.values))
            features['ADXR'] = process_feature(ta.ADXR(high.values, low.values, close.values))
            features['APO'] = process_feature(ta.APO(close.values))
            features['MINUS_DM'] = process_feature(ta.MINUS_DM(high.values, low.values))
            features['PLUS_DM'] = process_feature(ta.PLUS_DM(high.values, low.values))
            features['DX'] = process_feature(ta.DX(high.values, low.values, close.values))
            
            # 7. Volatility indicators
            features['ATR_14'] = process_feature(ta.ATR(high.values, low.values, close.values, timeperiod=14))
            features['NATR'] = process_feature(ta.NATR(high.values, low.values, close.values))
            features['TRANGE'] = process_feature(ta.TRANGE(high.values, low.values, close.values))
            
            # 8. Volume indicators
            features['OBV'] = process_feature(ta.OBV(close.values, volume.values))
            features['AD'] = process_feature(ta.AD(high.values, low.values, close.values, volume.values))
            
            # 9. Other oscillators
            features['CCI_14'] = process_feature(ta.CCI(high.values, low.values, close.values, timeperiod=14))
            features['CMO'] = process_feature(ta.CMO(close.values))
            features['PPO'] = process_feature(ta.PPO(close.values))
            features['ULTOSC'] = process_feature(ta.ULTOSC(high.values, low.values, close.values))
            
            # 10. Stochastic indicators
            slowk, slowd = ta.STOCH(high.values, low.values, close.values)
            features['STOCH_slowk'] = process_feature(slowk)
            features['STOCH_slowd'] = process_feature(slowd)
            
            fastk, fastd = ta.STOCHF(high.values, low.values, close.values)
            features['STOCHF_fastk'] = process_feature(fastk)
            features['STOCHF_fastd'] = process_feature(fastd)
            
            # 11. Aroon indicators
            aroon_down, aroon_up = ta.AROON(high.values, low.values)
            features['AROON_down'] = process_feature(aroon_down)
            features['AROON_up'] = process_feature(aroon_up)
            features['AROONOSC'] = process_feature(ta.AROONOSC(high.values, low.values))
            
            # 12. Adaptive Moving Average
            mama, fama = ta.MAMA(close.values)
            features['MAMA'] = process_feature(mama)
            features['FAMA'] = process_feature(fama)
            
            # 13. Other price indicators
            features['MIDPOINT'] = process_feature(ta.MIDPOINT(close.values))
            features['MIDPRICE'] = process_feature(ta.MIDPRICE(high.values, low.values))
            features['SAR'] = process_feature(ta.SAR(high.values, low.values))
            features['SAREXT'] = process_feature(ta.SAREXT(high.values, low.values))
            features['T3'] = process_feature(ta.T3(close.values))
            
            # 14. Money Flow Index
            features['MFI_14'] = process_feature(ta.MFI(high.values, low.values, close.values, volume.values, timeperiod=14))
            
            # 15. VWAP
            df['VWAP'] = (volume * close).cumsum() / volume.cumsum()
            
            # 16. Williams %R
            features['Williams_%R'] = process_feature(ta.WILLR(high.values, low.values, close.values))
            
            # 17. TRIX
            features['TRIX'] = process_feature(ta.TRIX(close.values))
            
            # Add all features at once
            result_df = pd.concat([df] + list(features.values()), axis=1)
            result_df.columns = list(df.columns) + list(features.keys())
            
            # Validate generated features
            self._validate_features(result_df)
            
            logging.info(f"Added technical indicators, new shape: {result_df.shape}")
            return result_df
            
        except Exception as e:
            logging.error(f"Error adding technical indicators: {str(e)}")
            raise
            
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        try:
            if not self.params.get('time_features', True):
                return df
                
            # Create feature dictionary
            features = {
                'hour': df.index.hour,
                'day_of_week': df.index.dayofweek,
                'day_of_month': df.index.day,
                'week_of_year': df.index.isocalendar().week,
                'month': df.index.month,
                'quarter': df.index.quarter
            }
            
            # Add periodic features
            features.update({
                'hour_sin': np.sin(2 * np.pi * features['hour']/24),
                'hour_cos': np.cos(2 * np.pi * features['hour']/24),
                'day_sin': np.sin(2 * np.pi * features['day_of_week']/7),
                'day_cos': np.cos(2 * np.pi * features['day_of_week']/7)
            })
            
            # Add all features at once
            result_df = pd.concat([df] + [pd.Series(v, index=df.index, name=k) 
                                        for k, v in features.items()], axis=1)
            
            logging.info(f"Added time features, new shape: {result_df.shape}")
            return result_df
            
        except Exception as e:
            logging.error(f"Error adding time features: {str(e)}")
            raise
            
    def add_lagged_features(self, df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """Add lagged features"""
        try:
            if not self.params.get('lagged_features', True):
                return df
                
            # Create feature dictionary
            features = {}
            for col in columns:
                for lag in lags:
                    features[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
            # Add all features at once
            result_df = pd.concat([df] + [pd.Series(v, index=df.index, name=k) 
                                        for k, v in features.items()], axis=1)
            
            # Fill NaN values
            result_df = result_df.fillna(method='ffill').fillna(method='bfill')
            
            logging.info(f"Added lagged features, new shape: {result_df.shape}")
            return result_df
            
        except Exception as e:
            logging.error(f"Error adding lagged features: {str(e)}")
            raise
            
    def add_rolling_features(self, df: pd.DataFrame,
                           columns: List[str],
                           windows: List[int]) -> pd.DataFrame:
        """
        Add rolling features
        
        Args:
            df: Original DataFrame
            columns: Columns that need rolling features
            windows: List of window sizes
            
        Returns:
            pd.DataFrame: DataFrame with added rolling features
        """
        try:
            # Create a dictionary to store all features
            features = {}
            
            for col in columns:
                for window in windows:
                    features[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
                    features[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
                    features[f'{col}_rolling_min_{window}'] = df[col].rolling(window).min()
                    features[f'{col}_rolling_max_{window}'] = df[col].rolling(window).max()
            
            # Add all features at once
            new_df = pd.concat([df] + list(features.values()), axis=1)
            new_df.columns = list(df.columns) + list(features.keys())
            
            # Fill NaN values
            new_df = new_df.fillna(method='ffill').fillna(method='bfill')
            
            logging.info(f"Added rolling features, new shape: {new_df.shape}")
            return new_df
            
        except Exception as e:
            logging.error(f"Error adding rolling features: {str(e)}")
            raise
            
    def _validate_features(self, df: pd.DataFrame) -> None:
        """Validate generated features"""
        try:
            # Check for infinite values
            inf_cols = df.columns[np.isinf(df).any()].tolist()
            if inf_cols:
                logging.warning(f"Infinite values found in columns: {inf_cols}")
            
            # Check for NaN values
            nan_cols = df.columns[df.isna().any()].tolist()
            if nan_cols:
                logging.warning(f"NaN values found in columns: {nan_cols}")
            
            # Check for constant columns
            constant_cols = df.columns[df.nunique() <= 1].tolist()
            if constant_cols:
                logging.warning(f"Constant columns found: {constant_cols}")
                
        except Exception as e:
            logging.error(f"Error validating features: {str(e)}")
            raise 