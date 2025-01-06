import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

class BacktestEngine:
    """Backtesting Engine"""
    def __init__(self, config=None):
        self.config = config
        self.results_dir = Path('results/backtest') if config is None else config.RESULTS_DIR / 'backtest'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    def run_backtest(self, model, data, initial_capital=10000, commission=0.001):
        """
        Run backtest
        
        Args:
            model: Prediction model
            data: Test data
            initial_capital: Initial capital
            commission: Trading commission rate
        """
        try:
            # Generate trading signals
            signals = self._generate_signals(model, data)
            
            # Calculate daily returns
            returns = self._calculate_returns(signals, commission)
            
            # Calculate strategy performance
            performance = self._evaluate_strategy(returns, initial_capital)
            
            # Save backtest results
            self.results = {
                'signals': signals,
                'returns': returns,
                'performance': performance
            }
            
            # Generate backtest report
            self._generate_report()
            
            return performance
            
        except Exception as e:
            logging.error(f"Error in backtest: {str(e)}")
            raise
            
    def _generate_signals(self, model, data):
        """Generate trading signals"""
        try:
            # Get prediction probabilities
            probs = model.predict_proba(data)[:, 1]
            
            # Generate signals
            signals = pd.DataFrame({
                'probability': probs,
                'signal': (probs > 0.5).astype(int)  # Buy signal when probability > 0.5
            })
            
            return signals
            
        except Exception as e:
            logging.error(f"Error generating signals: {str(e)}")
            raise
            
    def _calculate_returns(self, signals, commission):
        """Calculate strategy returns"""
        try:
            # Calculate positions
            positions = signals['signal'].diff()
            
            # Calculate trading costs
            trading_costs = abs(positions) * commission
            
            # Calculate strategy returns
            strategy_returns = signals['signal'] * signals['probability'] - trading_costs
            
            return strategy_returns
            
        except Exception as e:
            logging.error(f"Error calculating returns: {str(e)}")
            raise
            
    def _evaluate_strategy(self, returns, initial_capital):
        """Evaluate strategy performance"""
        try:
            # Calculate cumulative returns
            cumulative_returns = (1 + returns).cumprod()
            
            # Calculate maximum drawdown
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / rolling_max - 1
            max_drawdown = drawdowns.min()
            
            # Calculate other metrics
            total_return = cumulative_returns.iloc[-1] - 1
            annual_return = (1 + total_return) ** (252 / len(returns)) - 1
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
            
            performance = {
                'Total Return': total_return,
                'Annual Return': annual_return,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown': max_drawdown,
                'Final Capital': initial_capital * (1 + total_return)
            }
            
            return performance
            
        except Exception as e:
            logging.error(f"Error evaluating strategy: {str(e)}")
            raise
            
    def _generate_report(self):
        """Generate backtest report"""
        try:
            # Create backtest report charts
            fig, axes = plt.subplots(3, 1, figsize=(12, 15))
            
            # 1. Cumulative returns curve
            cumulative_returns = (1 + self.results['returns']).cumprod()
            cumulative_returns.plot(ax=axes[0])
            axes[0].set_title('Cumulative Returns')
            axes[0].set_xlabel('Time')
            axes[0].set_ylabel('Returns')
            
            # 2. Signal distribution
            sns.histplot(data=self.results['signals']['probability'], ax=axes[1])
            axes[1].set_title('Signal Distribution')
            axes[1].set_xlabel('Probability')
            
            # 3. Drawdown analysis
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / rolling_max - 1
            drawdowns.plot(ax=axes[2])
            axes[2].set_title('Drawdown Analysis')
            axes[2].set_xlabel('Time')
            axes[2].set_ylabel('Drawdown')
            
            # Save charts
            plt.tight_layout()
            save_path = self.results_dir / 'backtest_report.png'
            plt.savefig(save_path)
            plt.close()
            
            # Save performance metrics
            performance_df = pd.DataFrame.from_dict(
                self.results['performance'], 
                orient='index', 
                columns=['Value']
            )
            performance_df.to_csv(self.results_dir / 'performance_metrics.csv')
            
            logging.info(f"Backtest report generated and saved to {self.results_dir}")
            
        except Exception as e:
            logging.error(f"Error generating backtest report: {str(e)}")
            raise
            
    def plot_equity_curve(self):
        """Plot equity curve"""
        try:
            if not self.results:
                raise ValueError("No backtest results available")
                
            plt.figure(figsize=(12, 6))
            cumulative_returns = (1 + self.results['returns']).cumprod()
            cumulative_returns.plot()
            plt.title('Strategy Equity Curve')
            plt.xlabel('Time')
            plt.ylabel('Cumulative Returns')
            
            save_path = self.results_dir / 'equity_curve.png'
            plt.savefig(save_path)
            plt.close()
            
            logging.info(f"Equity curve plot saved to {save_path}")
            
        except Exception as e:
            logging.error(f"Error plotting equity curve: {str(e)}") 