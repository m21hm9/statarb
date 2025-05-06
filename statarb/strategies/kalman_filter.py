import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, List, Union
import logging
from pykalman import KalmanFilter
from ..utils.data_utils import calculate_zscore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KalmanPairsTrader:
    """
    Pairs trading strategy using Kalman Filter for dynamic hedge ratio estimation.
    """
    
    def __init__(
        self,
        lookback_period: int = 60,
        zscore_entry_threshold: float = 2.0,
        zscore_exit_threshold: float = 0.5,
        stop_loss_threshold: float = 3.0,
        max_position_days: int = 30,
        transaction_cost: float = 0.0003,
        delta: float = 1e-5,  # System noise
        varcov: float = 1e-4,  # Observation noise
        initial_state_mean: float = 0.0,
        initial_state_cov: float = 1.0
    ):
        """
        Initialize the Kalman Filter pairs trading strategy.
        
        Args:
            lookback_period: Period for calculating z-scores
            zscore_entry_threshold: Z-score threshold to enter a position
            zscore_exit_threshold: Z-score threshold to exit a position
            stop_loss_threshold: Z-score threshold for stop loss
            max_position_days: Maximum days to hold a position
            transaction_cost: Transaction cost per trade (percentage)
            delta: System noise parameter for Kalman filter
            varcov: Observation noise parameter for Kalman filter
            initial_state_mean: Initial state mean for Kalman filter
            initial_state_cov: Initial state covariance for Kalman filter
        """
        self.lookback_period = lookback_period
        self.zscore_entry_threshold = zscore_entry_threshold
        self.zscore_exit_threshold = zscore_exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.max_position_days = max_position_days
        self.transaction_cost = transaction_cost
        
        # Kalman filter parameters
        self.delta = delta
        self.varcov = varcov
        self.initial_state_mean = initial_state_mean
        self.initial_state_cov = initial_state_cov
        
        # Trading variables
        self.pairs = []
        self.kalman_filters = {}
        self.kalman_states = {}
        self.current_positions = {}
        self.position_history = []
    
    def estimate_dynamic_hedge_ratio(
        self,
        price_data: pd.DataFrame,
        ticker1: str,
        ticker2: str,
        include_intercept: bool = True
    ) -> pd.DataFrame:
        """
        Estimate dynamic hedge ratio using Kalman Filter.
        
        Args:
            price_data: DataFrame of price data
            ticker1: Y variable ticker (dependent)
            ticker2: X variable ticker (independent)
            include_intercept: Whether to include intercept in the model
            
        Returns:
            DataFrame with state means (hedge ratios) and state covariances
        """
        # Prepare data
        y = price_data[ticker1].values
        x = price_data[ticker2].values
        
        if include_intercept:
            # For model: y_t = alpha_t + beta_t * x_t + epsilon_t
            observation_matrices = np.vstack([np.ones(len(x)), x]).T[:, np.newaxis, :]
            transition_matrices = np.eye(2)
            observation_offsets = np.zeros(len(y))
            initial_state_mean = np.array([0.0, 0.0])
            initial_state_cov = np.eye(2)
            transition_covariance = self.delta * np.eye(2)
            observation_covariance = np.array([self.varcov])
            
            # Create and fit the Kalman filter
            kf = KalmanFilter(
                transition_matrices=transition_matrices,
                observation_matrices=observation_matrices,
                initial_state_mean=initial_state_mean,
                initial_state_cov=initial_state_cov,
                transition_covariance=transition_covariance,
                observation_covariance=observation_covariance,
                observation_offsets=observation_offsets
            )
            
            try:
                # Filter the data
                state_means, state_covs = kf.filter(y.reshape(-1, 1))
                
                # Store Kalman filter and states
                self.kalman_filters[(ticker1, ticker2)] = kf
                self.kalman_states[(ticker1, ticker2)] = (state_means, state_covs)
                
                # Create DataFrame with results
                results = pd.DataFrame(index=price_data.index)
                results['alpha'] = state_means[:, 0]
                results['beta'] = state_means[:, 1]
                results['alpha_var'] = state_covs[:, 0, 0]
                results['beta_var'] = state_covs[:, 1, 1]
                results['alpha_beta_cov'] = state_covs[:, 0, 1]
                
                return results
            
            except Exception as e:
                logger.error(f"Error estimating Kalman filter states: {str(e)}")
                return pd.DataFrame(index=price_data.index)
            
        else:
            # For model: y_t = beta_t * x_t + epsilon_t (no intercept)
            observation_matrices = x.reshape(-1, 1, 1)
            transition_matrices = np.eye(1)
            observation_offsets = np.zeros(len(y))
            initial_state_mean = np.array([self.initial_state_mean])
            initial_state_cov = np.array([[self.initial_state_cov]])
            transition_covariance = np.array([[self.delta]])
            observation_covariance = np.array([self.varcov])
            
            # Create and fit the Kalman filter
            kf = KalmanFilter(
                transition_matrices=transition_matrices,
                observation_matrices=observation_matrices,
                initial_state_mean=initial_state_mean,
                initial_state_cov=initial_state_cov,
                transition_covariance=transition_covariance,
                observation_covariance=observation_covariance,
                observation_offsets=observation_offsets
            )
            
            try:
                # Filter the data
                state_means, state_covs = kf.filter(y.reshape(-1, 1))
                
                # Store Kalman filter and states
                self.kalman_filters[(ticker1, ticker2)] = kf
                self.kalman_states[(ticker1, ticker2)] = (state_means, state_covs)
                
                # Create DataFrame with results
                results = pd.DataFrame(index=price_data.index)
                results['beta'] = state_means[:, 0]
                results['beta_var'] = state_covs[:, 0, 0]
                
                return results
            
            except Exception as e:
                logger.error(f"Error estimating Kalman filter states: {str(e)}")
                return pd.DataFrame(index=price_data.index)
    
    def calculate_dynamic_spread(
        self,
        price_data: pd.DataFrame,
        ticker1: str,
        ticker2: str,
        include_intercept: bool = True
    ) -> pd.DataFrame:
        """
        Calculate spread using dynamic hedge ratio from Kalman Filter.
        
        Args:
            price_data: DataFrame of price data
            ticker1: First ticker symbol
            ticker2: Second ticker symbol
            include_intercept: Whether to include intercept in the model
            
        Returns:
            DataFrame with spread and z-score
        """
        # Estimate hedge ratios
        kalman_results = self.estimate_dynamic_hedge_ratio(
            price_data, ticker1, ticker2, include_intercept
        )
        
        if kalman_results.empty:
            logger.warning(f"Could not estimate Kalman filter for {ticker1}-{ticker2}")
            return pd.DataFrame(index=price_data.index)
        
        # Calculate spread
        if include_intercept:
            spread = price_data[ticker1] - (kalman_results['alpha'] + kalman_results['beta'] * price_data[ticker2])
        else:
            spread = price_data[ticker1] - (kalman_results['beta'] * price_data[ticker2])
        
        # Calculate rolling z-score
        zscore = pd.Series(index=spread.index)
        
        for i in range(self.lookback_period, len(spread)):
            window = spread.iloc[i-self.lookback_period:i]
            mean = window.mean()
            std = window.std()
            
            if std > 0:
                zscore.iloc[i] = (spread.iloc[i] - mean) / std
        
        # Create results DataFrame
        results = pd.DataFrame(index=price_data.index)
        results['spread'] = spread
        results['zscore'] = zscore
        
        if include_intercept:
            results['alpha'] = kalman_results['alpha']
            results['beta'] = kalman_results['beta']
        else:
            results['beta'] = kalman_results['beta']
        
        return results
    
    def generate_signals(
        self,
        price_data: pd.DataFrame,
        ticker1: str,
        ticker2: str,
        include_intercept: bool = True
    ) -> pd.DataFrame:
        """
        Generate trading signals using Kalman Filter.
        
        Args:
            price_data: DataFrame of price data
            ticker1: First ticker symbol
            ticker2: Second ticker symbol
            include_intercept: Whether to include intercept in the model
            
        Returns:
            DataFrame with signals
        """
        # Calculate dynamic spread and z-score
        spread_data = self.calculate_dynamic_spread(
            price_data, ticker1, ticker2, include_intercept
        )
        
        if spread_data.empty:
            return pd.DataFrame(index=price_data.index)
        
        # Generate signals
        signals = pd.DataFrame(index=spread_data.index)
        signals['spread'] = spread_data['spread']
        signals['zscore'] = spread_data['zscore']
        
        if include_intercept:
            signals['alpha'] = spread_data['alpha']
            signals['beta'] = spread_data['beta']
        else:
            signals['beta'] = spread_data['beta']
        
        # Entry signals
        signals['long_entry'] = (spread_data['zscore'] < -self.zscore_entry_threshold) & (spread_data['zscore'].shift(1) >= -self.zscore_entry_threshold)
        signals['short_entry'] = (spread_data['zscore'] > self.zscore_entry_threshold) & (spread_data['zscore'].shift(1) <= self.zscore_entry_threshold)
        
        # Exit signals
        signals['long_exit'] = (spread_data['zscore'] > -self.zscore_exit_threshold) | (spread_data['zscore'] > self.stop_loss_threshold)
        signals['short_exit'] = (spread_data['zscore'] < self.zscore_exit_threshold) | (spread_data['zscore'] < -self.stop_loss_threshold)
        
        return signals
    
    def backtest(
        self,
        price_data: pd.DataFrame,
        ticker1: str,
        ticker2: str,
        include_intercept: bool = True,
        initial_capital: float = 100000.0,
        position_size: float = 0.05
    ) -> Dict:
        """
        Backtest the Kalman Filter pairs trading strategy.
        
        Args:
            price_data: DataFrame of price data
            ticker1: First ticker symbol
            ticker2: Second ticker symbol
            include_intercept: Whether to include intercept in the model
            initial_capital: Initial capital for the backtest
            position_size: Position size as a fraction of capital
            
        Returns:
            Dictionary with backtest results
        """
        # Generate signals
        signals = self.generate_signals(
            price_data, ticker1, ticker2, include_intercept
        )
        
        if signals.empty:
            logger.warning(f"Could not generate signals for {ticker1}-{ticker2}")
            return {}
        
        # Initialize backtest variables
        capital = initial_capital
        position = None
        trades = []
        daily_returns = []
        
        # Track portfolio value
        portfolio_values = pd.Series(index=price_data.index, dtype=float)
        portfolio_values.iloc[0] = capital
        
        # For each day in the backtest period (starting after lookback period)
        for i in range(self.lookback_period + 1, len(price_data)):
            day = price_data.index[i]
            prev_day = price_data.index[i-1]
            
            # Calculate daily P&L
            daily_pnl = 0.0
            
            # Check for exit signals if position exists
            if position is not None:
                entry_day = position['entry_day']
                entry_idx = price_data.index.get_loc(entry_day)
                position_days = i - entry_idx
                
                # Get current z-score and hedge ratios
                current_zscore = signals['zscore'].iloc[i]
                
                if include_intercept:
                    current_alpha = signals['alpha'].iloc[i]
                    current_beta = signals['beta'].iloc[i]
                else:
                    current_alpha = 0
                    current_beta = signals['beta'].iloc[i]
                
                # Check if we should exit the position
                should_exit = False
                
                if position['position'] == 'long':
                    # Exit long if z-score reverts back or exceeds stop loss
                    if signals['long_exit'].iloc[i]:
                        should_exit = True
                elif position['position'] == 'short':
                    # Exit short if z-score reverts back or exceeds stop loss
                    if signals['short_exit'].iloc[i]:
                        should_exit = True
                
                # Also exit if position has been held for too long
                if position_days >= self.max_position_days:
                    should_exit = True
                
                # Calculate P&L if exiting
                if should_exit:
                    # Calculate price changes
                    price1_entry = price_data[ticker1].iloc[entry_idx]
                    price2_entry = price_data[ticker2].iloc[entry_idx]
                    price1_exit = price_data[ticker1].iloc[i]
                    price2_exit = price_data[ticker2].iloc[i]
                    
                    # Apply hedge ratio at entry and exit
                    if position['position'] == 'long':
                        # Long spread: Long asset1, Short asset2*hedge_ratio
                        entry_value = price1_entry - (position['alpha'] + position['beta'] * price2_entry)
                        exit_value = price1_exit - (current_alpha + current_beta * price2_exit)
                        pnl = (exit_value - entry_value) / entry_value
                    else:
                        # Short spread: Short asset1, Long asset2*hedge_ratio
                        entry_value = price1_entry - (position['alpha'] + position['beta'] * price2_entry)
                        exit_value = price1_exit - (current_alpha + current_beta * price2_exit)
                        pnl = (entry_value - exit_value) / entry_value
                    
                    # Apply transaction costs
                    pnl -= 2 * self.transaction_cost  # Entry and exit costs
                    
                    # Apply position size
                    position_value = position['capital']
                    pnl_value = position_value * pnl
                    capital += position_value + pnl_value
                    daily_pnl += pnl_value
                    
                    # Record the trade
                    trades.append({
                        'pair': (ticker1, ticker2),
                        'position': position['position'],
                        'entry_date': entry_day,
                        'exit_date': day,
                        'duration': position_days,
                        'entry_zscore': position['entry_zscore'],
                        'exit_zscore': current_zscore,
                        'entry_alpha': position['alpha'] if include_intercept else 0,
                        'entry_beta': position['beta'],
                        'exit_alpha': current_alpha if include_intercept else 0,
                        'exit_beta': current_beta,
                        'pnl': pnl,
                        'pnl_value': pnl_value
                    })
                    
                    # Clear the position
                    position = None
            
            # Check for entry signals if no position exists
            if position is None:
                # Check for entry signals
                if signals['long_entry'].iloc[i]:
                    # Calculate position size
                    position_capital = capital * position_size
                    capital -= position_capital
                    
                    # Store entry parameters
                    if include_intercept:
                        entry_alpha = signals['alpha'].iloc[i]
                        entry_beta = signals['beta'].iloc[i]
                    else:
                        entry_alpha = 0
                        entry_beta = signals['beta'].iloc[i]
                    
                    # Open long position
                    position = {
                        'position': 'long',
                        'entry_day': day,
                        'alpha': entry_alpha,
                        'beta': entry_beta,
                        'entry_zscore': signals['zscore'].iloc[i],
                        'capital': position_capital
                    }
                    
                elif signals['short_entry'].iloc[i]:
                    # Calculate position size
                    position_capital = capital * position_size
                    capital -= position_capital
                    
                    # Store entry parameters
                    if include_intercept:
                        entry_alpha = signals['alpha'].iloc[i]
                        entry_beta = signals['beta'].iloc[i]
                    else:
                        entry_alpha = 0
                        entry_beta = signals['beta'].iloc[i]
                    
                    # Open short position
                    position = {
                        'position': 'short',
                        'entry_day': day,
                        'alpha': entry_alpha,
                        'beta': entry_beta,
                        'entry_zscore': signals['zscore'].iloc[i],
                        'capital': position_capital
                    }
            
            # Record portfolio value
            if position is not None:
                position_value = position['capital']
            else:
                position_value = 0
                
            portfolio_values[day] = capital + position_value
            
            # Calculate daily return
            daily_return = (portfolio_values[day] / portfolio_values[prev_day]) - 1
            daily_returns.append(daily_return)
        
        # Close any remaining position at the end of the backtest
        if position is not None:
            entry_day = position['entry_day']
            entry_idx = price_data.index.get_loc(entry_day)
            position_days = len(price_data) - 1 - entry_idx
            
            # Calculate price changes
            price1_entry = price_data[ticker1].iloc[entry_idx]
            price2_entry = price_data[ticker2].iloc[entry_idx]
            price1_exit = price_data[ticker1].iloc[-1]
            price2_exit = price_data[ticker2].iloc[-1]
            
            # Get final hedge ratios
            if include_intercept:
                final_alpha = signals['alpha'].iloc[-1]
                final_beta = signals['beta'].iloc[-1]
            else:
                final_alpha = 0
                final_beta = signals['beta'].iloc[-1]
            
            # Apply hedge ratio at entry and exit
            if position['position'] == 'long':
                # Long spread: Long asset1, Short asset2*hedge_ratio
                entry_value = price1_entry - (position['alpha'] + position['beta'] * price2_entry)
                exit_value = price1_exit - (final_alpha + final_beta * price2_exit)
                pnl = (exit_value - entry_value) / entry_value
            else:
                # Short spread: Short asset1, Long asset2*hedge_ratio
                entry_value = price1_entry - (position['alpha'] + position['beta'] * price2_entry)
                exit_value = price1_exit - (final_alpha + final_beta * price2_exit)
                pnl = (entry_value - exit_value) / entry_value
            
            # Apply transaction costs
            pnl -= 2 * self.transaction_cost  # Entry and exit costs
            
            # Apply position size
            position_value = position['capital']
            pnl_value = position_value * pnl
            capital += position_value + pnl_value
            
            # Record the trade
            trades.append({
                'pair': (ticker1, ticker2),
                'position': position['position'],
                'entry_date': entry_day,
                'exit_date': price_data.index[-1],
                'duration': position_days,
                'entry_zscore': position['entry_zscore'],
                'exit_zscore': signals['zscore'].iloc[-1],
                'entry_alpha': position['alpha'] if include_intercept else 0,
                'entry_beta': position['beta'],
                'exit_alpha': final_alpha if include_intercept else 0,
                'exit_beta': final_beta,
                'pnl': pnl,
                'pnl_value': pnl_value
            })
        
        # Calculate performance metrics
        portfolio_returns = portfolio_values.pct_change().dropna()
        
        results = {
            'portfolio_values': portfolio_values,
            'portfolio_returns': portfolio_returns,
            'signals': signals,
            'trades': trades,
            'final_capital': capital,
            'total_return': (capital / initial_capital) - 1,
            'sharpe_ratio': np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std() if len(portfolio_returns) > 0 else 0,
            'win_rate': sum(1 for t in trades if t['pnl'] > 0) / len(trades) if trades else 0,
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'num_trades': len(trades)
        }
        
        return results
    
    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """
        Calculate the maximum drawdown from peak to trough.
        
        Args:
            portfolio_values: Series of portfolio values
            
        Returns:
            Maximum drawdown as a percentage
        """
        # Calculate the cumulative maximum of the portfolio values
        running_max = portfolio_values.cummax()
        
        # Calculate the drawdown in percentage terms
        drawdown = (portfolio_values / running_max) - 1.0
        
        # Return the minimum drawdown (maximum loss)
        return drawdown.min()
    
    def plot_hedge_ratio(
        self,
        price_data: pd.DataFrame,
        ticker1: str,
        ticker2: str,
        include_intercept: bool = True
    ):
        """
        Plot the dynamic hedge ratio evolution.
        
        Args:
            price_data: DataFrame of price data
            ticker1: First ticker symbol
            ticker2: Second ticker symbol
            include_intercept: Whether to include intercept in the model
            
        Returns:
            Matplotlib figure
        """
        # Get Kalman filter results
        kalman_results = self.estimate_dynamic_hedge_ratio(
            price_data, ticker1, ticker2, include_intercept
        )
        
        if kalman_results.empty:
            logger.warning(f"Could not estimate Kalman filter for {ticker1}-{ticker2}")
            return None
        
        # Create figure
        fig, axes = plt.subplots(
            3 if include_intercept else 2, 
            1, 
            figsize=(12, 12 if include_intercept else 9),
            sharex=True
        )
        
        # Plot normalized prices
        normalized_price1 = price_data[ticker1] / price_data[ticker1].iloc[0]
        normalized_price2 = price_data[ticker2] / price_data[ticker2].iloc[0]
        
        axes[0].plot(normalized_price1, label=ticker1)
        axes[0].plot(normalized_price2, label=ticker2)
        axes[0].set_title(f"Normalized Prices: {ticker1} vs {ticker2}")
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot hedge ratio (beta)
        beta_index = 1 if include_intercept else 0
        axes[1].plot(kalman_results['beta'])
        axes[1].set_title(f"Dynamic Hedge Ratio (Beta)")
        axes[1].grid(True)
        
        # Plot standard deviation of beta
        axes[1].fill_between(
            kalman_results.index,
            kalman_results['beta'] - np.sqrt(kalman_results['beta_var']),
            kalman_results['beta'] + np.sqrt(kalman_results['beta_var']),
            alpha=0.2
        )
        
        # Plot intercept (alpha) if included
        if include_intercept:
            axes[2].plot(kalman_results['alpha'])
            axes[2].set_title(f"Dynamic Intercept (Alpha)")
            axes[2].grid(True)
            
            # Plot standard deviation of alpha
            axes[2].fill_between(
                kalman_results.index,
                kalman_results['alpha'] - np.sqrt(kalman_results['alpha_var']),
                kalman_results['alpha'] + np.sqrt(kalman_results['alpha_var']),
                alpha=0.2
            )
        
        plt.tight_layout()
        return fig

    def plot_pair_analysis(
        self,
        price_data: pd.DataFrame,
        ticker1: str,
        ticker2: str,
        include_intercept: bool = True
    ):
        """
        Plot analysis for a trading pair using Kalman Filter.
        
        Args:
            price_data: DataFrame of price data
            ticker1: First ticker symbol
            ticker2: Second ticker symbol
            include_intercept: Whether to include intercept in the model
            
        Returns:
            Matplotlib figure
        """
        # Calculate dynamic spread and z-score
        spread_data = self.calculate_dynamic_spread(
            price_data, ticker1, ticker2, include_intercept
        )
        
        if spread_data.empty:
            logger.warning(f"Could not calculate spread for {ticker1}-{ticker2}")
            return None
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot normalized prices
        normalized_price1 = price_data[ticker1] / price_data[ticker1].iloc[0]
        normalized_price2 = price_data[ticker2] / price_data[ticker2].iloc[0]
        
        ax1.plot(normalized_price1, label=ticker1)
        ax1.plot(normalized_price2, label=ticker2)
        ax1.set_title(f"Normalized Prices: {ticker1} vs {ticker2}")
        ax1.legend()
        ax1.grid(True)
        
        # Plot spread
        ax2.plot(spread_data['spread'])
        ax2.set_title(f"Dynamic Spread")
        ax2.grid(True)
        
        # Plot z-score
        ax3.plot(spread_data['zscore'])
        ax3.set_title(f"Z-Score (window = {self.lookback_period})")
        ax3.axhline(y=0, color='r', linestyle='--')
        ax3.axhline(y=self.zscore_entry_threshold, color='g', linestyle='--', label=f"Entry Threshold ({self.zscore_entry_threshold})")
        ax3.axhline(y=-self.zscore_entry_threshold, color='g', linestyle='--')
        ax3.axhline(y=self.stop_loss_threshold, color='m', linestyle='--', label=f"Stop Loss ({self.stop_loss_threshold})")
        ax3.axhline(y=-self.stop_loss_threshold, color='m', linestyle='--')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_backtest_results(self, results: Dict):
        """
        Plot the backtest results.
        
        Args:
            results: Dictionary with backtest results
            
        Returns:
            Tuple of Matplotlib figures
        """
        if not results:
            logger.warning("No backtest results to plot.")
            return None
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot portfolio value
        portfolio_values = results['portfolio_values']
        ax1.plot(portfolio_values)
        ax1.set_title("Portfolio Value")
        ax1.set_ylabel("Value")
        ax1.grid(True)
        
        # Plot equity curve with drawdowns
        running_max = portfolio_values.cummax()
        drawdown = (portfolio_values / running_max) - 1.0
        
        ax2.plot(drawdown)
        ax2.set_title("Drawdown")
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_ylim(-1, 0.05)
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Create a second figure for trade analysis
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot trade P&Ls
        if results['trades']:
            trades_df = pd.DataFrame(results['trades'])
            trades_df['duration'] = pd.to_numeric(trades_df['duration'])
            
            # Histogram of trade P&Ls
            sns.histplot(trades_df['pnl'], kde=True, ax=ax3)
            ax3.set_title("Distribution of Trade P&Ls")
            ax3.set_xlabel("P&L")
            
            # Scatter plot of P&L vs duration
            ax4.scatter(trades_df['duration'], trades_df['pnl'])
            ax4.set_title("P&L vs Duration")
            ax4.set_xlabel("Duration (days)")
            ax4.set_ylabel("P&L")
            ax4.grid(True)
        
        plt.tight_layout()
        
        # Create a third figure for signals and alpha/beta
        fig3 = None
        if 'signals' in results:
            signals = results['signals']
            
            if 'alpha' in signals.columns:
                # Create figure with 4 subplots
                fig3, axes = plt.subplots(4, 1, figsize=(12, 15), sharex=True)
                
                # Plot spread
                axes[0].plot(signals['spread'])
                axes[0].set_title("Spread")
                axes[0].grid(True)
                
                # Plot z-score
                axes[1].plot(signals['zscore'])
                axes[1].set_title("Z-Score")
                axes[1].axhline(y=0, color='r', linestyle='--')
                axes[1].axhline(y=self.zscore_entry_threshold, color='g', linestyle='--')
                axes[1].axhline(y=-self.zscore_entry_threshold, color='g', linestyle='--')
                axes[1].grid(True)
                
                # Plot beta
                axes[2].plot(signals['beta'])
                axes[2].set_title("Dynamic Hedge Ratio (Beta)")
                axes[2].grid(True)
                
                # Plot alpha
                axes[3].plot(signals['alpha'])
                axes[3].set_title("Dynamic Intercept (Alpha)")
                axes[3].grid(True)
                
            else:
                # Create figure with 3 subplots
                fig3, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
                
                # Plot spread
                axes[0].plot(signals['spread'])
                axes[0].set_title("Spread")
                axes[0].grid(True)
                
                # Plot z-score
                axes[1].plot(signals['zscore'])
                axes[1].set_title("Z-Score")
                axes[1].axhline(y=0, color='r', linestyle='--')
                axes[1].axhline(y=self.zscore_entry_threshold, color='g', linestyle='--')
                axes[1].axhline(y=-self.zscore_entry_threshold, color='g', linestyle='--')
                axes[1].grid(True)
                
                # Plot beta
                axes[2].plot(signals['beta'])
                axes[2].set_title("Dynamic Hedge Ratio (Beta)")
                axes[2].grid(True)
            
            plt.tight_layout()
        
        return fig, fig2, fig3 