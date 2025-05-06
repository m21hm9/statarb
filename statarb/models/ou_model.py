"""
Ornstein-Uhlenbeck Process for Statistical Arbitrage

This module implements the Ornstein-Uhlenbeck process modeling for spread series
in pairs trading. It includes estimation of mean reversion speed, half-life
calculation, and regime detection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from typing import Tuple, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class OUModel:
    """
    Ornstein-Uhlenbeck process model for mean-reverting time series.
    
    The OU process is defined by the SDE:
    dX_t = theta * (mu - X_t) * dt + sigma * dW_t
    
    where:
    - theta: speed of mean reversion
    - mu: long-term mean
    - sigma: volatility
    - W_t: Wiener process (Brownian motion)
    """
    
    def __init__(
        self,
        estimation_method: str = 'regression',
        min_half_life: float = 1.0,
        max_half_life: float = 100.0,
        confidence_level: float = 0.95
    ):
        """
        Initialize the OU Model.
        
        Args:
            estimation_method: Method to estimate parameters ('regression' or 'mle')
            min_half_life: Minimum acceptable half-life (days)
            max_half_life: Maximum acceptable half-life (days)
            confidence_level: Confidence level for statistical tests
        """
        self.estimation_method = estimation_method
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.confidence_level = confidence_level
        
        # Model parameters
        self.theta = None  # Mean reversion speed
        self.mu = None     # Long-term mean
        self.sigma = None  # Volatility
        self.half_life = None  # Half-life of mean reversion
        
        # Fit statistics
        self.is_mean_reverting = False
        self.p_value = None
        self.log_likelihood = None
        self.aic = None
        self.bic = None
        self.r_squared = None
        
    def fit(self, spread: pd.Series) -> Dict:
        """
        Fit the OU model to a spread series.
        
        Args:
            spread: Time series of spread values
            
        Returns:
            Dictionary of fitted parameters and statistics
        """
        # Clean data
        spread = spread.dropna()
        
        if len(spread) < 30:
            logger.warning("Not enough data to fit OU model (need at least 30 points)")
            return {
                'is_mean_reverting': False,
                'half_life': np.inf,
                'theta': 0,
                'mu': spread.mean(),
                'sigma': spread.std(),
                'p_value': 1.0
            }
        
        # Test for stationarity
        adf_result = adfuller(spread)
        self.p_value = adf_result[1]
        self.is_mean_reverting = self.p_value < (1 - self.confidence_level)
        
        # Estimate parameters based on the chosen method
        if self.estimation_method == 'regression':
            self._fit_regression(spread)
        elif self.estimation_method == 'mle':
            self._fit_mle(spread)
        else:
            raise ValueError(f"Unknown estimation method: {self.estimation_method}")
        
        # Calculate half-life
        if self.theta > 0:
            self.half_life = np.log(2) / self.theta
        else:
            self.half_life = np.inf
            
        # Check if half-life is within acceptable range
        if self.half_life < self.min_half_life or self.half_life > self.max_half_life:
            self.is_mean_reverting = False
        
        # Return parameters and statistics
        return {
            'is_mean_reverting': self.is_mean_reverting,
            'half_life': self.half_life,
            'theta': self.theta,
            'mu': self.mu,
            'sigma': self.sigma,
            'p_value': self.p_value,
            'r_squared': self.r_squared,
            'log_likelihood': self.log_likelihood,
            'aic': self.aic,
            'bic': self.bic
        }
    
    def _fit_regression(self, spread: pd.Series):
        """
        Estimate OU parameters using linear regression.
        
        Args:
            spread: Time series of spread values
        """
        # Calculate lagged values and differences
        lagged_spread = spread.shift(1)
        delta_spread = spread - lagged_spread
        
        # Remove NaN values
        valid_data = ~(lagged_spread.isnull() | delta_spread.isnull())
        X = lagged_spread[valid_data]
        y = delta_spread[valid_data]
        
        # Add constant term
        X = sm.add_constant(X)
        
        # Fit regression model
        model = OLS(y, X).fit()
        
        # Extract parameters
        self.theta = -model.params[1]  # Negative of the slope
        self.mu = model.params[0] / self.theta  # Intercept / theta
        
        # Calculate residuals and volatility
        residuals = model.resid
        self.sigma = np.sqrt(np.var(residuals) * 252)  # Annualized
        
        # Store fit statistics
        self.r_squared = model.rsquared
        self.log_likelihood = model.llf
        self.aic = model.aic
        self.bic = model.bic
    
    def _fit_mle(self, spread: pd.Series):
        """
        Estimate OU parameters using maximum likelihood estimation.
        
        Args:
            spread: Time series of spread values
        """
        # Define negative log-likelihood function
        def neg_log_likelihood(params):
            theta, mu, sigma = params
            
            if theta <= 0 or sigma <= 0:
                return 1e10  # Return large value for invalid parameters
            
            # Calculate log-likelihood
            n = len(spread) - 1
            dt = 1  # Assuming daily data, dt=1
            
            # Precompute values
            exp_theta_dt = np.exp(-theta * dt)
            X = np.array(spread[1:])
            X_prev = np.array(spread[:-1])
            
            # Expected value
            expected = mu + (X_prev - mu) * exp_theta_dt
            
            # Variance
            var = (sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta * dt))
            
            # Log-likelihood
            ll = -n/2 * np.log(2 * np.pi * var) - np.sum((X - expected)**2) / (2 * var)
            
            return -ll  # Return negative log-likelihood for minimization
        
        # Initial guess based on regression
        self._fit_regression(spread)
        initial_params = [self.theta, self.mu, self.sigma / np.sqrt(252)]
        
        # Bounds for parameters
        bounds = [
            (1e-5, 10),   # theta: positive, not too large
            (None, None), # mu: no bounds
            (1e-5, None)  # sigma: positive
        ]
        
        # Perform optimization
        try:
            result = minimize(
                neg_log_likelihood,
                initial_params,
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if result.success:
                # Extract parameters
                self.theta, self.mu, sigma_daily = result.x
                self.sigma = sigma_daily * np.sqrt(252)  # Annualize
                
                # Store log-likelihood
                self.log_likelihood = -result.fun
                
                # Calculate AIC and BIC
                k = 3  # Number of parameters
                n = len(spread)
                self.aic = 2 * k - 2 * self.log_likelihood
                self.bic = k * np.log(n) - 2 * self.log_likelihood
                
                # Calculate R-squared using the regression model (approximate)
                self._calculate_r_squared(spread)
            else:
                logger.warning(f"MLE optimization failed: {result.message}")
                # Fall back to regression estimates
        
        except Exception as e:
            logger.error(f"Error in MLE estimation: {str(e)}")
            # Keep regression estimates
    
    def _calculate_r_squared(self, spread: pd.Series):
        """
        Calculate R-squared for the fitted model.
        
        Args:
            spread: Time series of spread values
        """
        if self.theta is None or self.mu is None:
            self.r_squared = 0
            return
        
        # Calculate one-step ahead predictions
        lagged_spread = spread.shift(1)
        valid_data = ~lagged_spread.isnull()
        
        # Predicted changes
        dt = 1  # Assuming daily data
        pred_changes = self.theta * (self.mu - lagged_spread[valid_data]) * dt
        
        # Actual changes
        actual_changes = spread[valid_data] - lagged_spread[valid_data]
        
        # Calculate R-squared
        ss_total = np.sum((actual_changes - actual_changes.mean())**2)
        ss_residual = np.sum((actual_changes - pred_changes)**2)
        
        if ss_total > 0:
            self.r_squared = 1 - (ss_residual / ss_total)
        else:
            self.r_squared = 0
    
    def predict(
        self, 
        spread: pd.Series, 
        steps: int = 1, 
        num_simulations: int = 1000
    ) -> Dict:
        """
        Generate predictions and confidence intervals.
        
        Args:
            spread: Time series of spread values
            steps: Number of steps ahead to predict
            num_simulations: Number of Monte Carlo simulations
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        if self.theta is None or self.mu is None or self.sigma is None:
            logger.warning("Model not fitted. Call fit() first.")
            return {}
        
        # Get the last observed value
        last_value = spread.iloc[-1]
        
        # Time step
        dt = 1  # Assuming daily data
        
        # Precompute values
        exp_theta_dt = np.exp(-self.theta * dt)
        var = (self.sigma**2 / (2 * self.theta)) * (1 - np.exp(-2 * self.theta * dt))
        
        # Expected mean path
        means = np.zeros(steps + 1)
        means[0] = last_value
        
        for t in range(1, steps + 1):
            means[t] = self.mu + (means[t-1] - self.mu) * exp_theta_dt
        
        # Confidence intervals via Monte Carlo simulation
        simulations = np.zeros((num_simulations, steps + 1))
        simulations[:, 0] = last_value
        
        for t in range(1, steps + 1):
            # Expected value
            expected = self.mu + (simulations[:, t-1] - self.mu) * exp_theta_dt
            
            # Add random noise
            simulations[:, t] = expected + np.random.normal(0, np.sqrt(var), num_simulations)
        
        # Calculate percentiles for each step
        percentiles = {}
        for p in [5, 10, 25, 50, 75, 90, 95]:
            percentiles[p] = np.percentile(simulations[:, 1:], p, axis=0)
        
        return {
            'mean_path': means[1:],
            'percentiles': percentiles,
            'simulations': simulations[:, 1:]
        }
    
    def estimate_optimal_trading_horizon(self) -> int:
        """
        Estimate the optimal trading horizon based on mean reversion half-life.
        
        Returns:
            Optimal horizon in days
        """
        if self.half_life is None or self.half_life == np.inf:
            return 20  # Default value
        
        # Empirically, optimal holding period is often around 1.5-2x the half-life
        return int(np.ceil(self.half_life * 1.5))
    
    def calculate_expected_profit(
        self, 
        spread: pd.Series, 
        entry_zscore: float,
        transaction_cost: float = 0.001
    ) -> float:
        """
        Calculate expected profit from mean reversion.
        
        Args:
            spread: Time series of spread values
            entry_zscore: Z-score at entry
            transaction_cost: Round-trip transaction cost (fraction)
            
        Returns:
            Expected profit as percentage
        """
        if not self.is_mean_reverting or self.half_life == np.inf:
            return 0.0
        
        # Calculate spread statistics
        spread_std = spread.std()
        mean_spread = spread.mean()
        
        # Current spread level
        current_spread = mean_spread + entry_zscore * spread_std
        
        # Expected spread at optimal exit (mean)
        expected_spread = mean_spread
        
        # Calculate profit potential
        if entry_zscore < 0:  # Long spread position
            expected_profit = (expected_spread - current_spread) / abs(current_spread)
        else:  # Short spread position
            expected_profit = (current_spread - expected_spread) / abs(current_spread)
        
        # Adjust for transaction costs
        expected_profit -= transaction_cost
        
        return expected_profit
    
    def plot_fit(
        self, 
        spread: pd.Series, 
        title: str = "Ornstein-Uhlenbeck Process Fit",
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot the fitted OU process against the original spread.
        
        Args:
            spread: Time series of spread values
            title: Plot title
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure
        """
        if self.theta is None or self.mu is None:
            logger.warning("Model not fitted. Call fit() first.")
            return None
        
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Plot 1: Original spread and long-term mean
        axes[0].plot(spread, label='Spread', color='blue', alpha=0.7)
        axes[0].axhline(y=self.mu, color='red', linestyle='--', label=f'Long-term Mean ({self.mu:.4f})')
        axes[0].set_title(f"{title} - Half-life: {self.half_life:.2f} days")
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot 2: Detrended spread (subtract mean)
        detrended = spread - self.mu
        axes[1].plot(detrended, label='Detrended Spread', color='green', alpha=0.7)
        axes[1].axhline(y=0, color='black', linestyle='--')
        
        # Add bands at ±1, ±2 standard deviations
        equilibrium_std = self.sigma / np.sqrt(2 * self.theta)
        for i in [1, 2]:
            axes[1].axhline(y=i * equilibrium_std, color='orange', linestyle=':', alpha=0.7)
            axes[1].axhline(y=-i * equilibrium_std, color='orange', linestyle=':', alpha=0.7)
            
        axes[1].set_title(f"Detrended Spread - Equilibrium Std: {equilibrium_std:.4f}")
        axes[1].grid(True)
        
        # Plot 3: Mean-reverting tendency
        lagged_spread = spread.shift(1).dropna()
        delta_spread = spread.iloc[1:] - lagged_spread
        
        axes[2].scatter(lagged_spread, delta_spread, alpha=0.5, s=10)
        
        # Add regression line
        x_range = np.linspace(min(lagged_spread), max(lagged_spread), 100)
        y_range = self.theta * (self.mu - x_range)
        axes[2].plot(x_range, y_range, 'r-', 
                     label=f'θ(μ-x): θ={self.theta:.4f}, R²={self.r_squared:.2f}')
        
        axes[2].axhline(y=0, color='black', linestyle='--')
        axes[2].axvline(x=self.mu, color='red', linestyle='--')
        axes[2].set_xlabel('Spread(t-1)')
        axes[2].set_ylabel('Δ Spread(t)')
        axes[2].set_title('Mean-Reverting Behavior')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_predictions(
        self,
        spread: pd.Series,
        predictions: Dict,
        title: str = "Spread Forecast",
        figsize: Tuple[int, int] = (12, 6),
        show_simulations: bool = False,
        num_shown_sims: int = 50
    ) -> plt.Figure:
        """
        Plot spread predictions with confidence intervals.
        
        Args:
            spread: Time series of spread values
            predictions: Dictionary of predictions from predict()
            title: Plot title
            figsize: Figure size tuple
            show_simulations: Whether to show individual simulations
            num_shown_sims: Number of simulations to show
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Historical data
        historical_dates = spread.index[-min(30, len(spread)):]
        historical_values = spread.values[-min(30, len(spread)):]
        ax.plot(historical_dates, historical_values, 'b-', label='Historical')
        
        # Future dates
        last_date = spread.index[-1]
        if isinstance(last_date, pd.Timestamp):
            future_dates = pd.date_range(start=last_date, periods=len(predictions['mean_path'])+1)[1:]
        else:
            future_dates = np.arange(len(predictions['mean_path'])) + len(spread)
        
        # Plot mean prediction
        ax.plot(future_dates, predictions['mean_path'], 'r-', label='Expected Path')
        
        # Plot confidence intervals
        ax.fill_between(future_dates, 
                        predictions['percentiles'][5], 
                        predictions['percentiles'][95], 
                        color='red', alpha=0.1, label='90% CI')
        
        ax.fill_between(future_dates, 
                        predictions['percentiles'][25], 
                        predictions['percentiles'][75], 
                        color='red', alpha=0.2, label='50% CI')
        
        # Plot some simulations if requested
        if show_simulations and 'simulations' in predictions:
            # Choose random simulations
            sim_indices = np.random.choice(
                predictions['simulations'].shape[0], 
                min(num_shown_sims, predictions['simulations'].shape[0]), 
                replace=False
            )
            
            for idx in sim_indices:
                ax.plot(future_dates, predictions['simulations'][idx], 'gray', alpha=0.1)
        
        # Add long-term mean
        if isinstance(future_dates[-1], pd.Timestamp):
            ax.axhline(y=self.mu, xmin=pd.Timestamp(future_dates[0]), 
                      color='green', linestyle='--', label='Long-term Mean')
        else:
            ax.axhline(y=self.mu, color='green', linestyle='--', label='Long-term Mean')
        
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        return fig

class OUSpreadAnalyzer:
    """
    Analyzer for pairs trading spreads using OU process modeling.
    """
    
    def __init__(
        self,
        estimation_method: str = 'regression',
        min_half_life: float = 1.0,
        max_half_life: float = 100.0,
        lookback_window: int = 252,
        rolling_window: int = 60,
        confidence_level: float = 0.95
    ):
        """
        Initialize the spread analyzer.
        
        Args:
            estimation_method: Method to estimate parameters ('regression' or 'mle')
            min_half_life: Minimum acceptable half-life (days)
            max_half_life: Maximum acceptable half-life (days)
            lookback_window: Window for full analysis
            rolling_window: Window for rolling analysis
            confidence_level: Confidence level for statistical tests
        """
        self.estimation_method = estimation_method
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.lookback_window = lookback_window
        self.rolling_window = rolling_window
        self.confidence_level = confidence_level
        
        # Storage for analysis results
        self.full_results = {}
        self.rolling_results = {}
        
    def analyze_spread(
        self,
        spread: pd.Series,
        pair_id: str = None
    ) -> Dict:
        """
        Perform full analysis of a spread series.
        
        Args:
            spread: Time series of spread values
            pair_id: Optional identifier for the pair
            
        Returns:
            Dictionary of analysis results
        """
        # Clean data
        spread = spread.dropna()
        
        if len(spread) < 30:
            logger.warning("Not enough data to analyze spread (need at least 30 points)")
            return {'is_mean_reverting': False}
        
        # Create OU model
        model = OUModel(
            estimation_method=self.estimation_method,
            min_half_life=self.min_half_life,
            max_half_life=self.max_half_life,
            confidence_level=self.confidence_level
        )
        
        # Use the most recent lookback_window data points
        recent_spread = spread.iloc[-min(len(spread), self.lookback_window):]
        
        # Fit the model
        results = model.fit(recent_spread)
        
        # Calculate additional metrics
        results['optimal_horizon'] = model.estimate_optimal_trading_horizon()
        
        # Predictions
        if results['is_mean_reverting']:
            predictions = model.predict(recent_spread, steps=min(20, int(results['half_life'] * 2)))
            results['predictions'] = predictions
        
        # Store model
        results['model'] = model
        
        # Store results
        if pair_id is not None:
            self.full_results[pair_id] = results
        
        return results
    
    def analyze_rolling_spreads(
        self,
        spread: pd.Series,
        pair_id: str = None
    ) -> pd.DataFrame:
        """
        Perform rolling analysis of spread characteristics.
        
        Args:
            spread: Time series of spread values
            pair_id: Optional identifier for the pair
            
        Returns:
            DataFrame with rolling analysis results
        """
        # Clean data
        spread = spread.dropna()
        
        if len(spread) < self.rolling_window + 10:
            logger.warning(f"Not enough data for rolling analysis (need at least {self.rolling_window + 10} points)")
            return pd.DataFrame()
        
        # Initialize results
        results = []
        
        # For each rolling window
        for i in range(self.rolling_window, len(spread), 10):  # Step by 10 days for efficiency
            # Window data
            window_end = i
            window_start = max(0, window_end - self.rolling_window)
            window_data = spread.iloc[window_start:window_end]
            window_date = spread.index[window_end - 1]
            
            # Create and fit model
            model = OUModel(
                estimation_method=self.estimation_method,
                min_half_life=self.min_half_life,
                max_half_life=self.max_half_life,
                confidence_level=self.confidence_level
            )
            
            model_results = model.fit(window_data)
            
            # Store key results
            results.append({
                'date': window_date,
                'is_mean_reverting': model_results['is_mean_reverting'],
                'half_life': model_results['half_life'],
                'theta': model_results['theta'],
                'mu': model_results['mu'],
                'sigma': model_results['sigma'],
                'p_value': model_results['p_value'],
                'r_squared': model_results['r_squared']
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        results_df.set_index('date', inplace=True)
        
        # Store results
        if pair_id is not None:
            self.rolling_results[pair_id] = results_df
        
        return results_df
    
    def is_regime_favorable(
        self,
        pair_id: str,
        min_lookback_days: int = 20
    ) -> Tuple[bool, str]:
        """
        Determine if the current regime is favorable for trading the pair.
        
        Args:
            pair_id: Pair identifier
            min_lookback_days: Minimum days to look back
            
        Returns:
            Tuple of (is_favorable, reason)
        """
        if pair_id not in self.rolling_results:
            return False, "No rolling analysis available"
        
        # Get the most recent results
        recent_results = self.rolling_results[pair_id].iloc[-min_lookback_days:]
        
        # Check if mean-reverting
        pct_mean_reverting = recent_results['is_mean_reverting'].mean()
        if pct_mean_reverting < 0.7:
            return False, f"Pair is not consistently mean-reverting ({pct_mean_reverting:.1%})"
        
        # Check if half-life is stable
        recent_half_lives = recent_results['half_life'].values
        recent_half_lives = recent_half_lives[recent_half_lives < np.inf]
        
        if len(recent_half_lives) < 5:
            return False, "Not enough valid half-life estimates"
        
        half_life_std = np.std(recent_half_lives)
        half_life_mean = np.mean(recent_half_lives)
        half_life_cv = half_life_std / half_life_mean if half_life_mean > 0 else np.inf
        
        if half_life_cv > 0.5:
            return False, f"Half-life is unstable (CV = {half_life_cv:.2f})"
        
        # Check current half-life
        current_half_life = recent_half_lives[-1]
        if current_half_life < self.min_half_life:
            return False, f"Half-life too short ({current_half_life:.1f} days)"
        if current_half_life > self.max_half_life:
            return False, f"Half-life too long ({current_half_life:.1f} days)"
        
        return True, f"Favorable regime with half-life {current_half_life:.1f} days"
    
    def plot_rolling_analysis(
        self,
        pair_id: str,
        figsize: Tuple[int, int] = (12, 10)
    ) -> plt.Figure:
        """
        Plot rolling analysis results.
        
        Args:
            pair_id: Pair identifier
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if pair_id not in self.rolling_results:
            logger.warning(f"No rolling analysis available for {pair_id}")
            return None
        
        # Get results
        results = self.rolling_results[pair_id]
        
        # Create figure
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # Plot 1: Half-life
        axes[0].plot(results.index, results['half_life'].clip(0, 100), 'b-')
        axes[0].set_ylabel('Half-life (days)')
        axes[0].set_title(f'Rolling Analysis for {pair_id} - Window: {self.rolling_window} days')
        axes[0].axhline(y=self.min_half_life, color='red', linestyle='--', 
                       label=f'Min Half-life ({self.min_half_life})')
        axes[0].axhline(y=self.max_half_life, color='red', linestyle='--', 
                       label=f'Max Half-life ({self.max_half_life})')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot 2: Mean reversion speed (theta)
        axes[1].plot(results.index, results['theta'], 'g-')
        axes[1].set_ylabel('Theta')
        axes[1].set_title('Mean Reversion Speed')
        axes[1].grid(True)
        
        # Plot 3: R-squared
        axes[2].plot(results.index, results['r_squared'], 'purple')
        axes[2].set_ylabel('R-squared')
        axes[2].set_title('Model Fit Quality')
        axes[2].set_ylim(0, 1)
        axes[2].grid(True)
        
        # Plot 4: Is Mean Reverting
        axes[3].plot(results.index, results['is_mean_reverting'].astype(int), 'ro-')
        axes[3].set_ylabel('Is Mean Reverting')
        axes[3].set_title('Regime Classification')
        axes[3].set_ylim(-0.1, 1.1)
        axes[3].set_xlabel('Date')
        axes[3].grid(True)
        
        plt.tight_layout()
        return fig 