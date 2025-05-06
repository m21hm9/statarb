import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
import logging
from typing import List, Tuple, Dict, Optional, Union
import seaborn as sns
from ..utils.data_utils import calculate_zscore
from scipy.stats import johansen
from arch.unitroot import PhillipsPerron
import warnings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PairsTrader:
    def __init__(
        self,
        lookback_period: int = 252,
        zscore_entry_threshold: float = 2.0,
        zscore_exit_threshold: float = 0.5,
        stop_loss_threshold: float = 3.0,
        max_position_days: int = 30,
        formation_period: int = 252 * 2,
        hedge_ratio_recalculation_days: int = 30,
        use_adaptive_thresholds: bool = True,
        use_dynamic_stop_loss: bool = True,
        position_sizing_method: str = 'inverse_volatility',
        max_drawdown_limit: float = 0.15,
        max_pair_allocation: float = 0.05,
        max_portfolio_pairs: int = 10,
        min_half_life: int = 5,
        max_half_life: int = 100
    ):
        self.lookback_period = lookback_period
        self.zscore_entry_threshold = zscore_entry_threshold
        self.zscore_exit_threshold = zscore_exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.max_position_days = max_position_days
        self.formation_period = formation_period
        self.hedge_ratio_recalculation_days = hedge_ratio_recalculation_days
        self.use_adaptive_thresholds = use_adaptive_thresholds
        self.use_dynamic_stop_loss = use_dynamic_stop_loss
        self.position_sizing_method = position_sizing_method
        self.max_drawdown_limit = max_drawdown_limit
        self.max_pair_allocation = max_pair_allocation
        self.max_portfolio_pairs = max_portfolio_pairs
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        
        self.pairs = []
        self.hedge_ratios = {}
        self.rolling_hedge_ratios = {}
        self.pair_half_lives = {}
        self.current_positions = {}
        self.position_history = []
        self.regime_states = {}
        self.adaptive_thresholds = {}
        
    def find_cointegrated_pairs(
        self, 
        price_data: pd.DataFrame, 
        pvalue_threshold: float = 0.05,
        use_multiple_tests: bool = True,
        min_test_agreement: int = 2
    ) -> List[Tuple[str, str, float, Dict]]:
        """
        Find cointegrated pairs using multiple tests.
        
        Args:
            price_data: DataFrame of price data
            pvalue_threshold: Threshold for p-value to consider pairs cointegrated
            use_multiple_tests: Whether to use multiple cointegration tests
            min_test_agreement: Minimum number of tests that must agree for a pair to be considered cointegrated
            
        Returns:
            List of tuples (ticker1, ticker2, pvalue, test_results)
        """
        n = len(price_data.columns)
        keys = price_data.columns
        pairs = []
        
        # Progress tracking
        total_pairs = (n * (n-1)) // 2
        completed_pairs = 0
        
        # Perform cointegration test for each pair
        for i in range(n):
            for j in range(i+1, n):
                # Skip if either series has NaN values
                if price_data.iloc[:, i].isnull().any() or price_data.iloc[:, j].isnull().any():
                    completed_pairs += 1
                    continue
                
                ticker1, ticker2 = keys[i], keys[j]
                
                # Create test results dictionary
                test_results = {
                    'engle_granger': {'p_value': 1.0, 'is_cointegrated': False},
                    'phillips_perron': {'p_value': 1.0, 'is_cointegrated': False},
                    'johansen': {'p_value': 1.0, 'is_cointegrated': False, 'rank': 0},
                    'agreement_count': 0
                }
                
                # Always run Engle-Granger test
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    try:
                        result = coint(price_data.iloc[:, i], price_data.iloc[:, j])
                        pvalue = result[1]
                        test_results['engle_granger']['p_value'] = pvalue
                        test_results['engle_granger']['is_cointegrated'] = pvalue < pvalue_threshold
                        test_results['agreement_count'] += int(pvalue < pvalue_threshold)
                    except:
                        logger.warning(f"Error running Engle-Granger test for {ticker1}-{ticker2}")
                
                # Run additional tests if requested
                if use_multiple_tests:
                    # Phillips-Perron test on the residuals
                    try:
                        # First get the hedge ratio
                        y = price_data.iloc[:, i]
                        x = sm.add_constant(price_data.iloc[:, j])
                        model = OLS(y, x).fit()
                        residuals = model.resid
                        
                        # Run Phillips-Perron test on residuals
                        pp_test = PhillipsPerron(residuals)
                        pp_pvalue = pp_test.pvalue
                        test_results['phillips_perron']['p_value'] = pp_pvalue
                        test_results['phillips_perron']['is_cointegrated'] = pp_pvalue < pvalue_threshold
                        test_results['agreement_count'] += int(pp_pvalue < pvalue_threshold)
                    except:
                        logger.warning(f"Error running Phillips-Perron test for {ticker1}-{ticker2}")
                    
                    # Johansen test
                    try:
                        # Prepare data for Johansen test
                        pair_data = price_data.iloc[:, [i, j]].values
                        
                        # Run Johansen test
                        joh_result = johansen.coint_johansen(pair_data, 0, 1)
                        
                        # Get the trace test statistics
                        trace_stat = joh_result.lr1[0]
                        trace_crit = joh_result.cvt[0, 1]  # 5% critical value
                        
                        # Determine if cointegrated based on trace test
                        is_coint = trace_stat > trace_crit
                        
                        # Store results
                        test_results['johansen']['p_value'] = 0.05 if is_coint else 0.1  # Approximate
                        test_results['johansen']['is_cointegrated'] = is_coint
                        test_results['johansen']['rank'] = int(is_coint)
                        test_results['agreement_count'] += int(is_coint)
                    except:
                        logger.warning(f"Error running Johansen test for {ticker1}-{ticker2}")
                
                # Determine if pair is considered cointegrated based on agreement count
                is_cointegrated = test_results['agreement_count'] >= min_test_agreement if use_multiple_tests else test_results['engle_granger']['is_cointegrated']
                
                if is_cointegrated:
                    # Use the Engle-Granger p-value as the representative p-value
                    pvalue = test_results['engle_granger']['p_value']
                    pairs.append((ticker1, ticker2, pvalue, test_results))
                    logger.info(f"Found cointegrated pair: {ticker1} - {ticker2} (p-value: {pvalue:.4f}, agreement: {test_results['agreement_count']})")
                
                completed_pairs += 1
                if completed_pairs % 100 == 0:
                    logger.info(f"Completed {completed_pairs} out of {total_pairs} pairs ({(completed_pairs/total_pairs)*100:.1f}%)")
        
        # Sort pairs by p-value
        pairs.sort(key=lambda x: x[2])
        
        # Store the pairs
        self.pairs = pairs
        
        return pairs
    
    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate the half-life of mean reversion for a spread.
        
        Args:
            spread: Time series of the spread
            
        Returns:
            Half-life value (in days)
        """
        spread = spread.dropna()
        
        # Calculate lag of spread
        lag_spread = spread.shift(1)
        
        # Calculate change in spread
        delta_spread = spread - lag_spread
        
        # Remove NaN values
        lag_spread = lag_spread.dropna()
        delta_spread = delta_spread.dropna()
        
        # Regression: delta_spread = gamma * lag_spread + error
        X = sm.add_constant(lag_spread)
        model = OLS(delta_spread, X).fit()
        
        # Extract gamma coefficient
        gamma = model.params[1]
        
        # Calculate half-life: t_half = -log(2) / log(1 + gamma)
        # If gamma is positive, the process is not mean-reverting
        if gamma >= 0:
            half_life = 1000  # Set a very large value
        else:
            half_life = -np.log(2) / gamma
        
        return half_life
    
    def calculate_hedge_ratio(
        self, 
        price_data: pd.DataFrame, 
        ticker1: str, 
        ticker2: str,
        method: str = 'ols'
    ) -> Tuple[float, float]:
        """
        Calculate hedge ratio using various methods.
        
        Args:
            price_data: DataFrame of price data
            ticker1: First ticker symbol
            ticker2: Second ticker symbol
            method: Method to use ('ols', 'tls', 'rolling')
            
        Returns:
            Tuple of (alpha, beta)
        """
        y = price_data[ticker1]
        x = price_data[ticker2]
        
        if method == 'ols':
            # Add constant to x
            X = sm.add_constant(x)
            
            # Fit OLS model
            model = OLS(y, X).fit()
            
            # Extract coefficients
            alpha, beta = model.params[0], model.params[1]
            
        elif method == 'tls':
            # Total Least Squares (Orthogonal Regression)
            from sklearn.decomposition import PCA
            
            # Standardize data
            data = np.vstack([x, y]).T
            data_std = (data - data.mean(axis=0)) / data.std(axis=0)
            
            # Fit PCA
            pca = PCA(n_components=1)
            pca.fit(data_std)
            
            # Get the first principal component
            v = pca.components_[0]
            
            # Calculate slope (beta) and intercept (alpha)
            beta = v[1] / v[0]
            alpha = y.mean() - beta * x.mean()
            
        elif method == 'rolling':
            # Use a rolling window to calculate beta
            window = min(60, len(price_data) // 4)
            rolling_betas = []
            rolling_alphas = []
            
            for i in range(window, len(price_data)):
                window_data = price_data.iloc[i-window:i]
                X = sm.add_constant(window_data[ticker2])
                model = OLS(window_data[ticker1], X).fit()
                rolling_alphas.append(model.params[0])
                rolling_betas.append(model.params[1])
            
            # Use the median values for stability
            alpha, beta = np.median(rolling_alphas), np.median(rolling_betas)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Store the hedge ratio
        self.hedge_ratios[(ticker1, ticker2)] = (alpha, beta)
        
        return alpha, beta
    
    def calculate_rolling_hedge_ratios(
        self, 
        price_data: pd.DataFrame, 
        ticker1: str, 
        ticker2: str, 
        window: int = 60
    ) -> pd.DataFrame:
        """
        Calculate rolling hedge ratios.
        
        Args:
            price_data: DataFrame of price data
            ticker1: First ticker symbol
            ticker2: Second ticker symbol
            window: Rolling window size
            
        Returns:
            DataFrame with rolling alphas and betas
        """
        y = price_data[ticker1]
        x = price_data[ticker2]
        
        rolling_alphas = []
        rolling_betas = []
        index_values = []
        
        for i in range(window, len(price_data)):
            try:
                # Get window data
                window_y = y.iloc[i-window:i]
                window_x = x.iloc[i-window:i]
                
                # Add constant to x
                X = sm.add_constant(window_x)
                
                # Fit OLS model
                model = OLS(window_y, X).fit()
                
                # Extract coefficients
                alpha, beta = model.params[0], model.params[1]
                
                rolling_alphas.append(alpha)
                rolling_betas.append(beta)
                index_values.append(price_data.index[i])
            except:
                # Skip if there's an error
                continue
        
        # Create DataFrame
        rolling_hedge_ratios = pd.DataFrame({
            'alpha': rolling_alphas,
            'beta': rolling_betas
        }, index=index_values)
        
        # Store rolling hedge ratios
        self.rolling_hedge_ratios[(ticker1, ticker2)] = rolling_hedge_ratios
        
        return rolling_hedge_ratios
    
    def calculate_spread(
        self, 
        price_data: pd.DataFrame, 
        ticker1: str, 
        ticker2: str, 
        alpha: Optional[float] = None, 
        beta: Optional[float] = None,
        use_rolling: bool = False
    ) -> pd.Series:
        """
        Calculate the spread between two assets.
        
        Args:
            price_data: DataFrame of price data
            ticker1: First ticker symbol
            ticker2: Second ticker symbol
            alpha: Optional intercept
            beta: Optional slope coefficient
            use_rolling: Whether to use rolling hedge ratios
            
        Returns:
            Spread series
        """
        if alpha is None or beta is None:
            if (ticker1, ticker2) in self.hedge_ratios:
                alpha, beta = self.hedge_ratios[(ticker1, ticker2)]
            else:
                alpha, beta = self.calculate_hedge_ratio(price_data, ticker1, ticker2)
        
        if use_rolling and (ticker1, ticker2) in self.rolling_hedge_ratios:
            # Use rolling hedge ratios to calculate spread
            rolling_hr = self.rolling_hedge_ratios[(ticker1, ticker2)]
            spread = pd.Series(index=price_data.index)
            
            # Use fixed hedge ratio for dates before rolling window
            first_rolling_date = rolling_hr.index[0]
            early_data_mask = price_data.index < first_rolling_date
            if early_data_mask.any():
                spread.loc[early_data_mask] = price_data.loc[early_data_mask, ticker1] - (alpha + beta * price_data.loc[early_data_mask, ticker2])
            
            # Use rolling hedge ratios for later dates
            for date, row in rolling_hr.iterrows():
                if date in price_data.index:
                    alpha_t, beta_t = row['alpha'], row['beta']
                    spread.loc[date] = price_data.loc[date, ticker1] - (alpha_t + beta_t * price_data.loc[date, ticker2])
            
            return spread
        else:
            # Use fixed hedge ratio
            return price_data[ticker1] - (alpha + beta * price_data[ticker2])
    
    def test_stationarity(self, series: pd.Series) -> Tuple[bool, float]:
        """
        Test if a time series is stationary using ADF test.
        
        Args:
            series: Time series data
            
        Returns:
            Tuple of (is_stationary, p_value)
        """
        result = adfuller(series.dropna())
        pvalue = result[1]
        return pvalue < 0.05, pvalue
    
    def detect_market_regime(
        self, 
        spread: pd.Series, 
        lookback: int = 60
    ) -> pd.Series:
        """
        Detect market regime based on spread characteristics.
        
        Args:
            spread: Spread time series
            lookback: Lookback window
            
        Returns:
            Series of regime states (1 = mean-reverting, 0 = random walk, -1 = trending)
        """
        # Calculate Hurst exponent for rolling windows
        regimes = pd.Series(index=spread.index)
        
        for i in range(lookback, len(spread)):
            window = spread.iloc[i-lookback:i]
            h = self._calculate_hurst_exponent(window)
            
            # Classify regime:
            # h < 0.4: mean-reverting (good for pairs trading)
            # 0.4 <= h <= 0.6: random walk
            # h > 0.6: trending (not good for pairs trading)
            if h < 0.4:
                regimes.iloc[i] = 1  # Mean-reverting
            elif h > 0.6:
                regimes.iloc[i] = -1  # Trending
            else:
                regimes.iloc[i] = 0  # Random walk
        
        return regimes
    
    def _calculate_hurst_exponent(self, series: pd.Series, max_lag: int = 20) -> float:
        """
        Calculate the Hurst exponent of a time series.
        
        Args:
            series: Time series data
            max_lag: Maximum lag to consider
            
        Returns:
            Hurst exponent
        """
        series = series.dropna()
        
        # Return 0.5 (random walk) if series is too short
        if len(series) < max_lag + 10:
            return 0.5
        
        # Calculate range/standard deviation for different lags
        lags = range(2, max_lag)
        rs_values = []
        
        for lag in lags:
            # Cut the series to a multiple of lag
            cut_series = series[:len(series) - (len(series) % lag)]
            
            # Reshape the series and calculate means
            values = cut_series.values.reshape(-1, lag)
            means = values.mean(axis=1)
            
            # Calculate deviations from mean
            deviations = np.zeros_like(values)
            for i in range(lag):
                deviations[:, i] = values[:, i] - means
            
            # Calculate cumulative deviations
            z = np.cumsum(deviations, axis=1)
            
            # Calculate range (max - min of cumulative deviations)
            r = np.max(z, axis=1) - np.min(z, axis=1)
            
            # Calculate standard deviation
            s = np.std(deviations, axis=1)
            
            # Calculate R/S ratio
            rs = r / s
            rs_values.append(np.mean(rs))
        
        # Fit a line to log-log plot
        x = np.log10(lags)
        y = np.log10(rs_values)
        
        # Linear regression
        slope, _ = np.polyfit(x, y, 1)
        
        return slope
    
    def calculate_adaptive_thresholds(
        self, 
        zscore: pd.Series, 
        lookback: int = 252
    ) -> Dict[str, pd.Series]:
        """
        Calculate adaptive thresholds based on historical z-score volatility.
        
        Args:
            zscore: Z-score time series
            lookback: Lookback window
            
        Returns:
            Dictionary with adaptive threshold series
        """
        # Calculate z-score volatility (standard deviation of absolute z-score)
        zscore_abs = zscore.abs()
        zscore_vol = zscore_abs.rolling(window=lookback).std()
        
        # Base thresholds
        base_entry = self.zscore_entry_threshold
        base_exit = self.zscore_exit_threshold
        base_stop = self.stop_loss_threshold
        
        # Calculate adaptive thresholds
        # In high volatility regimes, widen thresholds
        # In low volatility regimes, narrow thresholds
        vol_scale = zscore_vol / zscore_vol.mean()
        vol_scale = vol_scale.clip(0.8, 1.5)  # Limit scaling
        
        entry_threshold = base_entry * vol_scale
        exit_threshold = base_exit * vol_scale
        stop_threshold = base_stop * vol_scale
        
        return {
            'entry': entry_threshold,
            'exit': exit_threshold,
            'stop': stop_threshold
        }
    
    def generate_signals(
        self, 
        price_data: pd.DataFrame, 
        ticker1: str, 
        ticker2: str,
        use_adaptive_thresholds: bool = None,
        use_regime_filter: bool = True
    ) -> pd.DataFrame:
        """
        Generate trading signals for a pair.
        
        Args:
            price_data: DataFrame of price data
            ticker1: First ticker symbol
            ticker2: Second ticker symbol
            use_adaptive_thresholds: Whether to use adaptive thresholds
            use_regime_filter: Whether to filter signals based on regime
            
        Returns:
            DataFrame with signals
        """
        # Use instance setting if not specified
        if use_adaptive_thresholds is None:
            use_adaptive_thresholds = self.use_adaptive_thresholds
        
        # Calculate hedge ratio and spread
        alpha, beta = self.calculate_hedge_ratio(price_data, ticker1, ticker2)
        spread = self.calculate_spread(price_data, ticker1, ticker2, alpha, beta)
        
        # Calculate spread half-life
        half_life = self.calculate_half_life(spread)
        logger.info(f"Half-life for {ticker1}-{ticker2}: {half_life:.1f} days")
        
        # Store half-life
        self.pair_half_lives[(ticker1, ticker2)] = half_life
        
        # Calculate z-score
        zscore = calculate_zscore(spread, window=self.lookback_period)
        
        # Detect market regime
        if use_regime_filter:
            regime = self.detect_market_regime(spread)
            self.regime_states[(ticker1, ticker2)] = regime
        else:
            regime = pd.Series(1, index=price_data.index)  # Always mean-reverting
        
        # Calculate adaptive thresholds if needed
        if use_adaptive_thresholds:
            thresholds = self.calculate_adaptive_thresholds(zscore)
            self.adaptive_thresholds[(ticker1, ticker2)] = thresholds
            
            entry_threshold = thresholds['entry']
            exit_threshold = thresholds['exit']
            stop_threshold = thresholds['stop']
        else:
            entry_threshold = pd.Series(self.zscore_entry_threshold, index=price_data.index)
            exit_threshold = pd.Series(self.zscore_exit_threshold, index=price_data.index)
            stop_threshold = pd.Series(self.stop_loss_threshold, index=price_data.index)
        
        # Generate signals
        signals = pd.DataFrame(index=price_data.index)
        signals['spread'] = spread
        signals['zscore'] = zscore
        signals['hedge_ratio'] = beta
        signals['alpha'] = alpha
        signals['half_life'] = half_life
        signals['regime'] = regime
        
        # Only generate signals in mean-reverting regime
        is_mean_reverting = (regime == 1)
        
        # Entry signals
        long_entry_condition = (zscore < -entry_threshold) & (zscore.shift(1) >= -entry_threshold) & is_mean_reverting
        short_entry_condition = (zscore > entry_threshold) & (zscore.shift(1) <= entry_threshold) & is_mean_reverting
        
        signals['long_entry'] = long_entry_condition
        signals['short_entry'] = short_entry_condition
        
        # Exit signals
        signals['long_exit'] = (zscore > -exit_threshold) | (zscore > stop_threshold)
        signals['short_exit'] = (zscore < exit_threshold) | (zscore < -stop_threshold)
        
        # Position sizing based on z-score magnitude
        signals['position_scale'] = 1.0
        
        # Scale based on z-score (higher absolute z-score means stronger signal)
        for i in range(len(signals)):
            z = signals['zscore'].iloc[i]
            if abs(z) > entry_threshold.iloc[i]:
                # Scale from 1.0 at entry_threshold to 2.0 at 2*entry_threshold
                scale = min(1.0 + (abs(z) - entry_threshold.iloc[i]) / entry_threshold.iloc[i], 2.0)
                signals['position_scale'].iloc[i] = scale
        
        return signals
    
    def backtest(
        self, 
        price_data: pd.DataFrame, 
        pairs: Optional[List[Tuple[str, str, float]]] = None, 
        initial_capital: float = 100000.0,
        position_size: float = 0.05
    ) -> Dict:
        """
        Backtest the pairs trading strategy.
        
        Args:
            price_data: DataFrame of price data
            pairs: Optional list of pairs to trade (if None, will use self.pairs)
            initial_capital: Initial capital for the backtest
            position_size: Position size as a fraction of capital
            
        Returns:
            Dictionary with backtest results
        """
        if pairs is None:
            pairs = self.pairs
            
        if not pairs:
            logger.warning("No pairs to trade. Run find_cointegrated_pairs first.")
            return {}
        
        # Initialize backtest variables
        capital = initial_capital
        positions = {}
        trades = []
        daily_returns = []
        
        # Track portfolio value
        portfolio_values = pd.Series(index=price_data.index, dtype=float)
        portfolio_values.iloc[0] = capital
        
        # For each day in the backtest period
        for day_idx in range(1, len(price_data)):
            day = price_data.index[day_idx]
            daily_pnl = 0.0
            
            # Process existing positions
            for pair_id, position in list(positions.items()):
                ticker1, ticker2 = pair_id
                entry_day = position['entry_day']
                entry_idx = price_data.index.get_loc(entry_day)
                position_days = day_idx - entry_idx
                
                # Get current z-score
                hedge_ratio = position['hedge_ratio']
                spread = self.calculate_spread(
                    price_data.iloc[day_idx:day_idx+1], 
                    ticker1, 
                    ticker2, 
                    hedge_ratio[0], 
                    hedge_ratio[1]
                ).iloc[0]
                
                zscore = (spread - position['mean']) / position['std']
                
                # Check if we should exit the position
                should_exit = False
                
                if position['position'] == 'long':
                    # Exit long if z-score reverts back or exceeds stop loss
                    if zscore > -position['exit'] or zscore > position['stop']:
                        should_exit = True
                elif position['position'] == 'short':
                    # Exit short if z-score reverts back or exceeds stop loss
                    if zscore < position['exit'] or zscore < -position['stop']:
                        should_exit = True
                
                # Also exit if position has been held for too long
                if position_days >= self.max_position_days:
                    should_exit = True
                
                # Calculate P&L if exiting
                if should_exit:
                    # Calculate price changes
                    price1_change = (price_data.iloc[day_idx][ticker1] / price_data.iloc[entry_idx][ticker1]) - 1
                    price2_change = (price_data.iloc[day_idx][ticker2] / price_data.iloc[entry_idx][ticker2]) - 1
                    
                    # Apply direction and hedge ratio
                    if position['position'] == 'long':
                        # Long spread: Long asset1, Short asset2*hedge_ratio
                        pnl = price1_change - (hedge_ratio[0] * price2_change)
                    else:
                        # Short spread: Short asset1, Long asset2*hedge_ratio
                        pnl = -price1_change + (hedge_ratio[0] * price2_change)
                    
                    # Apply position size
                    position_value = position['capital']
                    pnl_value = position_value * pnl
                    capital += position_value + pnl_value
                    daily_pnl += pnl_value
                    
                    # Record the trade
                    trades.append({
                        'pair': pair_id,
                        'position': position['position'],
                        'entry_date': entry_day,
                        'exit_date': day,
                        'duration': position_days,
                        'entry_zscore': position['entry_zscore'],
                        'exit_zscore': zscore,
                        'pnl': pnl,
                        'pnl_value': pnl_value
                    })
                    
                    # Remove the position
                    del positions[pair_id]
            
            # Look for new entry signals
            for ticker1, ticker2, _ in pairs:
                # Skip if we already have a position in this pair
                if (ticker1, ticker2) in positions:
                    continue
                
                # Generate signals for this pair
                signals = self.generate_signals(
                    price_data.iloc[:day_idx+1], 
                    ticker1, 
                    ticker2
                )
                
                # Get the latest signals
                latest_signals = signals.iloc[-1]
                
                # Check for entry signals
                if latest_signals['long_entry']:
                    # Calculate mean and std for z-score
                    spread_window = signals['spread'].iloc[-self.lookback_period:]
                    mean = spread_window.mean()
                    std = spread_window.std()
                    
                    # Calculate position size
                    position_capital = capital * position_size
                    capital -= position_capital
                    
                    # Open long position
                    positions[(ticker1, ticker2)] = {
                        'position': 'long',
                        'entry_day': day,
                        'hedge_ratio': (latest_signals['alpha'], latest_signals['hedge_ratio']),
                        'entry_zscore': latest_signals['zscore'],
                        'mean': mean,
                        'std': std,
                        'capital': position_capital
                    }
                    
                elif latest_signals['short_entry']:
                    # Calculate mean and std for z-score
                    spread_window = signals['spread'].iloc[-self.lookback_period:]
                    mean = spread_window.mean()
                    std = spread_window.std()
                    
                    # Calculate position size
                    position_capital = capital * position_size
                    capital -= position_capital
                    
                    # Open short position
                    positions[(ticker1, ticker2)] = {
                        'position': 'short',
                        'entry_day': day,
                        'hedge_ratio': (latest_signals['alpha'], latest_signals['hedge_ratio']),
                        'entry_zscore': latest_signals['zscore'],
                        'mean': mean,
                        'std': std,
                        'capital': position_capital
                    }
            
            # Record portfolio value
            portfolio_values[day] = capital + sum([p['capital'] for p in positions.values()])
            
            # Calculate daily return
            if day_idx > 0:
                daily_return = (portfolio_values[day] / portfolio_values[price_data.index[day_idx-1]]) - 1
                daily_returns.append(daily_return)
        
        # Close any remaining positions at the end of the backtest
        for pair_id, position in list(positions.items()):
            ticker1, ticker2 = pair_id
            entry_day = position['entry_day']
            entry_idx = price_data.index.get_loc(entry_day)
            position_days = len(price_data) - 1 - entry_idx
            
            # Calculate price changes
            price1_change = (price_data.iloc[-1][ticker1] / price_data.loc[entry_day][ticker1]) - 1
            price2_change = (price_data.iloc[-1][ticker2] / price_data.loc[entry_day][ticker2]) - 1
            
            hedge_ratio = position['hedge_ratio']
            
            # Apply direction and hedge ratio
            if position['position'] == 'long':
                pnl = price1_change - (hedge_ratio[0] * price2_change)
            else:
                pnl = -price1_change + (hedge_ratio[0] * price2_change)
            
            # Apply position size
            position_value = position['capital']
            pnl_value = position_value * pnl
            capital += position_value + pnl_value
            
            # Record the trade
            spread = self.calculate_spread(
                price_data.iloc[-1:], 
                ticker1, 
                ticker2, 
                hedge_ratio[0], 
                hedge_ratio[1]
            ).iloc[0]
            
            exit_zscore = (spread - position['mean']) / position['std']
            
            trades.append({
                'pair': pair_id,
                'position': position['position'],
                'entry_date': entry_day,
                'exit_date': price_data.index[-1],
                'duration': position_days,
                'entry_zscore': position['entry_zscore'],
                'exit_zscore': exit_zscore,
                'pnl': pnl,
                'pnl_value': pnl_value
            })
        
        # Calculate performance metrics
        portfolio_returns = portfolio_values.pct_change().dropna()
        
        results = {
            'portfolio_values': portfolio_values,
            'portfolio_returns': portfolio_returns,
            'trades': trades,
            'final_capital': capital,
            'total_return': (capital / initial_capital) - 1,
            'sharpe_ratio': np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if daily_returns else 0,
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
    
    def plot_pair_analysis(
        self, 
        price_data: pd.DataFrame, 
        ticker1: str, 
        ticker2: str,
        normalize: bool = True
    ):
        """
        Plot analysis for a trading pair.
        
        Args:
            price_data: DataFrame of price data
            ticker1: First ticker symbol
            ticker2: Second ticker symbol
            normalize: Whether to normalize prices
        """
        # Calculate hedge ratio and spread
        hedge_ratio = self.calculate_hedge_ratio(price_data, ticker1, ticker2)
        spread = self.calculate_spread(price_data, ticker1, ticker2, hedge_ratio[0], hedge_ratio[1])
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot normalized prices
        if normalize:
            normalized_price1 = price_data[ticker1] / price_data[ticker1].iloc[0]
            normalized_price2 = price_data[ticker2] / price_data[ticker2].iloc[0]
            ax1.plot(normalized_price1, label=ticker1)
            ax1.plot(normalized_price2, label=ticker2)
            ax1.set_title(f"Normalized Prices: {ticker1} vs {ticker2}")
        else:
            ax1.plot(price_data[ticker1], label=ticker1)
            ax1.plot(price_data[ticker2], label=ticker2)
            ax1.set_title(f"Prices: {ticker1} vs {ticker2}")
            
        ax1.legend()
        ax1.grid(True)
        
        # Plot spread
        ax2.plot(spread)
        ax2.set_title(f"Spread: {ticker1} - {hedge_ratio[1]:.2f}*{ticker2}")
        ax2.axhline(y=spread.mean(), color='r', linestyle='--', label="Mean")
        ax2.axhline(y=spread.mean() + 2*spread.std(), color='g', linestyle='--', label="Mean + 2*Std")
        ax2.axhline(y=spread.mean() - 2*spread.std(), color='g', linestyle='--', label="Mean - 2*Std")
        ax2.legend()
        ax2.grid(True)
        
        # Plot z-score
        zscore = calculate_zscore(spread, window=self.lookback_period)
        ax3.plot(zscore)
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
        """
        if not results:
            logger.warning("No backtest results to plot.")
            return
        
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
        
        return fig, fig2

def run_johansen_test(price_data: pd.DataFrame, significance_level: float = 0.05) -> pd.DataFrame:
    """
    Run the Johansen test for cointegration of multiple series.
    
    Args:
        price_data: DataFrame of price data
        significance_level: Significance level for the test
        
    Returns:
        DataFrame with results
    """
    try:
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
        
        # Convert to log prices if needed
        if price_data.min().min() > 0:  # Ensure all prices are positive
            log_prices = np.log(price_data)
        else:
            log_prices = price_data
        
        # Run Johansen test
        result = coint_johansen(log_prices, det_order=0, k_ar_diff=1)
        
        # Create DataFrame with results
        trace_stats = result.lr1
        critical_values = result.cvt[:, 0]  # 90% critical values
        
        n = len(price_data.columns)
        results_df = pd.DataFrame({
            'Rank': range(n),
            'Trace Statistic': trace_stats,
            'Critical Value (95%)': critical_values,
            'Is Significant': trace_stats > critical_values
        })
        
        # Determine cointegration rank
        cointegration_rank = sum(results_df['Is Significant'])
        logger.info(f"Johansen test suggests cointegration rank of {cointegration_rank}")
        
        return results_df
    
    except ImportError:
        logger.error("Could not run Johansen test. Make sure statsmodels is installed.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error in Johansen test: {str(e)}")
        return pd.DataFrame() 