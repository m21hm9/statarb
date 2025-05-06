import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
from scipy.stats import norm
import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PositionSizer:
    def __init__(
        self, 
        max_position_size: float = 0.05,
        max_pair_capital: float = 0.2,
        volatility_scaling: bool = True,
        target_volatility: float = 0.01,
        leverage_limit: float = 2.0,
        position_sizing_method: str = 'inverse_volatility',
        max_correlation: float = 0.5,
        max_drawdown_limit: float = 0.15,
        dynamic_position_sizing: bool = True,
        zscore_scaling: bool = True,
        half_life_scaling: bool = True
    ):
        self.max_position_size = max_position_size
        self.max_pair_capital = max_pair_capital
        self.volatility_scaling = volatility_scaling
        self.target_volatility = target_volatility
        self.leverage_limit = leverage_limit
        self.position_sizing_method = position_sizing_method
        self.max_correlation = max_correlation
        self.max_drawdown_limit = max_drawdown_limit
        self.dynamic_position_sizing = dynamic_position_sizing
        self.zscore_scaling = zscore_scaling
        self.half_life_scaling = half_life_scaling
        
        # Track portfolio state
        self.current_drawdown = 0.0
        self.high_water_mark = 1.0
        self.pair_allocations = {}
        self.pair_performances = {}
    
    def calculate_position_size(
        self, 
        capital: float,
        volatility: Optional[float] = None,
        signal_strength: Optional[float] = None,
        z_score: Optional[float] = None,
        half_life: Optional[float] = None,
        pair_id: Optional[str] = None,
        pair_sharpe: Optional[float] = None,
        current_drawdown: Optional[float] = None
    ) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            capital: Total portfolio capital
            volatility: Asset volatility (daily)
            signal_strength: Optional signal strength (0-1)
            z_score: Optional z-score for mean-reversion
            half_life: Optional half-life of mean reversion
            pair_id: Optional identifier for the pair
            pair_sharpe: Optional Sharpe ratio of the pair
            current_drawdown: Optional current drawdown percentage
            
        Returns:
            Position size in currency units
        """
        # Start with maximum position size
        position_fraction = self.max_position_size
        
        # Adjust position size based on method
        if self.position_sizing_method == 'equal':
            # Equal position sizing - use max_position_size directly
            pass
            
        elif self.position_sizing_method == 'inverse_volatility' and volatility is not None and volatility > 0:
            # Inverse volatility weighting
            if self.volatility_scaling:
                vol_scale = self.target_volatility / volatility
                # Cap the scaling to avoid excessive leverage
                vol_scale = min(vol_scale, 2.0)
                position_fraction *= vol_scale
            
        elif self.position_sizing_method == 'kelly' and pair_sharpe is not None and volatility is not None:
            # Kelly criterion (using Sharpe as an approximation of edge)
            # f* = (edge / volatility^2) = Sharpe / volatility
            kelly_fraction = pair_sharpe / (volatility * 252**0.5)  # Annualized Sharpe
            
            # Apply a fraction of Kelly (half-Kelly is more conservative)
            kelly_fraction = kelly_fraction * 0.5
            
            # Cap the Kelly fraction
            kelly_fraction = min(kelly_fraction, self.max_position_size)
            kelly_fraction = max(kelly_fraction, 0.01)  # Minimum position size
            
            position_fraction = kelly_fraction
        
        # Apply Z-score scaling if enabled
        if self.zscore_scaling and z_score is not None:
            # Use sigmoid function to scale by z-score
            z_scale = 2.0 / (1.0 + np.exp(-abs(z_score) + 2.0))
            position_fraction *= z_scale
        
        # Apply half-life scaling if enabled
        if self.half_life_scaling and half_life is not None:
            # Prefer pairs with shorter half-lives (faster mean reversion)
            if half_life > 5 and half_life < 100:
                # Scale from 1.0 at half_life=5 to 0.5 at half_life=100
                half_life_scale = 1.0 - 0.5 * ((half_life - 5) / 95)
                position_fraction *= half_life_scale
            elif half_life >= 100:
                # Strongly reduce position size for very slow mean-reverting pairs
                position_fraction *= 0.5
        
        # Apply drawdown control if enabled
        if self.dynamic_position_sizing and current_drawdown is not None:
            # Reduce position size as drawdown approaches limit
            if current_drawdown > 0:
                drawdown_ratio = current_drawdown / self.max_drawdown_limit
                if drawdown_ratio < 0.5:
                    # No reduction for small drawdowns
                    pass
                elif drawdown_ratio < 0.8:
                    # Linear reduction from 100% to 50% of position size
                    drawdown_scale = 1.0 - (drawdown_ratio - 0.5) * 1.0
                    position_fraction *= drawdown_scale
                else:
                    # Strong reduction for drawdowns approaching limit
                    position_fraction *= 0.5
        
        # Calculate position size in currency
        position_size = capital * position_fraction
        
        # Ensure position size doesn't exceed max pair capital
        max_size = capital * self.max_pair_capital
        position_size = min(position_size, max_size)
        
        # Store allocation for pair if pair_id is provided
        if pair_id is not None:
            self.pair_allocations[pair_id] = position_size / capital
        
        return position_size
    
    def calculate_pairs_position_sizes(
        self, 
        capital: float,
        price1: float,
        price2: float,
        hedge_ratio: float,
        volatility1: Optional[float] = None,
        volatility2: Optional[float] = None,
        correlation: Optional[float] = None,
        z_score: Optional[float] = None,
        half_life: Optional[float] = None,
        pair_id: Optional[str] = None,
        pair_sharpe: Optional[float] = None,
        current_drawdown: Optional[float] = None
    ) -> Dict:
        """
        Calculate position sizes for a pairs trade.
        
        Args:
            capital: Total portfolio capital
            price1: Price of first asset
            price2: Price of second asset
            hedge_ratio: Hedge ratio for the pair
            volatility1: Volatility of first asset
            volatility2: Volatility of second asset
            correlation: Correlation between assets
            z_score: Z-score of the spread
            half_life: Half-life of mean reversion
            pair_id: Identifier for the pair
            pair_sharpe: Sharpe ratio of the pair
            current_drawdown: Current drawdown percentage
            
        Returns:
            Dictionary with position sizes
        """
        # Calculate pair volatility if volatilities and correlation are provided
        if volatility1 is not None and volatility2 is not None and correlation is not None:
            pair_volatility = np.sqrt(
                volatility1**2 + (hedge_ratio * volatility2)**2 - 
                2 * correlation * volatility1 * hedge_ratio * volatility2
            )
        else:
            pair_volatility = None
        
        # Calculate position size for the pair
        pair_position = self.calculate_position_size(
            capital, 
            pair_volatility, 
            z_score=z_score,
            half_life=half_life,
            pair_id=pair_id,
            pair_sharpe=pair_sharpe,
            current_drawdown=current_drawdown
        )
        
        # Determine notional exposure for each leg
        notional1 = pair_position / (1 + hedge_ratio * price2 / price1)
        notional2 = notional1 * hedge_ratio * price2 / price1
        
        # Calculate quantity for each asset
        quantity1 = notional1 / price1
        quantity2 = notional2 / price2
        
        # Calculate total notional exposure
        total_notional = notional1 + notional2
        
        # Return position sizes
        positions = {
            'pair_position': pair_position,
            'notional1': notional1,
            'notional2': notional2,
            'quantity1': quantity1,
            'quantity2': quantity2,
            'total_notional': total_notional,
            'leverage': total_notional / pair_position,
            'allocation': pair_position / capital
        }
        
        return positions
    
    def update_portfolio_state(
        self,
        portfolio_value: float,
        previous_value: float,
        pair_performances: Dict[str, float]
    ):
        """
        Update portfolio state for dynamic position sizing.
        
        Args:
            portfolio_value: Current portfolio value
            previous_value: Previous portfolio value
            pair_performances: Dictionary of pair-level returns
        """
        # Update high water mark
        if portfolio_value > self.high_water_mark:
            self.high_water_mark = portfolio_value
        
        # Update current drawdown
        self.current_drawdown = max(0, 1 - portfolio_value / self.high_water_mark)
        
        # Update pair performances
        for pair_id, perf in pair_performances.items():
            if pair_id not in self.pair_performances:
                self.pair_performances[pair_id] = []
            self.pair_performances[pair_id].append(perf)
    
    def calculate_pair_diversification_score(
        self,
        new_pair_id: str,
        returns_data: Dict[str, pd.Series],
        existing_pairs: List[str]
    ) -> float:
        """
        Calculate a diversification score for adding a new pair to the portfolio.
        
        Args:
            new_pair_id: ID of the new pair
            returns_data: Dictionary of returns series by pair
            existing_pairs: List of existing pairs in the portfolio
            
        Returns:
            Diversification score (higher is better)
        """
        if not existing_pairs or new_pair_id not in returns_data:
            return 1.0  # Maximum score if no existing pairs or no data
        
        new_returns = returns_data[new_pair_id]
        
        # Calculate average correlation with existing pairs
        total_corr = 0.0
        count = 0
        
        for pair_id in existing_pairs:
            if pair_id in returns_data:
                existing_returns = returns_data[pair_id]
                
                # Calculate correlation if there's enough overlap
                common_index = new_returns.index.intersection(existing_returns.index)
                if len(common_index) > 20:  # Minimum required data points
                    corr = new_returns.loc[common_index].corr(existing_returns.loc[common_index])
                    total_corr += abs(corr)  # Use absolute correlation
                    count += 1
        
        # Calculate average correlation
        avg_corr = total_corr / count if count > 0 else 0.0
        
        # Convert to diversification score (1 - avg_corr)
        div_score = 1.0 - avg_corr
        
        return div_score
    
    def select_optimal_pairs(
        self,
        pair_candidates: List[str],
        returns_data: Dict[str, pd.Series],
        half_lives: Dict[str, float],
        sharpes: Dict[str, float],
        max_pairs: int = 10
    ) -> List[str]:
        """
        Select optimal pairs for the portfolio based on diversification and characteristics.
        
        Args:
            pair_candidates: List of candidate pair IDs
            returns_data: Dictionary of returns series by pair
            half_lives: Dictionary of half-lives by pair
            sharpes: Dictionary of Sharpe ratios by pair
            max_pairs: Maximum number of pairs to select
            
        Returns:
            List of selected pair IDs
        """
        if len(pair_candidates) <= max_pairs:
            return pair_candidates
        
        selected_pairs = []
        remaining_pairs = pair_candidates.copy()
        
        # Sort by Sharpe ratio and select the first pair
        sorted_by_sharpe = sorted(remaining_pairs, key=lambda p: -sharpes.get(p, 0))
        best_pair = sorted_by_sharpe[0]
        selected_pairs.append(best_pair)
        remaining_pairs.remove(best_pair)
        
        # Select remaining pairs based on diversification score
        while len(selected_pairs) < max_pairs and remaining_pairs:
            best_score = -1
            best_pair = None
            
            for pair_id in remaining_pairs:
                # Calculate diversification score
                div_score = self.calculate_pair_diversification_score(
                    pair_id, returns_data, selected_pairs
                )
                
                # Calculate half-life score (prefer lower half-lives)
                half_life = half_lives.get(pair_id, 50)
                half_life_score = max(0, 1 - (half_life / 100)) if half_life < 100 else 0
                
                # Calculate Sharpe score
                sharpe_score = min(1, sharpes.get(pair_id, 0) / 2)  # Normalize to 0-1
                
                # Combine scores with weights
                combined_score = (
                    0.5 * div_score +    # 50% weight on diversification
                    0.3 * sharpe_score + # 30% weight on Sharpe
                    0.2 * half_life_score # 20% weight on half-life
                )
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_pair = pair_id
            
            if best_pair:
                selected_pairs.append(best_pair)
                remaining_pairs.remove(best_pair)
            else:
                break
        
        return selected_pairs
    
    def adjust_for_portfolio_constraints(
        self, 
        proposed_positions: Dict[str, Dict],
        current_positions: Dict[str, Dict],
        capital: float,
        returns_data: Optional[Dict[str, pd.Series]] = None,
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> Dict[str, Dict]:
        """
        Adjust proposed positions to meet portfolio constraints.
        
        Args:
            proposed_positions: Dictionary of proposed positions by pair
            current_positions: Dictionary of current positions by pair
            capital: Total portfolio capital
            returns_data: Optional dictionary of returns data by pair
            correlation_matrix: Optional correlation matrix of pairs
            
        Returns:
            Dictionary of adjusted positions
        """
        # Calculate total proposed notional
        total_proposed_notional = sum(
            pos['total_notional'] for pos in proposed_positions.values()
        )
        
        # Check if leverage limit is exceeded
        if total_proposed_notional / capital > self.leverage_limit:
            # Scale down all positions proportionally
            scale_factor = (self.leverage_limit * capital) / total_proposed_notional
            
            for pair_id, position in proposed_positions.items():
                for key in ['pair_position', 'notional1', 'notional2', 'total_notional']:
                    position[key] *= scale_factor
                    
                position['quantity1'] *= scale_factor
                position['quantity2'] *= scale_factor
                position['leverage'] = position['total_notional'] / position['pair_position']
                position['allocation'] = position['pair_position'] / capital
        
        # Apply position limits based on drawdown
        if self.dynamic_position_sizing and self.current_drawdown > 0:
            # Calculate a global scale factor based on drawdown
            drawdown_ratio = self.current_drawdown / self.max_drawdown_limit
            if drawdown_ratio > 0.8:
                # Apply global scaling to all positions
                global_scale = 1.0 - (drawdown_ratio - 0.8) * 2.5  # Linear scaling to 0 at max drawdown
                global_scale = max(0.2, global_scale)  # Don't go below 20% allocation
                
                for pair_id, position in proposed_positions.items():
                    for key in ['pair_position', 'notional1', 'notional2', 'total_notional']:
                        position[key] *= global_scale
                        
                    position['quantity1'] *= global_scale
                    position['quantity2'] *= global_scale
                    position['allocation'] = position['pair_position'] / capital
        
        # Check correlation constraints if correlation matrix is provided
        if correlation_matrix is not None:
            # Identify highly correlated pairs
            high_corr_pairs = []
            
            # Look at all pairs of pairs
            pair_ids = list(proposed_positions.keys())
            for i, pair1 in enumerate(pair_ids):
                for pair2 in pair_ids[i+1:]:
                    if pair1 in correlation_matrix.index and pair2 in correlation_matrix.columns:
                        corr = abs(correlation_matrix.loc[pair1, pair2])
                        if corr > self.max_correlation:
                            high_corr_pairs.append((pair1, pair2, corr))
            
            # Sort by correlation (highest first)
            high_corr_pairs.sort(key=lambda x: -x[2])
            
            # Reduce position size for highly correlated pairs
            for pair1, pair2, corr in high_corr_pairs:
                if pair1 in proposed_positions and pair2 in proposed_positions:
                    # Scale down both positions based on excess correlation
                    excess_corr = (corr - self.max_correlation) / (1 - self.max_correlation)
                    scale_factor = 1.0 - excess_corr * 0.5  # Reduce by up to 50%
                    
                    # Apply to both positions
                    for pair_id in [pair1, pair2]:
                        position = proposed_positions[pair_id]
                        for key in ['pair_position', 'notional1', 'notional2', 'total_notional']:
                            position[key] *= scale_factor
                            
                        position['quantity1'] *= scale_factor
                        position['quantity2'] *= scale_factor
                        position['allocation'] = position['pair_position'] / capital
        
        return proposed_positions

class StopLossManager:
    def __init__(
        self, 
        z_score_stop: float = 3.0,
        max_loss_percentage: float = 0.05,
        time_stop_days: int = 30,
        trailing_stop_percentage: Optional[float] = None
    ):
        self.z_score_stop = z_score_stop
        self.max_loss_percentage = max_loss_percentage
        self.time_stop_days = time_stop_days
        self.trailing_stop_percentage = trailing_stop_percentage
    
    def check_stop_loss(
        self, 
        current_z_score: float,
        trade_z_score: float,
        entry_date: datetime.date,
        current_date: datetime.date,
        current_value: float,
        entry_value: float,
        high_water_mark: Optional[float] = None
    ) -> Tuple[bool, str]:

        if abs(current_z_score) > self.z_score_stop:
            return True, "z-score"

        if current_value / entry_value - 1 < -self.max_loss_percentage:
            return True, "max loss"
        
        # Check time stop
        holding_period = (current_date - entry_date).days
        if holding_period > self.time_stop_days:
            return True, "time stop"
        
        # Check trailing stop if applicable
        if (self.trailing_stop_percentage is not None and
            high_water_mark is not None and
            high_water_mark > entry_value):
            
            # Calculate drawdown from high-water mark
            drawdown = 1 - current_value / high_water_mark
            
            if drawdown > self.trailing_stop_percentage:
                return True, "trailing stop"
        
        # No stop-loss condition met
        return False, ""
    
    def calculate_stop_price(
        self, 
        entry_price: float,
        stop_type: str = 'fixed',
        stop_percentage: Optional[float] = None,
        atr: Optional[float] = None,
        atr_multiplier: float = 2.0
    ) -> float:
        """
        Calculate stop-loss price.
        
        Args:
            entry_price: Entry price
            stop_type: Type of stop ('fixed', 'percentage', 'atr')
            stop_percentage: Percentage for percentage-based stop
            atr: Average True Range for ATR-based stop
            atr_multiplier: Multiplier for ATR-based stop
            
        Returns:
            Stop-loss price
        """
        if stop_type == 'percentage' and stop_percentage is not None:
            stop_price = entry_price * (1 - stop_percentage)
        elif stop_type == 'atr' and atr is not None:
            stop_price = entry_price - atr * atr_multiplier
        else:
            # Default to fixed percentage stop
            stop_price = entry_price * (1 - self.max_loss_percentage)
        
        return stop_price

class RiskAnalyzer:
    """
    Risk analysis for statistical arbitrage strategies.
    """
    
    def __init__(
        self, 
        confidence_level: float = 0.95,
        var_window: int = 252,
        stress_test_scenarios: Optional[List[Dict]] = None
    ):
        """
        Initialize the risk analyzer.
        
        Args:
            confidence_level: Confidence level for VaR and ES calculations
            var_window: Window size for historical VaR
            stress_test_scenarios: Optional list of stress test scenarios
        """
        self.confidence_level = confidence_level
        self.var_window = var_window
        
        # Default stress test scenarios if none provided
        if stress_test_scenarios is None:
            self.stress_test_scenarios = [
                {'name': 'Market Crash', 'market_return': -0.15, 'volatility_multiplier': 2.0, 'correlation_increase': 0.2},
                {'name': 'Sector Rotation', 'market_return': -0.05, 'sector_rotation_strength': 0.1},
                {'name': 'Liquidity Crisis', 'spread_widening': 0.02, 'volume_reduction': 0.5}
            ]
        else:
            self.stress_test_scenarios = stress_test_scenarios
    
    def calculate_var(
        self, 
        returns: pd.Series,
        method: str = 'historical',
        portfolio_value: float = 1.0
    ) -> float:
        """
        Calculate Value at Risk for a return series.
        
        Args:
            returns: Series of returns
            method: Method for VaR calculation ('historical', 'parametric', 'monte_carlo')
            portfolio_value: Portfolio value
            
        Returns:
            Value at Risk
        """
        if method == 'historical':
            # Historical simulation method
            var = -np.percentile(returns, 100 * (1 - self.confidence_level))
            
        elif method == 'parametric':
            # Parametric (variance-covariance) method
            mean = returns.mean()
            std = returns.std()
            var = -mean - std * norm.ppf(self.confidence_level)
            
        elif method == 'monte_carlo':
            # Monte Carlo simulation
            mean = returns.mean()
            std = returns.std()
            n_simulations = 10000
            simulated_returns = np.random.normal(mean, std, n_simulations)
            var = -np.percentile(simulated_returns, 100 * (1 - self.confidence_level))
            
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        # Convert to monetary value
        var_monetary = portfolio_value * var
        
        return var_monetary
    
    def calculate_expected_shortfall(
        self, 
        returns: pd.Series,
        method: str = 'historical',
        portfolio_value: float = 1.0
    ) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR).
        
        Args:
            returns: Series of returns
            method: Method for ES calculation ('historical', 'parametric', 'monte_carlo')
            portfolio_value: Portfolio value
            
        Returns:
            Expected Shortfall
        """
        if method == 'historical':
            # Historical simulation method
            cutoff = np.percentile(returns, 100 * (1 - self.confidence_level))
            tail_returns = returns[returns <= cutoff]
            es = -tail_returns.mean()
            
        elif method == 'parametric':
            # Parametric method
            mean = returns.mean()
            std = returns.std()
            cutoff = norm.ppf(1 - self.confidence_level)
            es = -mean - std * norm.pdf(cutoff) / (1 - self.confidence_level)
            
        elif method == 'monte_carlo':
            # Monte Carlo simulation
            mean = returns.mean()
            std = returns.std()
            n_simulations = 10000
            simulated_returns = np.random.normal(mean, std, n_simulations)
            cutoff = np.percentile(simulated_returns, 100 * (1 - self.confidence_level))
            tail_returns = simulated_returns[simulated_returns <= cutoff]
            es = -tail_returns.mean()
            
        else:
            raise ValueError(f"Unknown ES method: {method}")
        
        # Convert to monetary value
        es_monetary = portfolio_value * es
        
        return es_monetary
    
    def calculate_portfolio_risk_metrics(
        self, 
        positions: Dict[str, Dict],
        returns_data: pd.DataFrame,
        correlations: pd.DataFrame,
        portfolio_value: float
    ) -> Dict:
        """
        Calculate risk metrics for a portfolio of positions.
        
        Args:
            positions: Dictionary of positions by pair
            returns_data: DataFrame of asset returns
            correlations: Correlation matrix of assets
            portfolio_value: Total portfolio value
            
        Returns:
            Dictionary with risk metrics
        """
        # Initialize weight vector and asset list
        weights = {}
        assets = []
        
        # Extract weights from positions
        for pair_id, position in positions.items():
            ticker1, ticker2 = pair_id
            assets.append(ticker1)
            assets.append(ticker2)
            
            # Calculate weights
            weight1 = position['notional1'] / portfolio_value
            weight2 = -position['notional2'] / portfolio_value  # Negative for short position
            
            weights[ticker1] = weights.get(ticker1, 0) + weight1
            weights[ticker2] = weights.get(ticker2, 0) + weight2
        
        # Create unique asset list
        unique_assets = list(set(assets))
        
        # Extract relevant returns and correlations
        returns_subset = returns_data[unique_assets]
        correlations_subset = correlations.loc[unique_assets, unique_assets]
        
        # Create weight vector
        weight_vector = np.array([weights.get(asset, 0) for asset in unique_assets])
        
        # Calculate portfolio volatility
        asset_vols = returns_subset.std()
        vol_vector = np.array([asset_vols[asset] for asset in unique_assets])
        
        # Calculate portfolio variance
        cov_matrix = np.outer(vol_vector, vol_vector) * correlations_subset.values
        portfolio_variance = weight_vector @ cov_matrix @ weight_vector
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Calculate portfolio expected return
        expected_returns = returns_subset.mean()
        portfolio_expected_return = sum(weights.get(asset, 0) * expected_returns[asset] for asset in unique_assets)
        
        # Calculate portfolio Sharpe ratio (annualized)
        risk_free_rate = 0.0  # Assuming zero risk-free rate
        sharpe_ratio = (portfolio_expected_return - risk_free_rate) / portfolio_volatility
        annualized_sharpe = sharpe_ratio * np.sqrt(252)  # Assuming daily returns
        
        # Calculate contribution to risk
        risk_contribution = np.zeros(len(unique_assets))
        for i, asset in enumerate(unique_assets):
            for j, other_asset in enumerate(unique_assets):
                risk_contribution[i] += weight_vector[i] * weight_vector[j] * cov_matrix[i, j]
        
        risk_contribution = risk_contribution / portfolio_variance
        
        # Calculate VaR and Expected Shortfall
        portfolio_returns = returns_subset @ weight_vector
        var = self.calculate_var(portfolio_returns, 'historical', portfolio_value)
        es = self.calculate_expected_shortfall(portfolio_returns, 'historical', portfolio_value)
        
        # Create risk metrics dictionary
        risk_metrics = {
            'portfolio_volatility': portfolio_volatility,
            'portfolio_expected_return': portfolio_expected_return,
            'annualized_sharpe_ratio': annualized_sharpe,
            'value_at_risk': var,
            'expected_shortfall': es,
            'asset_weights': {asset: weights.get(asset, 0) for asset in unique_assets},
            'risk_contribution': {asset: rc for asset, rc in zip(unique_assets, risk_contribution)}
        }
        
        return risk_metrics
    
    def run_stress_test(
        self, 
        positions: Dict[str, Dict],
        returns_data: pd.DataFrame,
        correlations: pd.DataFrame,
        portfolio_value: float,
        scenario: Dict
    ) -> Dict:
        """
        Run a stress test on the portfolio.
        
        Args:
            positions: Dictionary of positions by pair
            returns_data: DataFrame of asset returns
            correlations: Correlation matrix of assets
            portfolio_value: Total portfolio value
            scenario: Stress test scenario
            
        Returns:
            Dictionary with stress test results
        """
        # Create copies of data to modify
        stressed_returns = returns_data.copy()
        stressed_correlations = correlations.copy()
        
        # Apply market shock
        if 'market_return' in scenario:
            market_beta = 0.1  # Default beta to market
            for col in stressed_returns.columns:
                stressed_returns[col] += scenario['market_return'] * market_beta
        
        # Apply volatility shock
        if 'volatility_multiplier' in scenario:
            for col in stressed_returns.columns:
                col_mean = stressed_returns[col].mean()
                col_std = stressed_returns[col].std() * scenario['volatility_multiplier']
                stressed_returns[col] = np.random.normal(col_mean, col_std, len(stressed_returns))
        
        # Apply correlation shock
        if 'correlation_increase' in scenario:
            for i in range(len(stressed_correlations)):
                for j in range(i+1, len(stressed_correlations)):
                    if i != j:
                        stressed_correlations.iloc[i, j] = min(1.0, stressed_correlations.iloc[i, j] + scenario['correlation_increase'])
                        stressed_correlations.iloc[j, i] = stressed_correlations.iloc[i, j]
        
        # Calculate stressed risk metrics
        stressed_metrics = self.calculate_portfolio_risk_metrics(
            positions, stressed_returns, stressed_correlations, portfolio_value
        )
        
        # Add scenario name to results
        stressed_metrics['scenario'] = scenario['name']
        
        return stressed_metrics
    
    def run_all_stress_tests(
        self, 
        positions: Dict[str, Dict],
        returns_data: pd.DataFrame,
        correlations: pd.DataFrame,
        portfolio_value: float
    ) -> List[Dict]:
        """
        Run all stress tests on the portfolio.
        
        Args:
            positions: Dictionary of positions by pair
            returns_data: DataFrame of asset returns
            correlations: Correlation matrix of assets
            portfolio_value: Total portfolio value
            
        Returns:
            List of dictionaries with stress test results
        """
        stress_test_results = []
        
        # Run base case
        base_metrics = self.calculate_portfolio_risk_metrics(
            positions, returns_data, correlations, portfolio_value
        )
        base_metrics['scenario'] = 'Base Case'
        stress_test_results.append(base_metrics)
        
        # Run each stress test scenario
        for scenario in self.stress_test_scenarios:
            stressed_metrics = self.run_stress_test(
                positions, returns_data, correlations, portfolio_value, scenario
            )
            stress_test_results.append(stressed_metrics)
        
        return stress_test_results
    
    def plot_stress_test_results(self, stress_test_results: List[Dict]) -> plt.Figure:
        """
        Plot stress test results.
        
        Args:
            stress_test_results: List of dictionaries with stress test results
            
        Returns:
            Matplotlib figure
        """
        # Extract data for plotting
        scenarios = [result['scenario'] for result in stress_test_results]
        var_values = [result['value_at_risk'] for result in stress_test_results]
        es_values = [result['expected_shortfall'] for result in stress_test_results]
        volatilities = [result['portfolio_volatility'] for result in stress_test_results]
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot VaR
        ax1.bar(scenarios, var_values)
        ax1.set_title('Value at Risk')
        ax1.set_ylabel('VaR')
        ax1.set_xticklabels(scenarios, rotation=45, ha='right')
        
        # Plot Expected Shortfall
        ax2.bar(scenarios, es_values)
        ax2.set_title('Expected Shortfall')
        ax2.set_ylabel('ES')
        ax2.set_xticklabels(scenarios, rotation=45, ha='right')
        
        # Plot Volatility
        ax3.bar(scenarios, volatilities)
        ax3.set_title('Portfolio Volatility')
        ax3.set_ylabel('Volatility')
        ax3.set_xticklabels(scenarios, rotation=45, ha='right')
        
        plt.tight_layout()
        
        return fig 