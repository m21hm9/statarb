"""
Market Microstructure Adjustments for Statistical Arbitrage

This module handles market microstructure effects such as slippage,
liquidity constraints, and execution costs for statistical arbitrage strategies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import List, Dict, Tuple, Optional, Union, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SlippageModel:
    """
    Model for estimating slippage based on market impact.
    """
    
    def __init__(
        self, 
        impact_factor: float = 0.1,
        fixed_cost: float = 0.0001,
        use_volume: bool = True
    ):
        """
        Initialize the slippage model.
        
        Args:
            impact_factor: Factor for market impact calculation
            fixed_cost: Fixed cost component of slippage
            use_volume: Whether to use volume in slippage calculation
        """
        self.impact_factor = impact_factor
        self.fixed_cost = fixed_cost
        self.use_volume = use_volume
    
    def calculate_slippage(
        self, 
        price: float, 
        quantity: float, 
        average_volume: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> float:
        """
        Calculate slippage for a trade.
        
        Args:
            price: Asset price
            quantity: Trade quantity
            average_volume: Average daily volume
            volatility: Asset volatility (standard deviation of returns)
            
        Returns:
            Slippage as a percentage of price
        """
        # Base slippage is the fixed cost
        slippage = self.fixed_cost
        
        # Add market impact if volume is provided
        if self.use_volume and average_volume is not None and average_volume > 0:
            # Market impact is proportional to trade size relative to average volume
            volume_ratio = quantity / average_volume
            market_impact = self.impact_factor * np.sqrt(volume_ratio)
            slippage += market_impact
        
        # Adjust for volatility if provided
        if volatility is not None:
            slippage *= (1 + volatility)
        
        return slippage
    
    def adjust_trade_price(
        self, 
        price: float, 
        quantity: float, 
        is_buy: bool,
        average_volume: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> float:
        """
        Adjust trade price to account for slippage.
        
        Args:
            price: Asset price
            quantity: Trade quantity
            is_buy: Whether the trade is a buy (True) or sell (False)
            average_volume: Average daily volume
            volatility: Asset volatility
            
        Returns:
            Adjusted price
        """
        # Calculate slippage
        slippage = self.calculate_slippage(price, quantity, average_volume, volatility)
        
        # Adjust price based on trade direction
        if is_buy:
            adjusted_price = price * (1 + slippage)
        else:
            adjusted_price = price * (1 - slippage)
        
        return adjusted_price

class LiquidityChecker:
    """
    Check asset liquidity to filter out illiquid assets.
    """
    
    def __init__(
        self, 
        min_volume: float = 100000,
        min_market_cap: Optional[float] = None,
        max_spread_percentage: Optional[float] = None
    ):
        """
        Initialize the liquidity checker.
        
        Args:
            min_volume: Minimum average daily volume
            min_market_cap: Minimum market capitalization
            max_spread_percentage: Maximum bid-ask spread as percentage
        """
        self.min_volume = min_volume
        self.min_market_cap = min_market_cap
        self.max_spread_percentage = max_spread_percentage
    
    def is_liquid(
        self, 
        average_volume: float,
        market_cap: Optional[float] = None,
        spread_percentage: Optional[float] = None
    ) -> bool:
        """
        Check if an asset is liquid.
        
        Args:
            average_volume: Average daily volume
            market_cap: Market capitalization
            spread_percentage: Bid-ask spread as percentage
            
        Returns:
            True if asset is liquid, False otherwise
        """
        # Check volume
        if average_volume < self.min_volume:
            return False
        
        # Check market cap if provided
        if self.min_market_cap is not None and market_cap is not None:
            if market_cap < self.min_market_cap:
                return False
        
        # Check spread if provided
        if self.max_spread_percentage is not None and spread_percentage is not None:
            if spread_percentage > self.max_spread_percentage:
                return False
        
        return True
    
    def filter_liquid_assets(
        self, 
        asset_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Filter for liquid assets from a DataFrame.
        
        Args:
            asset_data: DataFrame with asset data including volume and optionally market_cap and spread
            
        Returns:
            DataFrame with only liquid assets
        """
        liquid_assets = []
        
        for _, asset in asset_data.iterrows():
            # Get required data
            if 'average_volume' in asset:
                average_volume = asset['average_volume']
            elif 'volume' in asset:
                average_volume = asset['volume']
            else:
                logger.warning(f"Volume data not found for {asset.name}, skipping")
                continue
            
            # Get optional data
            market_cap = asset.get('market_cap', None)
            spread_percentage = asset.get('spread_percentage', None)
            
            # Check liquidity
            if self.is_liquid(average_volume, market_cap, spread_percentage):
                liquid_assets.append(asset.name)
        
        # Filter the DataFrame
        return asset_data.loc[liquid_assets]

class ExecutionSimulator:
    """
    Simulate execution of trades with delays and partial fills.
    """
    
    def __init__(
        self, 
        delay_distribution: str = 'exponential',
        delay_scale: float = 1.0,
        fill_probability: float = 0.9,
        min_fill_ratio: float = 0.5
    ):
        """
        Initialize the execution simulator.
        
        Args:
            delay_distribution: Distribution for execution delays ('exponential', 'uniform', or 'fixed')
            delay_scale: Scale parameter for delay distribution
            fill_probability: Probability of a complete fill
            min_fill_ratio: Minimum ratio of quantity filled if not complete
        """
        self.delay_distribution = delay_distribution
        self.delay_scale = delay_scale
        self.fill_probability = fill_probability
        self.min_fill_ratio = min_fill_ratio
    
    def simulate_execution(
        self, 
        quantity: float, 
        is_market_order: bool = True
    ) -> Tuple[float, float]:
        """
        Simulate order execution.
        
        Args:
            quantity: Order quantity
            is_market_order: Whether the order is a market order
            
        Returns:
            Tuple of (execution delay in days, filled quantity)
        """
        # Generate execution delay
        if self.delay_distribution == 'exponential':
            delay = np.random.exponential(scale=self.delay_scale)
        elif self.delay_distribution == 'uniform':
            delay = np.random.uniform(0, self.delay_scale * 2)
        else:  # fixed delay
            delay = self.delay_scale
        
        # Convert delay to days (assuming delay_scale is in days)
        delay_days = delay
        
        # Determine fill quantity
        if is_market_order:
            # Market orders have higher fill probability
            fill_prob = self.fill_probability
        else:
            # Limit orders have lower fill probability
            fill_prob = self.fill_probability * 0.8
        
        # Simulate fill
        if np.random.rand() < fill_prob:
            # Complete fill
            filled_quantity = quantity
        else:
            # Partial fill
            fill_ratio = np.random.uniform(self.min_fill_ratio, 1.0)
            filled_quantity = quantity * fill_ratio
        
        return delay_days, filled_quantity
    
    def simulate_pair_execution(
        self, 
        quantity1: float, 
        quantity2: float,
        is_market_order: bool = True
    ) -> Dict:
        """
        Simulate execution for a pairs trade.
        
        Args:
            quantity1: Quantity for first asset
            quantity2: Quantity for second asset
            is_market_order: Whether the orders are market orders
            
        Returns:
            Dictionary with execution results
        """
        # Simulate execution for first asset
        delay1, filled1 = self.simulate_execution(quantity1, is_market_order)
        
        # Simulate execution for second asset
        delay2, filled2 = self.simulate_execution(quantity2, is_market_order)
        
        # Calculate fill ratios
        fill_ratio1 = filled1 / quantity1 if quantity1 > 0 else 0
        fill_ratio2 = filled2 / quantity2 if quantity2 > 0 else 0
        
        # Calculate max delay
        max_delay = max(delay1, delay2)
        
        # Return results
        results = {
            'delay1': delay1,
            'delay2': delay2,
            'max_delay': max_delay,
            'filled1': filled1,
            'filled2': filled2,
            'fill_ratio1': fill_ratio1,
            'fill_ratio2': fill_ratio2,
            'balanced_ratio': min(fill_ratio1, fill_ratio2)
        }
        
        return results

class TransactionCostModel:
    """
    Model for estimating transaction costs including commissions, fees, and market impact.
    """
    
    def __init__(
        self, 
        commission_rate: float = 0.0005,
        exchange_fee_rate: float = 0.0001,
        market_impact_factor: float = 0.1,
        fixed_cost: float = 0.0
    ):
        """
        Initialize the transaction cost model.
        
        Args:
            commission_rate: Commission rate as a fraction of trade value
            exchange_fee_rate: Exchange fee rate as a fraction of trade value
            market_impact_factor: Factor for market impact calculation
            fixed_cost: Fixed cost per trade
        """
        self.commission_rate = commission_rate
        self.exchange_fee_rate = exchange_fee_rate
        self.market_impact_factor = market_impact_factor
        self.fixed_cost = fixed_cost
        
        # Create a slippage model for market impact
        self.slippage_model = SlippageModel(
            impact_factor=market_impact_factor,
            fixed_cost=0.0,
            use_volume=True
        )
    
    def calculate_transaction_cost(
        self, 
        price: float, 
        quantity: float, 
        is_buy: bool,
        average_volume: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> Dict:
        """
        Calculate transaction costs for a trade.
        
        Args:
            price: Asset price
            quantity: Trade quantity
            is_buy: Whether the trade is a buy (True) or sell (False)
            average_volume: Average daily volume
            volatility: Asset volatility
            
        Returns:
            Dictionary with transaction cost components
        """
        # Calculate trade value
        trade_value = price * quantity
        
        # Calculate commission
        commission = trade_value * self.commission_rate
        
        # Calculate exchange fee
        exchange_fee = trade_value * self.exchange_fee_rate
        
        # Calculate market impact
        slippage_percentage = self.slippage_model.calculate_slippage(
            price, quantity, average_volume, volatility
        )
        market_impact = trade_value * slippage_percentage
        
        # Calculate total cost
        total_cost = commission + exchange_fee + market_impact + self.fixed_cost
        
        # Return cost components
        costs = {
            'commission': commission,
            'exchange_fee': exchange_fee,
            'market_impact': market_impact,
            'fixed_cost': self.fixed_cost,
            'total_cost': total_cost,
            'total_cost_percentage': total_cost / trade_value if trade_value > 0 else 0
        }
        
        return costs
    
    def adjust_pair_trade(
        self, 
        price1: float, 
        quantity1: float, 
        is_buy1: bool,
        price2: float, 
        quantity2: float, 
        is_buy2: bool,
        volume1: Optional[float] = None,
        volume2: Optional[float] = None,
        volatility1: Optional[float] = None,
        volatility2: Optional[float] = None
    ) -> Dict:
        """
        Adjust a pairs trade for transaction costs.
        
        Args:
            price1: Price of first asset
            quantity1: Quantity of first asset
            is_buy1: Whether first trade is a buy
            price2: Price of second asset
            quantity2: Quantity of second asset
            is_buy2: Whether second trade is a buy
            volume1: Volume of first asset
            volume2: Volume of second asset
            volatility1: Volatility of first asset
            volatility2: Volatility of second asset
            
        Returns:
            Dictionary with adjusted trade details
        """
        # Calculate costs for first asset
        costs1 = self.calculate_transaction_cost(
            price1, quantity1, is_buy1, volume1, volatility1
        )
        
        # Calculate costs for second asset
        costs2 = self.calculate_transaction_cost(
            price2, quantity2, is_buy2, volume2, volatility2
        )
        
        # Calculate adjusted prices
        adjusted_price1 = price1 * (1 + costs1['total_cost_percentage']) if is_buy1 else price1 * (1 - costs1['total_cost_percentage'])
        adjusted_price2 = price2 * (1 + costs2['total_cost_percentage']) if is_buy2 else price2 * (1 - costs2['total_cost_percentage'])
        
        # Calculate total value and cost
        value1 = price1 * quantity1
        value2 = price2 * quantity2
        total_value = value1 + value2
        total_cost = costs1['total_cost'] + costs2['total_cost']
        
        # Return adjusted trade details
        adjusted_trade = {
            'original_price1': price1,
            'adjusted_price1': adjusted_price1,
            'original_price2': price2,
            'adjusted_price2': adjusted_price2,
            'quantity1': quantity1,
            'quantity2': quantity2,
            'value1': value1,
            'value2': value2,
            'total_value': total_value,
            'costs1': costs1,
            'costs2': costs2,
            'total_cost': total_cost,
            'total_cost_percentage': total_cost / total_value if total_value > 0 else 0
        }
        
        return adjusted_trade 