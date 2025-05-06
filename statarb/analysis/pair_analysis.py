import numpy as np
import pandas as pd
import yfinance as yf
from typing import Tuple, Dict, List
from datetime import datetime, timedelta
from statarb.models.trading_fees import TradingFees

class PairAnalysis:
    def __init__(self, initial_capital: float = 1500.0):
        self.fees = TradingFees(initial_capital)
        
    def get_historical_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical price data for given tickers"""
        data = {}
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            data[ticker] = stock.history(start=start_date, end=end_date)['Close']
        return pd.DataFrame(data)
    
    def calculate_spread(self, data: pd.DataFrame, pair: Tuple[str, str]) -> pd.Series:
        """Calculate spread between two assets"""
        stock1, stock2 = pair
        return data[stock1] - data[stock2]
    
    def calculate_zscore(self, spread: pd.Series, window: int = 20) -> pd.Series:
        """Calculate z-score of the spread"""
        mean = spread.rolling(window=window).mean()
        std = spread.rolling(window=window).std()
        return (spread - mean) / std
    
    def analyze_pair(self, pair: Tuple[str, str], start_date: str, end_date: str) -> Dict:
        """Analyze trading pair and calculate potential P&L"""
        # Get historical data
        data = self.get_historical_data(list(pair), start_date, end_date)
        
        # Calculate correlation
        correlation = data[pair[0]].corr(data[pair[1]])
        
        # Calculate spread and z-score
        spread = self.calculate_spread(data, pair)
        zscore = self.calculate_zscore(spread, window=60)  # Increased window to 60 days
        
        # Calculate cointegration statistics
        from statsmodels.tsa.stattools import adfuller
        adf_result = adfuller(spread.dropna())
        
        # Calculate entry/exit points with more conservative thresholds
        entry_threshold = 2.5  # Increased from 2.0
        exit_threshold = 0.5   # Changed from 0.0 to 0.5
        
        # Initialize trading signals
        signals = pd.DataFrame(index=data.index)
        signals['position'] = 0
        
        # Generate trading signals
        signals.loc[zscore > entry_threshold, 'position'] = -1  # Short spread
        signals.loc[zscore < -entry_threshold, 'position'] = 1  # Long spread
        signals.loc[abs(zscore) < exit_threshold, 'position'] = 0  # Exit
        
        # Calculate returns
        returns = pd.DataFrame(index=data.index)
        returns['spread_return'] = spread.pct_change()
        returns['strategy_return'] = signals['position'].shift(1) * returns['spread_return']
        
        # Calculate P&L with fees and risk management
        initial_price1 = data[pair[0]].iloc[0]
        initial_price2 = data[pair[1]].iloc[0]
        
        # More conservative position sizing (30% of capital for each leg)
        shares1 = int((self.fees.initial_capital * 0.3) / initial_price1)
        shares2 = int((self.fees.initial_capital * 0.3) / initial_price2)
        
        # Calculate total P&L including fees
        total_pnl = 0
        current_position = 0
        trade_history = []
        
        for i in range(1, len(signals)):
            if signals['position'].iloc[i] != current_position:
                # Calculate fees for closing current position
                if current_position != 0:
                    if current_position == 1:
                        # Close long spread
                        fees1 = self.fees.calculate_stock_fees(shares1, data[pair[0]].iloc[i], False)
                        fees2 = self.fees.calculate_stock_fees(shares2, data[pair[1]].iloc[i], True)
                        trade_type = "Close Long"
                    else:
                        # Close short spread
                        fees1 = self.fees.calculate_stock_fees(shares1, data[pair[0]].iloc[i], True)
                        fees2 = self.fees.calculate_stock_fees(shares2, data[pair[1]].iloc[i], False)
                        trade_type = "Close Short"
                    total_pnl -= (fees1 + fees2)
                    
                    # Record trade
                    trade_history.append({
                        'date': data.index[i],
                        'type': trade_type,
                        'price1': data[pair[0]].iloc[i],
                        'price2': data[pair[1]].iloc[i],
                        'fees': fees1 + fees2,
                        'pnl': 0  # Will be updated with final P&L
                    })
                
                # Calculate fees for opening new position
                if signals['position'].iloc[i] != 0:
                    if signals['position'].iloc[i] == 1:
                        # Open long spread
                        fees1 = self.fees.calculate_stock_fees(shares1, data[pair[0]].iloc[i], True)
                        fees2 = self.fees.calculate_stock_fees(shares2, data[pair[1]].iloc[i], False)
                        trade_type = "Open Long"
                    else:
                        # Open short spread
                        fees1 = self.fees.calculate_stock_fees(shares1, data[pair[0]].iloc[i], False)
                        fees2 = self.fees.calculate_stock_fees(shares2, data[pair[1]].iloc[i], True)
                        trade_type = "Open Short"
                    total_pnl -= (fees1 + fees2)
                    
                    # Record trade
                    trade_history.append({
                        'date': data.index[i],
                        'type': trade_type,
                        'price1': data[pair[0]].iloc[i],
                        'price2': data[pair[1]].iloc[i],
                        'fees': fees1 + fees2,
                        'pnl': 0
                    })
                
                current_position = signals['position'].iloc[i]
            
            # Calculate daily P&L
            if current_position != 0:
                pnl = current_position * (shares1 * data[pair[0]].iloc[i] - shares2 * data[pair[1]].iloc[i])
                total_pnl += pnl
                
                # Update P&L for the last trade
                if trade_history:
                    trade_history[-1]['pnl'] = pnl
        
        # Calculate additional metrics
        trade_df = pd.DataFrame(trade_history)
        if not trade_df.empty:
            win_rate = len(trade_df[trade_df['pnl'] > 0]) / len(trade_df)
            avg_win = trade_df[trade_df['pnl'] > 0]['pnl'].mean() if len(trade_df[trade_df['pnl'] > 0]) > 0 else 0
            avg_loss = trade_df[trade_df['pnl'] < 0]['pnl'].mean() if len(trade_df[trade_df['pnl'] < 0]) > 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
        
        return {
            'total_pnl': total_pnl,
            'return_pct': (total_pnl / self.fees.initial_capital) * 100,
            'trades': len(signals[signals['position'] != 0]),
            'data': data,
            'signals': signals,
            'zscore': zscore,
            'trade_history': trade_history,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'correlation': correlation,
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1]
        }
    
    def analyze_leveraged_etfs(self, etfs: Tuple[str, str], start_date: str, end_date: str) -> Dict:
        """Analyze leveraged ETF pair"""
        # Get historical data
        data = self.get_historical_data(list(etfs), start_date, end_date)
        
        # Calculate daily returns
        returns = data.pct_change()
        
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Calculate P&L with fees
        initial_price1 = data[etfs[0]].iloc[0]
        initial_price2 = data[etfs[1]].iloc[0]
        
        # Calculate shares to trade (using 50% of capital for each ETF)
        shares1 = int((self.fees.initial_capital * 0.5) / initial_price1)
        shares2 = int((self.fees.initial_capital * 0.5) / initial_price2)
        
        # Calculate total P&L including fees
        total_pnl = 0
        
        # Buy both ETFs at the start
        fees1 = self.fees.calculate_stock_fees(shares1, initial_price1, True)
        fees2 = self.fees.calculate_stock_fees(shares2, initial_price2, True)
        total_pnl -= (fees1 + fees2)
        
        # Calculate final value
        final_value1 = shares1 * data[etfs[0]].iloc[-1]
        final_value2 = shares2 * data[etfs[1]].iloc[-1]
        
        # Sell both ETFs at the end
        fees1 = self.fees.calculate_stock_fees(shares1, data[etfs[0]].iloc[-1], False)
        fees2 = self.fees.calculate_stock_fees(shares2, data[etfs[1]].iloc[-1], False)
        total_pnl += (final_value1 + final_value2) - (fees1 + fees2)
        
        return {
            'total_pnl': total_pnl,
            'return_pct': (total_pnl / self.fees.initial_capital) * 100,
            'data': data,
            'returns': returns,
            'cum_returns': cum_returns
        } 