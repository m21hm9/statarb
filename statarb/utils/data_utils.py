import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Tuple, Dict, Optional, Union
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_stock_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    interval: str = "1d"
) -> pd.DataFrame:
    logger.info(f"Fetching data for {len(tickers)} stocks from {start_date} to {end_date}")
    try:
        data = yf.download(tickers, start=start_date, end=end_date, interval=interval)

        if len(tickers) == 1:
            data.columns = pd.MultiIndex.from_product([data.columns, tickers])
            
        logger.info(f"Successfully downloaded data with shape {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        raise

def calculate_returns(prices: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
    if method == 'simple':
        returns = prices.pct_change().dropna()
    elif method == 'log':
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        raise ValueError("method must be 'simple' or 'log'")
    
    return returns

def normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.div(prices.iloc[0])

def calculate_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return (series - mean) / std

def calculate_bollinger_bands(
    series: pd.Series, 
    window: int = 20, 
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    middle_band = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    return upper_band, middle_band, lower_band

def preprocess_for_pairs_trading(
    df: pd.DataFrame,
    adjust_for_market: bool = False,
    market_ticker: str = '^GSPC'
) -> pd.DataFrame:

    if ('Adj Close' in df.columns.levels[0] 
        if isinstance(df.columns, pd.MultiIndex) 
        else 'Adj Close' in df.columns):
        prices_col = 'Adj Close'
    else:
        prices_col = 'Close'

    if isinstance(df.columns, pd.MultiIndex):
        prices = df[prices_col]
    else:
        prices = df[[c for c in df.columns if prices_col in c]]

    returns = calculate_returns(prices)

    if adjust_for_market and market_ticker:
        try:
            if market_ticker not in prices.columns:
                market_data = yf.download(market_ticker, 
                                         start=prices.index[0], 
                                         end=prices.index[-1])
                market_returns = calculate_returns(market_data['Adj Close'])
            else:
                market_returns = returns[market_ticker]
            
            # Regress out market returns
            from statsmodels.api import OLS
            import statsmodels.api as sm
            
            adjusted_returns = pd.DataFrame(index=returns.index)
            
            for col in returns.columns:
                if col == market_ticker:
                    continue
                
                valid_data = ~(returns[col].isna() | market_returns.isna())
                
                if valid_data.sum() > 30:  # Minimum data requirement
                    X = sm.add_constant(market_returns[valid_data])
                    y = returns[col][valid_data]
                    
                    model = OLS(y, X).fit()
                    beta = model.params[1]
                    alpha = model.params[0]
                    
                    # Residuals plus alpha (to maintain return levels)
                    adjusted_returns[col] = alpha + returns[col] - (beta * market_returns)
            
            return adjusted_returns
        
        except Exception as e:
            logger.warning(f"Could not adjust for market: {str(e)}")
            return returns
    
    return returns 