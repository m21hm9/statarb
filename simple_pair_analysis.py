import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_stock_data(tickers, start_date, end_date):
    """
    Fetch stock data using yfinance.
    """
    import yfinance as yf
    data = yf.download(tickers, start=start_date, end=end_date)
    return data

def calculate_hedge_ratio(y, x):
    """
    Calculate hedge ratio using OLS regression.
    """
    # Add constant to x
    x = sm.add_constant(x)
    
    # Fit OLS model
    model = OLS(y, x).fit()
    
    # Extract beta coefficient
    beta = model.params[1]
    alpha = model.params[0]
    
    return alpha, beta

def calculate_spread(y, x, alpha, beta):
    """
    Calculate spread between two price series.
    """
    return y - (alpha + beta * x)

def calculate_zscore(spread, window=20):
    """
    Calculate z-score for a spread series.
    """
    mean = spread.rolling(window=window).mean()
    std = spread.rolling(window=window).std()
    return (spread - mean) / std

def analyze_pair(ticker1, ticker2, start_date='2018-01-01', end_date='2022-12-31', output_dir='results_pair'):
    """
    Analyze a trading pair.
    """
    # Fetch historical data
    logger.info(f"Fetching historical data for {ticker1} and {ticker2} from {start_date} to {end_date}")
    price_data = fetch_stock_data([ticker1, ticker2], start_date, end_date)
    
    # Use Adjusted Close prices
    if 'Adj Close' in price_data.columns:
        prices = price_data['Adj Close']
    else:
        prices = price_data['Close']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot normalized prices
    plt.figure(figsize=(12, 6))
    normalized_prices = prices / prices.iloc[0]
    normalized_prices.plot()
    plt.title(f"Normalized Prices: {ticker1} vs {ticker2}")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{ticker1}_{ticker2}_normalized_prices.png'))
    plt.close()
    
    # Test for cointegration
    result = coint(prices[ticker1], prices[ticker2])
    pvalue = result[1]
    
    logger.info(f"Cointegration test for {ticker1}-{ticker2}: p-value = {pvalue:.6f}")
    if pvalue < 0.1:
        logger.info(f"Pair is cointegrated at 10% significance level")
    else:
        logger.info(f"Pair is not cointegrated at 10% significance level")
    
    # Calculate hedge ratio
    alpha, beta = calculate_hedge_ratio(prices[ticker1], prices[ticker2])
    logger.info(f"Hedge ratio (beta): {beta:.4f}, Alpha: {alpha:.4f}")
    
    # Calculate spread
    spread = calculate_spread(prices[ticker1], prices[ticker2], alpha, beta)
    
    # Plot spread
    plt.figure(figsize=(12, 6))
    spread.plot()
    plt.title(f"Spread: {ticker1} - ({alpha:.4f} + {beta:.4f}*{ticker2})")
    plt.grid(True)
    plt.axhline(y=spread.mean(), color='r', linestyle='--', label="Mean")
    plt.axhline(y=spread.mean() + 2*spread.std(), color='g', linestyle='--', label="Mean + 2*Std")
    plt.axhline(y=spread.mean() - 2*spread.std(), color='g', linestyle='--', label="Mean - 2*Std")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{ticker1}_{ticker2}_spread.png'))
    plt.close()
    
    # Calculate z-score
    zscore = calculate_zscore(spread, window=20)
    
    # Plot z-score
    plt.figure(figsize=(12, 6))
    zscore.plot()
    plt.title(f"Z-Score (window = 20)")
    plt.grid(True)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.axhline(y=2, color='g', linestyle='--', label="Entry Threshold (2)")
    plt.axhline(y=-2, color='g', linestyle='--')
    plt.axhline(y=3, color='m', linestyle='--', label="Stop Loss (3)")
    plt.axhline(y=-3, color='m', linestyle='--')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{ticker1}_{ticker2}_zscore.png'))
    plt.close()
    
    # Calculate rolling correlation
    rolling_corr = prices[ticker1].rolling(window=60).corr(prices[ticker2])
    
    # Plot rolling correlation
    plt.figure(figsize=(12, 6))
    rolling_corr.plot()
    plt.title(f"60-Day Rolling Correlation: {ticker1} and {ticker2}")
    plt.grid(True)
    plt.axhline(y=rolling_corr.mean(), color='r', linestyle='--', label="Mean")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{ticker1}_{ticker2}_correlation.png'))
    plt.close()
    
    # Calculate rolling beta
    rolling_betas = []
    rolling_alphas = []
    for i in range(60, len(prices)):
        window = prices.iloc[i-60:i]
        alpha, beta = calculate_hedge_ratio(window[ticker1], window[ticker2])
        rolling_alphas.append(alpha)
        rolling_betas.append(beta)
    
    rolling_betas = pd.Series(rolling_betas, index=prices.index[60:])
    rolling_alphas = pd.Series(rolling_alphas, index=prices.index[60:])
    
    # Plot rolling beta
    plt.figure(figsize=(12, 6))
    rolling_betas.plot()
    plt.title(f"60-Day Rolling Hedge Ratio (Beta): {ticker1} and {ticker2}")
    plt.grid(True)
    plt.axhline(y=rolling_betas.mean(), color='r', linestyle='--', label="Mean")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{ticker1}_{ticker2}_beta.png'))
    plt.close()
    
    # Plot price ratio
    price_ratio = prices[ticker1] / prices[ticker2]
    plt.figure(figsize=(12, 6))
    price_ratio.plot()
    plt.title(f"Price Ratio: {ticker1} / {ticker2}")
    plt.grid(True)
    plt.axhline(y=price_ratio.mean(), color='r', linestyle='--', label="Mean")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{ticker1}_{ticker2}_ratio.png'))
    plt.close()
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Calculate return correlation 
    return_corr = returns.corr()
    logger.info(f"Return correlation: {return_corr.iloc[0,1]:.4f}")
    
    # Plot return correlation scatter
    plt.figure(figsize=(10, 8))
    plt.scatter(returns[ticker2], returns[ticker1], alpha=0.5)
    plt.title(f"Daily Return Scatter: {ticker1} vs {ticker2}")
    plt.xlabel(f"{ticker2} Daily Return")
    plt.ylabel(f"{ticker1} Daily Return")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{ticker1}_{ticker2}_return_scatter.png'))
    plt.close()
    
    logger.info(f"Analysis complete. Results saved to {output_dir}")
    
    return prices, spread, zscore

if __name__ == "__main__":
    # Current date for default end date
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    # Run for NVDA and AMD
    analyze_pair('NVDA', 'AMD', start_date='2018-01-01', end_date=end_date, output_dir='results_nvda_amd') 