import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_stock_data(tickers, start_date, end_date):

    import yfinance as yf
    data = yf.download(tickers, start=start_date, end=end_date)
    return data

def calculate_hedge_ratio(y, x):
    x = sm.add_constant(x)
    model = OLS(y, x).fit()
    beta = model.params.iloc[1]
    alpha = model.params.iloc[0]
    
    return alpha, beta

def calculate_spread(y, x, alpha, beta):
    return y - (alpha + beta * x)

def calculate_zscore(spread, window=20):
    mean = spread.rolling(window=window).mean()
    std = spread.rolling(window=window).std()
    return (spread - mean) / std

def analyze_leveraged_etf_pair(ticker1, ticker2, start_date='2018-01-01', end_date=None, output_dir='results_leveraged_etfs'):
    """
    Analyze a leveraged ETF pair like TQQQ and SQQQ.
    
    Args:
        ticker1: First ticker symbol
        ticker2: Second ticker symbol
        start_date: Start date for analysis
        end_date: End date for analysis (default: today)
        output_dir: Directory for output files
    
    Returns:
        prices, spread, zscore
    """
    # Set end date to today if not specified
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    # Fetch historical data
    logger.info(f"Fetching historical data for {ticker1} and {ticker2} from {start_date} to {end_date}")
    price_data = fetch_stock_data([ticker1, ticker2], start_date, end_date)
    
    # Use Adjusted Close prices
    if 'Adj Close' in price_data.columns:
        prices = price_data['Adj Close']
    else:
        prices = price_data['Close']
    
    # Check for missing data
    missing_data = prices.isna().sum()
    if missing_data.sum() > 0:
        logger.warning(f"Found missing data: {missing_data}")
        # Fill missing data with forward fill then backward fill
        prices = prices.fillna(method='ffill').fillna(method='bfill')
    
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
    window_size = 60
    
    for i in range(window_size, len(prices)):
        window = prices.iloc[i-window_size:i]
        try:
            alpha, beta = calculate_hedge_ratio(window[ticker1], window[ticker2])
            rolling_alphas.append(alpha)
            rolling_betas.append(beta)
        except:
            # Handle any errors in calculation
            if len(rolling_betas) > 0:
                rolling_alphas.append(rolling_alphas[-1])
                rolling_betas.append(rolling_betas[-1])
            else:
                rolling_alphas.append(0)
                rolling_betas.append(0)
    
    rolling_betas = pd.Series(rolling_betas, index=prices.index[window_size:])
    rolling_alphas = pd.Series(rolling_alphas, index=prices.index[window_size:])
    
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
    
    # Expected negative correlation for TQQQ-SQQQ
    logger.info(f"Expected direction of correlation: {'Correct' if return_corr.iloc[0,1] < 0 else 'Unexpected'}")
    
    # Plot return correlation scatter
    plt.figure(figsize=(10, 8))
    plt.scatter(returns[ticker2], returns[ticker1], alpha=0.5)
    plt.title(f"Daily Return Scatter: {ticker1} vs {ticker2}")
    plt.xlabel(f"{ticker2} Daily Return")
    plt.ylabel(f"{ticker1} Daily Return")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{ticker1}_{ticker2}_return_scatter.png'))
    plt.close()
    
    # Calculate potential profit for the last year
    # Find crossing points where z-score crosses the thresholds
    buy_signals = (zscore < -2) & (zscore.shift(1) >= -2)
    sell_signals = (zscore > 2) & (zscore.shift(1) <= 2)
    
    # Count signals in the last year
    if len(zscore) > 252:
        # Get the last year of data
        last_year_data = zscore[-252:]
        last_year_index = last_year_data.index
        
        # Filter signals to match the last year index
        last_year_buy = buy_signals.loc[last_year_index]
        last_year_sell = sell_signals.loc[last_year_index]
        
        num_buy_signals = last_year_buy.sum()
        num_sell_signals = last_year_sell.sum()
        
        logger.info(f"Number of buy signals in last year: {num_buy_signals}")
        logger.info(f"Number of sell signals in last year: {num_sell_signals}")
        
        # Estimate potential profit for last year (assuming perfect execution)
        # This is a very simplistic model assuming trades exit at mean reversion (z=0)
        if num_buy_signals > 0:
            buy_signal_dates = last_year_index[last_year_buy]
            avg_buy_z = zscore.loc[buy_signal_dates].mean()
        else:
            avg_buy_z = 0
            
        if num_sell_signals > 0:
            sell_signal_dates = last_year_index[last_year_sell]
            avg_sell_z = zscore.loc[sell_signal_dates].mean()
        else:
            avg_sell_z = 0
        
        # Estimate profit per $1000 invested
        est_profit_buy = 1000 * (-avg_buy_z/2 * 0.05) * num_buy_signals if num_buy_signals > 0 else 0
        est_profit_sell = 1000 * (avg_sell_z/2 * 0.05) * num_sell_signals if num_sell_signals > 0 else 0
        
        logger.info(f"Estimated profit for $1000 from buy signals: ${est_profit_buy:.2f}")
        logger.info(f"Estimated profit for $1000 from sell signals: ${est_profit_sell:.2f}")
        logger.info(f"Total estimated profit for $1000: ${est_profit_buy + est_profit_sell:.2f}")
    
    # Special analysis for leveraged ETFs
    logger.info("Performing special analysis for leveraged ETFs...")
    
    # Calculate decay effects
    if len(returns) > 252:
        # Calculate theoretical returns for inverse relationship (-3x)
        theoretical_ratio = prices[ticker1].iloc[0] / prices[ticker2].iloc[0]
        theoretical_prices = prices[ticker1].iloc[0] / (prices[ticker2] / prices[ticker2].iloc[0]) * theoretical_ratio
        
        # Plot theoretical vs actual prices for TQQQ
        plt.figure(figsize=(12, 6))
        plt.plot(prices[ticker1], label='Actual')
        plt.plot(theoretical_prices, label='Theoretical')
        plt.title(f"Leveraged ETF Decay Analysis: {ticker1}")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{ticker1}_{ticker2}_decay_analysis.png'))
        plt.close()
        
        # Calculate annual decay rate
        years = len(prices) / 252
        decay_ratio = (prices[ticker1].iloc[-1] / theoretical_prices.iloc[-1])
        annual_decay = (decay_ratio**(1/years) - 1) * 100
        
        logger.info(f"Estimated annual decay effect: {annual_decay:.2f}%")
        
    logger.info(f"Analysis complete. Results saved to {output_dir}")
    
    return prices, spread, zscore

if __name__ == "__main__":
    # Current date for default end date
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    # Run for TQQQ and SQQQ - these are 3x leveraged long and short ETFs for the Nasdaq-100
    analyze_leveraged_etf_pair('TQQQ', 'SQQQ', start_date='2018-01-01', end_date=end_date, output_dir='results_tqqq_sqqq') 