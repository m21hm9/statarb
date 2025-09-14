import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Alpaca credentials (env > fallback to provided) 
# Enter your Alpaca API credentials here or set as environment variables
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', ' ')
ALPACA_API_SECRET = os.getenv('ALPACA_API_SECRET', ' ')
ALPACA_DATA_BASE_URL = os.getenv('ALPACA_DATA_BASE_URL', ' ')
# Trading base (not used here, but provided by user): ...


def _iso8601_utc(date_str: str, end_of_day: bool) -> str:
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    if end_of_day:
        dt = dt + timedelta(hours=23, minutes=59, seconds=59)
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')


def fetch_stock_data_alpaca(tickers, start_date, end_date, timeframe='1Day', adjustment='split', feed='iex') -> pd.DataFrame:
    """
    Fetch OHLCV bars from Alpaca Data API and return a DataFrame of Close prices.

    Returns a DataFrame indexed by timestamp with one column per ticker containing Close prices.
    """
    if not tickers:
        raise ValueError('tickers must be a non-empty list')

    headers = {
        'APCA-API-KEY-ID': ALPACA_API_KEY,
        'APCA-API-SECRET-KEY': ALPACA_API_SECRET,
    }

    params = {
        'symbols': ','.join(tickers),
        'timeframe': timeframe,
        'start': _iso8601_utc(start_date, end_of_day=False),
        'end': _iso8601_utc(end_date, end_of_day=True),
        'limit': 10000,
        'adjustment': adjustment,
        'feed': feed,
        # 'sort': 'asc'  # default asc
    }

    all_rows = []
    next_page_token = None
    url = f"{ALPACA_DATA_BASE_URL}/stocks/bars"

    while True:
        if next_page_token:
            params['page_token'] = next_page_token
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        bars = payload.get('bars', [])
        # Handle two possible shapes: list of bars or dict keyed by symbol
        if isinstance(bars, dict):
            # {'AAPL': [{...}, ...], 'MSFT': [{...}, ...]}
            for sym, items in bars.items():
                for b in items:
                    all_rows.append({
                        'symbol': sym,
                        'timestamp': b.get('t'),
                        'open': b.get('o'),
                        'high': b.get('h'),
                        'low': b.get('l'),
                        'close': b.get('c'),
                        'volume': b.get('v'),
                        'trade_count': b.get('n'),
                        'vwap': b.get('vw'),
                    })
        elif isinstance(bars, list):
            # [{ 'S': 'AAPL', 't': ..., 'o': ..., 'c': ... }, ...]
            for b in bars:
                sym = b.get('S') or b.get('s') or b.get('symbol')
                all_rows.append({
                    'symbol': sym,
                    'timestamp': b.get('t'),
                    'open': b.get('o'),
                    'high': b.get('h'),
                    'low': b.get('l'),
                    'close': b.get('c'),
                    'volume': b.get('v'),
                    'trade_count': b.get('n'),
                    'vwap': b.get('vw'),
                })
        else:
            logger.warning('Unexpected bars payload shape from Alpaca')

        next_page_token = payload.get('next_page_token')
        if not next_page_token:
            break

    if not all_rows:
        raise RuntimeError('No data returned from Alpaca for the requested parameters')

    df = pd.DataFrame(all_rows)
    # Parse timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    # Pivot to wide close price matrix
    prices = df.pivot_table(index='timestamp', columns='symbol', values='close').sort_index()

    # Drop any duplicate indices and forward fill missing within same symbol if needed
    prices = prices[~prices.index.duplicated(keep='first')]

    # Restrict to requested tickers order
    prices = prices[[t for t in tickers if t in prices.columns]]

    # Convert to naive datetime for plotting simplicity
    prices.index = prices.index.tz_convert(None) if prices.index.tz is not None else prices.index

    return prices


def calculate_hedge_ratio(y: pd.Series, x: pd.Series):
    x_const = sm.add_constant(x)
    model = OLS(y, x_const).fit()
    beta = model.params[1]
    alpha = model.params[0]
    return alpha, beta


def calculate_spread(y: pd.Series, x: pd.Series, alpha: float, beta: float) -> pd.Series:
    return y - (alpha + beta * x)


def calculate_zscore(spread: pd.Series, window: int = 20) -> pd.Series:
    mean = spread.rolling(window=window).mean()
    std = spread.rolling(window=window).std()
    return (spread - mean) / std


def generate_positions_from_zscore(zscore: pd.Series, entry_z: float = 2.0, exit_z: float = 0.5) -> pd.Series:
    """
    Convert z-score into discrete positions for the spread.
    1 = long spread (buy y, sell x), -1 = short spread (sell y, buy x), 0 = flat.
    """
    position = pd.Series(0, index=zscore.index, dtype=int)
    current_position = 0
    for timestamp, z in zscore.items():
        if current_position == 0:
            if z is not None and z > entry_z:
                current_position = -1
            elif z is not None and z < -entry_z:
                current_position = 1
        else:
            if z is not None and abs(z) < exit_z:
                current_position = 0
        position.loc[timestamp] = current_position
    return position


def analyze_pair_with_alpaca(ticker_y, ticker_x, start_date='2018-01-01', end_date=None, output_dir='results_pair_alpaca'):
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    logger.info(f"Fetching historical data for {ticker_y} and {ticker_x} from {start_date} to {end_date} via Alpaca")
    prices = fetch_stock_data_alpaca([ticker_y, ticker_x], start_date, end_date)

    os.makedirs(output_dir, exist_ok=True)

    # Normalized price chart
    normalized_prices = prices / prices.iloc[0]
    plt.figure(figsize=(12, 6))
    normalized_prices.plot()
    plt.title(f"Normalized Prices: {ticker_y} vs {ticker_x}")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{ticker_y}_{ticker_x}_normalized_prices.png'))
    plt.close()

    # Cointegration test
    result = coint(prices[ticker_y].dropna(), prices[ticker_x].dropna())
    pvalue = result[1]
    logger.info(f"Cointegration test for {ticker_y}-{ticker_x}: p-value = {pvalue:.6f}")
    if pvalue < 0.1:
        logger.info("Pair is cointegrated at 10% significance level")
    else:
        logger.info("Pair is not cointegrated at 10% significance level")

    # Hedge ratio
    alpha, beta = calculate_hedge_ratio(prices[ticker_y], prices[ticker_x])
    logger.info(f"Hedge ratio (beta): {beta:.4f}, Alpha: {alpha:.4f}")

    # Spread and plots
    spread = calculate_spread(prices[ticker_y], prices[ticker_x], alpha, beta)

    plt.figure(figsize=(12, 6))
    spread.plot()
    plt.title(f"Spread: {ticker_y} - ({alpha:.4f} + {beta:.4f}*{ticker_x})")
    plt.grid(True)
    plt.axhline(y=spread.mean(), color='r', linestyle='--', label='Mean')
    plt.axhline(y=spread.mean() + 2*spread.std(), color='g', linestyle='--', label='Mean + 2*Std')
    plt.axhline(y=spread.mean() - 2*spread.std(), color='g', linestyle='--', label='Mean - 2*Std')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{ticker_y}_{ticker_x}_spread.png'))
    plt.close()

    # Z-score
    zscore = calculate_zscore(spread, window=20)

    plt.figure(figsize=(12, 6))
    zscore.plot()
    plt.title('Z-Score (window = 20)')
    plt.grid(True)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.axhline(y=2, color='g', linestyle='--', label='Entry Threshold (2)')
    plt.axhline(y=-2, color='g', linestyle='--')
    plt.axhline(y=3, color='m', linestyle='--', label='Stop Loss (3)')
    plt.axhline(y=-3, color='m', linestyle='--')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{ticker_y}_{ticker_x}_zscore.png'))
    plt.close()

    # Rolling correlation
    rolling_corr = prices[ticker_y].rolling(window=60).corr(prices[ticker_x])
    plt.figure(figsize=(12, 6))
    rolling_corr.plot()
    plt.title(f"60-Day Rolling Correlation: {ticker_y} and {ticker_x}")
    plt.grid(True)
    plt.axhline(y=rolling_corr.mean(), color='r', linestyle='--', label='Mean')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{ticker_y}_{ticker_x}_correlation.png'))
    plt.close()

    # Rolling hedge ratio
    rolling_betas = []
    rolling_alphas = []
    for i in range(60, len(prices)):
        window = prices.iloc[i-60:i]
        a, b = calculate_hedge_ratio(window[ticker_y], window[ticker_x])
        rolling_alphas.append(a)
        rolling_betas.append(b)

    if rolling_betas:
        rolling_betas = pd.Series(rolling_betas, index=prices.index[60:])
        plt.figure(figsize=(12, 6))
        rolling_betas.plot()
        plt.title(f"60-Day Rolling Hedge Ratio (Beta): {ticker_y} and {ticker_x}")
        plt.grid(True)
        plt.axhline(y=rolling_betas.mean(), color='r', linestyle='--', label='Mean')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{ticker_y}_{ticker_x}_beta.png'))
        plt.close()

    # Price ratio
    price_ratio = prices[ticker_y] / prices[ticker_x]
    plt.figure(figsize=(12, 6))
    price_ratio.plot()
    plt.title(f"Price Ratio: {ticker_y} / {ticker_x}")
    plt.grid(True)
    plt.axhline(y=price_ratio.mean(), color='r', linestyle='--', label='Mean')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{ticker_y}_{ticker_x}_ratio.png'))
    plt.close()

    # Returns and return correlation
    returns = prices.pct_change().dropna()
    return_corr = returns.corr()
    logger.info(f"Return correlation: {return_corr.loc[ticker_y, ticker_x]:.4f}")

    plt.figure(figsize=(10, 8))
    plt.scatter(returns[ticker_x], returns[ticker_y], alpha=0.5)
    plt.title(f"Daily Return Scatter: {ticker_y} vs {ticker_x}")
    plt.xlabel(f"{ticker_x} Daily Return")
    plt.ylabel(f"{ticker_y} Daily Return")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{ticker_y}_{ticker_x}_return_scatter.png'))
    plt.close()

    # Generate positions and buy/sell signals from z-score
    positions = generate_positions_from_zscore(zscore)
    signal_changes = positions.diff().fillna(positions)

    buy_entry_dates = signal_changes[signal_changes == 1].index  # 0 -> +1 (long spread: buy y)
    sell_entry_dates = signal_changes[signal_changes == -1].index  # 0 -> -1 (short spread: sell y)
    exit_dates = signal_changes[(positions.shift(1) != 0) & (positions == 0)].index

    # Stock chart with buy/sell/exit markers (use normalized y for markers)
    plt.figure(figsize=(12, 6))
    normalized_prices[ticker_y].plot(label=f'{ticker_y} (y, normalized)')
    normalized_prices[ticker_x].plot(label=f'{ticker_x} (x, normalized)')

    if len(buy_entry_dates) > 0:
        plt.scatter(buy_entry_dates, normalized_prices.loc[buy_entry_dates, ticker_y], marker='^', color='g', s=80, label=f'Buy y ({ticker_y}) / Sell x ({ticker_x})')
    if len(sell_entry_dates) > 0:
        plt.scatter(sell_entry_dates, normalized_prices.loc[sell_entry_dates, ticker_y], marker='v', color='r', s=80, label=f'Sell y ({ticker_y}) / Buy x ({ticker_x})')
    if len(exit_dates) > 0:
        plt.scatter(exit_dates, normalized_prices.loc[exit_dates, ticker_y], marker='x', color='k', s=70, label='Exit position')

    plt.title(f"Signals on Price Chart (y={ticker_y}, x={ticker_x})")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{ticker_y}_{ticker_x}_signals.png'))
    plt.close()

    logger.info(f"Analysis complete. Results saved to {output_dir}")
    return prices, spread, zscore, positions


if __name__ == '__main__':
    end_date_default = datetime.today().strftime('%Y-%m-%d')
    analyze_pair_with_alpaca('NVDA', 'AMD', start_date='2025-01-01', end_date=end_date_default, output_dir='results_nvda_amd_alpaca') 