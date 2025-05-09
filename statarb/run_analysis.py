from statarb.analysis.pair_analysis import PairAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.stattools import coint

def plot_pair_analysis(results: dict, pair: tuple):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

    ax1.plot(results['data'][pair[0]], label=pair[0])
    ax1.plot(results['data'][pair[1]], label=pair[1])
    ax1.set_title('Price Series')
    ax1.legend()

    ax2.plot(results['zscore'], label='Z-Score')
    ax2.axhline(y=2, color='r', linestyle='--', label='Entry Threshold')
    ax2.axhline(y=-2, color='r', linestyle='--')
    ax2.axhline(y=0, color='g', linestyle='--', label='Exit Threshold')
    ax2.set_title('Z-Score')
    ax2.legend()
    
    # Plot positions
    ax3.plot(results['signals']['position'], label='Position')
    ax3.set_title('Trading Signals')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

def plot_etf_analysis(results: dict, etfs: tuple):
    """Plot leveraged ETF analysis results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot prices
    ax1.plot(results['data'][etfs[0]], label=etfs[0])
    ax1.plot(results['data'][etfs[1]], label=etfs[1])
    ax1.set_title('Price Series')
    ax1.legend()
    
    # Plot cumulative returns
    ax2.plot(results['cum_returns'][etfs[0]], label=f'{etfs[0]} Returns')
    ax2.plot(results['cum_returns'][etfs[1]], label=f'{etfs[1]} Returns')
    ax2.set_title('Cumulative Returns')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_leveraged_etf_analysis(results: dict, etfs: tuple):
    """Plot leveraged ETF analysis results"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # Plot prices
    ax1.plot(results['data'][etfs[0]], label=etfs[0])
    ax1.plot(results['data'][etfs[1]], label=etfs[1])
    ax1.set_title('Price Series')
    ax1.legend()
    
    # Plot daily returns
    returns = results['data'].pct_change()
    ax2.plot(returns[etfs[0]], label=f'{etfs[0]} Daily Returns')
    ax2.plot(returns[etfs[1]], label=f'{etfs[1]} Daily Returns')
    ax2.set_title('Daily Returns')
    ax2.legend()
    
    # Plot cumulative returns
    cum_returns = (1 + returns).cumprod()
    ax3.plot(cum_returns[etfs[0]], label=f'{etfs[0]} Cumulative Returns')
    ax3.plot(cum_returns[etfs[1]], label=f'{etfs[1]} Cumulative Returns')
    ax3.set_title('Cumulative Returns')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

def plot_portfolio_allocation(allocations: dict):
    """Plot portfolio allocation as a pie chart"""
    labels = list(allocations.keys())
    sizes = list(allocations.values())
    
    plt.figure(figsize=(10, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Portfolio Allocation')
    plt.tight_layout()
    plt.show()

def plot_portfolio_performance(portfolio_data: pd.DataFrame):
    """Plot portfolio performance over time"""
    plt.figure(figsize=(12, 8))
    
    # Plot individual asset classes and total portfolio
    for column in portfolio_data.columns:
        plt.plot(portfolio_data.index, portfolio_data[column], label=column)
    
    plt.title('Portfolio Performance')
    plt.xlabel('Date')
    plt.ylabel('Value ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def analyze_asset_class(tickers: list, start_date: str, end_date: str, initial_allocation: float):
    """Analyze performance of an asset class"""
    # Download data
    raw_data = yf.download(tickers, start=start_date, end=end_date)
    
    # Check if 'Adj Close' is available, otherwise use 'Close'
    if 'Adj Close' in raw_data.columns:
        data = raw_data['Adj Close']
    else:
        data = raw_data['Close']
    
    # Handle single ticker case
    if len(tickers) == 1 and not isinstance(data, pd.DataFrame):
        data = data.to_frame(name=tickers[0])
    elif len(tickers) == 1 and isinstance(data, pd.DataFrame):
        # Ensure column is named correctly for single ticker case
        data.columns = [tickers[0]]
    
    # Calculate returns
    returns = data.pct_change().dropna()
    
    # Equal weight allocation within asset class
    weights = np.ones(len(tickers)) / len(tickers)
    
    # Calculate weighted returns
    weighted_returns = (returns * weights).sum(axis=1)
    
    # Calculate cumulative portfolio value
    cumulative_returns = (1 + weighted_returns).cumprod()
    portfolio_value = initial_allocation * cumulative_returns
    
    # Calculate performance metrics
    total_return = (portfolio_value.iloc[-1] / initial_allocation) - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = weighted_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    max_drawdown = (portfolio_value / portfolio_value.cummax() - 1).min()
    
    # Return results
    return {
        'data': data,
        'returns': returns,
        'portfolio_value': portfolio_value,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'final_value': portfolio_value.iloc[-1]
    }

def analyze_pairs_trading(tickers, start_date, end_date, initial_capital):
    """Analyze potential pairs trading opportunities"""
    from statsmodels.tsa.stattools import coint
    
    # Download price data
    data = yf.download(tickers, start=start_date, end=end_date)
    
    # Use close prices
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    else:
        prices = data['Close']
    
    # Find potential cointegrated pairs
    n = len(tickers)
    pairs = []
    pvalues = []
    
    print("\nAnalyzing potential pairs for trading:")
    print("-" * 50)
    
    # Test each pair for cointegration
    for i in range(n):
        for j in range(i+1, n):
            ticker1 = tickers[i]
            ticker2 = tickers[j]
            
            # Get price series for both tickers (drop any missing values)
            p1 = prices[ticker1].dropna()
            p2 = prices[ticker2].dropna()
            
            # Align the series (use common dates)
            common_dates = p1.index.intersection(p2.index)
            if len(common_dates) < 30:  # Need enough data for meaningful test
                continue
                
            p1 = p1.loc[common_dates]
            p2 = p2.loc[common_dates]
            
            # Test for cointegration
            result = coint(p1, p2)
            pvalue = result[1]
            
            # Store result if potentially cointegrated (p-value < 0.1)
            if pvalue < 0.1:
                pairs.append((ticker1, ticker2))
                pvalues.append(pvalue)
                print(f"Potential pair: {ticker1} - {ticker2}, p-value: {pvalue:.4f}")
    
    print("-" * 50)
    
    # Analyze the most promising pair if found
    best_pairs = []
    if pairs:
        # Sort by p-value (most significant first)
        sorted_pairs = [x for _, x in sorted(zip(pvalues, pairs))]
        best_pairs = sorted_pairs[:min(3, len(sorted_pairs))]
        
        # Analyze the top pairs
        pair_results = []
        
        for ticker1, ticker2 in best_pairs:
            print(f"\nDetailed analysis for pair: {ticker1} - {ticker2}")
            
            # Get aligned price series
            p1 = prices[ticker1].dropna()
            p2 = prices[ticker2].dropna()
            common_dates = p1.index.intersection(p2.index)
            p1 = p1.loc[common_dates]
            p2 = p2.loc[common_dates]
            
            # Calculate hedge ratio using OLS
            X = p2.values.reshape(-1, 1)
            X = np.concatenate([np.ones_like(X), X], axis=1)
            beta = np.linalg.lstsq(X, p1.values, rcond=None)[0]
            alpha, hedge_ratio = beta[0], beta[1]
            
            # Calculate spread
            spread = p1 - (alpha + hedge_ratio * p2)
            
            # Calculate z-score
            zscore = (spread - spread.mean()) / spread.std()
            
            # Simulate trading signals
            z_entry = 2.0  # Entry threshold
            z_exit = 0.0   # Exit threshold
            
            # Initialize positions and portfolio
            position = 0
            positions = []
            portfolio = initial_capital / 10  # Allocate a portion to this pair
            portfolio_values = []
            trade_history = []
            
            # Loop through z-scores to generate signals
            for i in range(len(zscore)):
                # Add current portfolio value (beginning of period)
                if i == 0:
                    portfolio_values.append(portfolio)
                
                # Store position
                positions.append(position)
                
                # Trading logic
                if position == 0:
                    # No position
                    if zscore.iloc[i] < -z_entry:
                        # Buy spread (long ticker1, short ticker2)
                        position = 1
                        entry_price = spread.iloc[i]
                        entry_date = zscore.index[i]
                        trade_history.append({"entry_date": entry_date, "entry_price": entry_price, "type": "long"})
                    elif zscore.iloc[i] > z_entry:
                        # Sell spread (short ticker1, long ticker2)
                        position = -1
                        entry_price = spread.iloc[i]
                        entry_date = zscore.index[i]
                        trade_history.append({"entry_date": entry_date, "entry_price": entry_price, "type": "short"})
                
                elif position == 1:
                    # Long spread position
                    if zscore.iloc[i] > z_exit:
                        # Exit position
                        exit_price = spread.iloc[i]
                        exit_date = zscore.index[i]
                        trade_pnl = exit_price - entry_price
                        portfolio = portfolio * (1 + trade_pnl / entry_price)
                        position = 0
                        trade_history[-1].update({"exit_date": exit_date, "exit_price": exit_price, "pnl": trade_pnl})
                
                elif position == -1:
                    # Short spread position
                    if zscore.iloc[i] < z_exit:
                        # Exit position
                        exit_price = spread.iloc[i]
                        exit_date = zscore.index[i]
                        trade_pnl = entry_price - exit_price
                        portfolio = portfolio * (1 + trade_pnl / entry_price)
                        position = 0
                        trade_history[-1].update({"exit_date": exit_date, "exit_price": exit_price, "pnl": trade_pnl})
                
                # Update portfolio value after this period
                portfolio_values.append(portfolio)
            
            # Complete the positions list for the last period if needed
            if len(positions) < len(zscore):
                positions.append(position)
            
            # Create Series with proper indexing
            positions = pd.Series(positions, index=zscore.index)
            
            # Create portfolio values series - make sure the length matches the index
            if len(portfolio_values) > len(zscore.index):
                # We have one extra value, create index with an extra date point
                portfolio_index = [zscore.index[0] - pd.Timedelta(days=1)] + list(zscore.index)
                portfolio_values = pd.Series(portfolio_values, index=portfolio_index)
            else:
                # Just use the zscore index (shouldn't happen with updated logic)
                portfolio_values = pd.Series(portfolio_values, index=zscore.index)
            
            # Calculate returns
            returns = portfolio_values.pct_change().dropna()
            total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
            
            # Extract completed trades
            completed_trades = [t for t in trade_history if "exit_date" in t]
            win_rate = sum(1 for t in completed_trades if t["pnl"] > 0) / len(completed_trades) if completed_trades else 0
            
            # Print pair trading results
            print(f"Hedge ratio: {hedge_ratio:.4f}")
            print(f"Number of trades: {len(completed_trades)}")
            print(f"Win rate: {win_rate:.2%}")
            print(f"Total return: {total_return:.2%}")
            
            # Plot results
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
            
            # Plot prices
            ax1.plot(p1, label=ticker1)
            ax1.plot(p2, label=ticker2)
            ax1.set_title(f'Price Series: {ticker1} vs {ticker2}')
            ax1.legend()
            
            # Plot spread
            ax2.plot(spread, label='Spread')
            ax2.set_title(f'Spread: {ticker1} - ({alpha:.2f} + {hedge_ratio:.4f}*{ticker2})')
            ax2.legend()
            
            # Plot z-score with positions
            ax3.plot(zscore, label='Z-Score')
            ax3.plot(positions, 'r--', label='Position')
            ax3.axhline(y=z_entry, color='g', linestyle='--', label='Entry')
            ax3.axhline(y=-z_entry, color='g', linestyle='--')
            ax3.axhline(y=z_exit, color='b', linestyle='--', label='Exit')
            ax3.axhline(y=-z_exit, color='b', linestyle='--')
            ax3.set_title('Z-Score with Trading Positions')
            ax3.legend()
            
            plt.tight_layout()
            plt.show()
            
            # Store results
            pair_results.append({
                'pair': (ticker1, ticker2),
                'hedge_ratio': hedge_ratio,
                'spread': spread,
                'zscore': zscore,
                'positions': positions,
                'portfolio_values': portfolio_values,
                'total_return': total_return,
                'win_rate': win_rate,
                'trades': completed_trades
            })
        
        return pair_results
    else:
        print("No cointegrated pairs found for pairs trading.")
        return []

def analyze_options_strategies(underlyings, start_date, end_date, initial_allocation):
    """
    Analyze performance of options strategies
    
    This function simulates the performance of various option strategies:
    1. Covered Calls on SPY
    2. Cash-Secured Puts on QQQ
    3. Long Calls on AAPL and NVDA
    4. Put Credit Spreads on broad market
    """
    print("Analyzing options strategies...")
    
    # Download underlying asset data
    data = yf.download(underlyings, start=start_date, end=end_date)
    
    # Use Close prices
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    else:
        prices = data['Close']
    
    # Handle single ticker case
    if len(underlyings) == 1 and not isinstance(prices, pd.DataFrame):
        prices = prices.to_frame(name=underlyings[0])
    elif len(underlyings) == 1 and isinstance(prices, pd.DataFrame):
        prices.columns = [underlyings[0]]
    
    # Create a date range to match the price data
    dates = prices.index
    
    # Simulate options strategies performance
    
    # 1. Covered Call Strategy (40% of options allocation) - SPY
    # Assumption: Monthly covered calls with ~2% premium, 30 delta
    covered_call_allocation = initial_allocation * 0.4
    monthly_premium_pct = 0.02  # 2% monthly premium
    assignment_probability = 0.3  # 30% chance of being assigned each month
    
    # Simulate monthly returns
    covered_call_returns = []
    
    # Get SPY data
    spy_prices = prices['SPY'] if 'SPY' in prices.columns else prices.iloc[:, 0]
    spy_returns = spy_prices.pct_change().dropna()
    
    # Generate monthly points - use 'ME' (month end) instead of 'M'
    monthly_dates = pd.date_range(start=dates[0], end=dates[-1], freq='ME')
    monthly_dates = monthly_dates[monthly_dates <= dates[-1]]
    
    # Helper function to find the closest date
    def find_closest_date(target_date, date_list):
        return date_list[np.abs([(d - target_date).total_seconds() for d in date_list]).argmin()]
    
    # For each month, calculate the return
    for i in range(len(monthly_dates)):
        # Calculate return for this month
        if i > 0:
            # Find closest dates in our actual data
            start_date = find_closest_date(monthly_dates[i-1], dates)
            end_date = find_closest_date(monthly_dates[i], dates)
            
            # Get indices for these dates
            start_idx = dates.get_indexer([start_date])[0]
            end_idx = dates.get_indexer([end_date])[0]
            
            # Calculate price return
            price_return = (spy_prices.iloc[end_idx] / spy_prices.iloc[start_idx]) - 1
            
            # If assigned (price went above strike), cap gains
            if np.random.random() < assignment_probability:
                # Cap gains at around 2% + premium
                covered_call_return = min(price_return, 0.02) + monthly_premium_pct
            else:
                # Not assigned, get price return + premium
                covered_call_return = price_return + monthly_premium_pct
                
            covered_call_returns.append(covered_call_return)
    
    # Annualize the returns (approximately)
    if covered_call_returns:  # Check if we have any returns
        annual_cc_return = np.mean(covered_call_returns) * 12
        cc_volatility = np.std(covered_call_returns) * np.sqrt(12)
    else:
        annual_cc_return = 0.08  # Default to 8% annual return if not enough data
        cc_volatility = 0.10  # Default volatility
    
    cc_sharpe = annual_cc_return / cc_volatility if cc_volatility > 0 else 0
    
    covered_call_value = covered_call_allocation * (1 + annual_cc_return) ** (len(monthly_dates) / 12)
    
    # 2. Cash-Secured Puts (30% of options allocation) - QQQ
    csp_allocation = initial_allocation * 0.3
    put_premium_pct = 0.015  # 1.5% monthly premium
    assignment_probability = 0.25  # 25% chance of being assigned
    
    # Simulate monthly returns
    csp_returns = []
    
    # Get QQQ data if available
    qqq_prices = prices['QQQ'] if 'QQQ' in prices.columns else spy_prices
    qqq_returns = qqq_prices.pct_change().dropna()
    
    # For each month, calculate the return
    for i in range(len(monthly_dates)):
        if i > 0:
            # Find closest dates in our actual data
            start_date = find_closest_date(monthly_dates[i-1], dates)
            end_date = find_closest_date(monthly_dates[i], dates)
            
            # Get indices for these dates
            start_idx = dates.get_indexer([start_date])[0]
            end_idx = dates.get_indexer([end_date])[0]
            
            # Calculate price return
            price_return = (qqq_prices.iloc[end_idx] / qqq_prices.iloc[start_idx]) - 1
            
            # If assigned (price dropped below strike)
            if np.random.random() < assignment_probability:
                # We get assigned the stock at a loss, assume ~2% below strike after premium
                csp_return = put_premium_pct - 0.02
            else:
                # Not assigned, keep the premium
                csp_return = put_premium_pct
                
            csp_returns.append(csp_return)
    
    # Annualize the returns
    if csp_returns:
        annual_csp_return = np.mean(csp_returns) * 12
        csp_volatility = np.std(csp_returns) * np.sqrt(12)
    else:
        annual_csp_return = 0.10  # Default to 10% annual return
        csp_volatility = 0.12     # Default volatility
        
    csp_sharpe = annual_csp_return / csp_volatility if csp_volatility > 0 else 0
    
    csp_value = csp_allocation * (1 + annual_csp_return) ** (len(monthly_dates) / 12)
    
    # 3. Long Calls (20% of options allocation) - High growth stocks
    long_call_allocation = initial_allocation * 0.2
    
    # Simulate quarterly options with 2-month expiry
    call_returns = []
    
    # Use NVDA or first stock as proxy
    growth_prices = prices['NVDA'] if 'NVDA' in prices.columns else prices.iloc[:, 0]
    growth_returns = growth_prices.pct_change().dropna()
    
    # Quarterly rebalance - use 'QE' (quarter end) instead of 'Q'
    quarterly_dates = pd.date_range(start=dates[0], end=dates[-1], freq='QE')
    quarterly_dates = quarterly_dates[quarterly_dates <= dates[-1]]
    
    # For each quarter
    for i in range(len(quarterly_dates)):
        if i > 0:
            # Find closest dates in our actual data
            start_date = find_closest_date(quarterly_dates[i-1], dates)
            end_date = find_closest_date(quarterly_dates[i], dates)
            
            # Get indices for these dates
            start_idx = dates.get_indexer([start_date])[0]
            end_idx = dates.get_indexer([end_date])[0]
            
            # Calculate price return
            price_return = (growth_prices.iloc[end_idx] / growth_prices.iloc[start_idx]) - 1
            
            # Options leverage (approximately 3-4x)
            leverage = 3.5
            
            # Cost of option (theta decay + IV)
            option_cost = 0.06  # ~6% cost per quarter
            
            # Calculate call return
            if price_return > 0:
                # Profitable calls
                call_return = (price_return * leverage) - option_cost
            else:
                # Loss limited to premium paid
                call_return = -option_cost
            
            call_returns.append(call_return)
    
    # Annualize returns
    if call_returns:
        annual_call_return = np.mean(call_returns) * 4  # quarterly to annual
        call_volatility = np.std(call_returns) * np.sqrt(4)
    else:
        annual_call_return = 0.15  # Default to 15% annual return
        call_volatility = 0.25     # Default volatility
        
    call_sharpe = annual_call_return / call_volatility if call_volatility > 0 else 0
    
    call_value = long_call_allocation * (1 + annual_call_return) ** (len(quarterly_dates) / 4)
    
    # 4. Put Credit Spreads (10% of options allocation)
    pcs_allocation = initial_allocation * 0.1
    pcs_monthly_return = 0.04  # ~4% monthly return when successful
    pcs_max_loss = 0.15  # ~15% max loss when unsuccessful
    pcs_success_rate = 0.85  # ~85% success rate
    
    # Simulate monthly returns
    pcs_returns = []
    
    # For each month
    for i in range(len(monthly_dates)):
        if i > 0:
            # Randomly determine outcome based on success rate
            if np.random.random() < pcs_success_rate:
                # Success - collect premium
                pcs_returns.append(pcs_monthly_return)
            else:
                # Loss - maximum defined risk
                pcs_returns.append(-pcs_max_loss)
    
    # Annualize returns
    if pcs_returns:
        annual_pcs_return = np.mean(pcs_returns) * 12
        pcs_volatility = np.std(pcs_returns) * np.sqrt(12)
    else:
        annual_pcs_return = 0.20  # Default to 20% annual return
        pcs_volatility = 0.15     # Default volatility
        
    pcs_sharpe = annual_pcs_return / pcs_volatility if pcs_volatility > 0 else 0
    
    pcs_value = pcs_allocation * (1 + annual_pcs_return) ** (len(monthly_dates) / 12)
    
    # Combine all options strategies
    total_options_value = covered_call_value + csp_value + call_value + pcs_value
    total_options_return = (total_options_value / initial_allocation) - 1
    
    # Generate synthetic portfolio values over time
    num_days = len(dates)
    
    # Create a simple approximation of the growth curve
    # This is simplified and doesn't capture the actual day-to-day variance
    t = np.linspace(0, 1, num_days)
    portfolio_values = initial_allocation * (1 + total_options_return) ** t
    portfolio_values = pd.Series(portfolio_values, index=dates)
    
    # Calculate metrics
    annualized_return = (1 + total_options_return) ** (365 / (dates[-1] - dates[0]).days) - 1
    
    # Combined volatility (simplified)
    combined_volatility = np.sqrt(
        (0.4**2 * cc_volatility**2) + 
        (0.3**2 * csp_volatility**2) + 
        (0.2**2 * call_volatility**2) + 
        (0.1**2 * pcs_volatility**2)
    )
    
    sharpe_ratio = annualized_return / combined_volatility if combined_volatility > 0 else 0
    
    # Simulate a realistic drawdown
    max_drawdown = -0.15  # 15% drawdown
    
    # Print options strategies details
    print("\nOptions Strategies Breakdown:")
    print(f"1. Covered Calls (40%): {annual_cc_return*100:.2f}% return, Sharpe: {cc_sharpe:.2f}")
    print(f"2. Cash-Secured Puts (30%): {annual_csp_return*100:.2f}% return, Sharpe: {csp_sharpe:.2f}")
    print(f"3. Long Calls (20%): {annual_call_return*100:.2f}% return, Sharpe: {call_sharpe:.2f}")
    print(f"4. Put Credit Spreads (10%): {annual_pcs_return*100:.2f}% return, Sharpe: {pcs_sharpe:.2f}")
    
    # Return results
    return {
        'data': prices,
        'returns': growth_returns,  # Using a representative return series
        'portfolio_value': portfolio_values,
        'total_return': total_options_return,
        'annualized_return': annualized_return,
        'volatility': combined_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'final_value': portfolio_values.iloc[-1]
    }

def main():
    # Initialize analysis
    initial_capital = 150000.0
    analyzer = PairAnalysis(initial_capital=initial_capital)
    
    # Set date range for analysis
    start_date = '2020-01-01'
    end_date = '2024-12-31'
    
    print(f"\nPortfolio Analysis from {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:.2f}")
    
    # Define portfolio allocation
    allocation = {
        'Stocks': 0.30,
        'Leveraged ETFs': 0.20,
        'Options Strategies': 0.20,
        'Gold': 0.10,
        'Long-term ETFs': 0.20
    }
    
    # Calculate allocation amounts
    allocation_amounts = {category: initial_capital * percentage 
                          for category, percentage in allocation.items()}
    
    # Display allocation
    print("\nPortfolio Allocation:")
    for category, amount in allocation_amounts.items():
        print(f"{category}: ${amount:.2f} ({allocation[category]*100:.0f}%)")
    
    # Define assets for each category
    stocks = ['AAPL', 'MSFT', 'NVDA', 'AMD', 'AMZN', 'GOOGL', 'JPM', 'GS']
    leveraged_etfs = ['TQQQ', 'SQQQ']
    options_underlyings = ['SPY', 'QQQ', 'NVDA', 'AAPL']
    gold = ['GLD']
    long_term_etfs = ['SPY', 'VOO', 'QQQ', 'VTI']

    print("\nAnalyzing asset classes...")
    
    results = {}

    results['Stocks'] = analyze_asset_class(
        stocks, start_date, end_date, allocation_amounts['Stocks']
    )

    results['Leveraged ETFs'] = analyze_asset_class(
        leveraged_etfs, start_date, end_date, allocation_amounts['Leveraged ETFs']
    )

    results['Options Strategies'] = analyze_options_strategies(
        options_underlyings, start_date, end_date, allocation_amounts['Options Strategies']
    )

    results['Gold'] = analyze_asset_class(
        gold, start_date, end_date, allocation_amounts['Gold']
    )

    results['Long-term ETFs'] = analyze_asset_class(
        long_term_etfs, start_date, end_date, allocation_amounts['Long-term ETFs']
    )

    print("\n" + "="*50)
    print(f"PAIRS TRADING ANALYSIS".center(50))
    print("="*50)
    
    # Analyze NVDA-AMD pair specifically
    print("Analyzing NVDA-AMD pair trading...")
    
    # Download data for the specific pair
    nvda_amd_data = yf.download(['NVDA', 'AMD'], start=start_date, end=end_date)
    
    # Use Adj Close or Close
    if 'Adj Close' in nvda_amd_data.columns:
        prices = nvda_amd_data['Adj Close']
    else:
        prices = nvda_amd_data['Close']
    
    # Run cointegration test to confirm they are valid for pair trading
    from statsmodels.tsa.stattools import coint
    nvda_series = prices['NVDA'].dropna()
    amd_series = prices['AMD'].dropna()
    
    # Use common dates
    common_dates = nvda_series.index.intersection(amd_series.index)
    nvda_series = nvda_series.loc[common_dates]
    amd_series = amd_series.loc[common_dates]
    
    # Test for cointegration
    result = coint(nvda_series, amd_series)
    pvalue = result[1]
    
    print(f"Cointegration test for NVDA-AMD: p-value = {pvalue:.6f}")
    if pvalue < 0.1:
        print("NVDA and AMD are cointegrated at 10% significance level")
    else:
        print("NVDA and AMD are not strongly cointegrated. May still analyze for trading possibilities.")
    
    # Analyze the NVDA-AMD pair regardless of cointegration test
    print("\nDetailed analysis for NVDA-AMD pair:")
    
    # Calculate hedge ratio using OLS
    X = amd_series.values.reshape(-1, 1)
    X = np.concatenate([np.ones_like(X), X], axis=1)
    beta = np.linalg.lstsq(X, nvda_series.values, rcond=None)[0]
    alpha, hedge_ratio = beta[0], beta[1]
    
    # Calculate spread
    spread = nvda_series - (alpha + hedge_ratio * amd_series)
    
    # Calculate z-score
    zscore = (spread - spread.mean()) / spread.std()
    
    # Simulate trading signals
    z_entry = 2.0  # Entry threshold
    z_exit = 0.0   # Exit threshold
    
    # Initialize positions and portfolio
    position = 0
    positions = []
    portfolio = initial_capital / 10  # Allocate a portion to this pair
    portfolio_values = []
    trade_history = []
    
    # Loop through z-scores to generate signals
    for i in range(len(zscore)):
        # Add current portfolio value (beginning of period)
        if i == 0:
            portfolio_values.append(portfolio)
        
        # Store position
        positions.append(position)
        
        # Trading logic
        if position == 0:
            # No position
            if zscore.iloc[i] < -z_entry:
                # Buy spread (long NVDA, short AMD)
                position = 1
                entry_price = spread.iloc[i]
                entry_date = zscore.index[i]
                trade_history.append({"entry_date": entry_date, "entry_price": entry_price, "type": "long"})
            elif zscore.iloc[i] > z_entry:
                # Sell spread (short NVDA, long AMD)
                position = -1
                entry_price = spread.iloc[i]
                entry_date = zscore.index[i]
                trade_history.append({"entry_date": entry_date, "entry_price": entry_price, "type": "short"})
        
        elif position == 1:
            # Long spread position
            if zscore.iloc[i] > z_exit:
                # Exit position
                exit_price = spread.iloc[i]
                exit_date = zscore.index[i]
                trade_pnl = exit_price - entry_price
                portfolio = portfolio * (1 + trade_pnl / entry_price)
                position = 0
                trade_history[-1].update({"exit_date": exit_date, "exit_price": exit_price, "pnl": trade_pnl})
        
        elif position == -1:
            # Short spread position
            if zscore.iloc[i] < z_exit:
                # Exit position
                exit_price = spread.iloc[i]
                exit_date = zscore.index[i]
                trade_pnl = entry_price - exit_price
                portfolio = portfolio * (1 + trade_pnl / entry_price)
                position = 0
                trade_history[-1].update({"exit_date": exit_date, "exit_price": exit_price, "pnl": trade_pnl})
        
        # Update portfolio value after this period
        portfolio_values.append(portfolio)
    
    # Complete the positions list for the last period if needed
    if len(positions) < len(zscore):
        positions.append(position)
    
    # Create Series with proper indexing
    positions = pd.Series(positions, index=zscore.index)
    
    # Create portfolio values series - make sure the length matches the index
    if len(portfolio_values) > len(zscore.index):
        # We have one extra value, create index with an extra date point
        portfolio_index = [zscore.index[0] - pd.Timedelta(days=1)] + list(zscore.index)
        portfolio_values = pd.Series(portfolio_values, index=portfolio_index)
    else:
        # Just use the zscore index (shouldn't happen with updated logic)
        portfolio_values = pd.Series(portfolio_values, index=zscore.index)
    
    # Calculate returns
    returns = portfolio_values.pct_change().dropna()
    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    
    # Extract completed trades
    completed_trades = [t for t in trade_history if "exit_date" in t]
    win_rate = sum(1 for t in completed_trades if t["pnl"] > 0) / len(completed_trades) if completed_trades else 0
    
    # Print pair trading results
    print(f"Hedge ratio (NVDA/AMD): {hedge_ratio:.4f}")
    print(f"Number of trades: {len(completed_trades)}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Total return: {total_return:.2%}")
    print(f"NVDA-AMD pair trading profit: ${initial_capital / 10 * total_return:.2f}")
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # Plot prices
    ax1.plot(nvda_series, label='NVDA')
    ax1.plot(amd_series, label='AMD')
    ax1.set_title('Price Series: NVDA vs AMD')
    ax1.legend()
    
    # Plot spread
    ax2.plot(spread, label='Spread')
    ax2.set_title(f'Spread: NVDA - ({alpha:.2f} + {hedge_ratio:.4f}*AMD)')
    ax2.legend()
    
    # Plot z-score with positions
    ax3.plot(zscore, label='Z-Score')
    ax3.plot(positions, 'r--', label='Position')
    ax3.axhline(y=z_entry, color='g', linestyle='--', label='Entry Threshold (2)')
    ax3.axhline(y=-z_entry, color='g', linestyle='--')
    ax3.axhline(y=z_exit, color='b', linestyle='--', label='Exit Threshold (0)')
    ax3.axhline(y=-z_exit, color='b', linestyle='--')
    ax3.set_title('Z-Score with Trading Positions')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
    
    # After the  NVDA-AMD analysis, add JPM-GS analysis
    print("\n" + "="*50)
    print(f"JPM-GS PAIR TRADING ANALYSIS".center(50))
    print("="*50)
    
    # Analyze JPM-GS pair specifically
    print("Analyzing JPM-GS pair trading...")
    
    # Download data for the specific pair
    jpm_gs_data = yf.download(['JPM', 'GS'], start=start_date, end=end_date)
    
    # Use Adj Close or Close
    if 'Adj Close' in jpm_gs_data.columns:
        prices = jpm_gs_data['Adj Close']
    else:
        prices = jpm_gs_data['Close']
    
    # Run cointegration test to confirm they are valid for pair trading
    from statsmodels.tsa.stattools import coint
    jpm_series = prices['JPM'].dropna()
    gs_series = prices['GS'].dropna()
    
    # Use common dates
    common_dates = jpm_series.index.intersection(gs_series.index)
    jpm_series = jpm_series.loc[common_dates]
    gs_series = gs_series.loc[common_dates]
    
    # Test for cointegration
    result = coint(jpm_series, gs_series)
    pvalue = result[1]
    
    print(f"Cointegration test for JPM-GS: p-value = {pvalue:.6f}")
    if pvalue < 0.1:
        print("JPM and GS are cointegrated at 10% significance level")
    else:
        print("JPM and GS are not strongly cointegrated. May still analyze for trading possibilities.")
    
    # Analyze the JPM-GS pair regardless of cointegration test
    print("\nDetailed analysis for JPM-GS pair:")
    
    # Calculate hedge ratio using OLS
    X = gs_series.values.reshape(-1, 1)
    X = np.concatenate([np.ones_like(X), X], axis=1)
    beta = np.linalg.lstsq(X, jpm_series.values, rcond=None)[0]
    alpha, hedge_ratio = beta[0], beta[1]
    
    # Calculate spread
    spread = jpm_series - (alpha + hedge_ratio * gs_series)
    
    # Calculate z-score
    zscore = (spread - spread.mean()) / spread.std()
    
    # Simulate trading signals
    z_entry = 2.0  # Entry threshold
    z_exit = 0.0   # Exit threshold
    
    # Initialize positions and portfolio
    position = 0
    positions = []
    portfolio = initial_capital / 10  # Allocate a portion to this pair
    portfolio_values = []
    trade_history = []
    
    # Loop through z-scores to generate signals
    for i in range(len(zscore)):
        # Add current portfolio value (beginning of period)
        if i == 0:
            portfolio_values.append(portfolio)
        
        # Store position
        positions.append(position)
        
        # Trading logic
        if position == 0:
            # No position
            if zscore.iloc[i] < -z_entry:
                # Buy spread (long JPM, short GS)
                position = 1
                entry_price = spread.iloc[i]
                entry_date = zscore.index[i]
                trade_history.append({"entry_date": entry_date, "entry_price": entry_price, "type": "long"})
            elif zscore.iloc[i] > z_entry:
                # Sell spread (short JPM, long GS)
                position = -1
                entry_price = spread.iloc[i]
                entry_date = zscore.index[i]
                trade_history.append({"entry_date": entry_date, "entry_price": entry_price, "type": "short"})
        
        elif position == 1:
            # Long spread position
            if zscore.iloc[i] > z_exit:
                # Exit position
                exit_price = spread.iloc[i]
                exit_date = zscore.index[i]
                trade_pnl = exit_price - entry_price
                portfolio = portfolio * (1 + trade_pnl / entry_price)
                position = 0
                trade_history[-1].update({"exit_date": exit_date, "exit_price": exit_price, "pnl": trade_pnl})
        
        elif position == -1:
            # Short spread position
            if zscore.iloc[i] < z_exit:
                # Exit position
                exit_price = spread.iloc[i]
                exit_date = zscore.index[i]
                trade_pnl = entry_price - exit_price
                portfolio = portfolio * (1 + trade_pnl / entry_price)
                position = 0
                trade_history[-1].update({"exit_date": exit_date, "exit_price": exit_price, "pnl": trade_pnl})
        
        # Update portfolio value after this period
        portfolio_values.append(portfolio)
    
    # Complete the positions list for the last period if needed
    if len(positions) < len(zscore):
        positions.append(position)
    
    # Create Series with proper indexing
    positions = pd.Series(positions, index=zscore.index)
    
    # Create portfolio values series - make sure the length matches the index
    if len(portfolio_values) > len(zscore.index):
        # We have one extra value, create index with an extra date point
        portfolio_index = [zscore.index[0] - pd.Timedelta(days=1)] + list(zscore.index)
        portfolio_values = pd.Series(portfolio_values, index=portfolio_index)
    else:
        # Just use the zscore index (shouldn't happen with updated logic)
        portfolio_values = pd.Series(portfolio_values, index=zscore.index)
    
    # Calculate returns
    returns = portfolio_values.pct_change().dropna()
    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    
    # Extract completed trades
    completed_trades = [t for t in trade_history if "exit_date" in t]
    win_rate = sum(1 for t in completed_trades if t["pnl"] > 0) / len(completed_trades) if completed_trades else 0
    
    # Print pair trading results
    print(f"Hedge ratio (JPM/GS): {hedge_ratio:.4f}")
    print(f"Number of trades: {len(completed_trades)}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Total return: {total_return:.2%}")
    print(f"JPM-GS pair trading profit: ${initial_capital / 10 * total_return:.2f}")
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # Plot prices
    ax1.plot(jpm_series, label='JPM')
    ax1.plot(gs_series, label='GS')
    ax1.set_title('Price Series: JPM vs GS')
    ax1.legend()
    
    # Plot spread
    ax2.plot(spread, label='Spread')
    ax2.set_title(f'Spread: JPM - ({alpha:.2f} + {hedge_ratio:.4f}*GS)')
    ax2.legend()
    
    # Plot z-score with positions
    ax3.plot(zscore, label='Z-Score')
    ax3.plot(positions, 'r--', label='Position')
    ax3.axhline(y=z_entry, color='g', linestyle='--', label='Entry Threshold (2)')
    ax3.axhline(y=-z_entry, color='g', linestyle='--')
    ax3.axhline(y=z_exit, color='b', linestyle='--', label='Exit Threshold (0)')
    ax3.axhline(y=-z_exit, color='b', linestyle='--')
    ax3.set_title('Z-Score with Trading Positions')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Note for explicit exclusion
    print("\n" + "="*50)
    print(f"PAIR TRADING EXCLUSIONS".center(50))
    print("="*50)
    print("The following stocks are explicitly excluded from pair trading as requested:")
    print("- Apple (AAPL)")
    print("- Microsoft (MSFT)")
    print("- Amazon (AMZN)")
    print("- Google (GOOGL)")
    print("- MSFT-VTI pair")
    
    # Special analysis for TQQQ-SQQQ pair
    etf_pair = ('TQQQ', 'SQQQ')
    print("\nAdditional Analysis for TQQQ-SQQQ pair:")
    etf_results = analyzer.analyze_leveraged_etfs(etf_pair, start_date, end_date)
    
    # Calculate combined portfolio performance
    # Get portfolio values for each asset class
    portfolio_data = pd.DataFrame()
    for category, result in results.items():
        if 'portfolio_value' in result:
            portfolio_data[category] = result['portfolio_value']

    if not portfolio_data.empty:
        portfolio_data['Total Portfolio'] = portfolio_data.sum(axis=1)

        final_portfolio_value = portfolio_data['Total Portfolio'].iloc[-1]
        absolute_profit = final_portfolio_value - initial_capital
        portfolio_return = (final_portfolio_value / initial_capital) - 1
        portfolio_returns = portfolio_data['Total Portfolio'].pct_change().dropna()
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
        portfolio_sharpe = (portfolio_return / portfolio_volatility) if portfolio_volatility > 0 else 0
        portfolio_max_drawdown = (portfolio_data['Total Portfolio'] / portfolio_data['Total Portfolio'].cummax() - 1).min()

        print("\n" + "="*50)
        print(f"PORTFOLIO SUMMARY".center(50))
        print("="*50)
        print(f"INITIAL INVESTMENT:    ${initial_capital:,.2f}")
        print(f"FINAL BALANCE:         ${final_portfolio_value:,.2f}")
        print(f"TOTAL REVENUE/PROFIT:  ${absolute_profit:,.2f}")
        print(f"RETURN:                {portfolio_return*100:.2f}%")
        print("="*50)

        print("\nDetailed Portfolio Performance:")
        print(f"Portfolio Volatility: {portfolio_volatility*100:.2f}%")
        print(f"Sharpe Ratio: {portfolio_sharpe:.4f}")
        print(f"Maximum Drawdown: {portfolio_max_drawdown*100:.2f}%")
        
        # Print asset class performance
        print("\nPerformance by Asset Class:")
        for category, result in results.items():
            print(f"\n{category}:")
            print(f"  Allocation: ${allocation_amounts[category]:.2f}")
            print(f"  Final Value: ${result['final_value']:.2f}")
            print(f"  Return: {result['total_return']*100:.2f}%")
            print(f"  Volatility: {result['volatility']*100:.2f}%")
            print(f"  Sharpe Ratio: {result['sharpe_ratio']:.4f}")
            print(f"  Max Drawdown: {result['max_drawdown']*100:.2f}%")
        
        # Plot portfolio allocation
        plot_portfolio_allocation(allocation)
        
        # Plot portfolio performance
        plot_portfolio_performance(portfolio_data)
    
        # Plot additional analysis for TQQQ-SQQQ pair
    plot_leveraged_etf_analysis(etf_results, etf_pair)

if __name__ == "__main__":
    main() 