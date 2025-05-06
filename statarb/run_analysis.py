from statarb.analysis.pair_analysis import PairAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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

def main():
    # Initialize analysis
    initial_capital = 1500.0
    analyzer = PairAnalysis(initial_capital=initial_capital)
    
    # Define leveraged ETF pair
    etf_pair = ('TQQQ', 'SQQQ')
    
    # Set date range for 2024 trading
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    
    print(f"\nAnalyzing TQQQ-SQQQ pair trading from {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:.2f}")
    
    # Run leveraged ETF analysis
    print("\nAnalyzing TQQQ-SQQQ pair:")
    etf_results = analyzer.analyze_leveraged_etfs(etf_pair, start_date, end_date)
    
    # Calculate additional metrics
    data = etf_results['data']
    returns = data.pct_change()
    daily_volatility = returns.std()
    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
    
    # Calculate final balance
    final_balance = initial_capital + etf_results['total_pnl']
    
    print(f"\nPerformance Metrics:")
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Total Profit: ${etf_results['total_pnl']:.2f}")
    print(f"Return: {etf_results['return_pct']:.2f}%")
    print(f"Final Balance: ${final_balance:.2f}")
    
    # Calculate monthly returns
    monthly_returns = data.resample('ME').last().pct_change()
    monthly_returns_pct = monthly_returns * 100
    
    print(f"\nMonthly Returns:")
    for month, ret in monthly_returns_pct.iterrows():
        print(f"{month.strftime('%B %Y')}:")
        print(f"  TQQQ: {ret['TQQQ']:.2f}%")
        print(f"  SQQQ: {ret['SQQQ']:.2f}%")
    
    print(f"\nRisk Metrics:")
    print(f"TQQQ Daily Volatility: {daily_volatility['TQQQ']:.4f}")
    print(f"SQQQ Daily Volatility: {daily_volatility['SQQQ']:.4f}")
    print(f"TQQQ Sharpe Ratio: {sharpe_ratio['TQQQ']:.4f}")
    print(f"SQQQ Sharpe Ratio: {sharpe_ratio['SQQQ']:.4f}")
    
    # Calculate drawdowns
    cum_returns = (1 + returns).cumprod()
    drawdown = (cum_returns / cum_returns.cummax() - 1) * 100
    max_drawdown = drawdown.min()
    
    print(f"\nDrawdown Analysis:")
    print(f"TQQQ Max Drawdown: {max_drawdown['TQQQ']:.2f}%")
    print(f"SQQQ Max Drawdown: {max_drawdown['SQQQ']:.2f}%")
    
    # Plot analysis
    plot_leveraged_etf_analysis(etf_results, etf_pair)

if __name__ == "__main__":
    main() 