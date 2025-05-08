import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union, Any

# Import modules
from statarb.utils.data_utils import fetch_stock_data, preprocess_for_pairs_trading
from statarb.strategies.pairs_trading import PairsTrader, run_johansen_test
from statarb.strategies.kalman_filter import KalmanPairsTrader
from statarb.models.ml_signal_generator import FeatureGenerator, XGBoostModel
from statarb.models.unsupervised_models import AssetClustering
from statarb.utils.market_microstructure import SlippageModel, LiquidityChecker, TransactionCostModel
from statarb.utils.risk_management import PositionSizer, StopLossManager, RiskAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('statarb.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Statistical Arbitrage Trading Strategy')
    parser.add_argument('--start_date', type=str, default='2018-01-01',
                        help='Start date for data (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None, 
                        help='End date for data (YYYY-MM-DD, default: today)')
    parser.add_argument('--tickers_file', type=str, default='data/tickers.csv', 
                        help='File with ticker symbols')
    parser.add_argument('--universe', type=str, default='sp500', 
                        help='Stock universe to use (sp500, nasdaq100, djia, custom)')

    parser.add_argument('--strategy', type=str, default='pairs', 
                        help='Strategy type (pairs, kalman, ml, ensemble)')
    parser.add_argument('--lookback_period', type=int, default=252, 
                        help='Lookback period for calculating statistics')
    parser.add_argument('--formation_period', type=int, default=504, 
                        help='Formation period for pairs selection')
    parser.add_argument('--z_score_entry', type=float, default=2.0, 
                        help='Z-score threshold for entry')
    parser.add_argument('--z_score_exit', type=float, default=0.5, 
                        help='Z-score threshold for exit')
    parser.add_argument('--stop_loss', type=float, default=3.0, 
                        help='Z-score threshold for stop loss')
    
    # Execution parameters
    parser.add_argument('--initial_capital', type=float, default=150000, 
                        help='Initial capital for backtest')
    parser.add_argument('--position_size', type=float, default=0.05, 
                        help='Position size as fraction of capital')
    parser.add_argument('--max_positions', type=int, default=20, 
                        help='Maximum number of open positions')
    parser.add_argument('--transaction_cost', type=float, default=0.0005, 
                        help='Transaction cost per trade')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='Directory for output files')
    parser.add_argument('--plot_results', action='store_true', 
                        help='Generate plots of results')
    parser.add_argument('--save_trades', action='store_true', 
                        help='Save detailed trade data')
    
    args = parser.parse_args()
    
    # Set end date to today if not specified
    if args.end_date is None:
        args.end_date = datetime.today().strftime('%Y-%m-%d')
    
    return args

def load_tickers(universe: str, tickers_file: str = None) -> List[str]:

    if universe == 'custom' and tickers_file:
        # Load custom tickers from file
        try:
            df = pd.read_csv(tickers_file)
            tickers = df['ticker'].tolist()
        except Exception as e:
            logger.error(f"Error loading custom tickers: {str(e)}")
            sys.exit(1)
    
    elif universe == 'sp500':
        tickers = [
            'AAPL', 'MSFT', 'AMZN', 'FB', 'GOOGL', 'GOOG', 'BRK-B', 'JNJ', 'JPM', 'PG', 
            'UNH', 'MA', 'INTC', 'V', 'HD', 'VZ', 'ADBE', 'CRM', 'NFLX', 'DIS',
            'PFE', 'KO', 'MRK', 'PEP', 'BAC', 'CSCO', 'T', 'ABT', 'WMT', 'CMCSA',
            'XOM', 'CVX', 'ABBV', 'COST', 'NKE', 'MCD', 'AMGN', 'MDT', 'LLY', 'TMO'
        ]
    
    elif universe == 'nasdaq100':

        tickers = [
            'AAPL', 'MSFT', 'AMZN', 'FB', 'GOOGL', 'GOOG', 'INTC', 'CSCO', 'CMCSA', 'PEP',
            'ADBE', 'NFLX', 'PYPL', 'NVDA', 'TSLA', 'COST', 'AMGN', 'TCOM', 'SBUX', 'BIDU',
            'AVGO', 'TXN', 'QCOM', 'INTU', 'CHTR', 'MDLZ', 'ISRG', 'ADP', 'AMD', 'ATVI'
        ]
    
    elif universe == 'djia':

        tickers = [
            'AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW',
            'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM',
            'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT'
        ]
    
    else:
        logger.error(f"Unknown universe: {universe}")
        sys.exit(1)
    
    logger.info(f"Loaded {len(tickers)} tickers from {universe} universe")
    return tickers

def run_pairs_trading_strategy(args):
    """
    Run the pairs trading strategy.
    
    Args:
        args: Command line arguments
    """
    # Load tickers
    tickers = load_tickers(args.universe, args.tickers_file)
    
    # Fetch historical data
    logger.info(f"Fetching historical data from {args.start_date} to {args.end_date}")
    price_data = fetch_stock_data(tickers, args.start_date, args.end_date)
    
    # Use Adjusted Close prices
    if 'Adj Close' in price_data.columns.levels[0]:
        prices = price_data['Adj Close']
    else:
        prices = price_data['Close']
    
    # Drop tickers with missing data
    missing_data = prices.isna().sum() > 0
    if missing_data.any():
        missing_tickers = missing_data[missing_data].index.tolist()
        logger.warning(f"Dropping {len(missing_tickers)} tickers with missing data: {missing_tickers}")
        prices = prices.drop(columns=missing_tickers)
    
    # Initialize pairs trading strategy
    pairs_trader = PairsTrader(
        lookback_period=args.lookback_period,
        zscore_entry_threshold=args.z_score_entry,
        zscore_exit_threshold=args.z_score_exit,
        stop_loss_threshold=args.stop_loss,
        max_position_days=args.lookback_period,
        formation_period=args.formation_period
    )
    
    # Find cointegrated pairs
    logger.info("Finding cointegrated pairs...")
    pairs = pairs_trader.find_cointegrated_pairs(prices, pvalue_threshold=0.1)  # Using a more relaxed p-value threshold
    
    if not pairs:
        logger.warning("No cointegrated pairs found")
        return
    
    logger.info(f"Found {len(pairs)} cointegrated pairs")
    for ticker1, ticker2, pvalue in pairs[:10]:
        logger.info(f"Pair: {ticker1} - {ticker2}, p-value: {pvalue:.6f}")
    
    # Run backtest
    logger.info("Running backtest...")
    results = pairs_trader.backtest(
        prices, 
        pairs=pairs[:args.max_positions],  # Use top N pairs
        initial_capital=args.initial_capital,
        position_size=args.position_size
    )
    
    # Print results
    total_return = results['total_return']
    sharpe_ratio = results['sharpe_ratio']
    max_drawdown = results['max_drawdown']
    num_trades = results['num_trades']
    win_rate = results['win_rate']
    
    logger.info(f"Backtest Results:")
    logger.info(f"Total Return: {total_return:.2%}")
    logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    logger.info(f"Max Drawdown: {max_drawdown:.2%}")
    logger.info(f"Number of Trades: {num_trades}")
    logger.info(f"Win Rate: {win_rate:.2%}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot results if requested
    if args.plot_results:
        logger.info("Generating plots...")
        fig = pairs_trader.plot_backtest_results(results)
        plt.savefig(os.path.join(args.output_dir, 'pairs_trading_results.png'))
        
        # Plot sample pairs
        for i, (ticker1, ticker2, _) in enumerate(pairs[:3]):
            fig = pairs_trader.plot_pair_analysis(prices, ticker1, ticker2)
            plt.savefig(os.path.join(args.output_dir, f'pair_analysis_{ticker1}_{ticker2}.png'))
    
    # Save detailed trade data if requested
    if args.save_trades and results['trades']:
        trades_df = pd.DataFrame(results['trades'])
        trades_df.to_csv(os.path.join(args.output_dir, 'pairs_trading_trades.csv'), index=False)
    
    # Save portfolio values
    results['portfolio_values'].to_csv(os.path.join(args.output_dir, 'portfolio_values.csv'))
    
    logger.info("Pairs trading strategy completed successfully")

def run_kalman_pairs_strategy(args):
    """
    Run the Kalman filter pairs trading strategy.
    
    Args:
        args: Command line arguments
    """
    # Load tickers
    tickers = load_tickers(args.universe, args.tickers_file)
    
    # Fetch historical data
    logger.info(f"Fetching historical data from {args.start_date} to {args.end_date}")
    price_data = fetch_stock_data(tickers, args.start_date, args.end_date)
    
    # Use Adjusted Close prices
    if 'Adj Close' in price_data.columns.levels[0]:
        prices = price_data['Adj Close']
    else:
        prices = price_data['Close']
    
    # Drop tickers with missing data
    missing_data = prices.isna().sum() > 0
    if missing_data.any():
        missing_tickers = missing_data[missing_data].index.tolist()
        logger.warning(f"Dropping {len(missing_tickers)} tickers with missing data: {missing_tickers}")
        prices = prices.drop(columns=missing_tickers)
    
    # First, find cointegrated pairs using standard pairs trading
    pairs_trader = PairsTrader(
        lookback_period=args.lookback_period,
        zscore_entry_threshold=args.z_score_entry,
        zscore_exit_threshold=args.z_score_exit,
        stop_loss_threshold=args.stop_loss,
        max_position_days=args.lookback_period,
        formation_period=args.formation_period
    )
    
    # Find cointegrated pairs
    logger.info("Finding cointegrated pairs...")
    pairs = pairs_trader.find_cointegrated_pairs(prices)
    
    if not pairs:
        logger.warning("No cointegrated pairs found")
        return
    
    logger.info(f"Found {len(pairs)} cointegrated pairs")
    
    # Initialize Kalman filter pairs trader
    kalman_trader = KalmanPairsTrader(
        lookback_period=args.lookback_period,
        zscore_entry_threshold=args.z_score_entry,
        zscore_exit_threshold=args.z_score_exit,
        stop_loss_threshold=args.stop_loss,
        max_position_days=args.lookback_period,
        transaction_cost=args.transaction_cost
    )
    
    # Run backtest on top pairs
    all_results = {}
    
    for i, (ticker1, ticker2, pvalue) in enumerate(pairs[:args.max_positions]):
        logger.info(f"Backtesting Kalman filter strategy for pair {ticker1}-{ticker2}")
        
        try:
            # Run backtest with Kalman filter
            pair_results = kalman_trader.backtest(
                prices, 
                ticker1, 
                ticker2,
                include_intercept=True,
                initial_capital=args.initial_capital / args.max_positions,
                position_size=args.position_size
            )
            
            if pair_results:
                all_results[(ticker1, ticker2)] = pair_results
                
                # Log pair results
                pair_return = pair_results['total_return']
                pair_sharpe = pair_results['sharpe_ratio']
                pair_trades = pair_results['num_trades']
                logger.info(f"Pair {ticker1}-{ticker2}: Return {pair_return:.2%}, Sharpe {pair_sharpe:.2f}, Trades {pair_trades}")
                
        except Exception as e:
            logger.error(f"Error backtesting pair {ticker1}-{ticker2}: {str(e)}")
    
    if not all_results:
        logger.warning("No successful backtest results")
        return
    
    # Aggregate results
    combined_portfolio_value = pd.DataFrame()
    total_trades = []
    
    for pair, results in all_results.items():
        ticker1, ticker2 = pair
        
        # Add pair identifier to trades
        for trade in results['trades']:
            trade['pair'] = f"{ticker1}-{ticker2}"
            total_trades.append(trade)
        
        # Combine portfolio values
        portfolio_values = results['portfolio_values']
        if combined_portfolio_value.empty:
            combined_portfolio_value = portfolio_values.to_frame(name=f"{ticker1}-{ticker2}")
        else:
            combined_portfolio_value[f"{ticker1}-{ticker2}"] = portfolio_values
    
    # Calculate combined portfolio
    if not combined_portfolio_value.empty:
        combined_portfolio = combined_portfolio_value.mean(axis=1)
        
        # Calculate performance metrics
        portfolio_returns = combined_portfolio.pct_change().dropna()
        total_return = (combined_portfolio.iloc[-1] / combined_portfolio.iloc[0]) - 1
        sharpe_ratio = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std()
        max_drawdown = (combined_portfolio / combined_portfolio.cummax() - 1.0).min()
        
        logger.info(f"Kalman Filter Strategy Results:")
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {max_drawdown:.2%}")
        logger.info(f"Number of Pairs: {len(all_results)}")
        logger.info(f"Total Trades: {len(total_trades)}")
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Plot results if requested
        if args.plot_results:
            # Plot portfolio
            plt.figure(figsize=(12, 6))
            plt.plot(combined_portfolio)
            plt.title("Kalman Filter Strategy - Portfolio Value")
            plt.xlabel("Date")
            plt.ylabel("Value")
            plt.grid(True)
            plt.savefig(os.path.join(args.output_dir, 'kalman_portfolio_value.png'))
            
            # Plot sample pair analysis
            for pair, results in list(all_results.items())[:3]:
                ticker1, ticker2 = pair
                fig = kalman_trader.plot_hedge_ratio(prices, ticker1, ticker2)
                plt.savefig(os.path.join(args.output_dir, f'kalman_hedge_ratio_{ticker1}_{ticker2}.png'))
                
                fig = kalman_trader.plot_pair_analysis(prices, ticker1, ticker2)
                plt.savefig(os.path.join(args.output_dir, f'kalman_pair_analysis_{ticker1}_{ticker2}.png'))
        
        # Save detailed trade data if requested
        if args.save_trades and total_trades:
            trades_df = pd.DataFrame(total_trades)
            trades_df.to_csv(os.path.join(args.output_dir, 'kalman_trades.csv'), index=False)
        
        # Save portfolio values
        combined_portfolio.to_csv(os.path.join(args.output_dir, 'kalman_portfolio_values.csv'))
        
        logger.info("Kalman filter strategy completed successfully")
    else:
        logger.warning("No portfolio data to analyze")

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which strategy to run
    if args.strategy == 'pairs':
        run_pairs_trading_strategy(args)
    elif args.strategy == 'kalman':
        run_kalman_pairs_strategy(args)
    else:
        logger.error(f"Unsupported strategy: {args.strategy}")
        sys.exit(1)
    
    logger.info("Statistical arbitrage strategy execution completed") 