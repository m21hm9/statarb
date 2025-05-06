"""
Machine Learning Strategy Runner for Statistical Arbitrage

This script implements the machine learning components of the statistical arbitrage strategy,
including supervised learning for signal generation and unsupervised learning for asset clustering.
"""

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
from statarb.utils.data_utils import fetch_stock_data, preprocess_for_pairs_trading, calculate_returns
from statarb.models.ml_signal_generator import FeatureGenerator, XGBoostModel
from statarb.models.unsupervised_models import AssetClustering
from statarb.utils.market_microstructure import TransactionCostModel
from statarb.utils.risk_management import PositionSizer, StopLossManager, RiskAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_strategy.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Machine Learning Strategy for Statistical Arbitrage')
    
    # Data parameters
    parser.add_argument('--start_date', type=str, default='2018-01-01', 
                        help='Start date for data (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None, 
                        help='End date for data (YYYY-MM-DD, default: today)')
    parser.add_argument('--tickers_file', type=str, default='data/tickers.csv', 
                        help='File with ticker symbols')
    parser.add_argument('--universe', type=str, default='sp500', 
                        help='Stock universe to use (sp500, nasdaq100, djia, custom)')
    
    # ML model parameters
    parser.add_argument('--model_type', type=str, default='xgboost', 
                        help='ML model type (xgboost, cluster)')
    parser.add_argument('--prediction_horizon', type=int, default=5, 
                        help='Prediction horizon in days')
    parser.add_argument('--train_test_split', type=float, default=0.8, 
                        help='Train-test split ratio')
    parser.add_argument('--feature_lookback', type=int, default=60, 
                        help='Lookback period for feature generation')
    parser.add_argument('--num_features', type=int, default=20, 
                        help='Number of features to use')
    
    # Execution parameters
    parser.add_argument('--initial_capital', type=float, default=1000000, 
                        help='Initial capital for backtest')
    parser.add_argument('--position_size', type=float, default=0.02, 
                        help='Position size as fraction of capital')
    parser.add_argument('--max_positions', type=int, default=20, 
                        help='Maximum number of open positions')
    parser.add_argument('--transaction_cost', type=float, default=0.0005, 
                        help='Transaction cost per trade')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results_ml', 
                        help='Directory for output files')
    parser.add_argument('--plot_results', action='store_true', 
                        help='Generate plots of results')
    parser.add_argument('--save_model', action='store_true', 
                        help='Save trained model')
    parser.add_argument('--save_trades', action='store_true', 
                        help='Save detailed trade data')
    
    args = parser.parse_args()
    
    # Set end date to today if not specified
    if args.end_date is None:
        args.end_date = datetime.today().strftime('%Y-%m-%d')
    
    return args

def load_tickers(universe: str, tickers_file: str = None) -> List[str]:
    """
    Load ticker symbols based on specified universe.
    
    Args:
        universe: Stock universe to use ('sp500', 'nasdaq100', 'djia', 'custom')
        tickers_file: File with ticker symbols for custom universe
        
    Returns:
        List of ticker symbols
    """
    if universe == 'custom' and tickers_file:
        # Load custom tickers from file
        try:
            df = pd.read_csv(tickers_file)
            tickers = df['ticker'].tolist()
        except Exception as e:
            logger.error(f"Error loading custom tickers: {str(e)}")
            sys.exit(1)
    
    elif universe == 'sp500':
        # S&P 500 components (sample)
        tickers = [
            'AAPL', 'MSFT', 'AMZN', 'FB', 'GOOGL', 'GOOG', 'BRK-B', 'JNJ', 'JPM', 'PG', 
            'UNH', 'MA', 'INTC', 'V', 'HD', 'VZ', 'ADBE', 'CRM', 'NFLX', 'DIS',
            'PFE', 'KO', 'MRK', 'PEP', 'BAC', 'CSCO', 'T', 'ABT', 'WMT', 'CMCSA',
            'XOM', 'CVX', 'ABBV', 'COST', 'NKE', 'MCD', 'AMGN', 'MDT', 'LLY', 'TMO'
        ]
    
    elif universe == 'nasdaq100':
        # NASDAQ 100 components (sample)
        tickers = [
            'AAPL', 'MSFT', 'AMZN', 'FB', 'GOOGL', 'GOOG', 'INTC', 'CSCO', 'CMCSA', 'PEP',
            'ADBE', 'NFLX', 'PYPL', 'NVDA', 'TSLA', 'COST', 'AMGN', 'TCOM', 'SBUX', 'BIDU',
            'AVGO', 'TXN', 'QCOM', 'INTU', 'CHTR', 'MDLZ', 'ISRG', 'ADP', 'AMD', 'ATVI'
        ]
    
    elif universe == 'djia':
        # Dow Jones components
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

def run_xgboost_strategy(args):
    """
    Run the XGBoost ML strategy for signal generation.
    
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
    
    # Calculate returns
    returns = calculate_returns(prices, method='log')
    
    # Initialize feature generator
    feature_gen = FeatureGenerator(lookback_periods=[5, 10, 20, 60, 120])
    
    # Create features for all tickers
    logger.info("Generating features...")
    features = feature_gen.create_price_features(prices)
    
    # Cutoff date for train-test split
    total_days = (prices.index[-1] - prices.index[0]).days
    train_days = int(total_days * args.train_test_split)
    cutoff_date = (prices.index[0] + timedelta(days=train_days)).strftime('%Y-%m-%d')
    
    # Train models for each ticker
    models = {}
    predictions = {}
    results = {}
    
    for ticker in prices.columns:
        logger.info(f"Training model for {ticker}...")
        
        try:
            # Create XGBoost model
            model = XGBoostModel(
                prediction_horizon=args.prediction_horizon,
                train_test_split_ratio=args.train_test_split,
                feature_selection_method='importance',
                num_features=args.num_features
            )
            
            # Train the model
            ticker_features = features.copy()
            ticker_results = model.train(
                ticker_features, 
                f'{ticker}_return_1d',
                cutoff_date=cutoff_date
            )
            
            # Make predictions
            ticker_predictions = model.predict(ticker_features)
            
            # Store model and predictions
            models[ticker] = model
            predictions[ticker] = ticker_predictions
            results[ticker] = ticker_results
            
            # Log model performance
            metrics = ticker_results['metrics']
            logger.info(f"Model for {ticker}: RMSE = {metrics['rmse']:.6f}, R² = {metrics['r2']:.6f}")
            
            # Save model if requested
            if args.save_model:
                model_dir = os.path.join(args.output_dir, 'models')
                os.makedirs(model_dir, exist_ok=True)
                model.save_model(model_dir, f"xgboost_{ticker}")
        
        except Exception as e:
            logger.error(f"Error training model for {ticker}: {str(e)}")
    
    if not models:
        logger.error("No models were successfully trained")
        return
    
    # Combine predictions and generate signals
    logger.info("Generating trading signals...")
    signals = pd.DataFrame(index=prices.index)
    
    for ticker, ticker_predictions in predictions.items():
        # Generate signals from predictions
        ticker_signals = models[ticker].generate_signals(
            ticker_predictions, 
            threshold=0.001,
            long_percentile=0.8,
            short_percentile=0.2
        )
        
        # Store signals
        signals[f'{ticker}_prediction'] = ticker_signals['prediction']
        signals[f'{ticker}_signal'] = ticker_signals['signal']
    
    # Backtest the strategy
    logger.info("Backtesting the strategy...")
    
    # Initialize position sizer and transaction cost model
    position_sizer = PositionSizer(
        max_position_size=args.position_size,
        max_pair_capital=0.1,
        volatility_scaling=True
    )
    
    transaction_model = TransactionCostModel(
        commission_rate=args.transaction_cost,
        exchange_fee_rate=0.0001,
        market_impact_factor=0.1
    )
    
    # Initialize backtest variables
    capital = args.initial_capital
    positions = {}
    trades = []
    daily_returns = []
    
    # Track portfolio value
    portfolio_values = pd.Series(index=prices.index, dtype=float)
    portfolio_values.iloc[0] = capital
    
    # Start after prediction horizon to avoid lookahead bias
    for day_idx in range(args.prediction_horizon + 60, len(prices)):
        day = prices.index[day_idx]
        prev_day = prices.index[day_idx-1]
        
        # Process existing positions
        for ticker, position in list(positions.items()):
            entry_day = position['entry_day']
            entry_idx = prices.index.get_loc(entry_day)
            position_days = day_idx - entry_idx
            
            # Calculate current value
            quantity = position['quantity']
            entry_price = position['entry_price']
            current_price = prices.loc[day, ticker]
            
            # Calculate P&L
            if position['side'] == 'long':
                pnl = (current_price / entry_price) - 1
            else:  # short
                pnl = 1 - (current_price / entry_price)
            
            position_value = position['value'] * (1 + pnl)
            
            # Check exit conditions
            should_exit = False
            exit_reason = ""
            
            # Exit if signal reverses
            signal = signals.loc[day, f'{ticker}_signal']
            if (position['side'] == 'long' and signal < 0) or (position['side'] == 'short' and signal > 0):
                should_exit = True
                exit_reason = "signal reversal"
            
            # Exit if held for too long
            if position_days >= args.prediction_horizon * 3:
                should_exit = True
                exit_reason = "time stop"
            
            # Exit if loss exceeds threshold
            if pnl < -0.05:
                should_exit = True
                exit_reason = "stop loss"
            
            # Process exit if needed
            if should_exit:
                # Apply transaction costs
                exit_costs = transaction_model.calculate_transaction_cost(
                    current_price, quantity, position['side'] == 'long'
                )
                
                # Update capital
                capital += position_value - exit_costs['total_cost']
                
                # Record trade
                trades.append({
                    'ticker': ticker,
                    'side': position['side'],
                    'entry_date': entry_day,
                    'exit_date': day,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'quantity': quantity,
                    'entry_value': position['value'],
                    'exit_value': position_value,
                    'pnl': pnl,
                    'pnl_value': position_value - position['value'],
                    'exit_reason': exit_reason,
                    'holding_period': position_days
                })
                
                # Remove position
                del positions[ticker]
        
        # Find new entry opportunities
        available_capital = capital * args.position_size
        max_new_positions = args.max_positions - len(positions)
        
        if max_new_positions > 0 and available_capital > 0:
            # Get signals for the day
            day_signals = signals.loc[day]
            
            # Find tickers with strong signals
            long_candidates = []
            short_candidates = []
            
            for col in day_signals.index:
                if '_signal' in col:
                    ticker = col.split('_')[0]
                    signal = day_signals[col]
                    
                    # Skip if already in a position
                    if ticker in positions:
                        continue
                    
                    # Add to candidates if signal is strong
                    if signal > 0:
                        long_candidates.append((ticker, signal))
                    elif signal < 0:
                        short_candidates.append((ticker, abs(signal)))
            
            # Sort candidates by signal strength
            long_candidates.sort(key=lambda x: x[1], reverse=True)
            short_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Take top candidates
            candidates = long_candidates[:max_new_positions//2] + short_candidates[:max_new_positions//2]
            
            # Open positions for each candidate
            for ticker, signal_strength in candidates:
                # Calculate position size based on volatility
                ticker_volatility = returns[ticker].rolling(window=20).std().iloc[day_idx]
                position_value = position_sizer.calculate_position_size(
                    available_capital / len(candidates),
                    volatility=ticker_volatility,
                    signal_strength=signal_strength
                )
                
                # Get current price
                current_price = prices.loc[day, ticker]
                
                # Calculate quantity
                quantity = position_value / current_price
                
                # Determine position side
                side = 'long' if (ticker, signal_strength) in long_candidates else 'short'
                
                # Apply transaction costs
                entry_costs = transaction_model.calculate_transaction_cost(
                    current_price, quantity, side == 'long'
                )
                
                # Adjust position value for costs
                adjusted_value = position_value - entry_costs['total_cost']
                
                # Open position
                positions[ticker] = {
                    'side': side,
                    'entry_day': day,
                    'entry_price': current_price,
                    'quantity': quantity,
                    'value': adjusted_value,
                    'signal_strength': signal_strength
                }
                
                # Update capital
                capital -= position_value
        
        # Calculate portfolio value
        current_portfolio_value = capital
        for ticker, position in positions.items():
            current_price = prices.loc[day, ticker]
            quantity = position['quantity']
            entry_price = position['entry_price']
            
            # Calculate position value
            if position['side'] == 'long':
                pnl = (current_price / entry_price) - 1
            else:  # short
                pnl = 1 - (current_price / entry_price)
                
            position_value = position['value'] * (1 + pnl)
            current_portfolio_value += position_value
        
        # Record portfolio value
        portfolio_values[day] = current_portfolio_value
        
        # Calculate daily return
        daily_return = (portfolio_values[day] / portfolio_values[prev_day]) - 1
        daily_returns.append(daily_return)
    
    # Close any remaining positions at the end of the backtest
    final_day = prices.index[-1]
    for ticker, position in list(positions.items()):
        entry_day = position['entry_day']
        entry_idx = prices.index.get_loc(entry_day)
        position_days = len(prices) - 1 - entry_idx
        
        # Calculate P&L
        quantity = position['quantity']
        entry_price = position['entry_price']
        current_price = prices.loc[final_day, ticker]
        
        if position['side'] == 'long':
            pnl = (current_price / entry_price) - 1
        else:  # short
            pnl = 1 - (current_price / entry_price)
        
        position_value = position['value'] * (1 + pnl)
        
        # Apply transaction costs
        exit_costs = transaction_model.calculate_transaction_cost(
            current_price, quantity, position['side'] == 'long'
        )
        
        # Update capital
        capital += position_value - exit_costs['total_cost']
        
        # Record trade
        trades.append({
            'ticker': ticker,
            'side': position['side'],
            'entry_date': entry_day,
            'exit_date': final_day,
            'entry_price': entry_price,
            'exit_price': current_price,
            'quantity': quantity,
            'entry_value': position['value'],
            'exit_value': position_value,
            'pnl': pnl,
            'pnl_value': position_value - position['value'],
            'exit_reason': 'end of backtest',
            'holding_period': position_days
        })
    
    # Calculate performance metrics
    if daily_returns:
        portfolio_returns = pd.Series(daily_returns, index=prices.index[args.prediction_horizon+61:])
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[args.prediction_horizon+60]) - 1
        sharpe_ratio = np.sqrt(252) * portfolio_returns.mean() / portfolio_returns.std()
        max_drawdown = (portfolio_values / portfolio_values.cummax() - 1.0).min()
        win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades) if trades else 0
        
        logger.info(f"XGBoost Strategy Results:")
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"Max Drawdown: {max_drawdown:.2%}")
        logger.info(f"Number of Trades: {len(trades)}")
        logger.info(f"Win Rate: {win_rate:.2%}")
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Plot results if requested
        if args.plot_results:
            # Plot portfolio value
            plt.figure(figsize=(12, 6))
            plt.plot(portfolio_values)
            plt.title("XGBoost Strategy - Portfolio Value")
            plt.xlabel("Date")
            plt.ylabel("Value")
            plt.grid(True)
            plt.savefig(os.path.join(args.output_dir, 'xgboost_portfolio_value.png'))
            
            # Plot sample feature importance
            for ticker, model in list(models.items())[:3]:
                fig = model.plot_feature_importance()
                plt.savefig(os.path.join(args.output_dir, f'xgboost_feature_importance_{ticker}.png'))
                
                fig = model.plot_predictions(results[ticker])
                plt.savefig(os.path.join(args.output_dir, f'xgboost_predictions_{ticker}.png'))
        
        # Save detailed trade data if requested
        if args.save_trades and trades:
            trades_df = pd.DataFrame(trades)
            trades_df.to_csv(os.path.join(args.output_dir, 'xgboost_trades.csv'), index=False)
        
        # Save portfolio values
        portfolio_values.to_csv(os.path.join(args.output_dir, 'xgboost_portfolio_values.csv'))
        
        logger.info("XGBoost strategy completed successfully")
    else:
        logger.warning("No trades executed")

def run_clustering_strategy(args):
    """
    Run the unsupervised clustering strategy for statistical arbitrage.
    
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
    
    # Calculate returns
    returns = calculate_returns(prices, method='log')
    
    # Initialize clustering model
    asset_clustering = AssetClustering(n_clusters=5)
    
    # Generate features for clustering
    logger.info("Generating features for clustering...")
    return_features = asset_clustering.generate_return_features(prices)
    correlation_features = asset_clustering.generate_correlation_features(returns)
    
    # Combine features
    all_features = pd.concat([return_features, correlation_features], axis=1)
    
    # Run clustering
    logger.info("Running asset clustering...")
    cluster_results = asset_clustering.fit(all_features, use_pca=True, n_components=5)
    
    # Find pairs within clusters
    logger.info("Finding pairs within clusters...")
    pairs = asset_clustering.find_pairs_within_clusters(returns, min_correlation=0.7)
    
    logger.info(f"Found {len(pairs)} highly correlated pairs within clusters")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot clustering results if requested
    if args.plot_results:
        logger.info("Generating cluster plots...")
        
        # Plot clusters in 2D
        fig = asset_clustering.plot_clusters_2d(all_features)
        plt.savefig(os.path.join(args.output_dir, 'clusters_2d.png'))
        
        # Plot dendrogram
        fig = asset_clustering.plot_dendrogram(all_features)
        plt.savefig(os.path.join(args.output_dir, 'clusters_dendrogram.png'))
        
        # Plot correlation network
        fig = asset_clustering.plot_correlation_network(returns, min_correlation=0.7)
        plt.savefig(os.path.join(args.output_dir, 'correlation_network.png'))
    
    # Save clustering results
    if args.save_model:
        model_dir = os.path.join(args.output_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        asset_clustering.save_model(model_dir, "asset_clustering")
    
    # Save cluster data
    clusters_df = cluster_results['clusters']
    clusters_df.to_csv(os.path.join(args.output_dir, 'asset_clusters.csv'), index=False)
    
    # Save pairs data
    pairs_df = pd.DataFrame(pairs, columns=['ticker1', 'ticker2', 'correlation'])
    pairs_df.to_csv(os.path.join(args.output_dir, 'cluster_pairs.csv'), index=False)
    
    logger.info("Clustering strategy completed successfully")

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which strategy to run
    if args.model_type == 'xgboost':
        run_xgboost_strategy(args)
    elif args.model_type == 'cluster':
        run_clustering_strategy(args)
    else:
        logger.error(f"Unsupported model type: {args.model_type}")
        sys.exit(1)
    
    logger.info("Machine learning strategy execution completed") 