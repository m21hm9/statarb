# StatArb: Statistical Arbitrage Trading Platform

A comprehensive Python platform for statistical arbitrage trading strategies, featuring basket-based stat arb, machine learning models, and advanced risk management tools.

## Features

- **Core Trading Engine**:
  - Regime-switching, factor-based **multivariate basket stat arb** (GMM + HMM)
  - Per-asset-class basket configs for **equities, FX, crypto, gold/silver**
  - Optional Machine Learning signal generation (XGBoost)

- **Data Sources**:
  - Alpaca Markets API integration
  - Yahoo Finance (`yfinance`) support
  - Real-time and historical data fetching

- **Advanced Analytics**:
  - Cointegration testing (Johansen test)
  - Z-score based entry/exit signals
  - Dynamic hedge ratio calculation
  - Correlation and spread analysis

- **Risk Management**:
  - Position sizing algorithms
  - Stop-loss management
  - Transaction cost modeling
  - Liquidity checking
  - Slippage modeling

- **Visualization**:
  - Interactive charts and plots
  - Performance metrics visualization
  - Signal analysis graphs
  - Correlation heatmaps

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

## Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd statarb
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (for Alpaca API):

   You can either export them directly:

   ```bash
   export ALPACA_API_KEY="your_api_key"
   export ALPACA_API_SECRET="your_secret_key"
   export ALPACA_DATA_BASE_URL="https://data.alpaca.markets"
   ```

   Or, more conveniently, create a `.env` file in the project root (next to `README.md`) using `.env.example` as a template:

   ```bash
   cp .env.example .env
   # then edit .env and fill in your real keys
   ```

   The library will load this file automatically via `statarb.config`.

## Quick Start

### Regime‑Switching Factor Model & Basket Analysis

From the project root (where `README.md` and `.env` live), run:

```bash
python run_analysis.py
```

This will:

- Analyze a diversified set of baskets (equities, gold/silver, FX, crypto).
- Build per-asset-class factor sets (momentum, volatility, dispersion).
- Fit a **GMM-HMM regime model** on the factor space for each basket.
- Apply a **regime-aware basket stat arb strategy** (e.g. mean reversion in MR regimes).
- Print key metrics including:
  - Total portfolio return (from basket stat arb).
  - Per‑basket P&L and returns.

The script also generates plots for:

- Portfolio allocation and portfolio equity curve.

## Project Structure

```text
statarb/
  statarb/                     # Main package
    analysis/                  # Analysis modules (e.g. pair_analysis)
    models/                    # ML models, regime detector, signal generators
    strategies/                # Trading strategies
    utils/                     # Utility functions
    main.py                    # (legacy) CLI interface
    run_analysis.py            # Library entry point (used by top-level run_analysis.py)
  run_analysis.py              # Top-level script you run (`python run_analysis.py`)
  requirements.txt             # Dependencies
  setup.py                     # Package setup
```

## Configuration

### Environment variables (.env)

For Alpaca and any other secrets, you can either:

- **Export real environment variables** (e.g. in your shell profile):

  ```bash
  export ALPACA_API_KEY="your_api_key"
  export ALPACA_API_SECRET="your_secret_key"
  export ALPACA_DATA_BASE_URL="https://data.alpaca.markets"
  ```

- **Or use a `.env` file** in the project root (next to `README.md`):
  - Copy the example: `cp .env.example .env`
  - Edit `.env` and fill in your real values.
  - The library loads this automatically via `statarb.config.load_env()`, and you can access structured Alpaca settings with:

    ```python
    from statarb.config import get_alpaca_credentials

    alpaca_cfg = get_alpaca_credentials()
    api_key = alpaca_cfg["api_key"]
    api_secret = alpaca_cfg["api_secret"]
    base_url = alpaca_cfg["data_base_url"]
    ```

### Strategy Parameters (Basket Stat Arb)

Key configuration lives in `statarb/run_analysis.py` and the following modules:

- `statarb/models/factor_models.py`:
  - Factor definitions per asset class (e.g. `EquityFactorModel`, `FXFactorModel`, `CryptoFactorModel`, `MetalFactorModel`).
- `statarb/models/unsupervised_models.py`:
  - `RegimeDetector`: GMM + HMM regime model (`n_gmm_components`, `n_regimes`).
- `statarb/strategies/basket_stat_arb.py`:
  - `BasketStatArbConfig`:
    - `tradable_regimes`: which regimes are allowed to trade (e.g. `[0]` for mean-reverting).
    - `gross_leverage`: target sum of absolute weights per basket.
    - `transaction_cost_bps`: transaction cost in basis points.

### Data Sources

- **Alpaca Markets**: Professional-grade market data
- **Yahoo Finance**: Free historical data via `yfinance`
- **Custom Data**: Support for CSV and other formats

## Example Outputs

The platform generates comprehensive analysis including:

- **Price Charts**: Normalized price movements and ratios
- **Statistical Analysis**: Correlation matrices, beta calculations
- **Trading Signals**: Z-score plots with entry/exit points
- **Performance Metrics**: Returns, Sharpe ratios, drawdowns
- **Risk Analysis**: Volatility, VaR, and stress testing

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m "Add amazing feature"`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Always do your own research and consider consulting with a financial advisor before making investment decisions.

## Support

For questions, issues, or contributions, please open an issue on GitHub or contact the maintainers.

---

**Happy Trading!**

