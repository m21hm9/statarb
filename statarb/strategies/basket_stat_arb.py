from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class BasketStatArbConfig:
    """
    Configuration for basket stat-arb strategy.

    Attributes:
        tradable_regimes: list of regime indices that are allowed to trade.
        gross_leverage: target sum of absolute weights per day.
        transaction_cost_bps: round-trip transaction cost in basis points.
    """

    tradable_regimes: List[int]
    gross_leverage: float = 1.0
    transaction_cost_bps: float = 0.0


class BasketStatArbStrategy:
    """
    Regime-aware multivariate basket statistical arbitrage.

    - Uses per-asset daily returns.
    - Uses regime probabilities to decide when to deploy capital.
    - Within tradable regimes, generates cross-sectional mean-reversion weights.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        factors: pd.DataFrame,
        regime_probs: pd.DataFrame,
        config: BasketStatArbConfig,
    ):
        self.prices = prices.dropna(how="all")
        self.factors = factors
        self.regime_probs = regime_probs
        self.config = config

        # Align all inputs on a common date index (intersection)
        common_index = (
            self.prices.index
            .intersection(self.factors.index)
            .intersection(self.regime_probs.index)
        )
        self.prices = self.prices.loc[common_index]
        self.factors = self.factors.loc[common_index]
        self.regime_probs = self.regime_probs.loc[common_index]

        # Simple daily returns
        self.returns = self.prices.pct_change().dropna()

        # Align again after returns calculation
        common_index = (
            self.returns.index
            .intersection(self.factors.index)
            .intersection(self.regime_probs.index)
        )
        self.returns = self.returns.loc[common_index]
        self.factors = self.factors.loc[common_index]
        self.regime_probs = self.regime_probs.loc[common_index]

    def generate_signals(self) -> pd.DataFrame:
        """
        Generate cross-sectional mean-reversion weights conditional on regimes.

        Logic:
            - For each day, pick the most likely regime.
            - If regime is in tradable_regimes:
                - Compute cross-sectional z-score of returns.
                - Go long underperformers, short outperformers.
                - Scale to match target gross leverage.
            - Else: weights are zero.
        """
        if self.returns.empty:
            raise ValueError("No returns available to generate signals.")

        index = self.returns.index
        weights = pd.DataFrame(0.0, index=index, columns=self.returns.columns)

        # Use 12-day cumulative returns for cross-sectional signal
        lookback = 12
        cum_ret = self.prices.pct_change(lookback)
        cum_ret = cum_ret.loc[index]  # align with returns index

        # Precompute most-likely regime per day and soft tradable probability
        regime_idx = self.regime_probs.values.argmax(axis=1)
        tradable_mask = pd.Series(False, index=self.regime_probs.index)
        for k in self.config.tradable_regimes:
            tradable_mask |= (regime_idx == k)

        prob_tradable = self.regime_probs[[f"regime_{k}" for k in self.config.tradable_regimes]].sum(axis=1)
        # Soft exposure: only trade when >= 60% confident in tradable regimes
        exposure = np.clip(prob_tradable - 0.6, 0.0, 1.0)
        exposure = exposure.reindex(index).fillna(0.0)

        # Dispersion filter: only trade when cross-sectional dispersion is high
        disp = None
        if "cross_section_dispersion_20d" in self.factors.columns:
            disp = self.factors["cross_section_dispersion_20d"].reindex(index)
            disp_threshold_series = disp.rolling(window=252, min_periods=60).quantile(0.65)
        else:
            disp_threshold_series = None

        for date in index:
            if exposure.loc[date] <= 0.0:
                continue

            if disp is not None:
                th = disp_threshold_series.loc[date]
                if pd.isna(th) or disp.loc[date] < th:
                    continue

            cs_ret = cum_ret.loc[date]
            if cs_ret.isna().all():
                continue

            # Percentile rank-based cross-sectional signal (robust for small N)
            pct_ranks = cs_ret.rank(method="average", pct=True)  # (0,1]
            # Map to [-1, 1]: underperformers (~0) -> +1, outperformers (~1) -> -1
            rank_scores = -(pct_ranks - 0.5) * 2.0

            raw_weights = rank_scores.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            gross = raw_weights.abs().sum()
            if gross == 0:
                continue

            scaled_weights = (raw_weights / gross) * self.config.gross_leverage
            # Apply soft exposure
            scaled_weights = scaled_weights * exposure.loc[date]
            weights.loc[date] = scaled_weights

        return weights

    def backtest(self, initial_capital: float) -> Dict[str, Any]:
        """
        Backtest the strategy and return portfolio equity and metrics.
        """
        weights = self.generate_signals()

        # Lag weights by one day to avoid look-ahead bias
        lagged_weights = weights.shift(1).fillna(0.0)

        # Strategy returns before costs
        strat_returns = (lagged_weights * self.returns).sum(axis=1)

        # Simple transaction cost model: cost proportional to daily turnover
        cost_perc = self.config.transaction_cost_bps / 10000.0
        turnover = (lagged_weights.diff().abs().sum(axis=1)).fillna(0.0)
        costs = turnover * cost_perc

        net_returns = strat_returns - costs

        equity = initial_capital * (1.0 + net_returns).cumprod()

        total_pnl = equity.iloc[-1] - initial_capital
        return_pct = (equity.iloc[-1] / initial_capital - 1.0) * 100.0 if initial_capital != 0 else 0.0

        return {
            "weights": weights,
            "returns": net_returns,
            "equity": equity,
            "total_pnl": total_pnl,
            "return_pct": return_pct,
        }

