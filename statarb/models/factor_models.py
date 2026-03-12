import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseFactorModel(ABC):
    """
    Base class for factor models.

    Each implementation takes a panel of prices (rows: dates, columns: tickers)
    and returns a DataFrame of time-series factors (rows: dates, columns: factors).
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config: Dict[str, Any] = config or {}

    @abstractmethod
    def build_factors(self, prices: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class _GenericBasketFactorModel(BaseFactorModel):
    """
    Generic basket factor model used as a simple default for all asset classes.

    Factors (per date):
    - mean_ret_5d, mean_ret_20d: average basket return over 5/20 days
    - vol_20d: realized basket volatility over 20 days
    - cross_section_dispersion_20d: average cross-sectional std of returns over 20 days
    """

    def build_factors(self, prices: pd.DataFrame) -> pd.DataFrame:
        prices = prices.dropna(how="all")
        returns = prices.pct_change().dropna()

        # Basket-level mean returns
        basket_ret = returns.mean(axis=1)
        mean_ret_5d = basket_ret.rolling(window=5).mean()
        mean_ret_20d = basket_ret.rolling(window=20).mean()

        # Basket-level volatility
        vol_20d = basket_ret.rolling(window=20).std()

        # Cross-sectional dispersion of returns:
        # std across assets each day, then 20-day rolling mean
        cs_std_daily = returns.std(axis=1)
        cs_dispersion_20d = cs_std_daily.rolling(window=20).mean()

        factors = pd.DataFrame(
            {
                "mean_ret_5d": mean_ret_5d,
                "mean_ret_20d": mean_ret_20d,
                "vol_20d": vol_20d,
                "cross_section_dispersion_20d": cs_dispersion_20d,
            }
        ).dropna()

        return factors


class EquityFactorModel(_GenericBasketFactorModel):
    """
    Equity basket factor model.

    Currently uses the generic basket factors; left as a separate class so
    equity-specific factors can be added later (e.g. value, size, quality).
    """

    pass


class FXFactorModel(_GenericBasketFactorModel):
    """
    FX basket factor model.

    Uses the same core factors for now; can be extended with carry, rate
    differentials, etc.
    """

    pass


class CryptoFactorModel(_GenericBasketFactorModel):
    """
    Crypto basket factor model.

    Uses the same core factors for now; can be extended with on-chain or
    funding-rate features.
    """

    pass


class MetalFactorModel(_GenericBasketFactorModel):
    """
    Metals (e.g., Gold/Silver) basket factor model.

    Uses the same core factors for now.
    """

    pass

