import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class TradingFees:
    def __init__(self, initial_capital: float = 1500.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.monthly_volume = 0  # For tiered fee calculation
        
    def calculate_stock_fees(self, shares: int, price: float, is_buy: bool = True) -> float:
        """
        Calculate total fees for stock/ETF trading including:
        - Commission
        - Platform fees (fixed)
        - Settlement fees
        - Regulatory fees (for sells)
        - Trading activity fees (for sells)
        """
        # Commission
        commission = max(0.0049 * shares, 0.99)
        
        # Platform fee (fixed)
        platform_fee = max(0.005 * shares, 1.0)
        
        # Settlement fee
        settlement_fee = 0.003 * shares
        
        # Total base fees
        total_fees = commission + platform_fee + settlement_fee
        
        # Additional fees for sells
        if not is_buy:
            # Regulatory fee
            reg_fee = max(0.0000278 * (shares * price), 0.01)
            
            # Trading activity fee
            activity_fee = max(0.000166 * shares, 0.01)
            activity_fee = min(activity_fee, 8.30)
            
            total_fees += reg_fee + activity_fee
        
        # Cap at 0.5% of transaction amount
        max_fee = 0.005 * (shares * price)
        total_fees = min(total_fees, max_fee)
        
        return total_fees
    
    def calculate_option_fees(self, contracts: int, premium: float, is_buy: bool = True) -> float:
        """
        Calculate total fees for options trading including:
        - Commission
        - Platform fees (fixed)
        - Regulatory fees
        - Trading activity fees
        - ORF
        - OCC fees
        - Settlement fees
        """
        # Commission
        if premium > 0.1:
            commission = max(0.65 * contracts, 1.99)
        else:
            commission = max(0.15 * contracts, 1.99)
        
        # Platform fee (fixed)
        platform_fee = 0.3 * contracts
        
        # ORF
        orf_fee = 0.013 * contracts
        
        # OCC fee
        occ_fee = min(0.02 * contracts, 55.0)
        
        # Settlement fee
        settlement_fee = 0.18 * contracts
        
        # Total base fees
        total_fees = commission + platform_fee + orf_fee + occ_fee + settlement_fee
        
        # Additional fees for sells
        if not is_buy:
            # Regulatory fee
            reg_fee = max(0.0000278 * (contracts * premium * 100), 0.01)
            
            # Trading activity fee
            activity_fee = max(0.00279 * contracts, 0.01)
            
            total_fees += reg_fee + activity_fee
        
        return total_fees
    
    def update_capital(self, amount: float, is_buy: bool = True) -> None:
        """Update the current capital after a trade"""
        if is_buy:
            self.current_capital -= amount
        else:
            self.current_capital += amount
    
    def get_current_capital(self) -> float:
        """Get the current capital"""
        return self.current_capital 