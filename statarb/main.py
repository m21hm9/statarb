"""
Legacy CLI entry point.

This module previously contained pairs-trading and Kalman filter strategies.
Those have been removed in favor of the new regime-switching multivariate
stat arb pipeline exposed via the top-level `run_analysis.py` script.

Usage:
    python run_analysis.py
"""

if __name__ == "__main__":
    print("This CLI is deprecated. Please run `python run_analysis.py` to use the new regime-switching multivariate stat arb pipeline.")