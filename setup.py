from setuptools import setup, find_packages

setup(
    name="statarb",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "yfinance>=0.1.70",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
) 