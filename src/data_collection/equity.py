"""
Equity Market Data Provider.

This module provides a specialized implementation of the BaseDataProvider
to fetch historical equity price data from Yahoo Finance using the 'yfinance' library.

It handles ticker mapping for global markets and calculates various return metrics.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, Dict, List, Union
from datetime import datetime
import logging
from pathlib import Path

from .base import BaseDataProvider
from config.settings import MARKETS, START_DATE, END_DATE

# Set up module logger
logger = logging.getLogger(__name__)


class EquityMarketProvider(BaseDataProvider):
    """
    Data provider for global equity market indices.
    
    This provider uses popular ETFs as proxies for major equity markets
    (e.g., SPY for USA, EEM for Emerging Markets).
    """
    
    # Map market names to liquid ETF tickers for historical price discovery
    MARKET_TICKER_MAP = {
        "USA": "SPY",      # S&P 500
        "Europe": "VGK",   # Vanguard FTSE Europe
        "Japan": "EWJ",    # iShares MSCI Japan
        "UK": "EWU",       # iShares MSCI United Kingdom
        "EM": "EEM",       # iShares MSCI Emerging Markets
    }
    
    def __init__(self, cache_directory: Optional[Union[str, Path]] = None):
        """Initialize the equity provider."""
        super().__init__("EquityMarkets", cache_directory)
        
    def fetch_data(
        self,
        target_markets: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: str = "ME",
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch historical price data for the specified markets.
        
        Args:
            target_markets: List of market names (USA, Europe, etc.).
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            frequency: Pandas frequency (ME=month-end).
            
        Returns:
            pd.DataFrame: A wide DataFrame of price levels.
        """
        if not target_markets:
            target_markets = list(self.MARKET_TICKER_MAP.keys())
        
        start = start_date or START_DATE
        end = end_date or END_DATE
        
        self.logger.info(f"Downloading equity data for {len(target_markets)} markets from Yahoo Finance")
        
        market_prices = {}
        
        for market in target_markets:
            ticker_symbol = self.MARKET_TICKER_MAP.get(market)
            if not ticker_symbol:
                self.logger.warning(f"No ticker mapping found for market: {market}")
                continue
            
            try:
                self.logger.debug(f"Fetching {market} ({ticker_symbol})...")
                
                # Use yfinance to download adjusted closing prices
                ticker_data = yf.download(
                    ticker_symbol,
                    start=start,
                    end=end,
                    progress=False,
                    auto_adjust=True  # Automatically adjusts for splits and dividends
                )
                
                if ticker_data is None or ticker_data.empty:
                    self.logger.warning(f"Yahoo Finance returned no data for {ticker_symbol}")
                    continue
                
                # Extract the price series (favoring 'Close' which is auto-adjusted)
                if 'Close' in ticker_data.columns:
                    price_series = ticker_data['Close']
                else:
                    # Fallback to the first available column
                    price_series = ticker_data.iloc[:, 0]
                
                # Handle cases where yfinance returns a DataFrame instead of a Series for one ticker
                if isinstance(price_series, pd.DataFrame):
                    price_series = price_series.iloc[:, 0]
                
                price_series.name = market
                market_prices[market] = price_series
                
                self.logger.debug(f"Successfully retrieved {len(price_series)} price points for {market}")
                
            except Exception as download_error:
                self.logger.error(f"Failed to download data for {market} ({ticker_symbol}): {str(download_error)}")
                continue
        
        if not market_prices:
            self.logger.error("Failed to retrieve any equity data")
            return pd.DataFrame()
        
        # Combine all market series into one DataFrame
        combined_df = pd.DataFrame(market_prices)
        combined_df.index = pd.to_datetime(combined_df.index)
        combined_df = combined_df.sort_index()
        
        # Resample to the requested frequency (typically month-end)
        # Use 'ME' to stay compliant with modern pandas
        target_freq = 'ME' if frequency == 'M' else frequency
        combined_df = combined_df.resample(target_freq).last()
        
        self.logger.info(f"Final equity dataset shape: {combined_df.shape}")
        return combined_df
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate the integrity of the equity price data.
        """
        if data is None or data.empty:
            self.logger.error("Validation failed: Empty dataset")
            return False
        
        # Prices should never be negative or zero
        if (data <= 0).any().any():
            self.logger.error("Validation failed: Non-positive prices detected")
            return False
            
        # Check for significant gaps
        missing_values_per_market = data.isnull().mean()
        for market, missing_pct in missing_values_per_market.items():
            if missing_pct > 0.1:
                self.logger.warning(f"Market '{market}' has {missing_pct:.1%} missing price points")
        
        return True

    @staticmethod
    def compute_forward_returns(
        prices: pd.DataFrame, 
        horizon_months: int = 36,
        annualized: bool = True
    ) -> pd.DataFrame:
        """
        Calculate future returns for use as ML targets.
        
        Args:
            prices: DataFrame of historical prices.
            horizon_months: How many months forward to look.
            annualized: Whether to return annualized returns or total returns.
            
        Returns:
            pd.DataFrame: Forward returns in percentage.
        """
        # Calculate the total change over the horizon
        # (Price at T + horizon / Price at T) - 1
        total_return = prices.shift(-horizon_months) / prices - 1
        
        if annualized:
            # Formula: (1 + total_return)^(12 / horizon) - 1
            calculated_returns = (1 + total_return) ** (12 / horizon_months) - 1
        else:
            calculated_returns = total_return
            
        return calculated_returns * 100  # Return as percentage (e.g., 8.5 instead of 0.085)
