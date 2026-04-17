"""
Equity market data collection using Yahoo Finance and other sources.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from datetime import datetime
import logging
import yfinance as yf

from .base import DataSource
from config.settings import MARKETS, START_DATE, END_DATE

logger = logging.getLogger(__name__)


class EquityDataSource(DataSource):
    """
    Data source for equity market indices and ETFs.
    Uses Yahoo Finance for historical price data.
    """
    
    # Fallback tickers for major markets (ETFs that track indices)
    MARKET_TICKERS = {
        "USA": "SPY",      # S&P 500 ETF
        "Europe": "VGK",   # Vanguard Europe ETF
        "Japan": "EWJ",    # iShares Japan ETF
        "UK": "EWU",       # iShares UK ETF
        "EM": "EEM",       # iShares Emerging Markets ETF
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__("EquityMarkets", cache_dir)
        
    def fetch(
        self,
        markets: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: str = "M"
    ) -> pd.DataFrame:
        """
        Fetch equity market data for specified markets.
        
        Args:
            markets: List of market names (e.g., ["USA", "Europe"])
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            frequency: Data frequency (D=daily, W=weekly, M=monthly)
            
        Returns:
            pd.DataFrame: Price data with markets as columns
        """
        if markets is None:
            markets = list(self.MARKET_TICKERS.keys())
        
        start = start_date or START_DATE
        end = end_date or END_DATE
        
        self.logger.info(f"Fetching equity data for {len(markets)} markets, {start} to {end}")
        
        price_data = {}
        
        for market in markets:
            ticker = self.MARKET_TICKERS.get(market)
            if not ticker:
                self.logger.warning(f"No ticker for market {market}")
                continue
            
            try:
                self.logger.debug(f"Downloading {market} ({ticker})")
                
                # Download from Yahoo Finance
                df = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    progress=False,
                    auto_adjust=True  # Adjust for splits/dividends
                )
                
                if df.empty:
                    self.logger.warning(f"No data for {market} ({ticker})")
                    continue
                
                # Use Adjusted Close if available, else Close
                if 'Adj Close' in df.columns:
                    prices = df['Adj Close']
                elif 'Close' in df.columns:
                    prices = df['Close']
                else:
                    prices = df.iloc[:, 0]  # First column
                
                # Handle MultiIndex columns from yfinance
                if isinstance(prices, pd.DataFrame):
                    prices = prices.iloc[:, 0]
                
                prices.name = market
                price_data[market] = prices
                
                self.logger.debug(f"Downloaded {len(prices)} prices for {market}")
                
            except Exception as e:
                self.logger.error(f"Error downloading {market}: {str(e)}")
                continue
        
        if not price_data:
            self.logger.error("No equity data retrieved")
            return pd.DataFrame()
        
        # Combine all markets
        df = pd.DataFrame(price_data)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Resample to monthly
        if frequency == "M":
            df = df.resample('M').last()
        elif frequency == "W":
            df = df.resample('W').last()
        
        self.logger.info(f"Successfully fetched {len(df)} observations for {len(df.columns)} markets")
        return df
    
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate equity price data."""
        if data.empty:
            self.logger.error("Data is empty")
            return False
        
        if len(data.columns) == 0:
            self.logger.error("No columns in data")
            return False
        
        # Check for negative or zero prices
        if (data <= 0).any().any():
            self.logger.error("Invalid prices (<= 0) found")
            return False
        
        # Check date span
        date_span = (data.index[-1] - data.index[0]).days / 365.25
        if date_span < 10:
            self.logger.warning(f"Short price history: {date_span:.1f} years")
        
        # Check for gaps in data
        for col in data.columns:
            null_pct = data[col].isnull().mean()
            if null_pct > 0.2:
                self.logger.warning(f"{col} has {null_pct:.1%} missing values")
        
        return True
    
    def calculate_returns(
        self,
        prices: pd.DataFrame,
        horizon_months: int = 36,
        annualize: bool = True
    ) -> pd.DataFrame:
        """
        Calculate forward returns for each market.
        
        Args:
            prices: Price DataFrame
            horizon_months: Forward looking period
            annualize: Whether to annualize returns
            
        Returns:
            pd.DataFrame: Forward returns (targets for ML model)
        """
        self.logger.info(f"Calculating {horizon_months}-month forward returns...")
        
        returns = {}
        
        for market in prices.columns:
            # Calculate total return over horizon
            prices_series = prices[market]
            
            # Forward-looking return: (Price(t+n) / Price(t)) - 1
            forward_return = prices_series.shift(-horizon_months) / prices_series - 1
            
            if annualize:
                # Annualize: (1 + total_return)^(12/horizon) - 1
                forward_return = (1 + forward_return) ** (12 / horizon_months) - 1
            
            returns[market] = forward_return
        
        returns_df = pd.DataFrame(returns)
        returns_df = returns_df * 100  # Convert to percentage
        
        self.logger.info(f"Calculated returns for {len(returns_df.columns)} markets")
        return returns_df
    
    def calculate_trailing_returns(
        self,
        prices: pd.DataFrame,
        months: int = 12
    ) -> pd.DataFrame:
        """
        Calculate trailing returns (momentum feature).
        
        Args:
            prices: Price DataFrame
            months: Lookback period
            
        Returns:
            pd.DataFrame: Trailing returns
        """
        trailing = {}
        
        for market in prices.columns:
            prices_series = prices[market]
            
            # Trailing return: (Price(t) / Price(t-n)) - 1
            trailing_return = prices_series / prices_series.shift(months) - 1
            trailing[market] = trailing_return
        
        trailing_df = pd.DataFrame(trailing)
        trailing_df = trailing_df * 100
        
        return trailing_df
    
    def calculate_volatility(
        self,
        prices: pd.DataFrame,
        months: int = 12
    ) -> pd.DataFrame:
        """
        Calculate rolling volatility (standard deviation of returns).
        
        Args:
            prices: Price DataFrame
            months: Rolling window
            
        Returns:
            pd.DataFrame: Rolling volatility (annualized)
        """
        vol = {}
        
        for market in prices.columns:
            # Calculate monthly returns
            monthly_returns = prices[market].pct_change()
            
            # Calculate rolling standard deviation
            rolling_vol = monthly_returns.rolling(window=months).std()
            
            # Annualize
            annualized_vol = rolling_vol * np.sqrt(12)
            
            vol[market] = annualized_vol * 100  # As percentage
        
        vol_df = pd.DataFrame(vol)
        return vol_df
