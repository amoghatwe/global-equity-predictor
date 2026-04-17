"""
Federal Reserve Economic Data (FRED) Provider.

This module provides a specialized implementation of the BaseDataProvider
to fetch economic data from the St. Louis Fed's FRED API using the 'fredapi' library.

It provides US-centric macroeconomic data as well as global indicators
like the US Dollar Index (DXY) and the VIX volatility index.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Union
from datetime import datetime
import logging
from pathlib import Path

from .base import BaseDataProvider
from config.settings import FRED_API_KEY, FRED_SERIES, START_DATE, END_DATE

# Set up module logger
logger = logging.getLogger(__name__)


class FredDataProvider(BaseDataProvider):
    """
    Data provider for FRED economic indicators.
    
    FRED provides high-frequency (daily/monthly) data for many indicators,
    making it a primary source for the US market and global risk indicators.
    """
    
    def __init__(self, api_key: Optional[str] = None, cache_directory: Optional[Union[str, Path]] = None):
        """
        Initialize the FRED provider.
        
        Args:
            api_key: FRED API key. If not provided, will look for FRED_API_KEY in env.
            cache_directory: Where to store cached data.
        """
        super().__init__("FRED", cache_directory)
        self.api_key = api_key or FRED_API_KEY
        
        # Check for placeholder or missing API key
        if not self.api_key or self.api_key == "YOUR-API-KEY":
            self.logger.warning(
                "FRED API key is not configured. Data collection from FRED will fail. "
                "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
            )
            self._is_active = False
        else:
            self._is_active = True
        
        # Lazy-loaded FRED client
        self._fred_client = None
        
    def _get_client(self):
        """
        Initialize and return the FRED API client.
        
        Returns:
            fredapi.Fred: The authenticated FRED client.
            
        Raises:
            ImportError: If fredapi is not installed.
            ValueError: If the API key is invalid.
        """
        if self._fred_client is None:
            if not self._is_active:
                raise ValueError("FRED API key is missing or invalid.")
                
            try:
                from fredapi import Fred
                self._fred_client = Fred(api_key=self.api_key)
            except ImportError:
                self.logger.error("The 'fredapi' library is not installed.")
                raise
            except Exception as e:
                self.logger.error(f"Failed to initialize FRED client: {str(e)}")
                raise
                
        return self._fred_client
    
    def fetch_data(
        self,
        series_definitions: Optional[Dict[str, str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        resampling_frequency: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch multiple data series from FRED.
        
        Args:
            series_definitions: Dict of {friendly_name: fred_id}.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            resampling_frequency: Pandas frequency string (e.g., 'ME' for month-end).
            
        Returns:
            pd.DataFrame: A wide DataFrame with each series as a column.
        """
        if not series_definitions:
            # Default to US series if none provided
            series_definitions = FRED_SERIES.get("US", {})
        
        start = start_date or START_DATE
        end = end_date or END_DATE
        
        self.logger.info(f"Fetching {len(series_definitions)} series from FRED ({start} to {end})")
        
        if not self._is_active:
            self.logger.error("Cannot fetch: FRED provider is inactive (no API key)")
            return pd.DataFrame()
            
        try:
            client = self._get_client()
        except Exception:
            return pd.DataFrame()
        
        downloaded_series = {}
        
        for friendly_name, fred_id in series_definitions.items():
            try:
                self.logger.debug(f"Downloading series: {friendly_name} ({fred_id})")
                
                # Fetch the series
                data = client.get_series(
                    fred_id,
                    observation_start=start,
                    observation_end=end
                )
                
                if data is not None and not data.empty:
                    downloaded_series[friendly_name] = data
                else:
                    self.logger.warning(f"No data returned for series {fred_id}")
                    
            except Exception as e:
                # Individual series failure shouldn't stop the whole process
                self.logger.error(f"Error fetching series '{friendly_name}' ({fred_id}): {str(e)}")
                continue
        
        if not downloaded_series:
            self.logger.error("Failed to retrieve any data from FRED")
            return pd.DataFrame()
        
        # Combine all series into one wide DataFrame
        df = pd.DataFrame(downloaded_series)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Apply resampling if requested (e.g., converting daily VIX to monthly)
        if resampling_frequency:
            # We use 'ME' instead of 'M' to avoid deprecation warnings in newer pandas
            target_freq = 'ME' if resampling_frequency == 'M' else resampling_frequency
            df = df.resample(target_freq).last()
            self.logger.info(f"Resampled FRED data to frequency: {target_freq}")
        
        self.logger.info(f"Successfully collected {len(df.columns)} FRED series")
        return df
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate the FRED dataset for completeness.
        """
        if data is None or data.empty:
            self.logger.error("Validation failed: Dataset is empty")
            return False
        
        if not data.columns.any():
            self.logger.error("Validation failed: No columns found")
            return False
        
        # Check if data is stale
        # FRED is usually updated frequently; warn if last data point is > 90 days old
        latest_date = data.index[-1]
        days_ago = (pd.Timestamp.now() - latest_date).days
        if days_ago > 90:
            self.logger.warning(f"FRED data might be stale. Last observation is from {days_ago} days ago.")
            
        return True
    
    def search_for_series(self, query: str, result_limit: int = 5) -> pd.DataFrame:
        """
        Utility to search for FRED series IDs by keyword.
        
        Args:
            query: Search term (e.g., "interest rates").
            result_limit: Max number of results to return.
            
        Returns:
            pd.DataFrame: Matching series with their IDs and metadata.
        """
        try:
            client = self._get_client()
            results = client.search(query, limit=result_limit)
            return results[['title', 'id', 'frequency', 'units']]
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            return pd.DataFrame()
