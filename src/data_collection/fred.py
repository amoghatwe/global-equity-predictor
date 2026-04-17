"""
Federal Reserve Economic Data (FRED) collection using fredapi.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from datetime import datetime
import logging

from .base import DataSource
from config.settings import FRED_API_KEY, FRED_SERIES, START_DATE, END_DATE

logger = logging.getLogger(__name__)


class FREDDataSource(DataSource):
    """
    Data source for FRED economic data.
    
    Provides US-centric macro data and global indicators
    available through FRED (Dollar index, VIX, commodity prices, etc.)
    """
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[str] = None):
        super().__init__("FRED", cache_dir)
        self.api_key = api_key or FRED_API_KEY
        
        if not self.api_key:
            self.logger.warning("FRED API key not set. Some features will be unavailable.")
        
        self._fred = None
        
    def _get_fred(self):
        """Lazy initialization of FRED API client."""
        if self._fred is None and self.api_key:
            try:
                from fredapi import Fred
                self._fred = Fred(api_key=self.api_key)
            except Exception as e:
                self.logger.error(f"Failed to initialize FRED API: {str(e)}")
                raise
        return self._fred
    
    def fetch(
        self,
        series_ids: Optional[Dict[str, str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch FRED data for specified series.
        
        Args:
            series_ids: Dictionary of {name: series_code}
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            frequency: Resampling frequency (D, W, M, Q, A)
            
        Returns:
            pd.DataFrame: Wide format DataFrame with series as columns
        """
        if series_ids is None:
            # Get all US series
            series_ids = FRED_SERIES.get("US", {})
        
        start = start_date or START_DATE
        end = end_date or END_DATE
        
        self.logger.info(f"Fetching FRED data for {len(series_ids)} series, {start} to {end}")
        
        if not self.api_key:
            self.logger.error("FRED API key required")
            return pd.DataFrame()
        
        fred = self._get_fred()
        if fred is None:
            return pd.DataFrame()
        
        data_dict = {}
        
        for name, series_id in series_ids.items():
            try:
                self.logger.debug(f"Fetching series {name} ({series_id})")
                series = fred.get_series(
                    series_id,
                    observation_start=start,
                    observation_end=end
                )
                
                if series is not None and not series.empty:
                    data_dict[name] = series
                    self.logger.debug(f"Retrieved {len(series)} observations for {name}")
                else:
                    self.logger.warning(f"No data for series {name} ({series_id})")
                    
            except Exception as e:
                self.logger.error(f"Error fetching {name} ({series_id}): {str(e)}")
                continue
        
        if not data_dict:
            self.logger.error("No data retrieved from FRED")
            return pd.DataFrame()
        
        # Combine all series
        df = pd.DataFrame(data_dict)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Resample if frequency specified
        if frequency:
            df = self._resample(df, frequency)
        
        self.logger.info(f"Successfully fetched {len(df)} rows with {len(df.columns)} series")
        return df
    
    def _resample(self, df: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """Resample data to specified frequency."""
        freq_map = {
            'D': 'D',
            'W': 'W',
            'M': 'M',
            'Q': 'Q',
            'A': 'A'
        }
        
        if frequency in freq_map:
            # Use last observation for each period
            df = df.resample(freq_map[frequency]).last()
            self.logger.info(f"Resampled to {frequency}")
        
        return df
    
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate FRED data."""
        if data.empty:
            self.logger.error("Data is empty")
            return False
        
        if len(data.columns) == 0:
            self.logger.error("No columns in data")
            return False
        
        # Check date range
        date_span = (data.index[-1] - data.index[0]).days / 365.25
        if date_span < 5:
            self.logger.warning(f"Short date span: {date_span:.1f} years")
        
        # Check for stale data (more than 90 days old)
        last_date = data.index[-1]
        days_since_update = (pd.Timestamp.now() - last_date).days
        if days_since_update > 90:
            self.logger.warning(f"Data may be stale. Last update: {last_date}")
        
        return True
    
    def search_series(self, search_text: str, limit: int = 10) -> pd.DataFrame:
        """Search for FRED series by text."""
        fred = self._get_fred()
        if fred is None:
            return pd.DataFrame()
        
        try:
            results = fred.search(search_text, limit=limit)
            return results[['title', 'id', 'frequency', 'units']]
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            return pd.DataFrame()
    
    def get_series_info(self, series_id: str) -> Dict:
        """Get metadata for a series."""
        fred = self._get_fred()
        if fred is None:
            return {}
        
        try:
            info = fred.get_series_info(series_id)
            return {
                'title': info['title'],
                'frequency': info['frequency'],
                'units': info['units'],
                'last_updated': info['last_updated']
            }
        except Exception as e:
            self.logger.error(f"Failed to get series info: {str(e)}")
            return {}
    
    def calculate_yield_curve(self) -> Optional[pd.Series]:
        """Calculate yield curve spread (10Y - 2Y)."""
        try:
            data = self.fetch(series_ids={
                'yield_10y': 'GS10',
                'yield_2y': 'GS2'
            })
            
            if not data.empty and 'yield_10y' in data.columns and 'yield_2y' in data.columns:
                spread = data['yield_10y'] - data['yield_2y']
                spread.name = 'yield_curve_spread'
                return spread
        except Exception as e:
            self.logger.error(f"Failed to calculate yield curve: {str(e)}")
        
        return None
