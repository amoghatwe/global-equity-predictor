"""
World Bank Data Provider.

This module provides a specialized implementation of the BaseDataProvider
to fetch macroeconomic indicators from the World Bank's Open Data API
using the 'wbdata' library.
"""

try:
    import wbdata
except SystemError:
    # Handle corrupted wbdata cache which happens on Python 3.12
    import os
    import shutil
    import appdirs
    cache_dir = appdirs.user_cache_dir('wbdata')
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
    import wbdata
import pandas as pd
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
import logging
from pathlib import Path
import time

from .base import BaseDataProvider
from config.settings import WB_INDICATORS, START_DATE, END_DATE

# Set up module logger
logger = logging.getLogger(__name__)


class WorldBankProvider(BaseDataProvider):
    """
    Data provider for World Bank macroeconomic indicators.
    """
    
    def __init__(self, cache_directory: Optional[Union[str, Path]] = None):
        """Initialize the World Bank provider."""
        super().__init__("WorldBank", cache_directory)
        
    def fetch_data(
        self,
        countries: List[str],
        indicators: Optional[Dict[str, str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch World Bank data with fallback to individual country requests.
        """
        if not indicators:
            indicators = WB_INDICATORS
            
        try:
            start_year = datetime.strptime(start_date or START_DATE, "%Y-%m-%d").year
            end_year = datetime.strptime(end_date or END_DATE, "%Y-%m-%d").year
        except ValueError:
            self.logger.error(f"Invalid date format: {start_date} or {end_date}")
            return pd.DataFrame()
        
        date_range = (datetime(start_year, 1, 1), datetime(end_year, 12, 31))
        
        # Try batch fetch first
        try:
            self.logger.info(f"Attempting batch fetch for {len(countries)} countries...")
            # We use skip_cache=True to avoid potential corrupted local cache issues
            raw_data = wbdata.get_dataframe(
                indicators=indicators, 
                country=countries, 
                date=date_range,
                skip_cache=True
            )
            if raw_data is not None and not raw_data.empty:
                return self._process_raw_response(raw_data)
        except Exception as e:
            self.logger.warning(f"Batch fetch failed: {str(e)}. Switching to individual requests...")
        
        # Fallback: Fetch one by one to isolate failures
        all_country_data = []
        for country in countries:
            try:
                self.logger.debug(f"Fetching data for {country}...")
                # Small delay to avoid hammering the API
                time.sleep(0.1)
                data = wbdata.get_dataframe(
                    indicators=indicators, 
                    country=[country], 
                    date=date_range,
                    skip_cache=True
                )
                if data is not None and not data.empty:
                    processed = self._process_raw_response(data)
                    all_country_data.append(processed)
                else:
                    self.logger.warning(f"No data returned for {country}")
            except Exception as e:
                self.logger.error(f"Failed to fetch data for {country}: {str(e)}")
                
        if not all_country_data:
            self.logger.error("All World Bank collection attempts failed.")
            return pd.DataFrame()
            
        # Combine all individual results
        # Since each 'processed' is a wide DF with "Country_Indicator" columns,
        # we can just concat along columns.
        combined = pd.concat(all_country_data, axis=1)
        # Handle duplicated columns if any
        combined = combined.loc[:, ~combined.columns.duplicated()]
        return combined.sort_index()
    
    def _process_raw_response(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms wbdata response into wide format.
        
        The input DF from wbdata.get_dataframe has:
        Index: MultiIndex(['country', 'date']) or Index(['date']) if only one country
        Columns: Indicator names (the values from the indicators dict)
        """
        try:
            # If it's a MultiIndex, we want to flatten it to columns
            if isinstance(df.index, pd.MultiIndex):
                # Reset index to move 'country' and 'date' to columns
                df = df.reset_index()
                
                # We want columns to be "Country_Indicator"
                # First, ensure 'date' is datetime
                df['date'] = pd.to_datetime(df['date'])
                
                # Pivot to get date as index and (country, indicator) as columns
                # But it's easier to just iterate if we have MultiIndex
                # Actually, let's use a simpler approach:
                # Group by country and rename columns
                
                countries = df['country'].unique()
                indicator_cols = [c for c in df.columns if c not in ['country', 'date']]
                
                results = []
                for country in countries:
                    country_df = df[df['country'] == country].copy()
                    country_df = country_df.set_index('date')
                    # Drop the country column as it's no longer needed in the data
                    country_df = country_df.drop(columns=['country'], errors='ignore')
                    # Prepend country to indicator names
                    country_df.columns = [f"{country}_{col}" for col in country_df.columns]
                    results.append(country_df)
                
                wide_df = pd.concat(results, axis=1)
                return wide_df.sort_index()
            else:
                # Single country response (only date in index)
                # We don't know the country name easily here unless we pass it in
                # But wait, if we call it with [country] (list), it might still return MultiIndex
                # Let's check if 'country' column exists after reset_index
                temp_df = df.reset_index()
                if 'country' in temp_df.columns:
                    # Same logic as above
                    return self._process_raw_response(df)
                else:
                    # Truly single index. This happens when only one country AND only one indicator 
                    # OR specific wbdata versions.
                    # This case is rare with our usage.
                    df.index = pd.to_datetime(df.index)
                    return df.sort_index()
            
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            return pd.DataFrame()

    def validate_data(self, data: pd.DataFrame) -> bool:
        if data is None or data.empty: 
            return False
        # Check if we have at least some non-NaN values
        if data.isna().all().all():
            return False
        return True


class MarketToCountryMapper:
    """
    Maps market regions to specific World Bank country codes.
    """
    MARKET_MAPPING = {
        "USA": ["USA"],
        "Europe": ["DEU", "FRA", "ITA", "ESP"],
        "Japan": ["JPN"],
        "UK": ["GBR"],
        "EM": ["CHN", "IND", "BRA", "ZAF", "MEX", "IDN", "TUR", "SAU", "ARG"],
    }
    
    @classmethod
    def get_country_codes_for_market(cls, market_name: str) -> List[str]:
        return cls.MARKET_MAPPING.get(market_name, [market_name])
    
    @classmethod
    def get_all_codes(cls, markets: List[str]) -> List[str]:
        codes = set()
        for market in markets:
            codes.update(cls.get_country_codes_for_market(market))
        return sorted(list(codes))
