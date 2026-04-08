"""
World Bank data collection using wbdata API.
"""
import pandas as pd
import wbdata
from typing import List, Dict, Optional
from datetime import datetime
import logging

from .base import DataSource
from config.settings import WB_INDICATORS, START_DATE, END_DATE

logger = logging.getLogger(__name__)


class WorldBankDataSource(DataSource):
    """
    Data source for World Bank macroeconomic indicators.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__("WorldBank", cache_dir)
        
    def fetch(
        self,
        countries: List[str],
        indicators: Optional[Dict[str, str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: str = "annual"
    ) -> pd.DataFrame:
        """
        Fetch World Bank data for specified countries and indicators.
        
        Args:
            countries: List of country codes (ISO 3-letter)
            indicators: Dictionary of {indicator_name: indicator_code}
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            frequency: Data frequency (annual, quarterly, monthly)
            
        Returns:
            pd.DataFrame: Wide format DataFrame with indicators as columns
        """
        if indicators is None:
            indicators = WB_INDICATORS
            
        start_year = datetime.strptime(start_date or START_DATE, "%Y-%m-%d").year
        end_year = datetime.strptime(end_date or END_DATE, "%Y-%m-%d").year
        
        self.logger.info(f"Fetching World Bank data for {len(countries)} countries, "
                        f"{len(indicators)} indicators, {start_year}-{end_year}")
        
        try:
            # Fetch data from World Bank
            # date parameter accepts tuple for date range, freq for periodicity
            data = wbdata.get_dataframe(
                indicators=indicators,
                country=countries,
                date=(datetime(start_year, 1, 1), datetime(end_year, 12, 31)),
                freq='M'  # Monthly frequency
            )
            
            if data.empty:
                self.logger.warning("No data returned from World Bank")
                return pd.DataFrame()
            
            # Reset index to make date and country columns
            data = data.reset_index()
            
            # Pivot to wide format: date x (country_indicator)
            data['indicator'] = data['indicator'].map({v: k for k, v in indicators.items()})
            data = data.pivot_table(
                index='date',
                columns=['country', 'indicator'],
                values='value'
            )
            
            # Flatten multi-level columns
            data.columns = [f"{col[0]}_{col[1]}" for col in data.columns]
            data.index = pd.to_datetime(data.index)
            data = data.sort_index()
            
            self.logger.info(f"Successfully fetched {len(data)} rows with {len(data.columns)} columns")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching World Bank data: {str(e)}")
            raise
    
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate World Bank data."""
        if data.empty:
            self.logger.error("Data is empty")
            return False
            
        # Check for required columns
        if len(data.columns) == 0:
            self.logger.error("No columns in data")
            return False
        
        # Check for reasonable data range
        if len(data) < 10:
            self.logger.warning(f"Very few data points: {len(data)}")
        
        # Check for excessive missing values
        missing_pct = data.isnull().mean().mean()
        if missing_pct > 0.8:
            self.logger.error(f"Too many missing values: {missing_pct:.1%}")
            return False
            
        return True
    
    def get_country_list(self) -> List[str]:
        """Get list of available countries."""
        countries = wbdata.get_country()
        return [c['id'] for c in countries]


class CountryMapper:
    """
    Maps market names to World Bank country codes.
    """
    
    MARKET_TO_COUNTRY = {
        "USA": "USA",
        "Europe": "DEU",  # Use Germany as proxy for Europe
        "Japan": "JPN",
        "UK": "GBR",
        "EM": "CHN",  # Use China as major EM proxy (will need composite for EM)
    }
    
    # For Emerging Markets, we'll need multiple countries
    EM_COUNTRIES = [
        "CHN", "IND", "BRA", "RUS", "ZAF",  # BRICS
        "MEX", "IDN", "TUR", "SAU", "ARG",  # Major EM
        "THA", "MYS", "PHL", "CHL", "POL",  # Other EM
    ]
    
    @classmethod
    def get_country_code(cls, market: str) -> str:
        """Get World Bank country code for a market."""
        return cls.MARKET_TO_COUNTRY.get(market, market)
    
    @classmethod
    def get_countries_for_market(cls, market: str) -> List[str]:
        """Get all country codes for a market (especially for EM)."""
        if market == "EM":
            return cls.EM_COUNTRIES
        else:
            return [cls.get_country_code(market)]
