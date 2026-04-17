"""
Data collection pipeline that orchestrates all data sources.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
import logging
from pathlib import Path

from config.settings import (
    DATA_RAW_PATH, 
    DATA_PROCESSED_PATH,
    MARKETS, 
    START_DATE, 
    END_DATE
)
from .base import DataCollector
from .world_bank import WorldBankDataSource
from .fred import FREDDataSource
from .equity import EquityDataSource

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Main pipeline for collecting and processing all data.
    """
    
    def __init__(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        markets: Optional[List[str]] = None,
        cache_dir: Optional[str] = None
    ):
        self.start_date = start_date or START_DATE
        self.end_date = end_date or END_DATE
        self.markets = self._parse_markets(markets)
        self.cache_dir = cache_dir or str(DATA_RAW_PATH)
        
        self.collector = DataCollector()
        self.logger = logging.getLogger(__name__)
        self._register_sources()
        
    def _parse_markets(self, markets: Optional[List[str]]) -> List[str]:
        """Parse market specification."""
        if markets is None or markets == ["all"]:
            return list(MARKETS.keys())
        return markets
    
    def _register_sources(self):
        """Register all data sources."""
        # World Bank - global macro data
        self.collector.register_source(
            "world_bank",
            WorldBankDataSource(cache_dir=self.cache_dir)
        )
        
        # FRED - US data and global indicators
        self.collector.register_source(
            "fred",
            FREDDataSource(cache_dir=self.cache_dir)
        )
        
        # Equity markets
        self.collector.register_source(
            "equity",
            EquityDataSource(cache_dir=self.cache_dir)
        )
        
        self.logger.info(f"Registered {len(self.collector.sources)} data sources")
    
    def collect_all(self) -> Dict[str, pd.DataFrame]:
        """
        Collect data from all sources.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of data by source
        """
        self.logger.info("Starting data collection pipeline...")
        self.logger.info(f"Markets: {self.markets}")
        self.logger.info(f"Period: {self.start_date} to {self.end_date}")
        
        results = {}
        
        # 1. Collect World Bank data
        try:
            self.logger.info("Collecting World Bank data...")
            
            # Get country codes for markets
            from .world_bank import CountryMapper
            countries = []
            for market in self.markets:
                countries.extend(CountryMapper.get_countries_for_market(market))
            
            wb_data = self.collector.collect(
                "world_bank",
                countries=countries,
                start_date=self.start_date,
                end_date=self.end_date
            )
            results["world_bank"] = wb_data
            
        except Exception as e:
            self.logger.error(f"World Bank collection failed: {str(e)}")
            results["world_bank"] = None
        
        # 2. Collect FRED data
        try:
            self.logger.info("Collecting FRED data...")
            
            fred_data = self.collector.collect(
                "fred",
                start_date=self.start_date,
                end_date=self.end_date,
                frequency="M"
            )
            results["fred"] = fred_data
            
        except Exception as e:
            self.logger.error(f"FRED collection failed: {str(e)}")
            results["fred"] = None
        
        # 3. Collect Equity data
        try:
            self.logger.info("Collecting Equity data...")
            
            equity_data = self.collector.collect(
                "equity",
                markets=self.markets,
                start_date=self.start_date,
                end_date=self.end_date,
                frequency="M"
            )
            results["equity"] = equity_data
            
        except Exception as e:
            self.logger.error(f"Equity collection failed: {str(e)}")
            results["equity"] = None
        
        # Summary
        for source, data in results.items():
            if data is not None:
                self.logger.info(f"✓ {source}: {data.shape}")
            else:
                self.logger.warning(f"✗ {source}: FAILED")
        
        return results
    
    def process(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Process raw data into unified format.
        
        Args:
            raw_data: Dictionary of DataFrames from different sources
            
        Returns:
            pd.DataFrame: Unified dataset
        """
        self.logger.info("Processing raw data...")
        
        processed_data = {}
        
        # Process each source
        for source_name, data in raw_data.items():
            if data is None or data.empty:
                continue
            
            if source_name == "world_bank":
                processed_data["macro"] = self._process_world_bank(data)
            elif source_name == "fred":
                processed_data["us_data"] = self._process_fred(data)
            elif source_name == "equity":
                if data.empty:
                    self.logger.error("Equity data is empty - critical for predictions")
                    raise ValueError("Equity data collection failed")
                processed_data["equity"] = data
        
        # Combine all sources
        unified = self._combine_sources(processed_data)
        
        self.logger.info(f"Processed data shape: {unified.shape}")
        return unified
    
    def _process_world_bank(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process World Bank data."""
        self.logger.info("Processing World Bank data...")
        
        # Interpolate annual data to monthly
        data_monthly = data.resample('M').interpolate(method='linear')
        
        # Forward fill any remaining gaps
        data_monthly = data_monthly.ffill(limit=3)
        
        return data_monthly
    
    def _process_fred(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process FRED data."""
        self.logger.info("Processing FRED data...")
        
        # Most FRED data is already monthly, just ensure consistency
        data = data.resample('M').last()
        
        # Calculate yield curve spread if both yields available
        if 'yield_10y' in data.columns and 'yield_2y' in data.columns:
            data['yield_curve_spread'] = data['yield_10y'] - data['yield_2y']
        
        # Forward fill sparse data
        data = data.ffill(limit=3)
        
        return data
    
    def _combine_sources(self, processed_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine all processed data sources."""
        self.logger.info("Combining data sources...")
        
        # Start with a date range
        date_range = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq='ME'
        )
        
        combined = pd.DataFrame(index=date_range)
        
        # Join each source
        for source_name, data in processed_data.items():
            if data is not None and not data.empty:
                # Ensure monthly frequency
                data = data.resample('M').last()
                
                # Prefix columns with source name
                data = data.add_prefix(f"{source_name}_")
                
                # Join to combined
                combined = combined.join(data, how='outer')
        
        # Clean up
        combined = combined.dropna(how='all')
        combined = combined.sort_index()
        
        self.logger.info(f"Combined data: {combined.shape[0]} rows, {combined.shape[1]} columns")
        
        return combined
    
    def save(self, data: pd.DataFrame, filename: str = "processed_data.csv"):
        """Save processed data."""
        output_path = DATA_PROCESSED_PATH / filename
        data.to_csv(output_path, index=True)
        self.logger.info(f"Saved processed data to {output_path}")
    
    def load(self, filename: str = "processed_data.csv") -> pd.DataFrame:
        """Load processed data."""
        input_path = DATA_PROCESSED_PATH / filename
        
        if not input_path.exists():
            self.logger.error(f"File not found: {input_path}")
            return pd.DataFrame()
        
        data = pd.read_csv(input_path, index_col=0, parse_dates=True)
        self.logger.info(f"Loaded processed data: {data.shape}")
        return data


def collect_single_source(source_name: str, **kwargs) -> Optional[pd.DataFrame]:
    """
    Collect data from a single source (utility function).
    
    Args:
        source_name: Name of the source (world_bank, fred, equity)
        **kwargs: Source-specific parameters
        
    Returns:
        DataFrame or None
    """
    pipeline = DataPipeline(**kwargs)
    
    try:
        return pipeline.collector.collect(source_name, **kwargs)
    except Exception as e:
        logger.error(f"Failed to collect from {source_name}: {str(e)}")
        return None
