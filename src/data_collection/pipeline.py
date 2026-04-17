"""
Data Collection Pipeline.

This module orchestrates the entire data gathering process by coordinating
multiple data providers (World Bank, FRED, Equity Markets). It handles:
1. Provider initialization and registration.
2. Sequential data collection with error handling.
3. Merging disparate data sources into a single unified time series.
4. Feature preprocessing (interpolation, cleaning).
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any
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
from .base import DataCollectionManager
from .world_bank import WorldBankProvider, MarketToCountryMapper
from .fred import FredDataProvider
from .equity import EquityMarketProvider

# Set up module logger
logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """
    Orchestrator for the data ingestion and transformation layer.
    
    This class is the main entry point for data collection. It manages the
    lifecycle of data from raw API responses to a cleaned, merged CSV file.
    """
    
    def __init__(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        target_markets: Optional[List[str]] = None,
        cache_directory: Optional[str] = None
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            start_date: Beginning of data period (YYYY-MM-DD).
            end_date: End of data period (YYYY-MM-DD).
            target_markets: List of market names to include.
            cache_directory: Directory for raw data storage.
        """
        self.start_date = start_date or START_DATE
        self.end_date = end_date or END_DATE
        self.markets = self._initialize_market_list(target_markets)
        self.cache_dir = cache_directory or str(DATA_RAW_PATH)
        
        # Initialize the collection manager and register providers
        self.manager = DataCollectionManager()
        self._setup_providers()
        
    def _initialize_market_list(self, markets: Optional[List[str]]) -> List[str]:
        """Convert input market specification to a validated list of names."""
        if markets is None or markets == ["all"]:
            return list(MARKETS.keys())
        return markets
    
    def _setup_providers(self):
        """Instantiate and register the specialized data providers."""
        # 1. World Bank - Global Macroeconomic Data
        self.manager.register_provider(
            "world_bank",
            WorldBankProvider(cache_directory=self.cache_dir)
        )
        
        # 2. FRED - Economic Indicators (US and Global)
        self.manager.register_provider(
            "fred",
            FredDataProvider(cache_directory=self.cache_dir)
        )
        
        # 3. Equity Markets - Historical Prices and Returns
        self.manager.register_provider(
            "equity",
            EquityMarketProvider(cache_directory=self.cache_dir)
        )
        
        logger.info(f"Initialized data pipeline with {len(self.manager.providers)} providers")
    
    def run_collection_sequence(self) -> Dict[str, pd.DataFrame]:
        """
        Execute the end-to-end collection for all providers.
        
        Returns:
            Dict: Collection results keyed by provider ID.
        """
        logger.info(f"Starting data collection for markets: {self.markets}")
        logger.info(f"Time range: {self.start_date} to {self.end_date}")
        
        results = {}
        
        # 1. Collect Macro data from World Bank
        try:
            country_codes = MarketToCountryMapper.get_all_codes(self.markets)
            results["world_bank"] = self.manager.collect_from_source(
                "world_bank",
                countries=country_codes,
                start_date=self.start_date,
                end_date=self.end_date
            )
        except Exception as e:
            logger.warning(f"World Bank collection failed (non-critical): {str(e)}")
            results["world_bank"] = None
            
        # 2. Collect Economic indicators from FRED
        try:
            results["fred"] = self.manager.collect_from_source(
                "fred",
                start_date=self.start_date,
                end_date=self.end_date,
                resampling_frequency="ME"
            )
        except Exception as e:
            logger.warning(f"FRED collection failed (non-critical): {str(e)}")
            results["fred"] = None
            
        # 3. Collect Price data from Equity Markets
        try:
            results["equity"] = self.manager.collect_from_source(
                "equity",
                target_markets=self.markets,
                start_date=self.start_date,
                end_date=self.end_date,
                frequency="ME"
            )
        except Exception as e:
            # Equity data is CRITICAL for the model; we let this exception propagate
            logger.error(f"CRITICAL: Equity data collection failed: {str(e)}")
            raise
            
        return results
    
    def transform_and_merge(self, raw_datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Transform raw datasets and merge them into a single analytical table.
        
        Workflow:
        - Upsample annual data (World Bank) to monthly.
        - Handle missing values (forward fill).
        - Prefix columns to maintain traceability.
        - Join everything on a monthly date index.
        """
        logger.info("Transforming and merging raw datasets...")
        
        processed_components = {}
        
        # Process World Bank (Macro)
        if raw_datasets.get("world_bank") is not None:
            wb_data = raw_datasets["world_bank"]
            # Annual to Monthly: linear interpolation for smooth macro trends
            wb_monthly = wb_data.resample('ME').interpolate(method='linear')
            # Limit forward fill to 12 months to avoid stale data propagation
            processed_components["macro"] = wb_monthly.ffill(limit=12)
            
        # Process FRED
        if raw_datasets.get("fred") is not None:
            fred_data = raw_datasets["fred"]
            # Calculate additional features like yield curve spread
            if 'yield_10y' in fred_data.columns and 'yield_2y' in fred_data.columns:
                fred_data['yield_curve_spread'] = fred_data['yield_10y'] - fred_data['yield_2y']
            processed_components["economic"] = fred_data.ffill(limit=3)
            
        # Process Equity
        if raw_datasets.get("equity") is not None:
            processed_components["prices"] = raw_datasets["equity"]
            
        # Combine all components
        return self._create_unified_dataset(processed_components)
    
    def _create_unified_dataset(self, components: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Joins multiple DataFrames on a shared monthly index."""
        # Create a master monthly index
        master_index = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq='ME'
        )
        unified_df = pd.DataFrame(index=master_index)
        
        for name, component_df in components.items():
            if component_df is not None and not component_df.empty:
                # Ensure component has the correct frequency
                component_df = component_df.resample('ME').last()
                
                # Add source prefix to column names (e.g., macro_USA_gdp)
                component_df = component_df.add_prefix(f"{name}_")
                
                # Outer join to preserve all dates
                unified_df = unified_df.join(component_df, how='outer')
        
        # Clean up rows that are entirely empty
        unified_df = unified_df.dropna(how='all').sort_index()
        
        logger.info(f"Created unified dataset with {unified_df.shape[1]} features across {len(unified_df)} months")
        return unified_df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_data.csv"):
        """Save the final cleaned dataset to the processed data directory."""
        output_path = DATA_PROCESSED_PATH / filename
        df.to_csv(output_path, index=True)
        logger.info(f"Analytical dataset saved to: {output_path}")

    @classmethod
    def load_processed_data(cls, filename: str = "processed_data.csv") -> pd.DataFrame:
        """Load an existing analytical dataset from disk."""
        input_path = DATA_PROCESSED_PATH / filename
        if not input_path.exists():
            logger.error(f"Analytical dataset not found at {input_path}")
            return pd.DataFrame()
            
        return pd.read_csv(input_path, index_col=0, parse_dates=True)
