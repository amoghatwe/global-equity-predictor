"""
Base classes and interfaces for the data collection layer.

This module defines the foundational abstractions used by all data providers
in the system. Following the Interface Segregation and Dependency Inversion
principles, it allows the pipeline to interact with various data sources
(World Bank, FRED, Yahoo Finance) through a unified API.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
import pandas as pd
import logging
from pathlib import Path

# Set up module-level logger
logger = logging.getLogger(__name__)


class BaseDataProvider(ABC):
    """
    Abstract base class for all data providers.
    
    A Data Provider is responsible for:
    1. Fetching raw data from a specific external API or service.
    2. Validating the integrity and quality of the fetched data.
    3. Providing a consistent interface (fetch/validate) for the pipeline.
    
    Attributes:
        provider_name: Human-readable name of the data source.
        cache_directory: Optional path to store/load cached CSV data.
        logger: Provider-specific logger instance.
    """
    
    def __init__(self, provider_name: str, cache_directory: Optional[Union[str, Path]] = None):
        """
        Initialize the data provider.
        
        Args:
            provider_name: Name of the source (e.g., "FRED", "WorldBank").
            cache_directory: Where to store cached data files.
        """
        self.provider_name = provider_name
        self.cache_directory = Path(cache_directory) if cache_directory else None
        
        # Create a specific logger for this provider (e.g., "src.data_collection.base.FRED")
        self.logger = logging.getLogger(f"{__name__}.{provider_name}")
        
    @abstractmethod
    def fetch_data(self, **kwargs) -> pd.DataFrame:
        """
        Fetch data from the external source.
        
        This method must be implemented by all subclasses. It should handle
        API-specific logic, authentication, and initial data parsing.
        
        Args:
            **kwargs: Provider-specific parameters (dates, tickers, etc.)
            
        Returns:
            pd.DataFrame: The fetched data in a pandas DataFrame format.
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Perform quality checks on the fetched data.
        
        Should check for:
        - Empty datasets
        - Missing columns
        - Excessive NaN values
        - Logical inconsistencies (e.g., negative prices)
        
        Args:
            data: The DataFrame to validate.
            
        Returns:
            bool: True if the data meets quality standards, False otherwise.
        """
        pass


class DataCollectionManager:
    """
    Orchestrator that manages multiple data providers.
    
    This class follows the Registry pattern, allowing different data sources
    to be plugged in and called through a centralized manager. It handles
    the high-level workflow of collection, validation, and error reporting.
    """
    
    def __init__(self):
        """Initialize the collection manager with an empty registry."""
        self.providers: Dict[str, BaseDataProvider] = {}
        self.logger = logging.getLogger(__name__)
        
    def register_provider(self, identifier: str, provider: BaseDataProvider):
        """
        Register a new data provider instance.
        
        Args:
            identifier: Unique string ID for the provider (e.g., "fred").
            provider: An instance of a class inheriting from BaseDataProvider.
        """
        self.providers[identifier] = provider
        self.logger.info(f"Registered data provider: {identifier} ({provider.provider_name})")
        
    def collect_from_source(self, provider_id: str, **kwargs) -> pd.DataFrame:
        """
        Execute the full collection cycle for a single source.
        
        Steps:
        1. Fetch data using the provider.
        2. Validate the result.
        3. Log successes/failures.
        
        Args:
            provider_id: ID of the registered provider to use.
            **kwargs: Parameters to pass to the provider's fetch_data method.
            
        Returns:
            pd.DataFrame: The collected data.
            
        Raises:
            ValueError: If the provider is unknown or validation fails.
        """
        if provider_id not in self.providers:
            error_msg = f"Unknown data provider: '{provider_id}'"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        provider = self.providers[provider_id]
        self.logger.info(f"Starting data collection from source: {provider_id}...")
        
        # 1. Fetch
        data = provider.fetch_data(**kwargs)
        
        # 2. Validate
        if provider.validate_data(data):
            self.logger.info(f"Successfully collected {len(data)} rows from {provider_id}")
            return data
        else:
            error_msg = f"Data validation failed for source '{provider_id}'"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
