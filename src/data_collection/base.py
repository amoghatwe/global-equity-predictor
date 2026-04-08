"""
Base classes for data collection.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """
    Abstract base class for data sources.
    
    All data source implementations should inherit from this class
    and implement the fetch method.
    """
    
    def __init__(self, name: str, cache_dir: Optional[str] = None):
        self.name = name
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
    @abstractmethod
    def fetch(self, **kwargs) -> pd.DataFrame:
        """
        Fetch data from the source.
        
        Args:
            **kwargs: Source-specific parameters
            
        Returns:
            pd.DataFrame: Fetched data
        """
        pass
    
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> bool:
        """
        Validate fetched data.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            bool: True if valid
        """
        pass
    
    def save_to_cache(self, data: pd.DataFrame, filename: str):
        """Save data to cache."""
        if self.cache_dir:
            import os
            os.makedirs(self.cache_dir, exist_ok=True)
            path = os.path.join(self.cache_dir, filename)
            data.to_csv(path, index=True)
            self.logger.info(f"Cached data to {path}")
    
    def load_from_cache(self, filename: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available."""
        if self.cache_dir:
            import os
            path = os.path.join(self.cache_dir, filename)
            if os.path.exists(path):
                self.logger.info(f"Loading cached data from {path}")
                return pd.read_csv(path, index_col=0, parse_dates=True)
        return None


class DataCollector:
    """
    Orchestrates data collection from multiple sources.
    """
    
    def __init__(self):
        self.sources: Dict[str, DataSource] = {}
        self.logger = logging.getLogger(__name__)
        
    def register_source(self, name: str, source: DataSource):
        """Register a data source."""
        self.sources[name] = source
        self.logger.info(f"Registered data source: {name}")
        
    def collect(self, source_name: str, **kwargs) -> pd.DataFrame:
        """Collect data from a specific source."""
        if source_name not in self.sources:
            raise ValueError(f"Unknown data source: {source_name}")
        
        source = self.sources[source_name]
        self.logger.info(f"Collecting data from {source_name}...")
        
        data = source.fetch(**kwargs)
        
        if source.validate(data):
            self.logger.info(f"Successfully collected {len(data)} rows from {source_name}")
            return data
        else:
            raise ValueError(f"Data validation failed for {source_name}")
    
    def collect_all(self, **kwargs) -> Dict[str, pd.DataFrame]:
        """Collect data from all registered sources."""
        results = {}
        for name in self.sources:
            try:
                results[name] = self.collect(name, **kwargs)
            except Exception as e:
                self.logger.error(f"Failed to collect from {name}: {str(e)}")
                results[name] = None
        return results
