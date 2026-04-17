"""
Feature Engineering Pipeline.

This module transforms processed time series data into analytical features
ready for machine learning models. It engineers indicators across several
categories: valuation, economic growth, inflation, interest rates, credit,
and momentum.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple, Union
from datetime import datetime
import logging
from pathlib import Path

from config.settings import (
    DATA_PROCESSED_PATH,
    DATA_FEATURES_PATH,
    MARKETS,
    FORECAST_HORIZON_MONTHS,
    FEATURE_CATEGORIES
)

# Set up module logger
logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """
    Orchestrates the creation of features and target variables.
    
    This pipeline takes cleaned monthly data and applies various statistical
    transformations (rolling windows, percentage changes, deviations) to
    generate the final dataset used for model training and prediction.
    """
    
    def __init__(
        self,
        target_markets: Optional[List[str]] = None,
        processed_data_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the feature pipeline.
        
        Args:
            target_markets: Markets to generate features for.
            processed_data_dir: Source directory for cleaned data.
        """
        self.markets = self._initialize_markets(target_markets)
        self.processed_data_dir = Path(processed_data_dir or DATA_PROCESSED_PATH)
        self.output_dir = Path(DATA_FEATURES_PATH)
        
        # State storage
        self.input_dataset: Optional[pd.DataFrame] = None
        self.engineered_features: Optional[pd.DataFrame] = None
        self.target_variables: Optional[pd.DataFrame] = None
        
        self.logger = logging.getLogger(__name__)
        
    def _initialize_markets(self, markets: Optional[List[str]]) -> List[str]:
        """Validate and return the list of target markets."""
        if markets is None or markets == ["all"]:
            return list(MARKETS.keys())
        return markets
    
    def load_input_data(self, filename: str = "processed_data.csv") -> bool:
        """
        Load the cleaned dataset from the processed directory.
        
        Returns:
            bool: True if loading was successful.
        """
        file_path = self.processed_data_dir / filename
        
        if not file_path.exists():
            self.logger.error(f"Required dataset not found: {file_path}")
            return False
        
        self.input_dataset = pd.read_csv(file_path, index_col=0, parse_dates=True)
        self.logger.info(f"Successfully loaded dataset: {self.input_dataset.shape}")
        return True
    
    def engineer_all_features(self) -> pd.DataFrame:
        """
        Main entry point for feature engineering.
        
        Transforms raw columns into analytical indicators across multiple categories.
        
        Returns:
            pd.DataFrame: The complete feature matrix.
        """
        self.logger.info("Starting feature engineering process...")
        
        if self.input_dataset is None:
            if not self.load_input_data():
                return pd.DataFrame()
        
        # We group indicators by their economic category
        feature_sets = {}
        
        # 1. Valuation: Is the market cheap or expensive relative to history?
        self.logger.info("  - Engineering valuation indicators...")
        valuation_df = self._engineer_valuation_indicators()
        if not valuation_df.empty:
            feature_sets['valuation'] = valuation_df
        
        # 2. Growth: Are economies expanding or contracting?
        self.logger.info("  - Engineering economic growth indicators...")
        growth_df = self._engineer_growth_indicators()
        if not growth_df.empty:
            feature_sets['growth'] = growth_df
        
        # 3. Inflation: What is the purchasing power trend?
        self.logger.info("  - Engineering inflation indicators...")
        inflation_df = self._engineer_inflation_indicators()
        if not inflation_df.empty:
            feature_sets['inflation'] = inflation_df
        
        # 4. Rates: What is the cost of capital?
        self.logger.info("  - Engineering interest rate indicators...")
        rates_df = self._engineer_rate_indicators()
        if not rates_df.empty:
            feature_sets['rates'] = rates_df
        
        # 5. Credit: Is liquidity increasing or decreasing?
        self.logger.info("  - Engineering credit and liquidity indicators...")
        credit_df = self._engineer_credit_indicators()
        if not credit_df.empty:
            feature_sets['credit'] = credit_df
        
        # 6. Momentum: What are the recent price trends?
        self.logger.info("  - Engineering market momentum indicators...")
        momentum_df = self._engineer_momentum_indicators()
        if not momentum_df.empty:
            feature_sets['momentum'] = momentum_df
        
        # Merge all generated indicator sets
        if not feature_sets:
            self.logger.error("No features were successfully generated.")
            return pd.DataFrame()
        
        # Start with the first available set and join the rest
        feature_matrix = list(feature_sets.values())[0]
        for name, df in list(feature_sets.items())[1:]:
            feature_matrix = feature_matrix.join(df, how='outer')
        
        # Final cleanup: sort and remove empty rows
        self.engineered_features = feature_matrix.dropna(how='all').sort_index()
        self.logger.info(f"Generated {len(self.engineered_features.columns)} features.")
        
        return self.engineered_features
    
    def _engineer_valuation_indicators(self) -> pd.DataFrame:
        """
        Creates indicators that measure price levels relative to historical norms.
        """
        features = pd.DataFrame(index=self.input_dataset.index)
        data = self.input_dataset
        
        for market in self.markets:
            price_col = f"prices_{market}"
            
            if price_col in data.columns:
                prices = data[price_col]
                
                # Price relative to Moving Averages (MA)
                # This shows if the market is overextended relative to its recent past
                for window in [12, 36, 60]:
                    # Use min_periods to handle shorter history (e.g., EM, Europe)
                    ma = prices.rolling(window=window, min_periods=min(window, 12)).mean()
                    features[f"{market}_price_to_ma{window}"] = prices / ma
                
                # Long-term trend deviation (%)
                # Use a smaller min_periods to allow 3+ years of data to be used
                long_term_trend = prices.rolling(window=120, min_periods=36).mean() # 10-year trend
                features[f"{market}_trend_deviation"] = (prices - long_term_trend) / long_term_trend * 100
        
        return features
    
    def _engineer_growth_indicators(self) -> pd.DataFrame:
        """
        Creates indicators based on economic output and production.
        """
        features = pd.DataFrame(index=self.input_dataset.index)
        data = self.input_dataset
        
        # Look for macro-economic columns (from World Bank)
        for col in data.columns:
            if 'macro_' in col and 'gdp_real_growth' in col:
                # 3-year smoothed trend
                features[f"{col}_smoothed"] = data[col].rolling(window=36, min_periods=12).mean()
                # Momentum of growth
                features[f"{col}_momentum"] = data[col].diff(12)
        
            if 'macro_' in col and 'industrial_production' in col:
                # Year-over-year change in production
                features[f"{col}_yoy"] = data[col].pct_change(12) * 100
        
        return features
    
    def _engineer_inflation_indicators(self) -> pd.DataFrame:
        """
        Creates indicators related to price stability and real rates.
        """
        features = pd.DataFrame(index=self.input_dataset.index)
        data = self.input_dataset
        
        for col in data.columns:
            if 'inflation' in col.lower():
                features[f"{col}_level"] = data[col]
                features[f"{col}_trend"] = data[col].rolling(window=24, min_periods=12).mean()
                features[f"{col}_acceleration"] = data[col].diff(12)
        
        # Real Interest Rate (10Y Yield - Inflation)
        # Note: Using US inflation as global proxy if local not available
        if 'economic_yield_10y' in data.columns:
            # Try to find a US inflation column
            us_inflation_col = [c for c in data.columns if 'cpi' in c.lower() and 'usa' in c.lower()]
            if us_inflation_col:
                features['global_real_rate'] = data['economic_yield_10y'] - data[us_inflation_col[0]]
        
        return features
    
    def _engineer_rate_indicators(self) -> pd.DataFrame:
        """
        Creates indicators derived from the yield curve and policy rates.
        """
        features = pd.DataFrame(index=self.input_dataset.index)
        data = self.input_dataset
        
        # US Yield Curve Spread (10Y - 2Y)
        if 'economic_yield_curve_spread' in data.columns:
            features['yield_curve_slope'] = data['economic_yield_curve_spread']
            features['yield_curve_change_6m'] = data['economic_yield_curve_spread'].diff(6)
        
        # Policy Rate (Fed Funds)
        if 'economic_fed_funds' in data.columns:
            features['interest_rate_level'] = data['economic_fed_funds']
            features['interest_rate_change_12m'] = data['economic_fed_funds'].diff(12)
        
        return features
    
    def _engineer_credit_indicators(self) -> pd.DataFrame:
        """
        Creates indicators related to liquidity and money supply.
        """
        features = pd.DataFrame(index=self.input_dataset.index)
        data = self.input_dataset
        
        # M2 Money Supply Growth
        for col in data.columns:
            if 'm2' in col.lower():
                features[f"{col}_growth"] = data[col].pct_change(12) * 100
                
        # Credit to GDP
        for col in data.columns:
            if 'credit_to_gdp' in col:
                features[f"{col}_level"] = data[col]
                features[f"{col}_change"] = data[col].diff(12)
        
        return features
    
    def _engineer_momentum_indicators(self) -> pd.DataFrame:
        """
        Creates technical and sentiment indicators like trailing returns and volatility.
        """
        features = pd.DataFrame(index=self.input_dataset.index)
        data = self.input_dataset
        
        for market in self.markets:
            price_col = f"prices_{market}"
            
            if price_col in data.columns:
                prices = data[price_col]
                
                # Trailing returns for multiple horizons
                for m in [3, 6, 12]:
                    features[f"{market}_momentum_{m}m"] = prices.pct_change(m) * 100
                
                # Realized Volatility (Rolling standard deviation of returns)
                returns = prices.pct_change()
                features[f"{market}_volatility_12m"] = returns.rolling(window=12, min_periods=6).std() * np.sqrt(12) * 100
        
        # VIX Index (Global fear gauge)
        if 'economic_vix' in data.columns:
            features['market_vol_vix'] = data['economic_vix']
            features['vix_trend_deviation'] = data['economic_vix'] - data['economic_vix'].rolling(window=12, min_periods=6).mean()
        
        return features
    
    def engineer_target_variables(self) -> pd.DataFrame:
        """
        Creates the 'ground truth' labels: 3-year annualized forward returns.
        
        Returns:
            pd.DataFrame: A matrix of targets for each market.
        """
        self.logger.info("Engineering target variables (3Y Forward Returns)...")
        
        if self.input_dataset is None:
            if not self.load_input_data():
                return pd.DataFrame()
        
        targets_df = pd.DataFrame(index=self.input_dataset.index)
        data = self.input_dataset
        
        for market in self.markets:
            price_col = f"prices_{market}"
            
            if price_col in data.columns:
                prices = data[price_col]
                
                # Forward-looking calculation:
                # We look at the price 36 months in the future
                future_prices = prices.shift(-FORECAST_HORIZON_MONTHS)
                total_return = (future_prices / prices) - 1
                
                # Annualize the total return
                # Formula: (1 + r)^(1/3) - 1
                annualized_return = ((1 + total_return) ** (12 / FORECAST_HORIZON_MONTHS)) - 1
                
                # Store as percentage
                col_name = f"{market}_target_return"
                targets_df[col_name] = annualized_return * 100
                
                valid_count = targets_df[col_name].notna().sum()
                self.logger.info(f"  - Created {col_name}: {valid_count} training labels.")
        
        self.target_variables = targets_df
        return targets_df
    
    def merge_and_filter_final_dataset(
        self, 
        features: pd.DataFrame, 
        targets: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Joins features and targets, removing dates that lack training labels.
        
        Args:
            features: Engineered feature matrix.
            targets: Target return matrix.
            
        Returns:
            pd.DataFrame: Cleaned training dataset.
        """
        self.logger.info("Merging features and targets for final training set...")
        
        # Align by date index
        combined_dataset = features.join(targets, how='inner')
        
        # Remove rows where all targets are missing (these are the most recent months
        # for which we don't yet know the 3-year future return)
        initial_len = len(combined_dataset)
        combined_dataset = combined_dataset.dropna(subset=targets.columns, how='all')
        
        self.logger.info(
            f"Final analytical dataset created: {len(combined_dataset)} rows, "
            f"{len(combined_dataset.columns)} columns. "
            f"({initial_len - len(combined_dataset)} future rows reserved for prediction)"
        )
        
        return combined_dataset
    
    def save_features(self, df: pd.DataFrame, filename: str = "features.csv"):
        """Save the engineered dataset to the features directory."""
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=True)
        self.logger.info(f"Feature matrix saved to: {output_path}")

    @classmethod
    def load_features(cls, filename: str = "features.csv") -> pd.DataFrame:
        """Load an existing feature matrix from disk."""
        input_path = Path(DATA_FEATURES_PATH) / filename
        if not input_path.exists():
            logger.error(f"Feature matrix not found at {input_path}")
            return pd.DataFrame()
            
        return pd.read_csv(input_path, index_col=0, parse_dates=True)
