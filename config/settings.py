"""
Configuration settings for Global Equity Market Return Predictor.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_PATH = Path(os.getenv("DATA_RAW_PATH", PROJECT_ROOT / "data" / "raw"))
DATA_PROCESSED_PATH = Path(os.getenv("DATA_PROCESSED_PATH", PROJECT_ROOT / "data" / "processed"))
DATA_FEATURES_PATH = Path(os.getenv("DATA_FEATURES_PATH", PROJECT_ROOT / "data" / "features"))

# Ensure directories exist
for path in [DATA_RAW_PATH, DATA_PROCESSED_PATH, DATA_FEATURES_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# API Keys
FRED_API_KEY = os.getenv("FRED_API_KEY")

# Date ranges
START_DATE = "1980-01-01"
END_DATE = "2024-12-31"

# Forecast horizon (in months)
FORECAST_HORIZON_MONTHS = 36

# Markets configuration
MARKETS = {
    "USA": {
        "index": "S&P 500",
        "ticker": "^GSPC",
        "country_code": "USA",
        "region": "North America",
        "developed": True,
    },
    "Europe": {
        "index": "MSCI Europe",
        "ticker": "IEV",  # iShares Europe ETF
        "country_code": "EUR",
        "region": "Europe",
        "developed": True,
    },
    "Japan": {
        "index": "MSCI Japan",
        "ticker": "EWJ",  # iShares Japan ETF
        "country_code": "JPN",
        "region": "Asia",
        "developed": True,
    },
    "UK": {
        "index": "MSCI UK",
        "ticker": "EWU",  # iShares UK ETF
        "country_code": "GBR",
        "region": "Europe",
        "developed": True,
    },
    "EM": {
        "index": "MSCI Emerging Markets",
        "ticker": "EEM",  # iShares EM ETF
        "country_code": "EM",
        "region": "Global",
        "developed": False,
    },
}

# Feature categories
FEATURE_CATEGORIES = {
    "valuation": ["cape_ratio", "pe_ratio", "dividend_yield", "market_cap_gdp"],
    "growth": ["gdp_growth", "industrial_production_growth", "gdp_trend"],
    "inflation": ["cpi_yoy", "inflation_trend", "real_rates"],
    "rates": ["policy_rate", "yield_10y", "yield_curve_slope"],
    "credit": ["credit_gdp", "m2_growth"],
    "momentum": ["trailing_12m_return", "volatility"],
    "global": ["dxy_index", "oil_price", "vix"],
}

# Model configuration
MODEL_CONFIG = {
    "random_state": int(os.getenv("RANDOM_STATE", 42)),
    "test_size": float(os.getenv("TEST_SIZE", 0.2)),
    "n_splits": 5,  # For time series cross-validation
    "min_train_years": 10,
    "max_train_years": 25,
}

# Linear Regression Parameters
LINEAR_PARAMS = {
    "alpha": 1.0,  # Ridge regularization
    "fit_intercept": True,
}

# Random Forest Parameters
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": MODEL_CONFIG["random_state"],
    "n_jobs": -1,
}

# XGBoost Parameters
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": MODEL_CONFIG["random_state"],
    "n_jobs": -1,
}

# World Bank indicators
WB_INDICATORS = {
    "gdp_current_usd": "NY.GDP.MKTP.CD",
    "gdp_real_growth": "NY.GDP.MKTP.KD.ZG",
    "gdp_per_capita": "NY.GDP.PCAP.CD",
    "inflation_cpi": "FP.CPI.TOTL.ZG",
    "industrial_production": "NV.IND.TOTL.ZS",
    "broad_money_m2": "FM.LBL.BMNY.GD.ZS",
    "credit_to_gdp": "GFDD.DI.14",
    "population_total": "SP.POP.TOTL",
    "population_65plus": "SP.POP.65UP.TO.ZS",
}

# FRED series codes
FRED_SERIES = {
    "US": {
        "yield_10y": "GS10",
        "yield_2y": "GS2",
        "fed_funds": "FEDFUNDS",
        "cpi_yoy": "CPIAUCSL",
        "industrial_production": "INDPRO",
        "unemployment": "UNRATE",
        "market_cap_gdp": "DDDM03USA156NWDB",
        "m2": "M2SL",
        "dxy": "DTWEXBGS",
        "vix": "VIXCLS",
    }
}

# Reporting
REPORT_CONFIG = {
    "title": "Global Equity Market Outlook Report",
    "author": "Global Equity Predictor System",
    "forecast_horizon_years": FORECAST_HORIZON_MONTHS // 12,
    "confidence_levels": ["Low", "Medium", "High"],
    "historical_periods": ["1980-1990", "1990-2000", "2000-2010", "2010-2020", "2020-Present"],
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": os.getenv("LOG_LEVEL", "INFO"),
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
        "file": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": PROJECT_ROOT / "logs" / "predictor.log",
            "mode": "a",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": os.getenv("LOG_LEVEL", "INFO"),
            "propagate": False,
        },
    },
}
