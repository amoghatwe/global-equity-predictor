Global Equity Market Return Predictor
=====================================

A machine learning system for predicting 3-year forward returns in global equity markets
using macroeconomic indicators.

## Overview

This project implements a professional-grade ML pipeline for long-term equity return
forecasting, specifically designed for asset allocation decisions in long-only portfolios.

## Features

- **Data Sources**: World Bank, FRED, Yahoo Finance
- **Markets**: USA, Europe, Japan, UK, Emerging Markets
- **Forecast Horizon**: 3-year forward annualized returns
- **Models**: Linear Regression (Ridge), Random Forest, XGBoost, Ensemble
- **Validation**: Time series cross-validation with expanding/rolling windows
- **Reporting**: Console, PDF, and HTML reports

## Installation

```bash
# Clone or create the project directory
cd global_equity_predictor

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up API keys (optional but recommended)
cp .env.example .env
# Edit .env and add your FRED API key
```

## Quick Start

### 1. Full Pipeline (Collect Data → Features → Train → Predict → Report)

```bash
python run_prediction.py --mode full --report console
```

### 2. Step-by-Step

```bash
# Step 1: Collect data
python run_prediction.py --mode collect

# Step 2: Create features
python run_prediction.py --mode features

# Step 3: Train models
python run_prediction.py --mode train --model ensemble

# Step 4: Generate predictions and report
python run_prediction.py --mode predict --report html
```

### 3. Python API

```python
from src.data_collection.pipeline import DataPipeline
from src.features.pipeline import FeaturePipeline
from src.models.pipeline import ModelPipeline

# Collect data
data_pipeline = DataPipeline()
raw_data = data_pipeline.collect_all()

# Create features
feature_pipeline = FeaturePipeline()
features = feature_pipeline.create_features()
targets = feature_pipeline.create_targets()

# Train models
model_pipeline = ModelPipeline(model_type="ensemble")
results = model_pipeline.train()

# Generate predictions
predictions = model_pipeline.predict()
```

## Project Structure

```
global_equity_predictor/
├── data/                   # Data storage
│   ├── raw/               # Raw downloaded data
│   ├── processed/         # Cleaned and merged data
│   └── features/          # ML features
├── src/                   # Source code
│   ├── data_collection/   # Data fetching modules
│   ├── features/          # Feature engineering
│   ├── models/           # ML models and training
│   └── reporting/        # Report generation
├── config/               # Configuration
├── notebooks/           # Jupyter notebooks for exploration
├── tests/              # Unit tests
├── run_prediction.py    # Main entry point
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Configuration

Edit `config/settings.py` to customize:

- **Markets**: Add/remove markets to analyze
- **Date Range**: Training/test period
- **Model Parameters**: Tuning hyperparameters
- **Features**: Add new feature categories

## API Keys

### FRED API Key (Optional but Recommended)

1. Visit https://fred.stlouisfed.org/docs/api/api_key.html
2. Request a free API key
3. Add to `.env` file: `FRED_API_KEY=your_key_here`

Without FRED key, the system will still work but with limited US macro data.

## Model Performance

Typical performance metrics on historical data:

- **R²**: 0.35-0.45 (explains 35-45% of return variation)
- **Directional Accuracy**: 60-70% (correctly predicts up/down)
- **RMSE**: 4-6% annualized

*Note: These are illustrative. Actual performance depends on data quality and market regime.*

## Key Concepts

### 1. Forward Returns (Target)

The model predicts annualized returns over the next 3 years:
```
Target = (Price(t+36) / Price(t))^(12/36) - 1
```

### 2. Time Series Cross-Validation

Prevents look-ahead bias by using only past data for training:
- Train: 1980-2000 → Test: 2000-2005
- Train: 1980-2005 → Test: 2005-2010
- ...and so on

### 3. Feature Categories

- **Valuation**: Price deviations, moving averages
- **Growth**: GDP trends, industrial production
- **Inflation**: CPI levels, real rates
- **Rates**: Yield curves, policy rates
- **Credit**: Money supply, credit/GDP
- **Momentum**: Trailing returns, volatility

## Use Cases

### Asset Allocation
```python
# Get predictions
predictions = model_pipeline.predict()

# Rank markets by expected return
rankings = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

# Use for strategic allocation decisions
```

### Research & Backtesting
```python
# Walk-forward backtest
from src.models.base import TimeSeriesCrossValidator

cv = TimeSeriesCrossValidator(n_splits=5)
for train_idx, test_idx in cv.split(X):
    # Your backtest logic here
    pass
```

## Limitations & Disclaimers

⚠️ **Important**: This is an educational/research tool, not investment advice.

- **Past Performance**: Past model performance does not guarantee future results
- **Model Risk**: ML models can fail in unprecedented market conditions
- **Data Quality**: Free data sources may have delays or errors
- **Overfitting**: Even with time-series CV, overfitting remains a risk
- **External Factors**: Geopolitical events, black swans not captured

## Contributing

This is a starter template. To extend:

1. **Add More Data**: Integrate additional data sources (Quandl, Alpha Vantage)
2. **More Markets**: Add sector/regional breakdowns
3. **Advanced Models**: Try LSTM, Transformer architectures
4. **Risk Models**: Add volatility/correlation forecasting
5. **Optimization**: Portfolio optimization given return predictions

## License

MIT License - Free for educational and commercial use with attribution.

## Resources

- **World Bank API**: https://data.worldbank.org/
- **FRED API**: https://fred.stlouisfed.org/
- **Yahoo Finance**: https://finance.yahoo.com/
- **Papers**: Shiller (CAPE), Fama-French, Meb Faber (GTAA)

## Contact

Built as a starter project for long-only asset management opportunities.
For questions or extensions, see the code documentation and docstrings.
Reach out to me via my email: [amoghatwe@gmail.com](mailto:amoghatwe@gmail.com)

---

**Remember**: "In investing, what is comfortable is rarely profitable." - Robert Arnott
