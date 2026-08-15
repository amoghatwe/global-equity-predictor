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
cd global-equity-predictor

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
global-equity-predictor/
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

## Model Performance (Out-of-Sample)

Walk-forward evaluation with 36-month train/test gap (no label leakage from
overlapping 3-year forward-return targets). Expanding-window CV, 5 folds per
market, min training: 10 years, 5 markets (USA, Europe, Japan, UK, EM).
Feature columns with <90% coverage dropped. Values pooled across all 25
folds; per-market detail in `reports/oos_results.json`.

| Model           | RMSE    | MAE     | R²       | Dir. Acc. | Notes                |
|-----------------|---------|---------|----------|-----------|----------------------|
| Ensemble        | 13.055  | 12.046  | -16.718  | 0.712     | Ridge + RF + XGBoost |
| ARIMA(1,1,1)    | 12.523  | 11.835  | -13.691  | 0.630     | Univariate baseline  |
| Historical Mean | 6.760   | 5.995   | -4.072   | 0.823     | Constant prediction  |

Protocol: expanding-window walk-forward, 36-month train/test gap.

**Honest finding: after fixing label leakage, the ensemble does not beat the
historical-mean baseline on RMSE/MAE/R² in any market.** Positive
directional accuracy (0.71) mostly reflects class imbalance (36-month
returns are positive ~80% of the time). Treat predictions as a ranking
signal at best, not calibrated return forecasts. Reproduce with:
`python -m pytest tests/test_oos_evaluation.py -v -s`.

## Mathematical Processes

See [ARCHITECTURE.md](ARCHITECTURE.md) for the complete mathematical pipeline
(data collection, feature engineering, model training, evaluation metrics,
and confidence classification).

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
