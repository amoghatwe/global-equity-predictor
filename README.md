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

## Mathematical Processes

This section describes the complete mathematical pipeline from raw data to final prediction.

### 1. Data Collection & Preprocessing

**Equity Prices** — Five market ETFs (SPY, VGK, EWJ, EWU, EEM) are downloaded from Yahoo Finance. Adjusted close prices are resampled to monthly via last-observation sampling: `P_month(t) = P(d_T)` where `d_T` is the last trading day of month `t`.

**World Bank Macro Data** — Nine annual indicators (GDP, inflation, industrial production, M2, credit-to-GDP, population) are fetched per country. Annual data is interpolated to monthly frequency using linear interpolation, then forward-filled up to 3 months:

```
X_monthly(t) = interpolate(X_annual),  then  X(t) = ffill(X(t), limit=3)
```

For Emerging Markets, 15 countries are aggregated into a single series.

**FRED US Data** — Nine monthly US series (yields, CPI, unemployment, M2, DXY, VIX) are resampled to month-end. A yield curve spread is computed:

```
yield_curve_spread(t) = yield_10y(t) - yield_2y(t)
```

All three sources are outer-joined on a monthly date index.

### 2. Target Variable

The prediction target is the **3-year forward annualized return**:

```
total_return(t) = P(t + 36) / P(t) - 1

annualized_return(t) = (1 + total_return(t))^(12/36) - 1

target(t) = annualized_return(t) × 100    [expressed as %]
```

The last 36 months of data have NaN targets since future prices are unknown.

### 3. Feature Engineering

Six feature categories are constructed from the merged monthly data:

#### 3a. Valuation Features

For each market equity price series `P(t)` and windows `w ∈ {12, 36, 60}` months:

```
price_ma_ratio_w(t) = P(t) / MA_w(t)

    where  MA_w(t) = (1/w) × Σ_{i=0}^{w-1} P(t - i)

trend_deviation_60(t) = [P(t) - MA_60(t)] / MA_60(t) × 100
```

A price above its moving average yields `ratio > 1` (overvalued relative to trend), while below yields `ratio < 1`.

#### 3b. Growth Features

For GDP growth columns:

```
gdp_trend(t)       = MA_36(gdp_growth(t))

gdp_deviation(t)   = gdp_growth(t) - MA_36(gdp_growth(t))

gdp_yoy_change(t)  = gdp_growth(t) - gdp_growth(t - 12)
```

For industrial production:

```
ip_growth(t) = [IP(t) / IP(t - 12) - 1] × 100

ip_trend(t)  = MA_36(ip_growth(t))
```

#### 3c. Inflation Features

For CPI/inflation columns:

```
inflation_trend(t)     = MA_36(inflation(t))

inflation_deviation(t) = inflation(t) - MA_36(inflation(t))
```

**Real interest rate** (Fisher equation approximation):

```
real_rate_10y(t) = yield_10y(t) - inflation(t)
```

#### 3d. Interest Rate Features

```
yield_10y_trend(t)       = MA_36(yield_10y(t))

yield_10y_change_12m(t)  = yield_10y(t) - yield_10y(t - 12)

fed_funds_change_12m(t)  = fed_funds(t) - fed_funds(t - 12)

yield_spread(t)           = yield_10y(t) - yield_2y(t)

yield_spread_change_12m(t) = spread(t) - spread(t - 12)
```

The yield curve spread captures term structure slope — a key recession signal when inverted.

#### 3e. Credit Features

```
m2_growth_12m(t) = [M2(t) / M2(t - 12) - 1] × 100

m2_growth_trend(t) = MA_36(m2_growth_12m(t))

credit_change_12m(t) = credit(t) - credit(t - 12)
```

#### 3f. Momentum Features

For each market, trailing returns over `m ∈ {1, 3, 6, 12}` months:

```
return_m(t) = [P(t) / P(t - m) - 1] × 100
```

Annualized rolling volatility over `w ∈ {6, 12}` months:

```
vol_w(t) = std(r, window=w) × √12 × 100

    where  r(t) = P(t) / P(t - 1) - 1
```

VIX features:

```
vix_level(t)  = VIX(t)

vix_trend(t)  = MA_12(VIX(t))
```

### 4. Model Training

#### 4a. Preprocessing

Rows with NaN targets are dropped. Remaining NaN feature values are imputed with the **column median** from the training set:

```
X_ij = median(X_col_j | X_col_j ≠ NaN)    if X_ij = NaN
```

#### 4b. Walk-Forward Time Series Cross-Validation

A 5-fold expanding-window CV scheme preserves temporal ordering:

```
Min training window: 120 months (10 years)
Fold size: (n_samples - 120) / 5

Fold 1: Train [0 .. 120+f)       → Test [120+f .. 120+2f)
Fold 2: Train [0 .. 120+2f)      → Test [120+2f .. 120+3f)
  ...
Fold 5: Train [0 .. 120+5f)      → Test [120+5f .. n)
```

No data from the future leaks into training — each fold only uses indices strictly before the test window.

#### 4c. Ridge Regression (LinearReturnModel)

```
min_β  Σ_i (y_i - X_i β - β_0)²  +  α × ||β||²

    α = 1.0   (L2 penalty)
    fit_intercept = True
```

Ridge shrinks coefficients toward zero, reducing variance at the cost of slight bias. With many correlated macro features, this prevents overfitting.

#### 4d. Random Forest (RandomForestReturnModel)

An ensemble of `B = 200` decision trees, each trained on a bootstrap sample:

```
ŷ = (1/B) × Σ_b ŷ_b(x)

    n_estimators = 200
    max_depth = 10
    min_samples_split = 5
    min_samples_leaf = 2
    max_features = √p    (p = total features; √p considered per split)
```

Each tree sees a random subset of rows and √p random features at each split, decorrelating predictions and reducing ensemble variance.

#### 4e. XGBoost (XGBoostReturnModel)

Gradient-boosted trees fit residuals sequentially:

```
ŷ(x) = Σ_b  η × f_b(x)

    where f_b is the b-th tree fit to the negative gradient of the loss

    n_estimators = 200
    max_depth = 6
    learning_rate (η) = 0.05
    subsample = 0.8         (row sampling per tree)
    colsample_bytree = 0.8  (feature sampling per tree)
    reg_alpha (λ_1) = 0.1   (L1 regularization on leaf weights)
    reg_lambda (λ_2) = 1.0  (L2 regularization on leaf weights)
```

Each new tree corrects the residual error of the ensemble so far. The low learning rate (0.05) and row/column subsampling prevent overfitting.

### 5. Ensemble Prediction

The three models are combined via an **equal-weight average**:

```
ŷ_ensemble = (1/3) × [ŷ_ridge + ŷ_rf + ŷ_xgb]
```

There is no stacking, meta-learner, or inverse-variance weighting — the ensemble is a simple arithmetic mean. This diversifies model risk since each model class captures different patterns (linear trends, interactions, nonlinear residual structure).

### 6. Evaluation Metrics

For predictions `ŷ` against true values `y` on each CV fold:

```
RMSE  = √( mean( (y_i - ŷ_i)² ) )

MAE   = mean( |y_i - ŷ_i| )

R²    = 1 - Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²

Dir. Accuracy = mean( sign(y_i) == sign(ŷ_i) )

Correlation   = Pearson r(y, ŷ)
```

CV metrics are aggregated by taking the **mean** and **standard deviation** across the 5 folds.

### 7. Confidence Classification

The standard deviation of the three model predictions determines confidence:

```
model_std = std(ŷ_ridge, ŷ_rf, ŷ_xgb)

model_std < 1.0   →  "High" confidence
1.0 ≤ model_std < 2.5  →  "Medium" confidence
model_std ≥ 2.5  →  "Low" confidence
```

When all three models agree closely (low dispersion), the prediction is considered more reliable.

### 8. Pipeline Summary

| Stage | Input | Operation | Output |
|-------|-------|-----------|--------|
| Data Collection | APIs | Download, resample to monthly, interpolate, merge | Monthly multi-source DataFrame |
| Feature Engineering | Monthly data | Rolling means, ratios, pct changes, volatilities | Feature matrix X |
| Target Creation | Monthly prices | `((P(t+36)/P(t)) - 1)^(12/36) - 1` in % | Target vector y |
| Model Training | X, y | Ridge + RF + XGBoost with 5-fold expanding-window CV | Three trained models per market |
| Prediction | Latest feature row | Each model predicts; ensemble = equal-weight mean | Scalar expected return per market |
| Reporting | Predictions | Rank by return, classify confidence, compare to 7% norm | Console / PDF / HTML report |

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
