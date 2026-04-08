"""
Quick Reference Guide for Global Equity Predictor
================================================

This file provides a quick reference for using the system.
For full documentation, see README.md and ARCHITECTURE.md

## Quick Start Commands

### 1. Setup (First Time Only)
```bash
cd global_equity_predictor
bash setup.sh              # Automated setup
# Or manually:
# python -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt
# cp .env.example .env    # Add your FRED API key
```

### 2. Run Full Pipeline

```bash
python3 run_prediction.py --mode full --report console
```

This will:
- Download data from World Bank, FRED, Yahoo Finance
- Create features and targets
- Train models with walk-forward validation
- Generate predictions
- Display results

### 3. Step-by-Step Execution
```bash
# Collect data only
python3 run_prediction.py --mode collect

# Create features only
python3 run_prediction.py --mode features

# Train models only
python3 run_prediction.py --mode train --model ensemble

# Generate predictions only (requires trained models)
python3 run_prediction.py --mode predict --report html
```

## Command Line Options

### Mode (--mode)
- `collect` - Download raw data from APIs
- `features` - Create ML features and targets
- `train` - Train models
- `predict` - Generate predictions
- `full` - Run complete pipeline (default)

### Report Format (--report)
- `console` - Text output in terminal (default)
- `pdf` - PDF report (requires reportlab)
- `html` - HTML report with charts (requires plotly)
- `none` - No report

### Model Type (--model)
- `linear` - Ridge regression only
- `rf` - Random Forest only
- `xgb` - XGBoost only
- `ensemble` - All models + ensemble average (default)
- `all` - Same as ensemble

### Markets (--market)
- `all` - All markets (default)
- Specific: `--market USA Europe Japan`
- Combinations: `--market USA Japan`

### Date Range
```bash
python run_prediction.py --start-date 2000-01-01 --end-date 2023-12-31
```

## Python API Usage

### Example 1: Full Pipeline in Python
```python
from src.data_collection.pipeline import DataPipeline
from src.features.pipeline import FeaturePipeline
from src.models.pipeline import ModelPipeline
from src.reporting.generator import ReportGenerator

# 1. Collect data
data_pipeline = DataPipeline(
    start_date="2000-01-01",
    end_date="2023-12-31"
)
raw_data = data_pipeline.collect_all()
processed = data_pipeline.process(raw_data)
data_pipeline.save(processed)

# 2. Create features
feature_pipeline = FeaturePipeline()
features = feature_pipeline.create_features()
targets = feature_pipeline.create_targets()
dataset = feature_pipeline.merge_features_targets(features, targets)
feature_pipeline.save(dataset)

# 3. Train models
model_pipeline = ModelPipeline(model_type="ensemble")
results = model_pipeline.train()
model_pipeline.save_models()

# 4. Generate predictions
predictions = model_pipeline.predict()

# 5. Create report
generator = ReportGenerator(format="html")
generator.generate_html_report(predictions)
```

### Example 2: Load Existing Data and Predict
```python
from src.models.pipeline import ModelPipeline
from src.reporting.generator import ReportGenerator

# Load trained models and generate predictions
model_pipeline = ModelPipeline()
predictions = model_pipeline.predict()

# Display results
generator = ReportGenerator(format="console")
generator.generate_console_report(predictions)
```

### Example 3: Custom Feature Engineering
```python
from src.features.pipeline import FeaturePipeline
import pandas as pd

# Load processed data
pipeline = FeaturePipeline()
pipeline.load_processed_data()

# Create features
features = pipeline.create_features()

# Access specific feature categories
valuation_features = [col for col in features.columns if 'price_ma' in col]
growth_features = [col for col in features.columns if 'gdp' in col]

print(f"Total features: {len(features.columns)}")
print(f"Valuation features: {len(valuation_features)}")
print(f"Growth features: {len(growth_features)}")
```

## Key Classes and Modules

### Data Collection
- `DataPipeline` - Main data collection orchestrator
- `WorldBankDataSource` - GDP, inflation, etc.
- `FREDDataSource` - US rates, yields, VIX
- `EquityDataSource` - Index prices from Yahoo Finance

### Feature Engineering
- `FeaturePipeline` - Creates all features
- Creates: valuation, growth, inflation, rates, credit, momentum features
- Creates: 3-year forward returns (targets)

### Models
- `LinearReturnModel` - Ridge regression
- `RandomForestReturnModel` - Random Forest
- `XGBoostReturnModel` - XGBoost
- `ModelEnsemble` - Combines multiple models
- `TimeSeriesCrossValidator` - Walk-forward CV

### Reporting
- `ReportGenerator` - Creates reports
- `generate_console_report()` - Terminal output
- `generate_pdf_report()` - PDF document
- `generate_html_report()` - HTML with charts

## Configuration

### Edit config/settings.py

```python
# Add new market
MARKETS["Asia_Ex_Japan"] = {
    "index": "MSCI Asia ex Japan",
    "ticker": "AAXJ",
    "country_code": "ASIA",
    "region": "Asia",
    "developed": False,
}

# Change forecast horizon
FORECAST_HORIZON_MONTHS = 60  # 5 years instead of 3

# Modify model parameters
RF_PARAMS = {
    "n_estimators": 500,  # More trees
    "max_depth": 15,     # Deeper trees
}
```

### Environment Variables (.env)

```bash
FRED_API_KEY=your_key_here
DATA_RAW_PATH=/custom/data/path
RANDOM_STATE=42
LOG_LEVEL=DEBUG
```

## Data Sources

### World Bank
- GDP (current USD, real growth)
- Inflation (CPI)
- Industrial production
- Broad money (M2)
- Credit to GDP
- Demographics

### FRED
- US 10Y Treasury yield
- US 2Y Treasury yield
- Federal Funds rate
- CPI inflation
- DXY (US Dollar index)
- VIX (volatility)

### Yahoo Finance
- S&P 500 (SPY) → USA
- Vanguard Europe (VGK) → Europe
- iShares Japan (EWJ) → Japan
- iShares UK (EWU) → UK
- iShares EM (EEM) → Emerging Markets

## Model Performance Metrics

### Interpretation
- **R²**: % of return variation explained (higher is better)
  - <0.2: Weak model
  - 0.2-0.5: Moderate model
  - >0.5: Strong model (rare in finance)

- **RMSE**: Root mean squared error in percentage points
  - <3%: Very accurate
  - 3-6%: Good
  - >6%: High uncertainty

- **Directional Accuracy**: % of correct up/down predictions
  - 50%: Random (coin flip)
  - 55-60%: Useful edge
  - >60%: Strong predictive power

- **Correlation**: Correlation between predicted and actual returns
  - 0.3-0.5: Typical for financial forecasting
  - >0.5: Excellent

## Common Issues and Solutions

### Issue: "FRED API key not set"
**Solution**: Get free key at https://fred.stlouisfed.org/docs/api/api_key.html
Add to .env: `FRED_API_KEY=your_key`

### Issue: "No data returned from World Bank"
**Solution**: Check internet connection. Some indicators may not be available for all countries.

### Issue: "Insufficient data for model training"
**Solution**: Extend date range: `--start-date 1990-01-01`

### Issue: "ImportError: No module named 'xgboost'"
**Solution**: Install all dependencies: `pip install -r requirements.txt`

### Issue: "MemoryError" during training
**Solution**: 
1. Reduce number of features in config
2. Use fewer cross-validation splits
3. Process fewer markets at once

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_system.py::TestModels -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Development Workflow

### 1. Add New Feature
```python
# In src/features/pipeline.py

def _create_my_feature(self) -> pd.DataFrame:
    features = pd.DataFrame(index=self.processed_data.index)
    
    # Your feature logic here
    features['my_feature'] = ...
    
    return features

# Add to create_features()
def create_features(self):
    ...
    my_feature = self._create_my_feature()
    if not my_feature.empty:
        features_dict['my_feature'] = my_feature
    ...
```

### 2. Add New Model
```python
# In src/models/implementations.py

class MyModel(BaseReturnModel):
    def __init__(self, **kwargs):
        from some_library import MyEstimator
        model = MyEstimator(**kwargs)
        super().__init__("MyModel", model)
    
    def train(self, X, y):
        # Training logic
        pass
    
    def predict(self, X):
        # Prediction logic
        pass

# Add to create_model()
def create_model(model_type: str, **kwargs):
    model_map = {
        ...
        'my_model': MyModel,
    }
```

### 3. Add New Report Format
```python
# In src/reporting/generator.py

def generate_json_report(self, predictions):
    import json
    data = self._prepare_data(predictions)
    with open('report.json', 'w') as f:
        json.dump(data, f)
```

## File Locations

### Data Files
- `data/raw/` - Raw downloaded data
- `data/processed/processed_data.csv` - Merged dataset
- `data/features/features.csv` - ML features and targets

### Model Files
- `models/{market}/{model_name}.pkl` - Trained models
- `models/metadata.json` - Model metadata

### Reports
- `reports/equity_outlook_YYYYMMDD.html` - HTML report
- `reports/equity_outlook_YYYYMMDD.pdf` - PDF report

### Logs
- `logs/predictor.log` - Application logs

## Tips for Asset Management Interviews

### 1. Understand the Theory
- Be ready to explain why valuation metrics predict long-term returns
- Understand the difference between time-series and cross-sectional prediction
- Know the limitations (overfitting, regime changes, etc.)

### 2. Know the Code
- Be able to explain the architecture
- Walk through the data flow
- Discuss design decisions (why walk-forward CV?)

### 3. Have Extensions Ready
- "How would you add sector allocation?"
- "How would you incorporate sentiment data?"
- "How would you handle real-time updates?"

### 4. Be Honest About Limitations
- Models can fail
- Past performance ≠ future results
- Free data has delays
- This is a starting point, not production-ready

### 5. Show You Can Extend It
- Have ideas for improvements
- Know how to add new features/models
- Understand how to integrate with portfolio optimization

## Further Reading

### Academic Papers
- Shiller (1981) - "Do Stock Prices Move Too Much to Be Justified by Subsequent Changes in Dividends?"
- Fama & French (1988) - "Dividend Yields and Expected Stock Returns"
- Cochrane (2008) - "The Dog That Did Not Bark: A Defense of Return Predictability"

### Books
- "Expected Returns" by Antti Ilmanen
- "Global Asset Allocation" by Meb Faber
- "Active Portfolio Management" by Grinold & Kahn

### Online Resources
- Shiller's Online Data: http://www.econ.yale.edu/~shiller/data.htm
- FRED: https://fred.stlouisfed.org/
- World Bank Data: https://data.worldbank.org/

---

## Support

For issues or questions:
1. Check ARCHITECTURE.md for detailed technical documentation
2. Review the code docstrings
3. Run tests: `pytest tests/ -v`
4. Check logs: `logs/predictor.log`

---

**Happy forecasting! 📈**
