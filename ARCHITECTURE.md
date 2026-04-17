# Global Equity Predictor - Architecture Overview

## Project Structure

```
global_equity_predictor/
├── config/                         # Configuration files
│   └── settings.py                # Main configuration (API keys, parameters, markets)
│
├── src/                           # Source code
│   ├── __init__.py
│   │
│   ├── data_collection/           # Data fetching modules
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract base classes for data sources
│   │   ├── world_bank.py         # World Bank macro data (GDP, inflation, etc.)
│   │   ├── fred.py               # FRED API (US data, yields, VIX)
│   │   ├── equity.py             # Yahoo Finance (index prices, returns)
│   │   └── pipeline.py           # Orchestrates all data collection
│   │
│   ├── features/                  # Feature engineering
│   │   ├── __init__.py
│   │   └── pipeline.py           # Creates ML features from raw data
│   │
│   ├── models/                    # Machine learning models
│   │   ├── __init__.py
│   │   ├── base.py               # Base classes, time series CV, ensembles
│   │   ├── implementations.py    # Linear, Random Forest, XGBoost models
│   │   └── pipeline.py           # Training pipeline with walk-forward validation
│   │
│   └── reporting/                 # Report generation
│       ├── __init__.py
│       └── generator.py          # Console, PDF, HTML reports
│
├── data/                          # Data storage (not in git)
│   ├── raw/                       # Downloaded raw data
│   ├── processed/                 # Cleaned and merged data
│   └── features/                  # ML feature matrices
│
├── notebooks/                     # Jupyter notebooks for exploration
│
├── tests/                         # Unit tests
│   ├── __init__.py
│   └── test_system.py            # Comprehensive test suite
│
├── logs/                          # Log files
│
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
├── setup.sh                       # Automated setup script
├── package.json                   # Project metadata
├── README.md                      # User documentation
├── ARCHITECTURE.md               # This file - technical architecture
└── run_prediction.py             # Main entry point and CLI

Total: 19 Python files, ~2,400 lines of code
```

## Architecture Principles

### 1. **Modularity**
- Each data source is independent and swappable
- Feature engineering is separate from model training
- Models can be trained individually or as ensemble
- Reports can be generated in multiple formats

### 2. **Separation of Concerns**
- **Data Collection**: Only fetches and validates data
- **Feature Engineering**: Only transforms raw data → features
- **Models**: Only training and prediction logic
- **Reporting**: Only presentation and visualization

### 3. **Configurability**
- All parameters in `config/settings.py`
- API keys via environment variables (.env file)
- Markets, date ranges, model parameters all configurable

### 4. **Time Series Safety**
- Walk-forward cross-validation prevents look-ahead bias
- No shuffling or random sampling of time series data
- Expanding/rolling windows for training sets

### 5. **Extensibility**
- Easy to add new data sources (inherit from `DataSource`)
- Easy to add new features (extend `FeaturePipeline`)
- Easy to add new models (inherit from `BaseReturnModel`)
- Easy to add new report formats (extend `ReportGenerator`)

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: Data Collection                                   │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  World Bank API ────┐                                       │
│                     │                                       │
│  FRED API ──────────┼───► DataPipeline ────► data/raw/      │
│                     │    (cleans & merges)                  │
│  Yahoo Finance ─────┘                                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 2: Feature Engineering                               │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  Raw Data ────► FeaturePipeline ────► data/features/        │
│  (processed/)    • Valuation features                       │
│                  • Growth features                          │
│                  • Inflation features                       │
│                  • Rate features                            │
│                  • Credit features                          │
│                  • Momentum features                        │
│                  • 3-year forward returns (targets)         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 3: Model Training                                    │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  Features ────► ModelPipeline ────► Trained models          │
│  + Targets      • Walk-forward CV                           │
│                 • Linear Regression (Ridge)                 │ 
│                 • Random Forest                             │
│                 • XGBoost                                   │
│                 • Ensemble average                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 4: Prediction & Reporting                            │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  Latest Data ────► Predict ────► ReportGenerator            │
│                    (3-year      • Console output            │
│                     returns)     • PDF report               │
│                                  • HTML report              │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### Data Collection

#### `DataSource` (Abstract Base Class)
All data sources inherit from this. Defines interface:
- `fetch(**kwargs)` - Get data from API
- `validate(data)` - Check data quality
- `save_to_cache()` / `load_from_cache()` - Disk persistence

#### `DataCollector`
Orchestrates multiple data sources:
- Registers multiple `DataSource` instances
- Coordinates fetching from all sources
- Handles errors gracefully (one source fails, others continue)

#### `DataPipeline`
Main entry point:
- Collects from all sources
- Processes (cleans, merges, resamples)
- Saves to `data/processed/`

### Feature Engineering

#### `FeaturePipeline`
Creates ML-ready features:
- **Valuation**: Price-to-moving average ratios, deviations
- **Growth**: GDP trend, industrial production growth
- **Inflation**: CPI levels, real rates
- **Rates**: Yield curve, policy rates
- **Credit**: M2 growth, credit/GDP
- **Momentum**: Trailing returns, volatility

Also creates **targets** (3-year forward annualized returns).

### Models

#### `BaseReturnModel` (Abstract)
Base class for all models:
- `train(X, y)` - Fit model
- `predict(X)` - Make predictions
- `evaluate(y_true, y_pred)` - Calculate metrics
- `get_feature_importance()` - Explainability
- `save()` / `load()` - Persistence

#### `TimeSeriesCrossValidator`
Specialized CV for financial time series:
- Walk-forward splits
- Expanding or rolling windows
- Gap between train/test to prevent leakage
- Embargo for overlapping periods

#### `ModelEnsemble`
Combines multiple models:
- Weighted average of predictions
- Configurable weights
- Evaluates all models + ensemble

#### Implementations
- **LinearReturnModel**: Ridge regression (interpretable baseline)
- **RandomForestReturnModel**: Non-linear, feature importance
- **XGBoostReturnModel**: Gradient boosting with regularization

### Reporting

#### `ReportGenerator`
Multiple output formats:
- **Console**: Text-based tables in terminal
- **PDF**: Professional document (requires reportlab)
- **HTML**: Interactive charts (requires plotly)

## Design Patterns

### 1. **Factory Pattern**
```python
# models/implementations.py
def create_model(model_type: str) -> BaseReturnModel:
    model_map = {
        'linear': LinearReturnModel,
        'rf': RandomForestReturnModel,
        'xgb': XGBoostReturnModel,
    }
    return model_map[model_type]()
```

### 2. **Template Method Pattern**
```python
# data_collection/base.py
class DataSource(ABC):
    def fetch_and_validate(self, **kwargs):
        data = self.fetch(**kwargs)
        if self.validate(data):
            return data
        raise ValueError("Validation failed")
    
    @abstractmethod
    def fetch(self, **kwargs): pass
    
    @abstractmethod
    def validate(self, data): pass
```

### 3. **Pipeline Pattern**
```python
# Main execution flow
DataPipeline() → FeaturePipeline() → ModelPipeline() → ReportGenerator()
```

### 4. **Strategy Pattern**
Different models implement same interface:
```python
models = {
    'linear': LinearReturnModel(),
    'rf': RandomForestReturnModel(),
    'xgb': XGBoostReturnModel(),
}
```

## Error Handling

### Data Collection
- Individual source failures don't stop pipeline
- Warnings logged for missing data
- Fallback to cached data if available
- Validation checks data quality before saving

### Feature Engineering
- Graceful handling of missing features
- Forward-fill for recent gaps
- Skip markets with insufficient data

### Model Training
- Try all models, continue if one fails
- Detailed error logging
- Validation prevents training on bad data

### Prediction
- Warn if models not trained
- Confidence scores based on model agreement

## Configuration System

### Environment Variables (.env)
```bash
FRED_API_KEY=your_key_here
DATA_RAW_PATH=custom/path
RANDOM_STATE=42
```

### Settings (config/settings.py)
```python
MARKETS = {
    "USA": {"ticker": "SPY", ...},
    "Europe": {"ticker": "VGK", ...},
    ...
}

FORECAST_HORIZON_MONTHS = 36

MODEL_CONFIG = {
    "n_splits": 5,
    "min_train_years": 15,
}
```

### CLI Arguments
```bash
python run_prediction.py --mode train --model xgb --market USA Europe
```

## Testing Strategy

### Unit Tests
- Test individual components in isolation
- Mock external APIs
- Test edge cases (empty data, missing values)

### Integration Tests
- Test data flow between components
- Test end-to-end with synthetic data

### Validation
- Time series CV prevents data leakage
- Out-of-sample testing
- Walk-forward validation

## Performance Considerations

### Memory
- Data loaded as needed, not all at once
- Streaming for large datasets
- Efficient pandas operations

### Speed
- Cached data to avoid repeated API calls
- Parallel processing where possible (n_jobs=-1)
- Lazy loading of heavy dependencies

### Scalability
- Easy to add more markets
- Easy to add more features
- Easy to add more models
- Modular design supports horizontal scaling

## Security

### API Keys
- Stored in `.env` file (not in git)
- `.env` in `.gitignore`
- Environment variables in production

### Data
- No sensitive data in code
- Local data storage only
- No external data sharing

## Deployment Options

### Development
```bash
python run_prediction.py --mode full
```

### Scheduled Execution (Cron)
```bash
# Monthly update
0 9 1 * * cd /path/to/project && python run_prediction.py --mode full --report html
```

### As Library
```python
from src.models.pipeline import ModelPipeline

pipeline = ModelPipeline()
results = pipeline.train()
predictions = pipeline.predict()
```

## Extension Points

### Adding a New Data Source
1. Create class inheriting from `DataSource`
2. Implement `fetch()` and `validate()`
3. Register in `DataPipeline._register_sources()`

### Adding a New Feature
1. Add method to `FeaturePipeline`
2. Call in `create_features()`
3. Add to `FEATURE_CATEGORIES` in settings

### Adding a New Model
1. Create class inheriting from `BaseReturnModel`
2. Implement `train()` and `predict()`
3. Add to `create_model()` factory
4. Add parameters to settings

### Adding a New Report Format
1. Add method to `ReportGenerator`
2. Implement formatting logic
3. Add to CLI options in `run_prediction.py`

## Best Practices Enforced

1. **Type Hints**: All functions have type annotations
2. **Docstrings**: Comprehensive documentation
3. **Logging**: Structured logging throughout
4. **Error Handling**: Graceful degradation
5. **Validation**: Data quality checks at each stage
6. **Immutability**: DataFrames copied before modification
7. **Configuration**: No hardcoded values
8. **Testing**: Unit tests for core functionality
9. **Modularity**: Single responsibility principle
10. **DRY**: Code reuse through inheritance and composition

## Future Enhancements

### Short Term
- [ ] Add more macro indicators (PMI, consumer confidence)
- [ ] Implement feature selection (RFE, SHAP)
- [ ] Add confidence intervals to predictions
- [ ] Create backtesting framework
- [ ] Improve the reports
- [ ] Increase time-frame and improve 

### Medium Term
- [ ] Add sector-level predictions
- [ ] Implement risk-parity allocation
- [ ] Add sentiment analysis (news, social media)
- [ ] Support for more markets (Australia, Canada, etc.)

### Long Term
- [ ] Deep learning models (LSTM, Transformer)
- [ ] Reinforcement learning for allocation
- [ ] Real-time data streaming
- [ ] Web dashboard for monitoring
- [ ] API service for predictions