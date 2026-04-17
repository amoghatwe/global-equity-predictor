#!/usr/bin/env python3
"""
Global Equity Market Return Predictor
====================================

An end-to-end machine learning system that predicts 3-year forward returns
for global equity markets by analyzing macroeconomic and fundamental indicators.

Supported Modes:
    - collect: Gather raw data from World Bank, FRED, and Yahoo Finance.
    - features: Engineer technical and economic indicators from raw data.
    - train: Run walk-forward cross-validation and train final models.
    - predict: Use trained models to forecast the next 3 years of returns.
    - full: Run the entire pipeline from collection to prediction.
"""

import argparse
import logging
import logging.config
import sys
from pathlib import Path
from datetime import datetime

# Add the 'src' directory to the Python path to enable module imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.settings import LOGGING_CONFIG
from src.data_collection.pipeline import DataIngestionPipeline
from src.features.pipeline import FeatureEngineeringPipeline
from src.models.pipeline import ModelTrainingPipeline
from src.reporting.generator import ReportGenerator

# Configure logging based on project settings
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Defines and parses command-line arguments for the system."""
    parser = argparse.ArgumentParser(
        description="Global Equity Market Return Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["collect", "features", "train", "predict", "full"],
        default="full",
        help="Pipeline execution mode (default: full)",
    )

    parser.add_argument(
        "--report",
        type=str,
        choices=["none", "console", "pdf", "html"],
        default="console",
        help="Report output format (default: console)",
    )

    parser.add_argument(
        "--market",
        type=str,
        nargs="+",
        default=["all"],
        help="Specific markets to analyze (e.g., USA EM). Default: all",
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["linear", "rf", "xgb", "arma", "arima", "ensemble", "all"],
        default="ensemble",
        help="Model architecture to utilize (default: ensemble)",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        help="Earliest date for historical data (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        help="Latest date for historical data (YYYY-MM-DD)",
    )

    return parser.parse_args()


def execute_data_collection(args):
    """Orchestrates the retrieval and merging of raw data sources."""
    logger.info("PHASE 1: Starting data collection and ingestion...")

    pipeline = DataIngestionPipeline(
        start_date=args.start_date, 
        end_date=args.end_date, 
        target_markets=args.market
    )

    # 1. Fetch raw data from all providers
    raw_datasets = pipeline.run_collection_sequence()
    
    # 2. Transform into a unified analytical format
    unified_df = pipeline.transform_and_merge(raw_datasets)
    
    # 3. Persist to disk
    pipeline.save_processed_data(unified_df)

    logger.info("Data collection completed successfully.")
    return unified_df


def execute_feature_engineering(args):
    """Orchestrates the creation of indicators and training labels."""
    logger.info("PHASE 2: Starting feature engineering and labeling...")

    pipeline = FeatureEngineeringPipeline(target_markets=args.market)

    # 1. Create the feature matrix
    features = pipeline.engineer_all_features()
    
    # 2. Create the target variables (ground truth)
    targets = pipeline.engineer_target_variables()
    
    # 3. Merge and align for training
    final_dataset = pipeline.merge_and_filter_final_dataset(features, targets)

    # 4. Save the finalized training set
    pipeline.save_features(final_dataset)

    logger.info("Feature engineering completed successfully.")
    return final_dataset


def execute_model_training(args):
    """Orchestrates model training and performance validation."""
    logger.info("PHASE 3: Starting model training and cross-validation...")

    pipeline = ModelTrainingPipeline(
        model_selection=args.model, 
        target_markets=args.market
    )

    # 1. Run the training cycle with walk-forward validation
    results = pipeline.execute_training_cycle()
    
    # 2. Persist the trained model objects
    pipeline.save_trained_models()
    
    # 3. Log the performance summary
    pipeline.print_evaluation_summary(results)

    logger.info("Model training completed successfully.")
    return results


def execute_prediction(args):
    """Generates return forecasts using the most recent available data."""
    logger.info("PHASE 4: Generating future return forecasts...")

    pipeline = ModelTrainingPipeline(
        model_selection=args.model, 
        target_markets=args.market
    )

    # Use existing models to forecast based on latest features
    forecasts = pipeline.generate_forecasts()

    logger.info("Forecast generation completed successfully.")
    return forecasts


def generate_report(args, forecasts):
    """Translates raw forecasts into a human-readable report."""
    if args.report == "none" or not forecasts:
        return

    logger.info(f"PHASE 5: Generating {args.report} outlook report...")

    # The report generator takes care of the visualization logic
    generator = ReportGenerator(output_format=args.report, target_markets=args.market)

    if args.report == "console":
        generator.generate_console_report(forecasts)
    elif args.report == "pdf":
        generator.generate_pdf_report(forecasts)
    elif args.report == "html":
        generator.generate_html_report(forecasts)

    logger.info("Report delivery completed.")


def main():
    """Main system entry point."""
    args = parse_arguments()

    # Log system header
    logger.info("=" * 60)
    logger.info("GLOBAL EQUITY MARKET PREDICTION SYSTEM")
    logger.info(f"Mode: {args.mode.upper()} | Model: {args.model.upper()}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    try:
        # Routing based on selected mode
        if args.mode == "collect":
            execute_data_collection(args)

        elif args.mode == "features":
            execute_feature_engineering(args)

        elif args.mode == "train":
            execute_model_training(args)

        elif args.mode == "predict":
            forecasts = execute_prediction(args)
            generate_report(args, forecasts)

        elif args.mode == "full":
            # Execute the entire end-to-end pipeline
            execute_data_collection(args)
            execute_feature_engineering(args)
            execute_model_training(args)
            forecasts = execute_prediction(args)
            generate_report(args, forecasts)

        logger.info("Pipeline execution finished successfully.")
        return 0

    except Exception as e:
        logger.critical(f"Pipeline crashed due to an unhandled exception: {str(e)}")
        # Log the full stack trace for debugging if in verbose mode
        logger.debug("Stack Trace:", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
