#!/usr/bin/env python3
"""
Global Equity Market Return Predictor
====================================

A machine learning system for predicting 3-year forward returns in global equity markets
using macroeconomic indicators.

Usage:
    python run_prediction.py --mode train
    python run_prediction.py --mode predict --report pdf
    python run_prediction.py --mode full
"""

import argparse
import logging
import logging.config
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.settings import LOGGING_CONFIG
from src.data_collection.pipeline import DataPipeline
from src.features.pipeline import FeaturePipeline
from src.models.pipeline import ModelPipeline
from src.reporting.generator import ReportGenerator

# Setup logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Global Equity Market Return Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode train              Train models on historical data
  %(prog)s --mode predict            Generate current predictions
  %(prog)s --mode full               Run full pipeline
  %(prog)s --mode predict --report pdf    Generate PDF report
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["collect", "features", "train", "predict", "full"],
        default="full",
        help="Execution mode (default: full)"
    )
    
    parser.add_argument(
        "--report",
        type=str,
        choices=["none", "console", "pdf", "html"],
        default="console",
        help="Report output format (default: console)"
    )
    
    parser.add_argument(
        "--market",
        type=str,
        nargs="+",
        choices=["USA", "Europe", "Japan", "UK", "EM", "all"],
        default=["all"],
        help="Specific markets to process (default: all)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["linear", "rf", "xgb", "ensemble", "all"],
        default="ensemble",
        help="Model type to use (default: ensemble)"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for data collection (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for data collection (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def run_data_collection(args):
    """Execute data collection phase."""
    logger.info("Starting data collection phase...")
    
    pipeline = DataPipeline(
        start_date=args.start_date,
        end_date=args.end_date,
        markets=args.market
    )
    
    raw_data = pipeline.collect_all()
    processed_data = pipeline.process(raw_data)
    pipeline.save(processed_data)
    
    logger.info("Data collection completed successfully")
    return processed_data


def run_feature_engineering(args):
    """Execute feature engineering phase."""
    logger.info("Starting feature engineering phase...")
    
    pipeline = FeaturePipeline(markets=args.market)
    
    features = pipeline.create_features()
    targets = pipeline.create_targets()
    dataset = pipeline.merge_features_targets(features, targets)
    
    pipeline.save(dataset)
    
    logger.info("Feature engineering completed successfully")
    return dataset


def run_model_training(args):
    """Execute model training phase."""
    logger.info("Starting model training phase...")
    
    pipeline = ModelPipeline(
        model_type=args.model,
        markets=args.market
    )
    
    results = pipeline.train()
    pipeline.save_models()
    pipeline.evaluate(results)
    
    logger.info("Model training completed successfully")
    return results


def run_prediction(args):
    """Execute prediction phase."""
    logger.info("Starting prediction phase...")
    
    pipeline = ModelPipeline(
        model_type=args.model,
        markets=args.market
    )
    
    predictions = pipeline.predict()
    
    logger.info("Prediction completed successfully")
    return predictions


def generate_report(args, predictions):
    """Generate output report."""
    if args.report == "none":
        return
    
    logger.info(f"Generating {args.report} report...")
    
    generator = ReportGenerator(
        format=args.report,
        markets=args.market
    )
    
    if args.report == "console":
        generator.generate_console_report(predictions)
    elif args.report == "pdf":
        generator.generate_pdf_report(predictions)
    elif args.report == "html":
        generator.generate_html_report(predictions)
    
    logger.info("Report generation completed")


def main():
    """Main execution function."""
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("=" * 60)
    logger.info("Global Equity Market Return Predictor")
    logger.info(f"Execution mode: {args.mode}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    try:
        if args.mode == "collect":
            run_data_collection(args)
        
        elif args.mode == "features":
            run_feature_engineering(args)
        
        elif args.mode == "train":
            run_model_training(args)
        
        elif args.mode == "predict":
            predictions = run_prediction(args)
            generate_report(args, predictions)
        
        elif args.mode == "full":
            # Run complete pipeline
            run_data_collection(args)
            dataset = run_feature_engineering(args)
            results = run_model_training(args)
            predictions = run_prediction(args)
            generate_report(args, predictions)
        
        logger.info("Execution completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
