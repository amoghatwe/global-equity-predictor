"""
Report Generation Module.

This module converts raw model predictions into human-readable reports
in various formats (Console, PDF, HTML). It calculates summary metrics,
determines confidence levels based on model agreement, and provides
economic context for the forecasts.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path
import logging

from config.settings import REPORT_CONFIG, MARKETS

# Set up module logger
logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Translates model output into professional reports.
    
    This class handles the presentation layer of the system, creating
    visualizations and tables that summarize the market outlook.
    """
    
    def __init__(
        self,
        output_format: str = "console",
        target_markets: Optional[List[str]] = None,
        report_directory: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the report generator.
        
        Args:
            output_format: 'console', 'pdf', or 'html'.
            target_markets: List of markets to include in the report.
            report_directory: Directory where report files will be saved.
        """
        self.output_format = output_format
        self.markets = self._initialize_markets(target_markets)
        self.output_dir = Path(report_directory or "reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
    def _initialize_markets(self, markets: Optional[List[str]]) -> List[str]:
        """Validate and return the list of markets to report on."""
        if markets is None or markets == ["all"]:
            return list(MARKETS.keys())
        return markets

    def generate_console_report(self, forecasts: Dict[str, pd.DataFrame]):
        """
        Prints a stylized, high-signal report to the standard output.
        
        Args:
            forecasts: Mapping of market names to DataFrames containing
                       individual model predictions.
        """
        # Header Section
        print("\n" + "=" * 70)
        print("  GLOBAL EQUITY MARKET OUTLOOK REPORT")
        print("=" * 70)
        print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"  Forecast Horizon: {REPORT_CONFIG['forecast_horizon_years']} Years")
        print("=" * 70)
        
        # Data aggregation for summary
        market_summaries = []
        for market in self.markets:
            if market not in forecasts:
                continue
            
            model_preds = forecasts[market]
            # Average across all successful models (Ensemble)
            mean_return = model_preds['predicted_return'].mean()
            # Consensus: How much do models agree?
            agreement_std = model_preds['predicted_return'].std()
            confidence = self._determine_confidence_level(agreement_std)
            
            # Contextual metric: comparison to historical 7% equity benchmark
            vs_historical = mean_return - 7.0
            
            market_summaries.append({
                'market': market,
                'mean_return': mean_return,
                'confidence': confidence,
                'vs_hist': vs_historical,
                'raw_preds': model_preds
            })
        
        if not market_summaries:
            print("\n  [!] No forecast data available to display.")
            return
            
        # Sort markets from most attractive to least attractive
        market_summaries.sort(key=lambda x: x['mean_return'], reverse=True)
        
        # Market Rankings Section
        print("\n  📊 MARKET RANKINGS (Expected 3-Year Annualized Return)")
        print("  " + "-" * 68)
        print(f"  {'Rank':<6}{'Market':<12}{'Expected Return':<18}{'Confidence':<12}{'vs History':<15}")
        print("  " + "-" * 68)
        
        for i, s in enumerate(market_summaries, 1):
            ret_text = f"{s['mean_return']:+.1f}%"
            hist_text = f"{s['vs_hist']:+.1f}%"
            print(f"  {i:<6}{s['market']:<12}{ret_text:<18}{s['confidence']:<12}{hist_text:<15}")
        
        print("  " + "-" * 68)
        
        # Detailed Breakdown Section
        print("\n  📈 DETAILED MODEL PREDICTIONS")
        print("  " + "-" * 68)
        
        for s in market_summaries:
            print(f"\n  {s['market']}:")
            for _, row in s['raw_preds'].iterrows():
                print(f"    • {row['model']:15s}: {row['predicted_return']:+6.2f}%")
            print(f"    {'─' * 40}")
            print(f"    • Ensemble Mean: {s['mean_return']:+6.2f}%")
            
        # Insights Section
        print("\n  🔍 KEY INSIGHTS")
        print("  " + "-" * 68)
        top = market_summaries[0]
        bottom = market_summaries[-1]
        print(f"  • Top Opportunity:    {top['market']} ({top['mean_return']:+.1f}%)")
        print(f"  • Maximum Risk:      {bottom['market']} ({bottom['mean_return']:+.1f}%)")
        print(f"  • Market Dispersion: {top['mean_return'] - bottom['mean_return']:.1f}%")
        
        # Risk Disclaimer Section
        print("\n  ⚠️  RISK DISCLAIMER")
        print("  " + "-" * 68)
        print("  These forecasts are probabilistic estimates based on historical data.")
        print("  ML models can fail in 'Black Swan' events or new market regimes.")
        print("  This report is for research purposes and is NOT investment advice.")
        
        print("\n" + "=" * 70)
        print(f"  Analytical Report | {datetime.now().year} Global Equity Predictor")
        print("=" * 70 + "\n")

    def _determine_confidence_level(self, std_dev: float) -> str:
        """Categorizes confidence based on the spread of model predictions."""
        if pd.isna(std_dev): return "N/A"
        if std_dev < 1.0: return "High"
        if std_dev < 2.5: return "Medium"
        return "Low"

    def generate_pdf_report(self, forecasts: Dict[str, pd.DataFrame]):
        """Place-holder for PDF generation using ReportLab."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
            
            output_file = self.output_dir / f"outlook_{datetime.now().strftime('%Y%m%d')}.pdf"
            doc = SimpleDocTemplate(str(output_file), pagesize=letter)
            styles = getSampleStyleSheet()
            
            story = [
                Paragraph("Global Equity Outlook", styles['Title']),
                Spacer(1, 12),
                Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d')}", styles['Normal']),
                Spacer(1, 12),
                Paragraph("PDF generation logic is active. Visual charts would be placed here.", styles['Italic'])
            ]
            
            doc.build(story)
            print(f"  [✓] PDF report exported to: {output_file}")
            
        except ImportError:
            logger.warning("PDF generation requires 'reportlab' library. Falling back to Console.")
            self.generate_console_report(forecasts)

    def generate_html_report(self, forecasts: Dict[str, pd.DataFrame]):
        """Generates an HTML report. Implementation follows similar logic to Console."""
        # For brevity in this exercise, I'm providing a minimal HTML bridge.
        # In a real system, this would use Jinja2 templates.
        print("  [!] HTML report generation is currently a placeholder for the web UI.")
        self.generate_console_report(forecasts)
