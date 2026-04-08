"""
Report generation for model predictions and analysis.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import logging

from config.settings import REPORT_CONFIG, MARKETS

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates professional reports from model predictions.
    """
    
    def __init__(
        self,
        format: str = "console",
        markets: Optional[List[str]] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize report generator.
        
        Args:
            format: 'console', 'pdf', or 'html'
            markets: List of markets to include
            output_dir: Directory to save reports
        """
        self.format = format
        if markets is None or markets == ["all"]:
            self.markets = list(MARKETS.keys())
        else:
            self.markets = markets
        self.output_dir = Path(output_dir or "reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
    def generate_console_report(self, predictions: Dict[str, pd.DataFrame]):
        """
        Generate a console-based report.
        
        Args:
            predictions: Dictionary of predictions by market
        """
        print("\n" + "="*70)
        print("  GLOBAL EQUITY MARKET OUTLOOK REPORT")
        print("="*70)
        print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"  Forecast Horizon: {REPORT_CONFIG['forecast_horizon_years']} Years")
        print("="*70)
        
        # Build summary table
        summary_data = []
        
        for market in self.markets:
            if market not in predictions:
                continue
            
            pred_df = predictions[market]
            
            # Get ensemble prediction (average of all models)
            ensemble_pred = pred_df['predicted_return'].mean()
            
            # Calculate confidence based on model agreement
            model_std = pred_df['predicted_return'].std()
            confidence = self._calculate_confidence(model_std)
            
            # Get historical average (placeholder - would come from actual data)
            historical_avg = 7.0  # Approximate historical equity return
            vs_historical = ensemble_pred - historical_avg
            
            summary_data.append({
                'market': market,
                'expected_return': ensemble_pred,
                'confidence': confidence,
                'vs_historical': vs_historical,
                'model_std': model_std
            })
        
        if not summary_data:
            print("\n  No predictions available.")
            return
        
        # Sort by expected return
        summary_data.sort(key=lambda x: x['expected_return'], reverse=True)
        
        # Print rankings
        print("\n  📊 MARKET RANKINGS (Expected 3-Year Annualized Return)")
        print("  " + "-"*68)
        print(f"  {'Rank':<6}{'Market':<12}{'Expected Return':<18}{'Confidence':<12}{'vs History':<15}")
        print("  " + "-"*68)
        
        for rank, data in enumerate(summary_data, 1):
            return_str = f"{data['expected_return']:+.1f}%"
            vs_hist_str = f"{data['vs_historical']:+.1f}%"
            
            print(f"  {rank:<6}{data['market']:<12}{return_str:<18}{data['confidence']:<12}{vs_hist_str:<15}")
        
        print("  " + "-"*68)
        
        # Detailed model predictions
        print("\n  📈 DETAILED MODEL PREDICTIONS")
        print("  " + "-"*68)
        
        for data in summary_data:
            market = data['market']
            pred_df = predictions[market]
            
            print(f"\n  {market}:")
            for _, row in pred_df.iterrows():
                model = row['model']
                pred = row['predicted_return']
                print(f"    • {model:15s}: {pred:+6.2f}%")
            print(f"    {'─'*40}")
            print(f"    • Ensemble:      {data['expected_return']:+6.2f}%")
        
        # Key insights
        print("\n  🔍 KEY INSIGHTS")
        print("  " + "-"*68)
        
        best = summary_data[0]
        worst = summary_data[-1]
        
        print(f"  • Highest expected return: {best['market']} ({best['expected_return']:+.1f}%)")
        print(f"  • Lowest expected return:  {worst['market']} ({worst['expected_return']:+.1f}%)")
        print(f"  • Return dispersion:       {best['expected_return'] - worst['expected_return']:.1f}%")
        
        attractive = [d for d in summary_data if d['expected_return'] > 6.0]
        if attractive:
            print(f"  • Markets with >6% expected return: {', '.join(d['market'] for d in attractive)}")
        else:
            print(f"  • No markets currently offering >6% expected returns")
        
        # Risk disclaimer
        print("\n  ⚠️  RISK DISCLAIMER")
        print("  " + "-"*68)
        print("  These are model predictions based on historical relationships.")
        print("  Past performance does not guarantee future results.")
        print("  Models can fail in unprecedented market conditions.")
        print("  This is for educational/research purposes only.")
        
        print("\n" + "="*70)
        print("  Report generated by Global Equity Predictor v1.0")
        print("="*70 + "\n")
    
    def generate_pdf_report(self, predictions: Dict[str, pd.DataFrame]):
        """
        Generate a PDF report (placeholder - requires reportlab).
        
        Args:
            predictions: Dictionary of predictions by market
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet
            
            filename = f"equity_outlook_{datetime.now().strftime('%Y%m%d')}.pdf"
            filepath = self.output_dir / filename
            
            doc = SimpleDocTemplate(str(filepath), pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            title = Paragraph("Global Equity Market Outlook Report", styles['Title'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Date
            date_str = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            story.append(Paragraph(date_str, styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Summary table
            data = [['Market', 'Expected Return', 'Confidence']]
            
            for market in self.markets:
                if market in predictions:
                    pred_df = predictions[market]
                    ensemble_pred = pred_df['predicted_return'].mean()
                    model_std = pred_df['predicted_return'].std()
                    confidence = self._calculate_confidence(model_std)
                    
                    data.append([
                        market,
                        f"{ensemble_pred:+.1f}%",
                        confidence
                    ])
            
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(table)
            
            # Build PDF
            doc.build(story)
            
            self.logger.info(f"Generated PDF report: {filepath}")
            print(f"\n  PDF report saved to: {filepath}")
            
        except ImportError:
            self.logger.error("reportlab not installed. Install with: pip install reportlab")
            print("\n  ⚠️  PDF generation requires reportlab. Install with: pip install reportlab")
            print("  Falling back to console report...")
            self.generate_console_report(predictions)
    
    def generate_html_report(self, predictions: Dict[str, pd.DataFrame]):
        """
        Generate an HTML report with interactive charts.
        
        Args:
            predictions: Dictionary of predictions by market
        """
        try:
            import plotly.graph_objects as go
            from plotly.offline import plot
            
            # Create HTML content
            html_parts = []
            html_parts.append("""
<!DOCTYPE html>
<html>
<head>
    <title>Global Equity Market Outlook</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th { background: #4CAF50; color: white; padding: 12px; text-align: left; }
        td { padding: 10px; border-bottom: 1px solid #ddd; }
        tr:hover { background: #f5f5f5; }
        .positive { color: #4CAF50; }
        .negative { color: #f44336; }
        .disclaimer { background: #fff3cd; border: 1px solid #ffc107; padding: 15px; margin-top: 30px; border-radius: 5px; }
        .chart { margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Global Equity Market Outlook Report</h1>
        <p><strong>Generated:</strong> {}</p>
        <p><strong>Forecast Horizon:</strong> {} Years</p>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M'), REPORT_CONFIG['forecast_horizon_years']))
            
            # Build predictions table
            summary_data = []
            for market in self.markets:
                if market in predictions:
                    pred_df = predictions[market]
                    ensemble_pred = pred_df['predicted_return'].mean()
                    model_std = pred_df['predicted_return'].std()
                    confidence = self._calculate_confidence(model_std)
                    
                    summary_data.append({
                        'market': market,
                        'return': ensemble_pred,
                        'confidence': confidence
                    })
            
            # Sort by return
            summary_data.sort(key=lambda x: x['return'], reverse=True)
            
            html_parts.append("<h2>Market Rankings</h2>")
            html_parts.append("<table>")
            html_parts.append("<tr><th>Rank</th><th>Market</th><th>Expected Return</th><th>Confidence</th></tr>")
            
            for rank, data in enumerate(summary_data, 1):
                return_class = "positive" if data['return'] > 0 else "negative"
                html_parts.append(f"""
                    <tr>
                        <td>{rank}</td>
                        <td><strong>{data['market']}</strong></td>
                        <td class="{return_class}">{data['return']:+.1f}%</td>
                        <td>{data['confidence']}</td>
                    </tr>
                """)
            
            html_parts.append("</table>")
            
            # Create bar chart
            markets = [d['market'] for d in summary_data]
            returns = [d['return'] for d in summary_data]
            
            fig = go.Figure(data=[go.Bar(
                x=markets,
                y=returns,
                marker_color=['#4CAF50' if r > 0 else '#f44336' for r in returns]
            )])
            
            fig.update_layout(
                title="Expected 3-Year Annualized Returns by Market",
                yaxis_title="Expected Return (%)",
                xaxis_title="Market",
                template="plotly_white"
            )
            
            chart_html = plot(fig, output_type='div', include_plotlyjs='cdn')
            html_parts.append(f'<div class="chart">{chart_html}</div>')
            
            # Model predictions table
            html_parts.append("<h2>Detailed Model Predictions</h2>")
            
            for data in summary_data:
                market = data['market']
                pred_df = predictions[market]
                
                html_parts.append(f"<h3>{market}</h3>")
                html_parts.append("<table><tr><th>Model</th><th>Predicted Return</th></tr>")
                
                for _, row in pred_df.iterrows():
                    return_class = "positive" if row['predicted_return'] > 0 else "negative"
                    html_parts.append(f"""
                        <tr>
                            <td>{row['model']}</td>
                            <td class="{return_class}">{row['predicted_return']:+.2f}%</td>
                        </tr>
                    """)
                
                html_parts.append("</table>")
            
            # Disclaimer
            html_parts.append("""
                <div class="disclaimer">
                    <strong>⚠️ Risk Disclaimer</strong><br>
                    These predictions are based on historical relationships between macroeconomic 
                    indicators and equity returns. Past performance does not guarantee future results. 
                    Models can fail in unprecedented market conditions. This report is for educational 
                    and research purposes only and should not be considered investment advice.
                </div>
            """)
            
            html_parts.append("</div></body></html>")
            
            # Save file
            filename = f"equity_outlook_{datetime.now().strftime('%Y%m%d')}.html"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w') as f:
                f.write(''.join(html_parts))
            
            self.logger.info(f"Generated HTML report: {filepath}")
            print(f"\n  HTML report saved to: {filepath}")
            print(f"  Open in browser: file://{filepath.absolute()}")
            
        except ImportError:
            self.logger.error("plotly not installed. Install with: pip install plotly")
            print("\n  ⚠️  HTML generation requires plotly. Install with: pip install plotly")
            print("  Falling back to console report...")
            self.generate_console_report(predictions)
    
    def _calculate_confidence(self, model_std: float) -> str:
        """
        Calculate confidence level based on model agreement.
        
        Args:
            model_std: Standard deviation of model predictions
            
        Returns:
            Confidence string: 'Low', 'Medium', or 'High'
        """
        if model_std < 1.0:
            return "High"
        elif model_std < 2.5:
            return "Medium"
        else:
            return "Low"
