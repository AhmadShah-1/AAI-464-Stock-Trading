"""
Visualization module for stock trading predictions.
Creates charts showing price movements, model predictions, and technical indicators.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
from pathlib import Path


class TradingVisualizer:
    """Creates visualizations for stock trading predictions and signals."""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save images (default: assets/images)
        """
        if output_dir is None:
            # Default to assets/images in the main directory
            main_dir = Path(__file__).parent.parent
            output_dir = main_dir / 'assets' / 'images'
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_trading_signals(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        confidence: np.ndarray,
        model_name: str,
        symbol: str,
        feature_cols: list = None
    ):
        """
        Create comprehensive trading visualization with multiple subplots.
        
        Args:
            df: DataFrame with OHLCV data and features
            predictions: Array of model predictions (0=sell, 1=hold, 2=buy)
            confidence: Array of confidence scores
            model_name: Name of the model
            symbol: Stock symbol
            feature_cols: List of feature columns to use (optional)
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 1, height_ratios=[3, 1.5, 1.5, 1], hspace=0.3)
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            dates = df['timestamp'].values
        else:
            dates = pd.to_datetime(df.index)
        
        # Add predictions and confidence to dataframe for plotting
        df_plot = df.copy()
        df_plot['prediction'] = predictions
        df_plot['confidence'] = confidence
        
        # Subplot 1: Price and Trading Signals
        ax1 = fig.add_subplot(gs[0])
        self._plot_price_signals(ax1, df_plot, dates, symbol)
        
        # Subplot 2: Moving Averages
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        self._plot_moving_averages(ax2, df_plot, dates)
        
        # Subplot 3: Volume
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        self._plot_volume(ax3, df_plot, dates)
        
        # Subplot 4: Confidence Scores
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        self._plot_confidence(ax4, df_plot, dates)
        
        # Format x-axis for all subplots
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            if ax != ax4:
                plt.setp(ax.get_xticklabels(), visible=False)
        
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        # Add title
        fig.suptitle(
            f'{symbol} Trading Analysis - {model_name} Model',
            fontsize=16,
            fontweight='bold',
            y=0.995
        )
        
        # Save figure
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{symbol}_{model_name.replace(" ", "_")}_{timestamp}.png'
        filepath = self.output_dir / filename
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved: {filepath}")
        
        return filepath
    
    def _plot_price_signals(self, ax, df, dates, symbol):
        """Plot price chart with buy/sell/hold signals."""
        # Plot closing price
        ax.plot(dates, df['close'].values, 
                label='Close Price', color='#2E86AB', linewidth=2, alpha=0.8)
        
        # Overlay predictions as colored markers
        buy_mask = df['prediction'] == 2
        hold_mask = df['prediction'] == 1
        sell_mask = df['prediction'] == 0
        
        # Buy signals (green triangles pointing up)
        if buy_mask.sum() > 0:
            ax.scatter(dates[buy_mask], df.loc[buy_mask, 'close'],
                      marker='^', s=100, color='#06D6A0', 
                      label='BUY Signal', zorder=5, edgecolors='black', linewidths=1)
        
        # Sell signals (red triangles pointing down)
        if sell_mask.sum() > 0:
            ax.scatter(dates[sell_mask], df.loc[sell_mask, 'close'],
                      marker='v', s=100, color='#EF476F', 
                      label='SELL Signal', zorder=5, edgecolors='black', linewidths=1)
        
        # Hold signals (gray circles)
        if hold_mask.sum() > 0:
            ax.scatter(dates[hold_mask], df.loc[hold_mask, 'close'],
                      marker='o', s=50, color='#A9A9A9', 
                      label='HOLD Signal', alpha=0.5, zorder=4)
        
        ax.set_ylabel('Price ($)', fontsize=11, fontweight='bold')
        ax.set_title('Price Action & Trading Signals', fontsize=12, fontweight='bold', pad=10)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Add signal statistics
        buy_count = buy_mask.sum()
        sell_count = sell_mask.sum()
        hold_count = hold_mask.sum()
        stats_text = f'Signals - Buy: {buy_count} | Hold: {hold_count} | Sell: {sell_count}'
        ax.text(0.99, 0.02, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _plot_moving_averages(self, ax, df, dates):
        """Plot moving averages."""
        if 'sma_5' in df.columns:
            ax.plot(dates, df['sma_5'].values, 
                   label='SMA-5', color='#F72585', linewidth=1.5, alpha=0.7)
        
        if 'sma_10' in df.columns:
            ax.plot(dates, df['sma_10'].values, 
                   label='SMA-10', color='#7209B7', linewidth=1.5, alpha=0.7)
        
        if 'sma_20' in df.columns:
            ax.plot(dates, df['sma_20'].values, 
                   label='SMA-20', color='#3A0CA3', linewidth=1.5, alpha=0.7)
        
        # Plot actual close price for reference
        ax.plot(dates, df['close'].values, 
               label='Close', color='#2E86AB', linewidth=1, alpha=0.5)
        
        ax.set_ylabel('Price ($)', fontsize=11, fontweight='bold')
        ax.set_title('Moving Averages', fontsize=12, fontweight='bold', pad=10)
        ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
        ax.grid(True, alpha=0.3)
    
    def _plot_volume(self, ax, df, dates):
        """Plot volume bars colored by price change."""
        # Calculate price change
        price_change = df['close'].diff()
        
        # Color bars by price change
        colors = ['#06D6A0' if x > 0 else '#EF476F' if x < 0 else '#A9A9A9' 
                  for x in price_change]
        
        ax.bar(dates, df['volume'].values, color=colors, alpha=0.6, width=0.8)
        
        ax.set_ylabel('Volume', fontsize=11, fontweight='bold')
        ax.set_title('Trading Volume', fontsize=12, fontweight='bold', pad=10)
        ax.ticklabel_format(style='plain', axis='y')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_confidence(self, ax, df, dates):
        """Plot confidence scores."""
        # Plot confidence as area chart
        ax.fill_between(dates, 0, df['confidence'].values, 
                        alpha=0.3, color='#4361EE', label='Confidence')
        ax.plot(dates, df['confidence'].values, 
               color='#4361EE', linewidth=2, label='Confidence Score')
        
        # Add threshold line
        ax.axhline(y=0.6, color='#EF476F', linestyle='--', 
                  linewidth=1.5, label='Threshold (60%)', alpha=0.7)
        
        ax.set_ylabel('Confidence', fontsize=11, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax.set_title('Prediction Confidence', fontsize=12, fontweight='bold', pad=10)
        ax.set_ylim([0, 1])
        ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add average confidence
        avg_conf = df['confidence'].mean()
        ax.text(0.99, 0.95, f'Avg: {avg_conf:.2%}', transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    def create_summary_report(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        confidence: np.ndarray,
        model_name: str,
        symbol: str,
        test_metrics: dict
    ):
        """
        Create a text-based summary report with statistics.
        
        Args:
            df: DataFrame with data
            predictions: Model predictions
            confidence: Confidence scores
            model_name: Name of the model
            symbol: Stock symbol
            test_metrics: Dictionary of test metrics
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{symbol}_{model_name.replace(" ", "_")}_report_{timestamp}.txt'
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write(f"TRADING ANALYSIS REPORT - {symbol}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            
            # Signal distribution
            f.write("SIGNAL DISTRIBUTION\n")
            f.write("-" * 70 + "\n")
            buy_count = (predictions == 2).sum()
            hold_count = (predictions == 1).sum()
            sell_count = (predictions == 0).sum()
            total = len(predictions)
            
            f.write(f"  BUY signals:  {buy_count:4d} ({buy_count/total*100:5.1f}%)\n")
            f.write(f"  HOLD signals: {hold_count:4d} ({hold_count/total*100:5.1f}%)\n")
            f.write(f"  SELL signals: {sell_count:4d} ({sell_count/total*100:5.1f}%)\n")
            f.write(f"  Total:        {total:4d}\n\n")
            
            # Confidence statistics
            f.write("CONFIDENCE STATISTICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Average:    {confidence.mean():.2%}\n")
            f.write(f"  Median:     {np.median(confidence):.2%}\n")
            f.write(f"  Std Dev:    {confidence.std():.2%}\n")
            f.write(f"  Min:        {confidence.min():.2%}\n")
            f.write(f"  Max:        {confidence.max():.2%}\n")
            f.write(f"  Above 60%:  {(confidence >= 0.6).sum()} ({(confidence >= 0.6).sum()/len(confidence)*100:.1f}%)\n\n")
            
            # Model performance
            f.write("MODEL PERFORMANCE\n")
            f.write("-" * 70 + "\n")
            for key, value in test_metrics.items():
                if isinstance(value, (int, float)):
                    if 'accuracy' in key.lower():
                        f.write(f"  {key.replace('_', ' ').title()}: {value:.2%}\n")
                    else:
                        f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("End of Report\n")
            f.write("=" * 70 + "\n")
        
        print(f"✓ Report saved: {filepath}")
        return filepath


