#!/usr/bin/env python3
"""
Lomb-Scargle Periodogram Analysis
=================================

This module implements Lomb-Scargle periodogram analysis for time series data,
specifically applied to Tesla stock price data for periodicity detection.

Author: BE21B032
Course: Mathematical Foundations of Data Science (MFDS)
Institution: IIT Madras
Faculty: Dr. Arun Tangirala
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class LombScarglePeriodogram:
    """
    Implementation of Lomb-Scargle periodogram for unevenly sampled time series.
    
    The Lomb-Scargle periodogram is particularly useful for detecting periodic
    signals in unevenly sampled data, such as financial time series.
    """
    
    def __init__(self, oversampling_factor=4):
        """
        Initialize the Lomb-Scargle periodogram analyzer.
        
        Parameters:
        -----------
        oversampling_factor : int
            Factor by which to oversample the frequency grid
        """
        self.oversampling_factor = oversampling_factor
        self.frequencies = None
        self.power = None
        self.periods = None
        
    def compute_periodogram(self, times, values, min_freq=None, max_freq=None, n_freq=1000):
        """
        Compute the Lomb-Scargle periodogram.
        
        Parameters:
        -----------
        times : np.ndarray
            Time points (can be unevenly spaced)
        values : np.ndarray
            Corresponding values at each time point
        min_freq : float, optional
            Minimum frequency to consider
        max_freq : float, optional
            Maximum frequency to consider
        n_freq : int
            Number of frequency points to compute
            
        Returns:
        --------
        tuple : (frequencies, power, periods)
        """
        # Remove any NaN values
        valid_mask = ~(np.isnan(times) | np.isnan(values))
        times = times[valid_mask]
        values = values[valid_mask]
        
        # Center the data
        values = values - np.mean(values)
        
        # Set frequency range if not provided
        if min_freq is None:
            min_freq = 1.0 / (times[-1] - times[0])
        if max_freq is None:
            max_freq = 0.5 / np.min(np.diff(times))
        
        # Create frequency grid
        self.frequencies = np.logspace(np.log10(min_freq), np.log10(max_freq), n_freq)
        
        # Compute periodogram using scipy
        self.power = signal.lombscargle(times, values, self.frequencies)
        
        # Convert frequencies to periods
        self.periods = 1.0 / self.frequencies
        
        return self.frequencies, self.power, self.periods
    
    def find_peaks(self, n_peaks=5, min_distance=None):
        """
        Find the most significant peaks in the periodogram.
        
        Parameters:
        -----------
        n_peaks : int
            Number of peaks to find
        min_distance : int, optional
            Minimum distance between peaks in frequency indices
            
        Returns:
        --------
        tuple : (peak_frequencies, peak_powers, peak_periods)
        """
        if self.power is None:
            raise ValueError("Must compute periodogram first")
        
        # Find peaks
        if min_distance is None:
            min_distance = len(self.frequencies) // (n_peaks * 4)
        
        peak_indices = signal.find_peaks(self.power, distance=min_distance)[0]
        
        # Sort by power and take top n_peaks
        sorted_indices = peak_indices[np.argsort(self.power[peak_indices])[::-1]]
        top_indices = sorted_indices[:n_peaks]
        
        peak_frequencies = self.frequencies[top_indices]
        peak_powers = self.power[top_indices]
        peak_periods = self.periods[top_indices]
        
        return peak_frequencies, peak_powers, peak_periods
    
    def plot_periodogram(self, highlight_peaks=True, n_peaks=5):
        """
        Plot the Lomb-Scargle periodogram.
        
        Parameters:
        -----------
        highlight_peaks : bool
            Whether to highlight significant peaks
        n_peaks : int
            Number of peaks to highlight
        """
        if self.power is None:
            raise ValueError("Must compute periodogram first")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot power vs frequency
        ax1.semilogx(self.frequencies, self.power)
        ax1.set_xlabel('Frequency')
        ax1.set_ylabel('Power')
        ax1.set_title('Lomb-Scargle Periodogram (Power vs Frequency)')
        ax1.grid(True, alpha=0.3)
        
        # Plot power vs period
        ax2.semilogx(self.periods, self.power)
        ax2.set_xlabel('Period')
        ax2.set_ylabel('Power')
        ax2.set_title('Lomb-Scargle Periodogram (Power vs Period)')
        ax2.grid(True, alpha=0.3)
        
        # Highlight peaks if requested
        if highlight_peaks:
            try:
                peak_freqs, peak_powers, peak_periods = self.find_peaks(n_peaks)
                
                # Mark peaks on frequency plot
                ax1.plot(peak_freqs, peak_powers, 'ro', markersize=8, label=f'Top {n_peaks} Peaks')
                ax1.legend()
                
                # Mark peaks on period plot
                ax2.plot(peak_periods, peak_powers, 'ro', markersize=8, label=f'Top {n_peaks} Peaks')
                ax2.legend()
                
                # Annotate peaks
                for i, (freq, power, period) in enumerate(zip(peak_freqs, peak_powers, peak_periods)):
                    ax1.annotate(f'P{i+1}', (freq, power), xytext=(10, 10), 
                                textcoords='offset points', fontsize=10)
                    ax2.annotate(f'P{i+1}', (period, power), xytext=(10, 10), 
                                textcoords='offset points', fontsize=10)
                    
            except Exception as e:
                print(f"Could not highlight peaks: {e}")
        
        plt.tight_layout()
        plt.show()
        
        return fig


def load_tesla_data():
    """
    Load Tesla stock price data.
    
    Returns:
    --------
    tuple : (dates, prices, returns)
    """
    try:
        # Try to load the actual Tesla data
        data = pd.read_csv('Tesla Stock Price.csv')
        
        # Convert date column
        if 'Date' in data.columns:
            dates = pd.to_datetime(data['Date'])
        elif 'date' in data.columns:
            dates = pd.to_datetime(data['date'])
        else:
            # Assume first column is date
            dates = pd.to_datetime(data.iloc[:, 0])
        
        # Find price column
        price_cols = [col for col in data.columns if 'price' in col.lower() or 'close' in col.lower()]
        if price_cols:
            prices = data[price_cols[0]].values
        else:
            # Assume second column is price
            prices = data.iloc[:, 1].values
        
        # Convert to numeric and handle missing values
        prices = pd.to_numeric(prices, errors='coerce')
        valid_mask = ~(np.isnan(prices) | np.isnan(dates))
        dates = dates[valid_mask]
        prices = prices[valid_mask]
        
        # Convert dates to numeric (days since start)
        start_date = dates.min()
        times = (dates - start_date).dt.days.values
        
        # Calculate returns
        returns = np.diff(np.log(prices))
        
        return times, prices, returns
        
    except Exception as e:
        print(f"Could not load Tesla data: {e}")
        print("Generating synthetic periodic data instead...")
        
        # Generate synthetic data with known periodicity
        np.random.seed(42)
        n_points = 1000
        
        # Time vector (unevenly spaced to simulate real data)
        times = np.sort(np.random.uniform(0, 1000, n_points))
        
        # Generate periodic signal with noise
        period1 = 50  # 50-day cycle
        period2 = 200  # 200-day cycle
        
        signal1 = np.sin(2 * np.pi * times / period1)
        signal2 = 0.5 * np.sin(2 * np.pi * times / period2)
        noise = 0.1 * np.random.randn(n_points)
        
        prices = 100 + 10 * signal1 + 5 * signal2 + noise
        returns = np.diff(np.log(prices))
        
        return times, prices, returns


def evaluate_periodicity(times, values, detected_periods, true_periods=None):
    """
    Evaluate the quality of periodicity detection.
    
    Parameters:
    -----------
    times : np.ndarray
        Time points
    values : np.ndarray
        Values at each time point
    detected_periods : np.ndarray
        Detected periods
    true_periods : np.ndarray, optional
        True periods (if known)
        
    Returns:
    --------
    dict : Evaluation metrics
    """
    results = {}
    
    # Reconstruct signal using detected periods
    reconstructed = np.zeros_like(values)
    for period in detected_periods:
        if period > 0:
            frequency = 1.0 / period
            # Simple harmonic reconstruction
            reconstructed += np.sin(2 * np.pi * frequency * times)
    
    # Normalize
    if np.std(reconstructed) > 0:
        reconstructed = reconstructed * (np.std(values) / np.std(reconstructed))
    
    # Add mean
    reconstructed = reconstructed + np.mean(values)
    
    # Calculate metrics
    mse = mean_squared_error(values, reconstructed)
    mape = np.mean(np.abs((values - reconstructed) / values)) * 100
    
    results['MSE'] = mse
    results['MAPE'] = mape
    results['Detected_Periods'] = detected_periods
    
    # Compare with true periods if available
    if true_periods is not None:
        period_errors = []
        for detected in detected_periods:
            errors = [abs(detected - true) / true for true in true_periods]
            period_errors.append(min(errors))
        
        results['Period_Accuracy'] = 1 - np.mean(period_errors)
    
    return results, reconstructed


def main():
    """
    Main function demonstrating Lomb-Scargle periodogram analysis.
    """
    print("=" * 70)
    print("Lomb-Scargle Periodogram Analysis")
    print("Mathematical Foundations of Data Science (MFDS)")
    print("IIT Madras - Dr. Arun Tangirala")
    print("=" * 70)
    
    # Load data
    print("\nLoading Tesla stock price data...")
    times, prices, returns = load_tesla_data()
    
    print(f"Data loaded: {len(times)} data points")
    print(f"Time range: {times[0]:.1f} to {times[-1]:.1f} days")
    print(f"Price range: ${prices.min():.2f} to ${prices.max():.2f}")
    
    # Analyze price data
    print("\n" + "=" * 70)
    print("Analyzing Price Data")
    print("=" * 70)
    
    ls_price = LombScarglePeriodogram()
    freq_price, power_price, periods_price = ls_price.compute_periodogram(times, prices)
    
    print("Price periodogram computed successfully!")
    
    # Find peaks in price data
    peak_freqs_price, peak_powers_price, peak_periods_price = ls_price.find_peaks(n_peaks=5)
    
    print("\nTop 5 detected periods in price data:")
    for i, (freq, power, period) in enumerate(zip(peak_freqs_price, peak_powers_price, peak_periods_price)):
        print(f"Peak {i+1}: Period = {period:.1f} days, Frequency = {freq:.6f} 1/day, Power = {power:.4f}")
    
    # Analyze returns data
    print("\n" + "=" * 70)
    print("Analyzing Returns Data")
    print("=" * 70)
    
    ls_returns = LombScarglePeriodogram()
    freq_returns, power_returns, periods_returns = ls_returns.compute_periodogram(times[:-1], returns)
    
    print("Returns periodogram computed successfully!")
    
    # Find peaks in returns data
    peak_freqs_returns, peak_powers_returns, peak_periods_returns = ls_returns.find_peaks(n_peaks=5)
    
    print("\nTop 5 detected periods in returns data:")
    for i, (freq, power, period) in enumerate(zip(peak_freqs_returns, peak_powers_returns, peak_periods_returns)):
        print(f"Peak {i+1}: Period = {period:.1f} days, Frequency = {freq:.6f} 1/day, Power = {power:.4f}")
    
    # Evaluate periodicity detection
    print("\n" + "=" * 70)
    print("Periodicity Detection Evaluation")
    print("=" * 70)
    
    # For synthetic data, we know the true periods
    if len(times) == 1000:  # Synthetic data
        true_periods = np.array([50, 200])
        print("Using synthetic data with known periods: 50 and 200 days")
        
        # Evaluate price analysis
        price_results, price_reconstructed = evaluate_periodicity(
            times, prices, peak_periods_price, true_periods
        )
        
        # Evaluate returns analysis
        returns_results, returns_reconstructed = evaluate_periodicity(
            times[:-1], returns, peak_periods_returns, true_periods
        )
        
        print(f"\nPrice Analysis - MSE: {price_results['MSE']:.4f}, MAPE: {price_results['MAPE']:.2f}%")
        print(f"Returns Analysis - MSE: {returns_results['MSE']:.4f}, MAPE: {returns_results['MAPE']:.2f}%")
        
        # Plot original vs reconstructed
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        ax1.plot(times, prices, 'b-', alpha=0.7, label='Original Prices')
        ax1.plot(times, price_reconstructed, 'r--', label='Reconstructed (Periodic)')
        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('Price ($)')
        ax1.set_title('Tesla Stock Prices: Original vs Reconstructed')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(times[:-1], returns, 'b-', alpha=0.7, label='Original Returns')
        ax2.plot(times[:-1], returns_reconstructed, 'r--', label='Reconstructed (Periodic)')
        ax2.set_xlabel('Time (days)')
        ax2.set_ylabel('Log Returns')
        ax2.set_title('Tesla Stock Returns: Original vs Reconstructed')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Plot periodograms
    print("\nPlotting periodograms...")
    
    # Price periodogram
    ls_price.plot_periodogram(highlight_peaks=True, n_peaks=5)
    
    # Returns periodogram
    ls_returns.plot_periodogram(highlight_peaks=True, n_peaks=5)
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    
    return ls_price, ls_returns


if __name__ == "__main__":
    ls_price, ls_returns = main()
