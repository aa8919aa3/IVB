#!/usr/bin/env python3
"""
Enhanced Josephson Junction Analysis with Differentiated Initialization Strategies
=================================================================================

This script implements truly different initialization strategies for Josephson fitting:
1. Simple Mean Initialization (baseline)
2. RMS-Based Initialization (uses RMS statistics for better amplitude estimation)  
3. Period-Aware Initialization (uses period detection for optimized parameters)

Author: GitHub Copilot
Date: 2025-06-03
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy import signal
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from Fit import JosephsonFitter

class EnhancedJosephsonFitter(JosephsonFitter):
    """Enhanced Josephson fitter with differentiated initialization strategies"""
    
    def __init__(self, initialization_strategy='simple_mean'):
        """
        Initialize with specific strategy
        
        Args:
            initialization_strategy: 'simple_mean', 'rms_based', or 'period_aware'
        """
        super().__init__()
        self.initialization_strategy = initialization_strategy
        self._strategy_params = {}
    
    def set_strategy_parameters(self, **kwargs):
        """Set parameters specific to the initialization strategy"""
        self._strategy_params.update(kwargs)
    
    def estimate_initial_parameters(self, phi_ext, I_s, lomb_scargle_result=None):
        """Enhanced parameter estimation with strategy-specific methods"""
        from lmfit import Parameters
        
        params = Parameters()
        
        if self.initialization_strategy == 'simple_mean':
            return self._simple_mean_initialization(phi_ext, I_s, lomb_scargle_result)
        elif self.initialization_strategy == 'rms_based':
            return self._rms_based_initialization(phi_ext, I_s, lomb_scargle_result)
        elif self.initialization_strategy == 'period_aware':
            return self._period_aware_initialization(phi_ext, I_s, lomb_scargle_result)
        else:
            # Fallback to original method
            return super().estimate_initial_parameters(phi_ext, I_s, lomb_scargle_result)
    
    def _simple_mean_initialization(self, phi_ext, I_s, lomb_scargle_result=None):
        """Strategy 1: Simple mean-based initialization (original approach)"""
        from lmfit import Parameters
        
        params = Parameters()
        
        print(f"🔧 Using Simple Mean Initialization Strategy")
        
        # Use simple statistics
        initial_C = np.mean(I_s)
        initial_I_c = np.std(I_s) * 2  # Conservative amplitude estimate
        
        # Use frequency from Lomb-Scargle if available
        if lomb_scargle_result is not None:
            initial_f = lomb_scargle_result.get('best_frequency', 1.0 / (2 * np.pi))
            initial_phi_0 = lomb_scargle_result.get('phase', 0.0)
        else:
            initial_f = 1.0 / (2 * np.pi)
            initial_phi_0 = 0.0
        
        # Standard estimates
        initial_d = np.mean(phi_ext)
        trend_coeffs = np.polyfit(phi_ext, I_s, 1)
        initial_r = trend_coeffs[0]
        
        print(f"   Initial C: {initial_C:.6e} (simple mean)")
        print(f"   Initial I_c: {initial_I_c:.6e} (2 × std)")
        print(f"   Initial f: {initial_f:.6e}")
        
        # Set parameters
        params.add('I_c', value=initial_I_c, min=0.1 * abs(initial_I_c), max=10 * abs(initial_I_c))
        params.add('f', value=initial_f, min=0.01 * initial_f, max=100 * initial_f)
        params.add('d', value=initial_d, min=initial_d - abs(initial_d), max=initial_d + abs(initial_d))
        params.add('phi_0', value=initial_phi_0, min=-2*np.pi, max=2*np.pi)
        params.add('T', value=0.5, min=0.01, max=0.99)
        params.add('r', value=initial_r, min=-10*abs(initial_r), max=10*abs(initial_r))
        params.add('C', value=initial_C, min=initial_C - 5*np.std(I_s), max=initial_C + 5*np.std(I_s))
        
        self.initial_params = params
        return params
    
    def _rms_based_initialization(self, phi_ext, I_s, lomb_scargle_result=None):
        """Strategy 2: RMS-based initialization for better amplitude estimation"""
        from lmfit import Parameters
        
        params = Parameters()
        
        print(f"🔧 Using RMS-Based Initialization Strategy")
        
        # Use RMS value for more accurate amplitude estimation
        rms_value = self._strategy_params.get('rms', np.sqrt(np.mean(I_s**2)))
        initial_I_c = rms_value * 1.5  # RMS-based amplitude estimate
        
        # For periodic signals, RMS gives better amplitude estimation than std
        # Use envelope detection for more sophisticated baseline
        try:
            analytic_signal = signal.hilbert(I_s)
            envelope = np.abs(analytic_signal)
            initial_C = np.mean(envelope) * 0.7  # Baseline from envelope
        except:
            initial_C = np.mean(I_s)
        
        # Use frequency from Lomb-Scargle if available
        if lomb_scargle_result is not None:
            initial_f = lomb_scargle_result.get('best_frequency', 1.0 / (2 * np.pi))
            initial_phi_0 = lomb_scargle_result.get('phase', 0.0)
        else:
            initial_f = 1.0 / (2 * np.pi)
            initial_phi_0 = 0.0
        
        # Improved phase offset estimation using cross-correlation
        initial_d = self._estimate_phase_offset_correlation(phi_ext, I_s)
        
        # Robust linear trend estimation (less sensitive to outliers)
        trend_coeffs = np.polyfit(phi_ext, I_s, 1)
        initial_r = trend_coeffs[0] * 0.8  # Slightly reduced to account for RMS adjustment
        
        print(f"   Initial C: {initial_C:.6e} (envelope-based)")
        print(f"   Initial I_c: {initial_I_c:.6e} (1.5 × RMS = {rms_value:.6e})")
        print(f"   Initial d: {initial_d:.6e} (correlation-based)")
        print(f"   Initial r: {initial_r:.6e} (robust trend)")
        
        # Set parameters with RMS-adjusted bounds
        params.add('I_c', value=initial_I_c, min=0.2 * abs(initial_I_c), max=8 * abs(initial_I_c))
        params.add('f', value=initial_f, min=0.01 * initial_f, max=100 * initial_f)
        params.add('d', value=initial_d, min=initial_d - 2*abs(initial_d), max=initial_d + 2*abs(initial_d))
        params.add('phi_0', value=initial_phi_0, min=-2*np.pi, max=2*np.pi)
        params.add('T', value=0.6, min=0.01, max=0.99)  # Slightly higher transmission
        params.add('r', value=initial_r, min=-8*abs(initial_r), max=8*abs(initial_r))
        params.add('C', value=initial_C, min=initial_C - 3*np.std(I_s), max=initial_C + 3*np.std(I_s))
        
        self.initial_params = params
        return params
    
    def _period_aware_initialization(self, phi_ext, I_s, lomb_scargle_result=None):
        """Strategy 3: Period-aware initialization using detected periodicity"""
        from lmfit import Parameters
        
        params = Parameters()
        
        print(f"🔧 Using Period-Aware Initialization Strategy")
        
        # Get period information from strategy parameters
        period_samples = self._strategy_params.get('estimated_period_samples', None)
        period_aware_mean = self._strategy_params.get('period_aware_mean', np.mean(I_s))
        n_complete_periods = self._strategy_params.get('n_complete_periods', 0)
        
        if period_samples is not None and n_complete_periods >= 1:
            print(f"   Detected period: {period_samples} samples ({n_complete_periods} complete periods)")
            
            # Use period information for better parameter estimation
            # Frequency based on detected period
            phi_range = np.max(phi_ext) - np.min(phi_ext)
            initial_f = n_complete_periods / phi_range
            
            # Amplitude estimation from period-segmented data
            complete_data = I_s[:n_complete_periods * period_samples]
            periods_2d = complete_data.reshape(n_complete_periods, period_samples)
            
            # Use the variation within periods for amplitude
            period_ranges = np.max(periods_2d, axis=1) - np.min(periods_2d, axis=1)
            initial_I_c = np.mean(period_ranges) * 0.8  # Conservative period-based amplitude
            
            # Use period-aware mean for baseline
            initial_C = period_aware_mean
            
            # Phase offset from period alignment
            initial_phi_0 = self._estimate_phase_from_periods(periods_2d)
            
            print(f"   Period-based f: {initial_f:.6e}")
            print(f"   Period-based I_c: {initial_I_c:.6e}")
            print(f"   Period-aware C: {initial_C:.6e}")
            
        else:
            print(f"   No clear periodicity detected, using enhanced fallback")
            # Fallback to improved estimates
            initial_f = lomb_scargle_result.get('best_frequency', 1.0 / (2 * np.pi)) if lomb_scargle_result else 1.0 / (2 * np.pi)
            initial_I_c = np.std(I_s) * 2.5  # Slightly higher for period-aware
            initial_C = np.median(I_s)  # Use median for robustness
            initial_phi_0 = 0.0
        
        # Optimized d estimation for period-aware approach
        initial_d = self._estimate_optimal_d(phi_ext, I_s, initial_f)
        
        # Linear trend with period consideration
        if period_samples is not None:
            # Detrend each period separately for better linear estimation
            trend_coeffs = self._period_aware_trend_estimation(phi_ext, I_s, period_samples)
            initial_r = trend_coeffs
        else:
            trend_coeffs = np.polyfit(phi_ext, I_s, 1)
            initial_r = trend_coeffs[0]
        
        print(f"   Optimized d: {initial_d:.6e}")
        print(f"   Period-aware r: {initial_r:.6e}")
        
        # Set parameters with period-optimized bounds
        params.add('I_c', value=initial_I_c, min=0.3 * abs(initial_I_c), max=5 * abs(initial_I_c))
        params.add('f', value=initial_f, min=0.1 * initial_f, max=50 * initial_f)
        params.add('d', value=initial_d, min=initial_d - 0.5*abs(initial_d), max=initial_d + 0.5*abs(initial_d))
        params.add('phi_0', value=initial_phi_0, min=-np.pi, max=np.pi)
        params.add('T', value=0.7, min=0.1, max=0.95)  # Higher transmission for period-aware
        params.add('r', value=initial_r, min=-5*abs(initial_r), max=5*abs(initial_r))
        params.add('C', value=initial_C, min=initial_C - 2*np.std(I_s), max=initial_C + 2*np.std(I_s))
        
        self.initial_params = params
        return params
    
    def _estimate_phase_offset_correlation(self, phi_ext, I_s):
        """Estimate phase offset using cross-correlation"""
        try:
            # Create a reference sinusoidal signal
            ref_signal = np.sin(2 * np.pi * (phi_ext - np.mean(phi_ext)))
            
            # Cross-correlate with data
            correlation = np.correlate(I_s - np.mean(I_s), ref_signal, mode='same')
            max_idx = np.argmax(np.abs(correlation))
            
            # Estimate phase offset from correlation peak
            offset = phi_ext[max_idx] - np.mean(phi_ext)
            return np.mean(phi_ext) + offset * 0.5  # Conservative adjustment
        except:
            return np.mean(phi_ext)
    
    def _estimate_phase_from_periods(self, periods_2d):
        """Estimate initial phase from period alignment"""
        try:
            # Find the average phase where the signal starts
            avg_period = np.mean(periods_2d, axis=0)
            
            # Find the phase where signal crosses mean
            mean_val = np.mean(avg_period)
            cross_indices = np.where(np.diff(np.sign(avg_period - mean_val)))[0]
            
            if len(cross_indices) > 0:
                # Use first crossing as phase reference
                return 2 * np.pi * cross_indices[0] / len(avg_period)
            else:
                return 0.0
        except:
            return 0.0
    
    def _estimate_optimal_d(self, phi_ext, I_s, frequency):
        """Estimate optimal d parameter to minimize linear contribution"""
        try:
            # Try different d values and find one that minimizes linear trend
            d_candidates = np.linspace(np.min(phi_ext), np.max(phi_ext), 20)
            linear_trends = []
            
            for d_test in d_candidates:
                # Calculate what the linear contribution would be
                linear_contrib = np.mean(phi_ext - d_test)
                linear_trends.append(abs(linear_contrib))
            
            # Choose d that minimizes linear contribution
            optimal_idx = np.argmin(linear_trends)
            return d_candidates[optimal_idx]
        except:
            return np.mean(phi_ext)
    
    def _period_aware_trend_estimation(self, phi_ext, I_s, period_samples):
        """Estimate linear trend with period awareness"""
        try:
            n_complete_periods = len(I_s) // period_samples
            if n_complete_periods >= 2:
                # Calculate trend between period means
                complete_data = I_s[:n_complete_periods * period_samples]
                periods_2d = complete_data.reshape(n_complete_periods, period_samples)
                period_means = np.mean(periods_2d, axis=1)
                
                # Corresponding phi_ext positions
                period_phi = phi_ext[:n_complete_periods * period_samples].reshape(n_complete_periods, period_samples)
                period_phi_means = np.mean(period_phi, axis=1)
                
                # Linear fit between period means
                trend_coeffs = np.polyfit(period_phi_means, period_means, 1)
                return trend_coeffs[0]
            else:
                # Fallback to standard estimation
                trend_coeffs = np.polyfit(phi_ext, I_s, 1)
                return trend_coeffs[0]
        except:
            trend_coeffs = np.polyfit(phi_ext, I_s, 1)
            return trend_coeffs[0]


class PeriodicSignalAnalyzer:
    """Enhanced analyzer for periodic Josephson signals"""
    
    def __init__(self):
        self.results = {}
    
    def calculate_signal_statistics(self, phi_ext, I_s):
        """Calculate various statistical measures appropriate for periodic signals"""
        stats = {}
        
        # 1. Simple mean (baseline, potentially biased for incomplete periods)
        stats['simple_mean'] = np.mean(I_s)
        
        # 2. RMS value (more appropriate for periodic signals)
        stats['rms'] = np.sqrt(np.mean(I_s**2))
        
        # 3. Try to identify period and calculate period-aware mean
        period_stats = self._estimate_period_aware_stats(phi_ext, I_s)
        stats.update(period_stats)
        
        # 4. Robust statistics
        stats['median'] = np.median(I_s)
        stats['trimmed_mean'] = self._trimmed_mean(I_s, trim_percent=0.1)
        
        # 5. Signal envelope statistics
        envelope_stats = self._calculate_envelope_stats(I_s)
        stats.update(envelope_stats)
        
        return stats
    
    def _estimate_period_aware_stats(self, phi_ext, I_s):
        """Estimate period and calculate statistics based on complete periods"""
        try:
            # Estimate period using autocorrelation
            autocorr = np.correlate(I_s - np.mean(I_s), I_s - np.mean(I_s), mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Find peaks in autocorrelation to estimate period
            peaks, _ = signal.find_peaks(autocorr[1:], height=0.1*np.max(autocorr))
            
            if len(peaks) > 0:
                estimated_period_samples = peaks[0] + 1
                
                # Calculate how many complete periods we have
                n_complete_periods = len(I_s) // estimated_period_samples
                
                if n_complete_periods >= 1:
                    # Take only complete periods for averaging
                    complete_data = I_s[:n_complete_periods * estimated_period_samples]
                    
                    # Reshape into periods and average
                    periods = complete_data.reshape(n_complete_periods, estimated_period_samples)
                    period_means = np.mean(periods, axis=1)
                    
                    return {
                        'period_aware_mean': np.mean(period_means),
                        'estimated_period_samples': estimated_period_samples,
                        'n_complete_periods': n_complete_periods,
                        'period_std': np.std(period_means)
                    }
        except:
            pass
        
        return {
            'period_aware_mean': np.nan,
            'estimated_period_samples': np.nan,
            'n_complete_periods': 0,
            'period_std': np.nan
        }
    
    def _trimmed_mean(self, data, trim_percent=0.1):
        """Calculate trimmed mean by removing extreme values"""
        sorted_data = np.sort(data)
        n = len(sorted_data)
        trim_count = int(n * trim_percent)
        if trim_count > 0:
            return np.mean(sorted_data[trim_count:-trim_count])
        return np.mean(sorted_data)
    
    def _calculate_envelope_stats(self, I_s):
        """Calculate statistics based on signal envelope"""
        try:
            # Calculate envelope using Hilbert transform
            analytic_signal = signal.hilbert(I_s)
            envelope = np.abs(analytic_signal)
            
            return {
                'envelope_mean': np.mean(envelope),
                'envelope_std': np.std(envelope)
            }
        except:
            return {
                'envelope_mean': np.nan,
                'envelope_std': np.nan
            }


def analyze_dataset_with_strategies(dataset_id, analyzer):
    """Analyze a single dataset with all three initialization strategies"""
    
    # Load data
    file_path = f"/Users/albert-mac/Code/GitHub/IVB/Ic/{dataset_id}Ic.csv"
    try:
        data = pd.read_csv(file_path)
        phi_ext = data['y_field'].values
        I_s = data['Ic'].values
    except Exception as e:
        print(f"Error loading dataset {dataset_id}: {e}")
        return None
    
    print(f"\n{'='*70}")
    print(f"Enhanced Analysis: Dataset {dataset_id}")
    print(f"{'='*70}")
    print(f"Data points: {len(I_s)}")
    print(f"Field range: {phi_ext.min():.6f} to {phi_ext.max():.6f}")
    print(f"Current range: {I_s.min():.2e} to {I_s.max():.2e}")
    
    # Calculate enhanced statistics
    stats = analyzer.calculate_signal_statistics(phi_ext, I_s)
    
    print(f"\n📊 Enhanced Signal Statistics:")
    print(f"Simple mean: {stats['simple_mean']:.6e}")
    print(f"RMS value: {stats['rms']:.6e}")
    print(f"Median: {stats['median']:.6e}")
    print(f"Trimmed mean (10%): {stats['trimmed_mean']:.6e}")
    
    if not np.isnan(stats['period_aware_mean']):
        print(f"Period-aware mean: {stats['period_aware_mean']:.6e}")
        print(f"Estimated period: {stats['estimated_period_samples']} samples")
        print(f"Complete periods: {stats['n_complete_periods']}")
        print(f"Period-to-period std: {stats['period_std']:.6e}")
    else:
        print("Period-aware analysis: Could not detect clear periodicity")
    
    if not np.isnan(stats['envelope_mean']):
        print(f"Envelope mean: {stats['envelope_mean']:.6e}")
        print(f"Envelope std: {stats['envelope_std']:.6e}")
    
    # Perform fitting with differentiated initialization strategies
    fitting_results = {}
    
    # Strategy 1: Simple Mean Initialization
    print(f"\n🔧 Strategy 1: Simple Mean Initialization")
    print("-" * 50)
    try:
        fitter1 = EnhancedJosephsonFitter(initialization_strategy='simple_mean')
        params1, result1 = fitter1.fit_josephson_relation(phi_ext, I_s)
        
        if result1 is not None and params1 is not None:
            fitted_curve1 = fitter1.calculate_fitted_curve(phi_ext)
            
            fitting_results['simple_mean'] = {
                'params': params1,
                'result': result1,
                'fitted_curve': fitted_curve1,
                'residual_std': np.std(I_s - fitted_curve1),
                'r_squared': 1 - np.sum((I_s - fitted_curve1)**2) / np.sum((I_s - np.mean(I_s))**2),
                'fitter': fitter1
            }
            print(f"✅ R²: {fitting_results['simple_mean']['r_squared']:.6f}")
            print(f"✅ Residual std: {fitting_results['simple_mean']['residual_std']:.6e}")
        else:
            print(f"❌ Fitting failed: Could not converge")
            fitting_results['simple_mean'] = None
        
    except Exception as e:
        print(f"❌ Fitting failed: {e}")
        fitting_results['simple_mean'] = None
    
    # Strategy 2: RMS-Based Initialization  
    print(f"\n🔧 Strategy 2: RMS-Based Initialization")
    print("-" * 50)
    try:
        fitter2 = EnhancedJosephsonFitter(initialization_strategy='rms_based')
        fitter2.set_strategy_parameters(rms=stats['rms'])
        params2, result2 = fitter2.fit_josephson_relation(phi_ext, I_s)
        
        if result2 is not None and params2 is not None:
            fitted_curve2 = fitter2.calculate_fitted_curve(phi_ext)
            
            fitting_results['rms_based'] = {
                'params': params2,
                'result': result2,
                'fitted_curve': fitted_curve2,
                'residual_std': np.std(I_s - fitted_curve2),
                'r_squared': 1 - np.sum((I_s - fitted_curve2)**2) / np.sum((I_s - np.mean(I_s))**2),
                'fitter': fitter2
            }
            print(f"✅ R²: {fitting_results['rms_based']['r_squared']:.6f}")
            print(f"✅ Residual std: {fitting_results['rms_based']['residual_std']:.6e}")
        else:
            print(f"❌ Fitting failed: Could not converge")
            fitting_results['rms_based'] = None
        
    except Exception as e:
        print(f"❌ Fitting failed: {e}")
        fitting_results['rms_based'] = None
    
    # Strategy 3: Period-Aware Initialization
    print(f"\n🔧 Strategy 3: Period-Aware Initialization")
    print("-" * 50)
    try:
        fitter3 = EnhancedJosephsonFitter(initialization_strategy='period_aware')
        fitter3.set_strategy_parameters(
            estimated_period_samples=stats['estimated_period_samples'],
            period_aware_mean=stats['period_aware_mean'],
            n_complete_periods=stats['n_complete_periods']
        )
        params3, result3 = fitter3.fit_josephson_relation(phi_ext, I_s)
        
        if result3 is not None and params3 is not None:
            fitted_curve3 = fitter3.calculate_fitted_curve(phi_ext)
            
            fitting_results['period_aware'] = {
                'params': params3,
                'result': result3,
                'fitted_curve': fitted_curve3,
                'residual_std': np.std(I_s - fitted_curve3),
                'r_squared': 1 - np.sum((I_s - fitted_curve3)**2) / np.sum((I_s - np.mean(I_s))**2),
                'fitter': fitter3
            }
            print(f"✅ R²: {fitting_results['period_aware']['r_squared']:.6f}")
            print(f"✅ Residual std: {fitting_results['period_aware']['residual_std']:.6e}")
        else:
            print(f"❌ Fitting failed: Could not converge")
            fitting_results['period_aware'] = None
        
    except Exception as e:
        print(f"❌ Fitting failed: {e}")
        fitting_results['period_aware'] = None
    
    # Create enhanced visualization
    create_enhanced_analysis_plot(dataset_id, phi_ext, I_s, stats, fitting_results)
    
    return {
        'dataset_id': dataset_id,
        'stats': stats,
        'fitting_results': fitting_results,
        'data': {'phi_ext': phi_ext, 'I_s': I_s}
    }


def create_enhanced_analysis_plot(dataset_id, phi_ext, I_s, stats, fitting_results):
    """Create enhanced comprehensive analysis visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Enhanced Josephson Analysis: Dataset {dataset_id}', fontsize=16, fontweight='bold')
    
    # Plot 1: Data and statistical measures
    ax1 = axes[0, 0]
    ax1.plot(phi_ext, I_s, 'ko', alpha=0.6, markersize=3, label='Data')
    ax1.axhline(y=stats['simple_mean'], color='blue', linestyle='-', alpha=0.7, label=f"Simple Mean: {stats['simple_mean']:.2e}")
    ax1.axhline(y=stats['rms'], color='green', linestyle='--', alpha=0.7, label=f"RMS: {stats['rms']:.2e}")
    ax1.axhline(y=stats['median'], color='orange', linestyle=':', alpha=0.7, label=f"Median: {stats['median']:.2e}")
    if not np.isnan(stats['period_aware_mean']):
        ax1.axhline(y=stats['period_aware_mean'], color='red', linestyle='-.', alpha=0.7, 
                   label=f"Period-aware: {stats['period_aware_mean']:.2e}")
    ax1.set_xlabel('Magnetic Field')
    ax1.set_ylabel('Critical Current (A)')
    ax1.set_title('Data and Statistical Measures')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Fitting strategy comparison
    ax2 = axes[0, 1]
    ax2.plot(phi_ext, I_s, 'ko', alpha=0.4, markersize=2, label='Data')
    
    colors = ['blue', 'green', 'red']
    strategies = ['simple_mean', 'rms_based', 'period_aware']
    strategy_names = ['Simple Mean', 'RMS-Based', 'Period-Aware']
    
    for i, (strategy, name) in enumerate(zip(strategies, strategy_names)):
        if strategy in fitting_results and fitting_results[strategy] is not None:
            fitted_curve = fitting_results[strategy]['fitted_curve']
            r_squared = fitting_results[strategy]['r_squared']
            ax2.plot(phi_ext, fitted_curve, color=colors[i], linestyle='-', 
                    linewidth=2, alpha=0.8, label=f'{name} (R²={r_squared:.4f})')
    
    ax2.set_xlabel('Magnetic Field')
    ax2.set_ylabel('Critical Current (A)')
    ax2.set_title('Initialization Strategy Comparison')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Residuals comparison
    ax3 = axes[0, 2]
    for i, (strategy, name) in enumerate(zip(strategies, strategy_names)):
        if strategy in fitting_results and fitting_results[strategy] is not None:
            residuals = I_s - fitting_results[strategy]['fitted_curve']
            ax3.plot(phi_ext, residuals, color=colors[i], alpha=0.7, 
                    linewidth=1.5, label=f'{name} Residuals')
    
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_xlabel('Magnetic Field')
    ax3.set_ylabel('Residuals (A)')
    ax3.set_title('Fitting Residuals Comparison')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Parameter comparison
    ax4 = axes[1, 0]
    
    # Collect parameters for comparison
    param_names = ['I_c', 'f', 'C', 'r']
    param_data = {name: [] for name in param_names}
    strategy_labels = []
    
    for strategy, name in zip(strategies, strategy_names):
        if strategy in fitting_results and fitting_results[strategy] is not None:
            params = fitting_results[strategy]['params']
            strategy_labels.append(name)
            for param_name in param_names:
                if param_name in params:
                    param_data[param_name].append(params[param_name]['value'])
                else:
                    param_data[param_name].append(np.nan)
    
    if strategy_labels:
        x = np.arange(len(strategy_labels))
        width = 0.2
        
        for i, param_name in enumerate(param_names):
            values = param_data[param_name]
            if not all(np.isnan(values)):
                ax4.bar(x + i*width, values, width, label=param_name, alpha=0.7)
        
        ax4.set_xlabel('Strategy')
        ax4.set_ylabel('Parameter Value')
        ax4.set_title('Parameter Values by Strategy')
        ax4.set_xticks(x + width * 1.5)
        ax4.set_xticklabels(strategy_labels, rotation=45)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: R² comparison
    ax5 = axes[1, 1]
    r2_values = []
    r2_labels = []
    
    for strategy, name in zip(strategies, strategy_names):
        if strategy in fitting_results and fitting_results[strategy] is not None:
            r2_values.append(fitting_results[strategy]['r_squared'])
            r2_labels.append(name)
    
    if r2_values:
        bars = ax5.bar(r2_labels, r2_values, color=colors[:len(r2_values)], alpha=0.7)
        ax5.set_ylabel('R² Value')
        ax5.set_title('Fitting Quality Comparison')
        ax5.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, r2_values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 6: Summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Create summary text
    summary_text = f"""Dataset {dataset_id} Enhanced Analysis

📊 Data Statistics:
• Points: {len(I_s)}
• Field range: {phi_ext.min():.6f} to {phi_ext.max():.6f}
• Current range: {I_s.min():.2e} to {I_s.max():.2e}

📈 Signal Statistics:
• Simple mean: {stats['simple_mean']:.6e}
• RMS: {stats['rms']:.6e}
• Median: {stats['median']:.6e}
• Trimmed mean: {stats['trimmed_mean']:.6e}
"""
    
    if not np.isnan(stats['period_aware_mean']):
        summary_text += f"""
🔄 Periodicity:
• Period-aware mean: {stats['period_aware_mean']:.6e}
• Estimated period: {stats['estimated_period_samples']} samples
• Complete periods: {stats['n_complete_periods']}
• Period std: {stats['period_std']:.6e}
"""
    else:
        summary_text += "\n🔄 Periodicity: No clear period detected"
    
    # Add fitting results summary
    summary_text += "\n🎯 Fitting Results:"
    for strategy, name in zip(strategies, strategy_names):
        if strategy in fitting_results and fitting_results[strategy] is not None:
            r2 = fitting_results[strategy]['r_squared']
            res_std = fitting_results[strategy]['residual_std']
            summary_text += f"\n• {name}: R²={r2:.4f}, σ={res_std:.2e}"
        else:
            summary_text += f"\n• {name}: Failed"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    filename = f'dataset_{dataset_id}_enhanced_strategies_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Enhanced analysis plot saved: {filename}")
    
    plt.show()


def generate_enhanced_summary_report(all_results):
    """Generate enhanced comprehensive summary report"""
    
    if not all_results:
        print("❌ No results to summarize")
        return
    
    print(f"\n{'='*80}")
    print("ENHANCED JOSEPHSON ANALYSIS SUMMARY REPORT")
    print(f"{'='*80}")
    
    # Create summary DataFrame
    summary_data = []
    
    for result in all_results:
        dataset_id = result['dataset_id']
        stats = result['stats']
        fitting_results = result['fitting_results']
        
        row = {
            'dataset_id': dataset_id,
            'n_points': len(result['data']['I_s']),
            'simple_mean': stats['simple_mean'],
            'rms': stats['rms'],
            'median': stats['median'],
            'trimmed_mean': stats['trimmed_mean'],
            'period_aware_mean': stats['period_aware_mean'],
            'estimated_period': stats['estimated_period_samples'],
            'n_complete_periods': stats['n_complete_periods'],
            'envelope_mean': stats['envelope_mean']
        }
        
        # Add fitting results for each strategy
        strategies = ['simple_mean', 'rms_based', 'period_aware']
        for strategy in strategies:
            if strategy in fitting_results and fitting_results[strategy]:
                row[f'{strategy}_r2'] = fitting_results[strategy]['r_squared']
                row[f'{strategy}_residual_std'] = fitting_results[strategy]['residual_std']
            else:
                row[f'{strategy}_r2'] = np.nan
                row[f'{strategy}_residual_std'] = np.nan
        
        # Calculate relative differences between averaging methods
        simple_mean = stats['simple_mean']
        if simple_mean != 0:
            row['rms_vs_simple_diff_pct'] = 100 * (stats['rms'] - simple_mean) / abs(simple_mean)
            row['median_vs_simple_diff_pct'] = 100 * (stats['median'] - simple_mean) / abs(simple_mean)
            if not np.isnan(stats['period_aware_mean']):
                row['period_vs_simple_diff_pct'] = 100 * (stats['period_aware_mean'] - simple_mean) / abs(simple_mean)
            else:
                row['period_vs_simple_diff_pct'] = np.nan
        else:
            row['rms_vs_simple_diff_pct'] = np.nan
            row['median_vs_simple_diff_pct'] = np.nan
            row['period_vs_simple_diff_pct'] = np.nan
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    csv_filename = 'enhanced_strategies_analysis_summary.csv'
    summary_df.to_csv(csv_filename, index=False)
    print(f"📄 Enhanced summary saved to: {csv_filename}")
    
    # Print key statistics
    print(f"\n📊 Enhanced Analysis Statistics:")
    print(f"Datasets processed: {len(summary_df)}")
    print(f"Average data points per dataset: {summary_df['n_points'].mean():.1f}")
    
    print(f"\n📈 Statistical Measure Differences (vs Simple Mean):")
    print(f"RMS difference: {summary_df['rms_vs_simple_diff_pct'].mean():.2f}% ± {summary_df['rms_vs_simple_diff_pct'].std():.2f}%")
    print(f"Median difference: {summary_df['median_vs_simple_diff_pct'].mean():.2f}% ± {summary_df['median_vs_simple_diff_pct'].std():.2f}%")
    
    period_diffs = summary_df['period_vs_simple_diff_pct'].dropna()
    if len(period_diffs) > 0:
        print(f"Period-aware difference: {period_diffs.mean():.2f}% ± {period_diffs.std():.2f}%")
        print(f"Datasets with detectable periods: {len(period_diffs)}/{len(summary_df)}")
    
    print(f"\n🎯 Fitting Performance Comparison:")
    strategies = ['simple_mean', 'rms_based', 'period_aware']
    strategy_names = ['Simple Mean', 'RMS-Based', 'Period-Aware']
    
    for strategy, name in zip(strategies, strategy_names):
        r2_col = f'{strategy}_r2'
        if r2_col in summary_df.columns:
            valid_r2 = summary_df[r2_col].dropna()
            if len(valid_r2) > 0:
                print(f"{name:15}: R² = {valid_r2.mean():.6f} ± {valid_r2.std():.6f} (n={len(valid_r2)})")
            else:
                print(f"{name:15}: No successful fits")
    
    # Find best performing strategy for each dataset
    print(f"\n🏆 Best Strategy by Dataset:")
    best_strategies = []
    for _, row in summary_df.iterrows():
        r2_values = {}
        for strategy in strategies:
            r2_col = f'{strategy}_r2'
            if not np.isnan(row[r2_col]):
                r2_values[strategy] = row[r2_col]
        
        if r2_values:
            best_strategy = max(r2_values, key=r2_values.get)
            best_r2 = r2_values[best_strategy]
            best_strategies.append(best_strategy)
            strategy_name = dict(zip(strategies, strategy_names))[best_strategy]
            print(f"Dataset {int(row['dataset_id']):3d}: {strategy_name:15} (R² = {best_r2:.6f})")
        else:
            print(f"Dataset {int(row['dataset_id']):3d}: No successful fits")
    
    if best_strategies:
        from collections import Counter
        strategy_counts = Counter(best_strategies)
        print(f"\n📋 Strategy Performance Summary:")
        for strategy in strategies:
            count = strategy_counts.get(strategy, 0)
            percentage = 100 * count / len(best_strategies)
            strategy_name = dict(zip(strategies, strategy_names))[strategy]
            print(f"{strategy_name:15}: {count:2d}/{len(best_strategies)} datasets ({percentage:5.1f}%)")


def main():
    """Main function for enhanced analysis"""
    dataset_ids = [317, 346, 435, 338, 337, 439, 336, 352, 335, 341]
    
    print("🚀 ENHANCED JOSEPHSON JUNCTION ANALYSIS")
    print("Differentiated Initialization Strategies Implementation")
    print("="*80)
    print("Strategies:")
    print("1. Simple Mean Initialization (baseline)")
    print("2. RMS-Based Initialization (improved amplitude estimation)")
    print("3. Period-Aware Initialization (optimized for periodic signals)")
    print("="*80)
    
    analyzer = PeriodicSignalAnalyzer()
    all_results = []
    
    for dataset_id in dataset_ids:
        result = analyze_dataset_with_strategies(dataset_id, analyzer)
        if result:
            all_results.append(result)
    
    # Generate enhanced summary report
    generate_enhanced_summary_report(all_results)
    
    print(f"\n{'='*80}")
    print("🎉 ENHANCED ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"✅ Processed {len(all_results)} datasets successfully")
    print(f"📊 Individual enhanced plots: dataset_XXX_enhanced_strategies_analysis.png")
    print(f"📄 Summary report: enhanced_strategies_analysis_summary.csv")
    print(f"🔬 Three differentiated initialization strategies implemented and compared")


if __name__ == "__main__":
    main()
