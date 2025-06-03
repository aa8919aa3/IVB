#!/usr/bin/env python3
"""
Advanced Analysis of 10 Specific Josephson Junction Datasets
Addresses the periodic signal averaging issue raised by the user.

Dataset IDs: 317, 346, 435, 338, 337, 439, 336, 352, 335, 341

This script implements improved methods for handling periodic signals:
1. Simple mean (baseline)
2. RMS averaging 
3. Period-aware averaging
4. Robust statistical measures
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

def analyze_dataset(dataset_id, analyzer):
    """Analyze a single dataset with enhanced periodic signal handling"""
    
    # Load data
    file_path = f"/Users/albert-mac/Code/GitHub/IVB/Ic/{dataset_id}Ic.csv"
    try:
        data = pd.read_csv(file_path)
        phi_ext = data['y_field'].values
        I_s = data['Ic'].values
    except Exception as e:
        print(f"Error loading dataset {dataset_id}: {e}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Analyzing Dataset {dataset_id}")
    print(f"{'='*60}")
    print(f"Data points: {len(I_s)}")
    print(f"Field range: {phi_ext.min():.6f} to {phi_ext.max():.6f}")
    print(f"Current range: {I_s.min():.2e} to {I_s.max():.2e}")
    
    # Calculate enhanced statistics
    stats = analyzer.calculate_signal_statistics(phi_ext, I_s)
    
    print(f"\nSignal Statistics:")
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
    
    # Perform Josephson fitting with different initialization strategies
    fitting_results = {}
    
    # Strategy 1: Original simple mean initialization
    print(f"\nFitting Strategy 1: Simple Mean Initialization")
    try:
        fitter1 = JosephsonFitter()
        params1, result1 = fitter1.fit_josephson_relation(phi_ext, I_s)
        fitted_curve1 = fitter1.josephson_relation(phi_ext, **{p: params1[p].value for p in params1})
        
        fitting_results['simple_mean'] = {
            'params': params1,
            'result': result1,
            'fitted_curve': fitted_curve1,
            'residual_std': np.std(I_s - fitted_curve1),
            'r_squared': 1 - np.sum((I_s - fitted_curve1)**2) / np.sum((I_s - np.mean(I_s))**2)
        }
        print(f"  R²: {fitting_results['simple_mean']['r_squared']:.6f}")
        print(f"  Residual std: {fitting_results['simple_mean']['residual_std']:.6e}")
        
    except Exception as e:
        print(f"  Fitting failed: {e}")
        fitting_results['simple_mean'] = None
    
    # Strategy 2: RMS-based initialization
    print(f"\nFitting Strategy 2: RMS-Based Initialization")
    try:
        fitter2 = JosephsonFitter()
        # Modify the fitter to use RMS for C initialization
        fitter2._use_rms_init = True
        fitter2._rms_value = stats['rms']
        params2, result2 = fitter2.fit_josephson_relation(phi_ext, I_s)
        fitted_curve2 = fitter2.josephson_relation(phi_ext, **{p: params2[p].value for p in params2})
        
        fitting_results['rms_based'] = {
            'params': params2,
            'result': result2,
            'fitted_curve': fitted_curve2,
            'residual_std': np.std(I_s - fitted_curve2),
            'r_squared': 1 - np.sum((I_s - fitted_curve2)**2) / np.sum((I_s - np.mean(I_s))**2)
        }
        print(f"  R²: {fitting_results['rms_based']['r_squared']:.6f}")
        print(f"  Residual std: {fitting_results['rms_based']['residual_std']:.6e}")
        
    except Exception as e:
        print(f"  Fitting failed: {e}")
        fitting_results['rms_based'] = None
    
    # Strategy 3: Period-aware initialization (if period detected)
    if not np.isnan(stats['period_aware_mean']):
        print(f"\nFitting Strategy 3: Period-Aware Initialization")
        try:
            fitter3 = JosephsonFitter()
            fitter3._use_period_aware_init = True
            fitter3._period_aware_mean = stats['period_aware_mean']
            params3, result3 = fitter3.fit_josephson_relation(phi_ext, I_s)
            fitted_curve3 = fitter3.josephson_relation(phi_ext, **{p: params3[p].value for p in params3})
            
            fitting_results['period_aware'] = {
                'params': params3,
                'result': result3,
                'fitted_curve': fitted_curve3,
                'residual_std': np.std(I_s - fitted_curve3),
                'r_squared': 1 - np.sum((I_s - fitted_curve3)**2) / np.sum((I_s - np.mean(I_s))**2)
            }
            print(f"  R²: {fitting_results['period_aware']['r_squared']:.6f}")
            print(f"  Residual std: {fitting_results['period_aware']['residual_std']:.6e}")
            
        except Exception as e:
            print(f"  Fitting failed: {e}")
            fitting_results['period_aware'] = None
    
    # Create visualization
    create_analysis_plot(dataset_id, phi_ext, I_s, stats, fitting_results)
    
    return {
        'dataset_id': dataset_id,
        'stats': stats,
        'fitting_results': fitting_results,
        'data': {'phi_ext': phi_ext, 'I_s': I_s}
    }

def create_analysis_plot(dataset_id, phi_ext, I_s, stats, fitting_results):
    """Create comprehensive analysis visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Dataset {dataset_id} - Enhanced Josephson Analysis', fontsize=16)
    
    # Plot 1: Original data with different statistical measures
    ax1 = axes[0, 0]
    ax1.plot(phi_ext, I_s, 'b-', linewidth=1, alpha=0.7, label='Original Data')
    ax1.axhline(y=stats['simple_mean'], color='r', linestyle='--', label=f'Simple Mean: {stats["simple_mean"]:.2e}')
    ax1.axhline(y=stats['rms'], color='g', linestyle='--', label=f'RMS: {stats["rms"]:.2e}')
    ax1.axhline(y=stats['median'], color='orange', linestyle='--', label=f'Median: {stats["median"]:.2e}')
    if not np.isnan(stats['period_aware_mean']):
        ax1.axhline(y=stats['period_aware_mean'], color='purple', linestyle='--', 
                   label=f'Period-Aware: {stats["period_aware_mean"]:.2e}')
    
    ax1.set_xlabel('Magnetic Field')
    ax1.set_ylabel('Critical Current (A)')
    ax1.set_title('Data with Statistical Measures')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Fitting comparison
    ax2 = axes[0, 1]
    ax2.plot(phi_ext, I_s, 'b-', linewidth=2, alpha=0.7, label='Original Data')
    
    colors = ['red', 'green', 'purple']
    strategies = ['simple_mean', 'rms_based', 'period_aware']
    strategy_names = ['Simple Mean Init', 'RMS Init', 'Period-Aware Init']
    
    for i, (strategy, name) in enumerate(zip(strategies, strategy_names)):
        if strategy in fitting_results and fitting_results[strategy] is not None:
            fitted_curve = fitting_results[strategy]['fitted_curve']
            r_squared = fitting_results[strategy]['r_squared']
            ax2.plot(phi_ext, fitted_curve, color=colors[i], linestyle='--', 
                    linewidth=2, label=f'{name} (R²={r_squared:.4f})')
    
    ax2.set_xlabel('Magnetic Field')
    ax2.set_ylabel('Critical Current (A)')
    ax2.set_title('Fitting Strategy Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Residuals analysis
    ax3 = axes[1, 0]
    if 'simple_mean' in fitting_results and fitting_results['simple_mean'] is not None:
        residuals = I_s - fitting_results['simple_mean']['fitted_curve']
        ax3.plot(phi_ext, residuals, 'r-', alpha=0.7, label='Simple Mean Residuals')
    
    if 'rms_based' in fitting_results and fitting_results['rms_based'] is not None:
        residuals = I_s - fitting_results['rms_based']['fitted_curve']
        ax3.plot(phi_ext, residuals, 'g-', alpha=0.7, label='RMS Residuals')
    
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_xlabel('Magnetic Field')
    ax3.set_ylabel('Residuals (A)')
    ax3.set_title('Fitting Residuals')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistics summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create text summary
    summary_text = f"""
Dataset {dataset_id} Analysis Summary

Data Statistics:
• Data points: {len(I_s)}
• Field range: {phi_ext.min():.6f} to {phi_ext.max():.6f}
• Current range: {I_s.min():.2e} to {I_s.max():.2e}

Statistical Measures:
• Simple mean: {stats['simple_mean']:.6e}
• RMS value: {stats['rms']:.6e}
• Median: {stats['median']:.6e}
• Trimmed mean: {stats['trimmed_mean']:.6e}
"""
    
    if not np.isnan(stats['period_aware_mean']):
        summary_text += f"""
Periodicity Analysis:
• Period-aware mean: {stats['period_aware_mean']:.6e}
• Est. period: {stats['estimated_period_samples']} samples
• Complete periods: {stats['n_complete_periods']}
"""
    else:
        summary_text += "\nPeriodicity: No clear period detected"
    
    # Add fitting results
    if 'simple_mean' in fitting_results and fitting_results['simple_mean'] is not None:
        r2 = fitting_results['simple_mean']['r_squared']
        summary_text += f"\nSimple Mean Fit R²: {r2:.6f}"
    
    if 'rms_based' in fitting_results and fitting_results['rms_based'] is not None:
        r2 = fitting_results['rms_based']['r_squared']
        summary_text += f"\nRMS-Based Fit R²: {r2:.6f}"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(f'/Users/albert-mac/Code/GitHub/IVB/Sim/dataset_{dataset_id}_enhanced_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main analysis function"""
    dataset_ids = [317, 346, 435, 338, 337, 439, 336, 352, 335, 341]
    
    print("Enhanced Josephson Junction Analysis")
    print("Addressing Periodic Signal Averaging Issues")
    print("="*80)
    
    analyzer = PeriodicSignalAnalyzer()
    all_results = []
    
    for dataset_id in dataset_ids:
        result = analyze_dataset(dataset_id, analyzer)
        if result:
            all_results.append(result)
    
    # Generate summary report
    generate_summary_report(all_results)
    
    print(f"\n{'='*80}")
    print("Analysis Complete!")
    print(f"Processed {len(all_results)} datasets successfully")
    print("Individual plots saved as dataset_XXX_enhanced_analysis.png")
    print("Summary report saved as enhanced_analysis_summary.csv")

def generate_summary_report(all_results):
    """Generate comprehensive summary report"""
    
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
            'period_aware_mean': stats.get('period_aware_mean', np.nan),
            'estimated_period': stats.get('estimated_period_samples', np.nan),
            'n_complete_periods': stats.get('n_complete_periods', 0),
            'envelope_mean': stats.get('envelope_mean', np.nan)
        }
        
        # Add fitting quality metrics
        if 'simple_mean' in fitting_results and fitting_results['simple_mean']:
            row['simple_fit_r2'] = fitting_results['simple_mean']['r_squared']
            row['simple_fit_residual_std'] = fitting_results['simple_mean']['residual_std']
        else:
            row['simple_fit_r2'] = np.nan
            row['simple_fit_residual_std'] = np.nan
            
        if 'rms_based' in fitting_results and fitting_results['rms_based']:
            row['rms_fit_r2'] = fitting_results['rms_based']['r_squared']
            row['rms_fit_residual_std'] = fitting_results['rms_based']['residual_std']
        else:
            row['rms_fit_r2'] = np.nan
            row['rms_fit_residual_std'] = np.nan
        
        # Calculate relative differences between averaging methods
        simple_mean = stats['simple_mean']
        row['rms_vs_simple_diff_pct'] = 100 * (stats['rms'] - simple_mean) / abs(simple_mean) if simple_mean != 0 else np.nan
        row['median_vs_simple_diff_pct'] = 100 * (stats['median'] - simple_mean) / abs(simple_mean) if simple_mean != 0 else np.nan
        
        if not np.isnan(stats.get('period_aware_mean', np.nan)):
            row['period_vs_simple_diff_pct'] = 100 * (stats['period_aware_mean'] - simple_mean) / abs(simple_mean) if simple_mean != 0 else np.nan
        else:
            row['period_vs_simple_diff_pct'] = np.nan
        
        summary_data.append(row)
    
    # Save to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('/Users/albert-mac/Code/GitHub/IVB/Sim/enhanced_analysis_summary.csv', index=False)
    
    print(f"\nSummary Report Generated:")
    print(f"Columns: {list(summary_df.columns)}")
    print(f"\nMean differences (vs simple mean):")
    print(f"RMS difference: {summary_df['rms_vs_simple_diff_pct'].mean():.2f}% ± {summary_df['rms_vs_simple_diff_pct'].std():.2f}%")
    print(f"Median difference: {summary_df['median_vs_simple_diff_pct'].mean():.2f}% ± {summary_df['median_vs_simple_diff_pct'].std():.2f}%")
    
    period_diffs = summary_df['period_vs_simple_diff_pct'].dropna()
    if len(period_diffs) > 0:
        print(f"Period-aware difference: {period_diffs.mean():.2f}% ± {period_diffs.std():.2f}%")
        print(f"Datasets with detectable periods: {len(period_diffs)}/{len(summary_df)}")
    else:
        print("No datasets showed clear periodicity")

if __name__ == "__main__":
    main()
