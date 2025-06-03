#!/usr/bin/env python3
"""
Fit the simulated data to the model
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.timeseries import LombScargle
from lmfit import Model, Parameters
from sklearn.metrics import r2_score, mean_squared_error

# Set matplotlib font to support better rendering
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['font.size'] = 10


class ModelStatistics:
    """模型統計評估類"""
    
    def __init__(self, y_true, y_pred, n_params, model_name="Unknown"):
        """
        初始化模型統計
        
        Args:
            y_true: 真實值
            y_pred: 預測值
            n_params: 模型參數數量
            model_name: 模型名稱
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.n_params = n_params
        self.model_name = model_name
        self.n_samples = len(y_true)
        
        # 計算統計指標
        self._calculate_statistics()
    
    def _calculate_statistics(self):
        """計算統計指標"""
        # R²
        self.r_squared = r2_score(self.y_true, self.y_pred)
        
        # 調整 R²
        if self.n_samples > self.n_params + 1:
            self.adjusted_r_squared = 1 - (1 - self.r_squared) * (self.n_samples - 1) / (self.n_samples - self.n_params - 1)
        else:
            self.adjusted_r_squared = self.r_squared
        
        # 均方誤差
        self.mse = mean_squared_error(self.y_true, self.y_pred)
        self.rmse = np.sqrt(self.mse)
        
        # 平均絕對誤差
        self.mae = np.mean(np.abs(self.y_true - self.y_pred))
        
        # 殘差
        self.residuals = self.y_true - self.y_pred
        
        # 信噪比 (SNR) 計算
        self._calculate_snr()
        
        # AIC 和 BIC (需要假設誤差分佈)
        if self.mse > 0:
            log_likelihood = -0.5 * self.n_samples * (np.log(2 * np.pi * self.mse) + 1)
            self.aic = 2 * self.n_params - 2 * log_likelihood
            self.bic = self.n_params * np.log(self.n_samples) - 2 * log_likelihood
        else:
            self.aic = np.inf
            self.bic = np.inf
    
    def _calculate_snr(self):
        """計算信噪比 (Signal-to-Noise Ratio)"""
        # 方法1: 信號功率 vs 噪聲功率
        signal_power = np.var(self.y_true)
        noise_power = np.var(self.residuals)
        
        if noise_power > 0:
            self.snr_power = signal_power / noise_power
            self.snr_db = 10 * np.log10(self.snr_power)
        else:
            self.snr_power = np.inf
            self.snr_db = np.inf
        
        # 方法2: 信號RMS vs 噪聲RMS
        signal_rms = np.sqrt(np.mean(self.y_true**2))
        noise_rms = np.sqrt(np.mean(self.residuals**2))
        
        if noise_rms > 0:
            self.snr_rms = signal_rms / noise_rms
            self.snr_rms_db = 20 * np.log10(self.snr_rms)
        else:
            self.snr_rms = np.inf
            self.snr_rms_db = np.inf
        
        # 方法3: 信號範圍 vs 噪聲標準差 (Peak SNR)
        signal_range = np.max(self.y_true) - np.min(self.y_true)
        noise_std = np.std(self.residuals)
        
        if noise_std > 0:
            self.snr_peak = signal_range / noise_std
            self.snr_peak_db = 20 * np.log10(self.snr_peak)
        else:
            self.snr_peak = np.inf
            self.snr_peak_db = np.inf
        
        # 方法4: 去趨勢信號的SNR (適用於週期性信號)
        signal_mean = np.mean(self.y_true)
        signal_detrended = self.y_true - signal_mean
        signal_detrended_power = np.var(signal_detrended)
        
        if noise_power > 0:
            self.snr_detrended = signal_detrended_power / noise_power
            self.snr_detrended_db = 10 * np.log10(self.snr_detrended)
        else:
            self.snr_detrended = np.inf
            self.snr_detrended_db = np.inf
    
    def __str__(self):
        """字符串表示"""
        return f"""
ModelStatistics for {self.model_name}:
  R²: {self.r_squared:.6f}
  Adjusted R²: {self.adjusted_r_squared:.6f}
  RMSE: {self.rmse:.6f}
  MAE: {self.mae:.6f}
  AIC: {self.aic:.2f}
  BIC: {self.bic:.2f}
  SNR (Power): {self.snr_power:.2f} ({self.snr_db:.2f} dB)
  SNR (RMS): {self.snr_rms:.2f} ({self.snr_rms_db:.2f} dB)
  SNR (Peak): {self.snr_peak:.2f} ({self.snr_peak_db:.2f} dB)
  SNR (Detrended): {self.snr_detrended:.2f} ({self.snr_detrended_db:.2f} dB)
"""

def load_simulation_data(file_path):
    """Load simulation data from a CSV file."""
    return pd.read_csv(file_path)

def load_simulation_parameters(file_path):
    """Load simulation parameters from a CSV file."""
    return pd.read_csv(file_path)

class JosephsonAnalyzer:
    """Josephson 結數據分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.simulation_results = {}
        self.analysis_results = {}
    
    def add_simulation_data(self, model_type, data, parameters, model_name):
        """
        添加模擬數據
        
        Args:
            model_type: 模型類型
            data: 包含 Phi_ext, I_s 等數據的字典
            parameters: 模型參數
            model_name: 模型名稱
        """
        self.simulation_results[model_type] = {
            'Phi_ext': data['Phi_ext'],
            'I_s': data['I_s'],
            'I_s_error': data.get('I_s_error', np.zeros_like(data['I_s'])),
            'parameters': parameters,
            'model_name': model_name
        }
    
    def analyze_with_lomb_scargle(self, model_type, detrend_order=1):
        """
        使用 Lomb-Scargle 分析 Josephson 數據
        
        Args:
            model_type: 模型類型
            detrend_order: 去趨勢多項式階數
            
        Returns:
            分析結果字典
        """
        if model_type not in self.simulation_results:
            print(f"❌ Simulation data not found for {model_type} model")
            return None
        
        data = self.simulation_results[model_type]
        times = data['Phi_ext']  # 將外部磁通當作"時間"軸
        values = data['I_s']  # 將超導電流當作"值"軸I_s
        errors = data.get('I_s_error', np.zeros_like(values))  # 如果有誤差，則使用，否則默認為0
        
        
        print(f"\n🔬 Starting Lomb-Scargle Analysis - {data['model_name']}")
        
        # Detrending
        detrended_values = values.copy()
        trend_coeffs = None
        if detrend_order > 0:
            trend_coeffs = np.polyfit(times, values, detrend_order)
            trend = np.polyval(trend_coeffs, times)
            detrended_values = values - trend
            print(f"✅ Applied {detrend_order}-order polynomial detrending")
        
        # Lomb-Scargle analysis
        ls = LombScargle(times, detrended_values, dy=errors, 
                        fit_mean=True, center_data=True)
        
        # Automatically determine frequency range
        time_span = times.max() - times.min()
        min_freq = 0.01 / time_span  # Much lower minimum frequency
        median_dt = np.median(np.diff(np.sort(times)))
        max_freq = 1.0 / median_dt  # More conservative maximum frequency
        
        # Calculate periodogram with higher resolution
        frequency, power = ls.autopower(minimum_frequency=min_freq,
                                      maximum_frequency=max_freq,
                                      samples_per_peak=50)  # Much higher resolution
        
        # 找到最佳頻率
        best_idx = np.argmax(power)
        best_frequency = frequency[best_idx]
        best_period = 1.0 / best_frequency
        best_power = power[best_idx]
        
        # 計算模型參數
        model_params = ls.model_parameters(best_frequency)
        amplitude = np.sqrt(model_params[0]**2 + model_params[1]**2)
        phase = np.arctan2(model_params[1], model_params[0])
        offset = ls.offset()
        
        # 計算擬合值
        ls_model_detrended = ls.model(times, best_frequency)
        if trend_coeffs is not None:
            ls_model_original = ls_model_detrended + np.polyval(trend_coeffs, times)
        else:
            ls_model_original = ls_model_detrended
        
        # 統計評估
        stats = ModelStatistics(
            y_true=values,
            y_pred=ls_model_original,
            n_params=3,  # 頻率、振幅、相位
            model_name=f"LS-{data['model_name']}"
        )
        
        # 保存分析結果
        analysis_result = {
            'frequency': frequency,
            'power': power,
            'best_frequency': best_frequency,
            'best_period': best_period,
            'best_power': best_power,
            'amplitude': amplitude,
            'phase': phase,
            'offset': offset,
            'ls_model': ls_model_original,
            'statistics': stats,
            'true_frequency': data['parameters']['f'],  # 真實頻率
            'ls_object': ls
        }
        
        self.analysis_results[model_type] = analysis_result
        
        # Print results
        print(f"\n📊 Lomb-Scargle Analysis Results:")
        print(f"   True Frequency: {data['parameters']['f']:.6f}")
        print(f"   Detected Frequency: {best_frequency:.6f}")
        print(f"   Frequency Error: {abs(best_frequency - data['parameters']['f']):.6f}")
        print(f"   Best Period: {best_period:.6f}")
        print(f"   Detected Amplitude: {amplitude:.6f}")
        print(f"   R²: {stats.r_squared:.6f}")
        print(f"   SNR (Power): {stats.snr_power:.2f} ({stats.snr_db:.2f} dB)")
        print(f"   SNR (RMS): {stats.snr_rms:.2f} ({stats.snr_rms_db:.2f} dB)")
        print(f"   SNR (Peak): {stats.snr_peak:.2f} ({stats.snr_peak_db:.2f} dB)")
        
        return analysis_result
    
    def plot_analysis_results(self, model_type, save_plot=True):
        """
        繪製分析結果
        
        Args:
            model_type: 模型類型
            save_plot: 是否保存圖片
        """
        if model_type not in self.analysis_results:
            print(f"❌ Analysis results not found for {model_type}")
            return
        
        result = self.analysis_results[model_type]
        data = self.simulation_results[model_type]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Lomb-Scargle Analysis: {data["model_name"]}', fontsize=16)
        
        # 1. 原始數據和擬合
        ax1 = axes[0, 0]
        ax1.plot(data['Phi_ext'], data['I_s'], 'b.', alpha=0.6, label='Data')
        ax1.plot(data['Phi_ext'], result['ls_model'], 'r-', linewidth=2, label='LS Fit')
        ax1.set_xlabel('External Flux (Φ_ext)')
        ax1.set_ylabel('Supercurrent (I_s)')
        ax1.set_title('Data and Lomb-Scargle Fit')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 週期圖
        ax2 = axes[0, 1]
        ax2.plot(result['frequency'], result['power'], 'k-', linewidth=1.5)
        ax2.axvline(result['best_frequency'], color='r', linestyle='--', 
                   label=f'Best freq: {result["best_frequency"]:.6f}')
        ax2.axvline(data['parameters']['f'], color='g', linestyle='--', 
                   label=f'True freq: {data["parameters"]["f"]:.6f}')
        ax2.set_xlabel('Frequency')
        ax2.set_ylabel('Power')
        ax2.set_title('Lomb-Scargle Periodogram')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 殘差分析
        ax3 = axes[1, 0]
        residuals = data['I_s'] - result['ls_model']
        ax3.plot(data['Phi_ext'], residuals, 'k.', alpha=0.6)
        ax3.axhline(0, color='r', linestyle='-', alpha=0.7)
        ax3.set_xlabel('External Flux (Φ_ext)')
        ax3.set_ylabel('Residuals')
        ax3.set_title('Residual Analysis')
        ax3.grid(True, alpha=0.3)
        
        # 4. Statistics Information
        ax4 = axes[1, 1]
        ax4.axis('off')
        stats_text = f"""
Statistical Evaluation:

R² = {result['statistics'].r_squared:.6f}
Adjusted R² = {result['statistics'].adjusted_r_squared:.6f}
RMSE = {result['statistics'].rmse:.6e}
MAE = {result['statistics'].mae:.6e}

Frequency Comparison:
True Frequency: {data['parameters']['f']:.6f}
Detected Frequency: {result['best_frequency']:.6f}
Relative Error: {abs(result['best_frequency'] - data['parameters']['f']) / data['parameters']['f'] * 100:.2f}%

Model Parameters:
Amplitude: {result['amplitude']:.6f}
Phase: {result['phase']:.6f}
Offset: {result['offset']:.6f}
"""
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plot:
            filename = f'lomb_scargle_analysis_{model_type}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Analysis plot saved: {filename}")
        
        plt.show()
    
    def generate_comparison_report(self):
        """生成比較報告"""
        if not self.analysis_results:
            print("❌ No analysis results available")
            return
        
        print("\n" + "="*80)
        print("📊 LOMB-SCARGLE ANALYSIS SUMMARY REPORT")
        print("="*80)
        
        for model_type, result in self.analysis_results.items():
            data = self.simulation_results[model_type]
            print(f"\n🔬 Model: {data['model_name']} ({model_type})")
            print("-" * 60)
            print(f"   True Frequency:     {data['parameters']['f']:.6f}")
            print(f"   Detected Frequency: {result['best_frequency']:.6f}")
            print(f"   Frequency Error:    {abs(result['best_frequency'] - data['parameters']['f']):.6f}")
            print(f"   Relative Error:     {abs(result['best_frequency'] - data['parameters']['f']) / data['parameters']['f'] * 100:.2f}%")
            print(f"   R²:                 {result['statistics'].r_squared:.6f}")
            print(f"   Detected Amplitude: {result['amplitude']:.6f}")
            print(f"   Detected Phase:     {result['phase']:.6f}")
            print(f"   SNR (Power):        {result['statistics'].snr_power:.2f} ({result['statistics'].snr_db:.2f} dB)")
            print(f"   SNR (RMS):          {result['statistics'].snr_rms:.2f} ({result['statistics'].snr_rms_db:.2f} dB)")
            print(f"   SNR (Peak):         {result['statistics'].snr_peak:.2f} ({result['statistics'].snr_peak_db:.2f} dB)")
            print(f"   SNR (Detrended):    {result['statistics'].snr_detrended:.2f} ({result['statistics'].snr_detrended_db:.2f} dB)")
    
    def save_detrended_data_to_csv(self, model_type):
        """
        Save detrended data to CSV file
        
        Args:
            model_type: Model type to save data for
        """
        if model_type not in self.analysis_results:
            print(f"❌ Analysis results not found for {model_type}")
            return
        
        result = self.analysis_results[model_type]
        data = self.simulation_results[model_type]
        
        print(f"\n💾 Saving detrended data for {data['model_name']}...")
        
        # Get original data
        times = data['Phi_ext']
        values = data['I_s']
        
        # Calculate trend and detrended data
        detrended_values = values.copy()
        trend_line = np.zeros_like(values)
        
        # Apply same detrending as in analysis
        trend_coeffs = np.polyfit(times, values, 1)  # 1st order
        trend_line = np.polyval(trend_coeffs, times)
        detrended_values = values - trend_line
        
        # Create output dataframe
        output_data = {
            'Phi_ext': times,
            'I_s_original': values,
            'I_s_trend': trend_line,
            'I_s_detrended': detrended_values,
            'I_s_ls_fit': result['ls_model']
        }
        
        output_df = pd.DataFrame(output_data)
        
        # Save to CSV
        filename = f'detrended_data_{model_type}.csv'
        output_df.to_csv(filename, index=False)
        
        print(f"✅ Detrended data saved to: {filename}")
        print(f"   Columns: {list(output_df.columns)}")
        print(f"   Data points: {len(output_df)}")
        
        # Also save SNR analysis summary
        self._save_snr_analysis_summary(model_type, result)
        
        return filename
    
    def _save_snr_analysis_summary(self, model_type, result):
        """Save SNR analysis summary to CSV"""
        data = self.simulation_results[model_type]
        stats = result['statistics']
        
        summary_data = {
            'Model': [data['model_name']],
            'Model_Type': [model_type],
            'True_Frequency': [data['parameters']['f']],
            'Detected_Frequency': [result['best_frequency']],
            'Frequency_Error': [abs(result['best_frequency'] - data['parameters']['f'])],
            'Relative_Error_Percent': [abs(result['best_frequency'] - data['parameters']['f']) / data['parameters']['f'] * 100],
            'R_Squared': [stats.r_squared],
            'RMSE': [stats.rmse],
            'MAE': [stats.mae],
            'SNR_Power': [stats.snr_power],
            'SNR_Power_dB': [stats.snr_db],
            'SNR_RMS': [stats.snr_rms],
            'SNR_RMS_dB': [stats.snr_rms_db],
            'SNR_Peak': [stats.snr_peak],
            'SNR_Peak_dB': [stats.snr_peak_db],
            'SNR_Detrended': [stats.snr_detrended],
            'SNR_Detrended_dB': [stats.snr_detrended_db],
            'Detected_Amplitude': [result['amplitude']],
            'Detected_Phase': [result['phase']]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_filename = f'snr_analysis_summary_{model_type}.csv'
        summary_df.to_csv(summary_filename, index=False)
        
        print(f"✅ SNR analysis summary saved to: {summary_filename}")
    
    def plot_detrended_data_comparison(self, model_type, save_plot=True):
        """
        Plot comparison of original, trend, and detrended data
        
        Args:
            model_type: Model type to plot
            save_plot: Whether to save the plot
        """
        if model_type not in self.analysis_results:
            print(f"❌ Analysis results not found for {model_type}")
            return
        
        result = self.analysis_results[model_type]
        data = self.simulation_results[model_type]
        
        print(f"\n📊 Creating detrended data comparison plot for {data['model_name']}...")
        
        # Get data
        times = data['Phi_ext']
        values = data['I_s']
        
        # Calculate trend and detrended data
        trend_coeffs = np.polyfit(times, values, 1)  # 1st order
        trend_line = np.polyval(trend_coeffs, times)
        detrended_values = values - trend_line
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Detrending Analysis: {data["model_name"]}', fontsize=16)
        
        # 1. Original data with trend line
        ax1 = axes[0, 0]
        ax1.plot(times, values, 'b.', alpha=0.6, label='Original Data')
        ax1.plot(times, trend_line, 'r-', linewidth=2, label='1st-Order Trend')
        ax1.set_xlabel('External Flux (Φ_ext)')
        ax1.set_ylabel('Supercurrent (I_s)')
        ax1.set_title('Original Data with Trend Line')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Detrended data
        ax2 = axes[0, 1]
        ax2.plot(times, detrended_values, 'g.', alpha=0.6, label='Detrended Data')
        ax2.plot(times, result['ls_model'] - trend_line, 'r-', linewidth=2, label='LS Fit (Detrended)')
        ax2.set_xlabel('External Flux (Φ_ext)')
        ax2.set_ylabel('Detrended Supercurrent')
        ax2.set_title('Detrended Data with Lomb-Scargle Fit')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Comparison: Before vs After Detrending
        ax3 = axes[1, 0]
        ax3.plot(times, values, 'b-', alpha=0.7, label='Original')
        ax3.plot(times, detrended_values + np.mean(values), 'g-', alpha=0.7, label='Detrended (shifted)')
        ax3.set_xlabel('External Flux (Φ_ext)')
        ax3.set_ylabel('Supercurrent (I_s)')
        ax3.set_title('Original vs Detrended Data')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Statistical summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate statistics
        original_std = np.std(values)
        detrended_std = np.std(detrended_values)
        trend_slope = trend_coeffs[0]
        trend_intercept = trend_coeffs[1]
        improvement = (original_std - detrended_std) / original_std * 100
        
        stats_text = f"""
Detrending Statistics:

Original Data:
  Std Dev: {original_std:.6e}
  Range: {np.max(values) - np.min(values):.6e}

Trend Line (1st-order):
  Slope: {trend_slope:.6e}
  Intercept: {trend_intercept:.6e}

Detrended Data:
  Std Dev: {detrended_std:.6e}
  Range: {np.max(detrended_values) - np.min(detrended_values):.6e}
  Improvement: {improvement:.1f}%

Lomb-Scargle Analysis:
  Best Frequency: {result['best_frequency']:.6f}
  R²: {result['statistics'].r_squared:.6f}
  RMSE: {result['statistics'].rmse:.6e}

Signal-to-Noise Ratio:
  SNR (Power): {result['statistics'].snr_power:.2f} ({result['statistics'].snr_db:.2f} dB)
  SNR (RMS): {result['statistics'].snr_rms:.2f} ({result['statistics'].snr_rms_db:.2f} dB)
  SNR (Peak): {result['statistics'].snr_peak:.2f} ({result['statistics'].snr_peak_db:.2f} dB)
"""
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plot:
            filename = f'detrending_comparison_{model_type}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"✅ Detrending comparison plot saved: {filename}")
        
        plt.show()
        return fig
    
    def fit_complete_josephson_equation(self, model_type, use_lbfgsb=True, save_results=True):
        """
        Fit complete Josephson equation using lmfit + L-BFGS-B with Lomb-Scargle initial parameters
        
        Args:
            model_type: Model type to fit
            use_lbfgsb: Whether to use L-BFGS-B method (default: True)
            save_results: Whether to save results to CSV files
            
        Returns:
            JosephsonFitter object with fit results
        """
        if model_type not in self.analysis_results:
            print(f"❌ No Lomb-Scargle analysis results found for {model_type}")
            print("   Please run analyze_with_lomb_scargle() first to generate initial parameters")
            return None
        
        if model_type not in self.simulation_results:
            print(f"❌ No simulation data found for {model_type}")
            return None
        
        # Get data and Lomb-Scargle results
        data = self.simulation_results[model_type]
        ls_result = self.analysis_results[model_type]
        
        phi_ext = data['Phi_ext']
        I_s = data['I_s']
        I_s_error = data.get('I_s_error', None)
        
        print(f"\n🚀 Fitting Complete Josephson Equation for {data['model_name']}")
        print("="*70)
        print(f"   Using Lomb-Scargle results as initial parameters")
        print(f"   Method: {'L-BFGS-B' if use_lbfgsb else 'default lmfit'}")
        
        # Create Josephson fitter
        fitter = JosephsonFitter()
        
        # Perform the fit
        method = 'lbfgsb' if use_lbfgsb else 'leastsq'
        fit_result = fitter.fit(
            phi_ext=phi_ext,
            I_s=I_s,
            I_s_error=I_s_error,
            lomb_scargle_result=ls_result,
            method=method
        )
        
        if fit_result is None:
            print("❌ Fitting failed")
            return None
        
        # Print fit results
        fitted_params = fitter.get_fitted_parameters()
        if fitted_params is not None:
            print(f"\n📊 Complete Josephson Equation Fit Results:")
            print("-"*50)
            for param_name, param_info in fitted_params.items():
                initial_val = param_info['initial']
                fitted_val = param_info['value']
                stderr = param_info['stderr']
                
                print(f"   {param_name}:")
                print(f"      Initial:  {initial_val:.6f}")
                print(f"      Fitted:   {fitted_val:.6f} ± {stderr:.6f}")
                if param_name == 'f' and 'f' in data['parameters']:
                    true_val = data['parameters']['f']
                    error = abs(fitted_val - true_val)
                    rel_error = error / true_val * 100
                    print(f"      True:     {true_val:.6f}")
                    print(f"      Error:    {error:.6f} ({rel_error:.2f}%)")
        
        # Generate plots
        print(f"\n📈 Generating complete Josephson fit plots...")
        stats = fitter.plot_fit_results(phi_ext, I_s, save_plot=True, 
                                       filename_suffix=f"_{model_type}")
        
        # Save results if requested
        if save_results:
            print(f"\n💾 Saving complete Josephson fit results...")
            result_files = fitter.save_fit_results_to_csv(
                phi_ext, I_s, filename_suffix=f"_{model_type}")
            if result_files is not None:
                data_file, param_file = result_files
        
        # Store fitter object for future use
        if not hasattr(self, 'josephson_fitters'):
            self.josephson_fitters = {}
        self.josephson_fitters[model_type] = fitter
        
        return fitter
    
    def compare_lomb_scargle_vs_josephson_fit(self, model_type, save_plot=True):
        """
        Compare Lomb-Scargle analysis results with complete Josephson equation fit
        
        Args:
            model_type: Model type to compare
            save_plot: Whether to save the comparison plot
        """
        if model_type not in self.analysis_results:
            print(f"❌ No Lomb-Scargle analysis results found for {model_type}")
            return
        
        if not hasattr(self, 'josephson_fitters') or model_type not in self.josephson_fitters:
            print(f"❌ No complete Josephson fit results found for {model_type}")
            print("   Please run fit_complete_josephson_equation() first")
            return
        
        data = self.simulation_results[model_type]
        ls_result = self.analysis_results[model_type]
        fitter = self.josephson_fitters[model_type]
        
        phi_ext = data['Phi_ext']
        I_s = data['I_s']
        
        # Calculate fitted curves
        ls_fit = ls_result['ls_model']
        josephson_fit = fitter.calculate_fitted_curve(phi_ext)
        
        if josephson_fit is None:
            print("❌ Unable to calculate Josephson fit curve")
            return
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Lomb-Scargle vs Complete Josephson Fit: {data["model_name"]}', fontsize=16)
        
        # 1. Data with both fits
        ax1 = axes[0, 0]
        ax1.plot(phi_ext, I_s, 'k.', alpha=0.4, label='Data', markersize=2)
        ax1.plot(phi_ext, ls_fit, 'b-', linewidth=2, label='Lomb-Scargle Fit', alpha=0.8)
        ax1.plot(phi_ext, josephson_fit, 'r-', linewidth=2, label='Complete Josephson Fit', alpha=0.8)
        ax1.set_xlabel('External Flux (Φ_ext)')
        ax1.set_ylabel('Supercurrent (I_s)')
        ax1.set_title('Data and Model Fits')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuals comparison
        ax2 = axes[0, 1]
        ls_residuals = I_s - ls_fit
        josephson_residuals = I_s - josephson_fit
        ax2.plot(phi_ext, ls_residuals, 'b.', alpha=0.6, label='LS Residuals', markersize=3)
        ax2.plot(phi_ext, josephson_residuals, 'r.', alpha=0.6, label='Josephson Residuals', markersize=3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('External Flux (Φ_ext)')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Parameter comparison
        ax3 = axes[1, 0]
        fitted_params = fitter.get_fitted_parameters()
        
        if fitted_params is not None:
            # Compare frequencies and amplitudes
            ls_freq = ls_result['best_frequency']
            ls_amp = ls_result['amplitude']
            josephson_freq = fitted_params['f']['value']
            josephson_amp = fitted_params['I_c']['value']
            
            param_comparison = {
                'Frequency': [ls_freq, josephson_freq],
                'Amplitude/I_c': [ls_amp, josephson_amp]
            }
            
            x_pos = np.arange(len(param_comparison))
            width = 0.35
            
            ls_values = [param_comparison['Frequency'][0], param_comparison['Amplitude/I_c'][0]]
            josephson_values = [param_comparison['Frequency'][1], param_comparison['Amplitude/I_c'][1]]
            
            ax3.bar(x_pos - width/2, ls_values, width, label='Lomb-Scargle', alpha=0.8)
            ax3.bar(x_pos + width/2, josephson_values, width, label='Complete Josephson', alpha=0.8)
            ax3.set_xlabel('Parameters')
            ax3.set_ylabel('Values')
            ax3.set_title('Key Parameter Comparison')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(list(param_comparison.keys()))
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Statistical comparison
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate statistics for both fits
        ls_stats = ls_result['statistics']
        josephson_stats = ModelStatistics(
            y_true=I_s,
            y_pred=josephson_fit,
            n_params=len(fitted_params) if fitted_params else 7,
            model_name="Complete Josephson"
        )
        
        stats_text = f"""
Comparison Statistics:

Lomb-Scargle Analysis:
  R²: {ls_stats.r_squared:.6f}
  RMSE: {ls_stats.rmse:.6e}
  MAE: {ls_stats.mae:.6e}
  SNR (Power): {ls_stats.snr_power:.2f} ({ls_stats.snr_db:.1f} dB)
  SNR (RMS): {ls_stats.snr_rms:.2f} ({ls_stats.snr_rms_db:.1f} dB)

Complete Josephson Fit:
  R²: {josephson_stats.r_squared:.6f}
  RMSE: {josephson_stats.rmse:.6e}
  MAE: {josephson_stats.mae:.6e}
  SNR (Power): {josephson_stats.snr_power:.2f} ({josephson_stats.snr_db:.1f} dB)
  SNR (RMS): {josephson_stats.snr_rms:.2f} ({josephson_stats.snr_rms_db:.1f} dB)
  Chi-square: {fitter.result.chisqr:.6f}
  AIC: {fitter.result.aic:.2f}
  BIC: {fitter.result.bic:.2f}

Improvement:
  ΔR²: {josephson_stats.r_squared - ls_stats.r_squared:.6f}
  ΔRMSE: {ls_stats.rmse - josephson_stats.rmse:.6e}
  ΔSNR (dB): {josephson_stats.snr_db - ls_stats.snr_db:.1f}
"""
        
        if fitted_params is not None:
            stats_text += f"""
Josephson Parameters:
  I_c = {fitted_params['I_c']['value']:.6f} ± {fitted_params['I_c']['stderr']:.6f}
  f = {fitted_params['f']['value']:.6f} ± {fitted_params['f']['stderr']:.6f}
  T = {fitted_params['T']['value']:.6f} ± {fitted_params['T']['stderr']:.6f}
  φ₀ = {fitted_params['phi_0']['value']:.6f} ± {fitted_params['phi_0']['stderr']:.6f}
"""
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plot:
            filename = f'lomb_scargle_vs_josephson_comparison_{model_type}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Comparison plot saved: {filename}")
        
        plt.show()
        
        return josephson_stats


def analyze_with_lomb_scargle_standalone(times, values, errors=None, true_frequency=None, 
                                       detrend_order=1, model_name="Unknown"):
    """
    獨立的 Lomb-Scargle 分析函數
    
    Args:
        times: 時間序列（或外部磁通）
        values: 測量值（或超導電流）
        errors: 測量誤差（可選）
        true_frequency: 真實頻率（用於比較，可選）
        detrend_order: 去趨勢多項式階數
        model_name: 模型名稱
        
    Returns:
        分析結果字典
    """
    times = np.array(times)
    values = np.array(values)
    
    if errors is None:
        errors = np.zeros_like(values)
    else:
        errors = np.array(errors)
    
    print(f"\n🔬 Start Lomb-Scargle Analysis - {model_name}")
    
    # Detrending
    detrended_values = values.copy()
    trend_coeffs = None
    if detrend_order > 0:
        trend_coeffs = np.polyfit(times, values, detrend_order)
        trend = np.polyval(trend_coeffs, times)
        detrended_values = values - trend
        print(f"✅ Applied {detrend_order}-order polynomial detrending")
    
    # Lomb-Scargle analysis
    ls = LombScargle(times, detrended_values, dy=errors, 
                    fit_mean=True, center_data=True)
    
    # Automatically determine frequency range
    time_span = times.max() - times.min()
    min_freq = 0.01 / time_span  # Much lower minimum frequency
    median_dt = np.median(np.diff(np.sort(times)))
    max_freq = 1.0 / median_dt  # More conservative maximum frequency
    
    # Calculate periodogram with higher resolution
    frequency, power = ls.autopower(minimum_frequency=min_freq,
                                  maximum_frequency=max_freq,
                                  samples_per_peak=50)  # Much higher resolution
    
    # 找到最佳頻率
    best_idx = np.argmax(power)
    best_frequency = frequency[best_idx]
    best_period = 1.0 / best_frequency
    best_power = power[best_idx]
    
    # 計算模型參數
    model_params = ls.model_parameters(best_frequency)
    amplitude = np.sqrt(model_params[0]**2 + model_params[1]**2)
    phase = np.arctan2(model_params[1], model_params[0])
    offset = ls.offset()
    
    # 計算擬合值
    ls_model_detrended = ls.model(times, best_frequency)
    if trend_coeffs is not None:
        ls_model_original = ls_model_detrended + np.polyval(trend_coeffs, times)
    else:
        ls_model_original = ls_model_detrended
    
    # 統計評估
    stats = ModelStatistics(
        y_true=values,
        y_pred=ls_model_original,
        n_params=3,  # 頻率、振幅、相位
        model_name=f"LS-{model_name}"
    )
    
    # 組織結果
    analysis_result = {
        'frequency': frequency,
        'power': power,
        'best_frequency': best_frequency,
        'best_period': best_period,
        'best_power': best_power,
        'amplitude': amplitude,
        'phase': phase,
        'offset': offset,
        'ls_model': ls_model_original,
        'statistics': stats,
        'ls_object': ls
    }
    
    if true_frequency is not None:
        analysis_result['true_frequency'] = true_frequency
        frequency_error = abs(best_frequency - true_frequency)
        relative_error = frequency_error / true_frequency * 100
        analysis_result['frequency_error'] = frequency_error
        analysis_result['relative_error'] = relative_error
    
    # Print results
    print(f"\n📊 Lomb-Scargle Analysis Results:")
    if true_frequency is not None:
        print(f"   True Frequency: {true_frequency:.6e}")
        print(f"   Detected Frequency: {best_frequency:.6e}")
        print(f"   Frequency Error: {frequency_error:.6e}")
        print(f"   Relative Error: {relative_error:.2f}%")
    else:
        print(f"   Detected Frequency: {best_frequency:.6e}")
    print(f"   Best Period: {best_period:.6e}")
    print(f"   Detected Amplitude: {amplitude:.6e}")
    print(f"   R²: {stats.r_squared:.6f}")
    print(f"   SNR (Power): {stats.snr_power:.2f} ({stats.snr_db:.2f} dB)")
    print(f"   SNR (RMS): {stats.snr_rms:.2f} ({stats.snr_rms_db:.2f} dB)")
    print(f"   SNR (Peak): {stats.snr_peak:.2f} ({stats.snr_peak_db:.2f} dB)")

    return analysis_result


def complete_josephson_equation(phi_ext, I_c, f, d, phi_0, T, r, C):
    """
    Complete Josephson junction equation with finite transmission
    
    Args:
        phi_ext: External flux (independent variable)
        I_c: Critical current
        f: Frequency
        d: Phase offset
        phi_0: Initial phase
        T: Transmission coefficient (0 < T < 1)
        r: Linear trend slope
        C: Constant offset
        
    Returns:
        Current I_s calculated from the complete Josephson equation
    """
    # Calculate the phase
    phase = 2 * np.pi * f * (phi_ext - d) - phi_0
    
    # Avoid numerical issues with square root
    sin_half_phase = np.sin(phase / 2)
    denominator = np.sqrt(1 - T * sin_half_phase**2)
    
    # Main Josephson term
    josephson_term = I_c * np.sin(phase) / denominator
    
    # Linear background term
    linear_term = r * (phi_ext - d)
    
    # Total current
    I_s = josephson_term + linear_term + C
    
    return I_s


class JosephsonFitter:
    """
    Complete Josephson equation fitter using lmfit + L-BFGS-B
    """
    
    def __init__(self):
        """Initialize the fitter"""
        self.model = None
        self.result = None
        self.initial_params = None
        
    def create_model(self):
        """Create the lmfit Model for complete Josephson equation"""
        self.model = Model(complete_josephson_equation)
        return self.model
    
    def estimate_initial_parameters(self, phi_ext, I_s, lomb_scargle_result=None):
        """
        Estimate initial parameters using Lomb-Scargle results and data analysis
        
        Args:
            phi_ext: External flux data
            I_s: Current data
            lomb_scargle_result: Results from Lomb-Scargle analysis
            
        Returns:
            lmfit Parameters object with initial estimates
        """
        params = Parameters()
        
        if lomb_scargle_result is not None:
            # Use frequency from Lomb-Scargle (detected from detrended data)
            initial_f = lomb_scargle_result.get('best_frequency', 1.0 / (2 * np.pi))
            initial_phi_0 = lomb_scargle_result.get('phase', 0.0)
            
            # CRITICAL FIX: Re-estimate amplitude and offset from ORIGINAL data
            # Lomb-Scargle was done on detrended data, but Josephson fit uses original data
            initial_I_c = np.std(I_s) * 2  # Conservative estimate from original data variation
            initial_C = np.mean(I_s)  # True baseline from original data
            
            print(f"📊 Using hybrid initial parameters:")
            print(f"   Initial f: {initial_f:.6e} (from Lomb-Scargle analysis)")
            print(f"   Initial phi_0: {initial_phi_0:.6f} (from Lomb-Scargle analysis)")
            print(f"   Initial I_c: {initial_I_c:.6e} (re-estimated from original data)")
            print(f"   Initial C: {initial_C:.6e} (mean of original data)")
            print("   ✅ Using original data for amplitude and baseline estimation")
        else:
            # Fallback estimates from data
            initial_I_c = np.std(I_s) * 2
            initial_f = 1.0 / (2 * np.pi)  # Default Josephson frequency
            initial_phi_0 = 0.0
            initial_C = np.mean(I_s)
            
            print(f"📊 Using data-based initial parameters (no Lomb-Scargle results)")
        
        # Estimate linear trend
        trend_coeffs = np.polyfit(phi_ext, I_s, 1)
        initial_r = trend_coeffs[0]
        
        # Set parameters with bounds
        params.add('I_c', value=initial_I_c, min=0.1 * abs(initial_I_c), max=10 * abs(initial_I_c))
        params.add('f', value=initial_f, min=0.01 * initial_f, max=100 * initial_f)
        params.add('d', value=0.0, min=-10, max=10)
        params.add('phi_0', value=initial_phi_0, min=-2*np.pi, max=2*np.pi)
        params.add('T', value=0.5, min=0.01, max=0.99)  # Transmission coefficient
        params.add('r', value=initial_r, min=-10*abs(initial_r), max=10*abs(initial_r))
        params.add('C', value=initial_C, min=initial_C - 5*np.std(I_s), max=initial_C + 5*np.std(I_s))
        
        self.initial_params = params
        return params
    
    def fit(self, phi_ext, I_s, I_s_error=None, lomb_scargle_result=None, method='lbfgsb'):
        """
        Fit the complete Josephson equation to data
        
        Args:
            phi_ext: External flux data
            I_s: Current data
            I_s_error: Current measurement errors (optional)
            lomb_scargle_result: Results from Lomb-Scargle analysis
            method: Fitting method (default: 'lbfgsb' for L-BFGS-B)
            
        Returns:
            Fit result object
        """
        if self.model is None:
            self.create_model()
        
        # Estimate initial parameters
        params = self.estimate_initial_parameters(phi_ext, I_s, lomb_scargle_result)
        
        # Prepare weights
        weights = None
        if I_s_error is not None and np.any(I_s_error > 0):
            weights = 1.0 / I_s_error
        
        print(f"\n🔧 Fitting complete Josephson equation using {method.upper()}...")
        
        # Perform the fit
        try:
            self.result = self.model.fit(
                I_s, 
                params, 
                phi_ext=phi_ext,
                weights=weights,
                method=method
            )
            
            print(f"✅ Fitting completed successfully!")
            print(f"   Chi-square: {self.result.chisqr:.6f}")
            print(f"   Reduced Chi-square: {self.result.redchi:.6f}")
            print(f"   AIC: {self.result.aic:.2f}")
            print(f"   BIC: {self.result.bic:.2f}")
            
            return self.result
            
        except Exception as e:
            print(f"❌ Fitting failed: {str(e)}")
            return None
    
    def get_fitted_parameters(self):
        """Get fitted parameters with uncertainties"""
        if self.result is None:
            return None
        
        fitted_params = {}
        for name, param in self.result.params.items():
            fitted_params[name] = {
                'value': param.value,
                'stderr': param.stderr if param.stderr is not None else 0.0,
                'initial': self.initial_params[name].value
            }
        
        return fitted_params
    
    def calculate_fitted_curve(self, phi_ext):
        """Calculate the fitted curve"""
        if self.result is None:
            return None
        
        return self.result.eval(phi_ext=phi_ext)
    
    def plot_fit_results(self, phi_ext, I_s, save_plot=True, filename_suffix=""):
        """
        Plot the fitting results
        
        Args:
            phi_ext: External flux data
            I_s: Current data
            save_plot: Whether to save the plot
            filename_suffix: Suffix for the filename
        """
        if self.result is None:
            print("❌ No fit results to plot")
            return
        
        # Calculate fitted curve
        fitted_curve = self.calculate_fitted_curve(phi_ext)
        residuals = I_s - fitted_curve
        
        # Create statistics
        stats = ModelStatistics(
            y_true=I_s,
            y_pred=fitted_curve,
            n_params=len(self.result.params),
            model_name="Complete Josephson Fit"
        )
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Complete Josephson Equation Fit Results', fontsize=16)
        
        # 1. Data and fit
        ax1 = axes[0, 0]
        ax1.plot(phi_ext, I_s, 'b.', alpha=0.6, label='Data', markersize=3)
        ax1.plot(phi_ext, fitted_curve, 'r-', linewidth=2, label='Complete Josephson Fit')
        ax1.set_xlabel('External Flux (Φ_ext)')
        ax1.set_ylabel('Supercurrent (I_s)')
        ax1.set_title('Data vs Complete Josephson Fit')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuals
        ax2 = axes[0, 1]
        ax2.plot(phi_ext, residuals, 'g.', alpha=0.6, markersize=3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('External Flux (Φ_ext)')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Fit Residuals')
        ax2.grid(True, alpha=0.3)
        
        # 3. Parameter comparison
        ax3 = axes[1, 0]
        fitted_params = self.get_fitted_parameters()
        
        param_names = []
        initial_values = []
        fitted_values = []
        errors = []
        
        for name, param_info in fitted_params.items():
            param_names.append(name)
            initial_values.append(param_info['initial'])
            fitted_values.append(param_info['value'])
            errors.append(param_info['stderr'])
        
        x_pos = np.arange(len(param_names))
        ax3.bar(x_pos - 0.2, initial_values, 0.4, label='Initial', alpha=0.7)
        ax3.errorbar(x_pos + 0.2, fitted_values, yerr=errors, fmt='o', 
                    label='Fitted', capsize=5, capthick=2)
        ax3.set_xlabel('Parameters')
        ax3.set_ylabel('Values')
        ax3.set_title('Parameter Comparison')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(param_names, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""
Fit Statistics:

Chi-square: {self.result.chisqr:.6f}
Reduced Chi-square: {self.result.redchi:.6f}
R²: {stats.r_squared:.6f}
Adjusted R²: {stats.adjusted_r_squared:.6f}
RMSE: {stats.rmse:.6e}
MAE: {stats.mae:.6e}
AIC: {self.result.aic:.2f}
BIC: {self.result.bic:.2f}

SNR Metrics:
Power SNR: {stats.snr_power:.2f} ({stats.snr_db:.1f} dB)
RMS SNR: {stats.snr_rms:.2f} ({stats.snr_rms_db:.1f} dB)
Peak SNR: {stats.snr_peak:.2f} ({stats.snr_peak_db:.1f} dB)

Fitted Parameters:
I_c = {fitted_params['I_c']['value']:.6f} ± {fitted_params['I_c']['stderr']:.6f}
f = {fitted_params['f']['value']:.6f} ± {fitted_params['f']['stderr']:.6f}
d = {fitted_params['d']['value']:.6f} ± {fitted_params['d']['stderr']:.6f}
φ₀ = {fitted_params['phi_0']['value']:.6f} ± {fitted_params['phi_0']['stderr']:.6f}
T = {fitted_params['T']['value']:.6f} ± {fitted_params['T']['stderr']:.6f}
r = {fitted_params['r']['value']:.6f} ± {fitted_params['r']['stderr']:.6f}
C = {fitted_params['C']['value']:.6f} ± {fitted_params['C']['stderr']:.6f}
"""
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plot:
            filename = f'complete_josephson_fit{filename_suffix}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Complete Josephson fit plot saved: {filename}")
        
        plt.show()
        
        return stats
    
    def save_fit_results_to_csv(self, phi_ext, I_s, filename_suffix=""):
        """Save fit results to CSV file"""
        if self.result is None:
            print("❌ No fit results to save")
            return None
        
        fitted_curve = self.calculate_fitted_curve(phi_ext)
        residuals = I_s - fitted_curve
        
        # Create output dataframe
        output_data = {
            'Phi_ext': phi_ext,
            'I_s_measured': I_s,
            'I_s_josephson_fit': fitted_curve,
            'residuals': residuals
        }
        
        output_df = pd.DataFrame(output_data)
        
        # Save to CSV
        filename = f'complete_josephson_fit_results{filename_suffix}.csv'
        output_df.to_csv(filename, index=False)
        
        # Also save parameter summary
        fitted_params = self.get_fitted_parameters()
        param_summary = []
        
        for name, param_info in fitted_params.items():
            param_summary.append({
                'Parameter': name,
                'Initial_Value': param_info['initial'],
                'Fitted_Value': param_info['value'],
                'Standard_Error': param_info['stderr']
            })
        
        param_df = pd.DataFrame(param_summary)
        param_filename = f'complete_josephson_fit_parameters{filename_suffix}.csv'
        param_df.to_csv(param_filename, index=False)
        
        print(f"✅ Fit results saved to: {filename}")
        print(f"✅ Fit parameters saved to: {param_filename}")
        print(f"   Data points: {len(output_df)}")
        
        return filename, param_filename
    
    def fit_complete_josephson_equation(self, model_type, use_lbfgsb=True, save_results=True):
        """
        Fit complete Josephson equation using lmfit + L-BFGS-B with Lomb-Scargle initial parameters
        
        Args:
            model_type: Model type to fit
            use_lbfgsb: Whether to use L-BFGS-B method (default: True)
            save_results: Whether to save results to CSV files
            
        Returns:
            JosephsonFitter object with fit results
        """
        if model_type not in self.analysis_results:
            print(f"❌ No Lomb-Scargle analysis results found for {model_type}")
            print("   Please run analyze_with_lomb_scargle() first to generate initial parameters")
            return None
        
        if model_type not in self.simulation_results:
            print(f"❌ No simulation data found for {model_type}")
            return None
        
        # Get data and Lomb-Scargle results
        data = self.simulation_results[model_type]
        ls_result = self.analysis_results[model_type]
        
        phi_ext = data['Phi_ext']
        I_s = data['I_s']
        I_s_error = data.get('I_s_error', None)
        
        print(f"\n🚀 Fitting Complete Josephson Equation for {data['model_name']}")
        print("="*70)
        print(f"   Using Lomb-Scargle results as initial parameters")
        print(f"   Method: {'L-BFGS-B' if use_lbfgsb else 'default lmfit'}")
        
        # Create Josephson fitter
        fitter = JosephsonFitter()
        
        # Perform the fit
        method = 'lbfgsb' if use_lbfgsb else 'leastsq'
        fit_result = fitter.fit(
            phi_ext=phi_ext,
            I_s=I_s,
            I_s_error=I_s_error,
            lomb_scargle_result=ls_result,
            method=method
        )
        
        if fit_result is None:
            print("❌ Fitting failed")
            return None
        
        # Print fit results
        fitted_params = fitter.get_fitted_parameters()
        print(f"\n📊 Complete Josephson Equation Fit Results:")
        print("-"*50)
        for param_name, param_info in fitted_params.items():
            initial_val = param_info['initial']
            fitted_val = param_info['value']
            stderr = param_info['stderr']
            
            print(f"   {param_name}:")
            print(f"      Initial:  {initial_val:.6f}")
            print(f"      Fitted:   {fitted_val:.6f} ± {stderr:.6f}")
            if param_name == 'f' and 'f' in data['parameters']:
                true_val = data['parameters']['f']
                error = abs(fitted_val - true_val)
                rel_error = error / true_val * 100
                print(f"      True:     {true_val:.6f}")
                print(f"      Error:    {error:.6f} ({rel_error:.2f}%)")
        
        # Generate plots
        print(f"\n📈 Generating complete Josephson fit plots...")
        stats = fitter.plot_fit_results(phi_ext, I_s, save_plot=True, 
                                       filename_suffix=f"_{model_type}")
        
        # Save results if requested
        if save_results:
            print(f"\n💾 Saving complete Josephson fit results...")
            data_file, param_file = fitter.save_fit_results_to_csv(
                phi_ext, I_s, filename_suffix=f"_{model_type}")
        
        # Store fitter object for future use
        if not hasattr(self, 'josephson_fitters'):
            self.josephson_fitters = {}
        self.josephson_fitters[model_type] = fitter
        
        return fitter
    
    def compare_lomb_scargle_vs_josephson_fit(self, model_type, save_plot=True):
        """
        Compare Lomb-Scargle analysis results with complete Josephson equation fit
        
        Args:
            model_type: Model type to compare
            save_plot: Whether to save the comparison plot
        """
        if model_type not in self.analysis_results:
            print(f"❌ No Lomb-Scargle analysis results found for {model_type}")
            return
        
        if not hasattr(self, 'josephson_fitters') or model_type not in self.josephson_fitters:
            print(f"❌ No complete Josephson fit results found for {model_type}")
            print("   Please run fit_complete_josephson_equation() first")
            return
        
        data = self.simulation_results[model_type]
        ls_result = self.analysis_results[model_type]
        fitter = self.josephson_fitters[model_type]
        
        phi_ext = data['Phi_ext']
        I_s = data['I_s']
        
        # Calculate fitted curves
        ls_fit = ls_result['ls_model']
        josephson_fit = fitter.calculate_fitted_curve(phi_ext)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Lomb-Scargle vs Complete Josephson Fit: {data["model_name"]}', fontsize=16)
        
        # 1. Data with both fits
        ax1 = axes[0, 0]
        ax1.plot(phi_ext, I_s, 'k.', alpha=0.4, label='Data', markersize=2)
        ax1.plot(phi_ext, ls_fit, 'b-', linewidth=2, label='Lomb-Scargle Fit', alpha=0.8)
        ax1.plot(phi_ext, josephson_fit, 'r-', linewidth=2, label='Complete Josephson Fit', alpha=0.8)
        ax1.set_xlabel('External Flux (Φ_ext)')
        ax1.set_ylabel('Supercurrent (I_s)')
        ax1.set_title('Data and Model Fits')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuals comparison
        ax2 = axes[0, 1]
        ls_residuals = I_s - ls_fit
        josephson_residuals = I_s - josephson_fit
        ax2.plot(phi_ext, ls_residuals, 'b.', alpha=0.6, label='LS Residuals', markersize=3)
        ax2.plot(phi_ext, josephson_residuals, 'r.', alpha=0.6, label='Josephson Residuals', markersize=3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('External Flux (Φ_ext)')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Parameter comparison
        ax3 = axes[1, 0]
        fitted_params = fitter.get_fitted_parameters()
        
        # Compare frequencies and amplitudes
        ls_freq = ls_result['best_frequency']
        ls_amp = ls_result['amplitude']
        josephson_freq = fitted_params['f']['value']
        josephson_amp = fitted_params['I_c']['value']
        
        param_comparison = {
            'Frequency': [ls_freq, josephson_freq],
            'Amplitude/I_c': [ls_amp, josephson_amp]
        }
        
        x_pos = np.arange(len(param_comparison))
        width = 0.35
        
        ls_values = [param_comparison['Frequency'][0], param_comparison['Amplitude/I_c'][0]]
        josephson_values = [param_comparison['Frequency'][1], param_comparison['Amplitude/I_c'][1]]
        
        ax3.bar(x_pos - width/2, ls_values, width, label='Lomb-Scargle', alpha=0.8)
        ax3.bar(x_pos + width/2, josephson_values, width, label='Complete Josephson', alpha=0.8)
        ax3.set_xlabel('Parameters')
        ax3.set_ylabel('Values')
        ax3.set_title('Key Parameter Comparison')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(list(param_comparison.keys()))
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Statistical comparison
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate statistics for both fits
        ls_stats = ls_result['statistics']
        josephson_stats = ModelStatistics(
            y_true=I_s,
            y_pred=josephson_fit,
            n_params=len(fitted_params),
            model_name="Complete Josephson"
        )
        
        stats_text = f"""
Comparison Statistics:

Lomb-Scargle Analysis:
  R²: {ls_stats.r_squared:.6f}
  RMSE: {ls_stats.rmse:.6e}
  MAE: {ls_stats.mae:.6e}
  SNR (Power): {ls_stats.snr_power:.2f} ({ls_stats.snr_db:.1f} dB)
  SNR (RMS): {ls_stats.snr_rms:.2f} ({ls_stats.snr_rms_db:.1f} dB)

Complete Josephson Fit:
  R²: {josephson_stats.r_squared:.6f}
  RMSE: {josephson_stats.rmse:.6e}
  MAE: {josephson_stats.mae:.6e}
  SNR (Power): {josephson_stats.snr_power:.2f} ({josephson_stats.snr_db:.1f} dB)
  SNR (RMS): {josephson_stats.snr_rms:.2f} ({josephson_stats.snr_rms_db:.1f} dB)
  Chi-square: {fitter.result.chisqr:.6f}
  AIC: {fitter.result.aic:.2f}
  BIC: {fitter.result.bic:.2f}

Improvement:
  ΔR²: {josephson_stats.r_squared - ls_stats.r_squared:.6f}
  ΔRMSE: {ls_stats.rmse - josephson_stats.rmse:.6e}
  ΔSNR (dB): {josephson_stats.snr_db - ls_stats.snr_db:.1f}

Josephson Parameters:
  I_c = {fitted_params['I_c']['value']:.6f} ± {fitted_params['I_c']['stderr']:.6f}
  f = {fitted_params['f']['value']:.6f} ± {fitted_params['f']['stderr']:.6f}
  T = {fitted_params['T']['value']:.6f} ± {fitted_params['T']['stderr']:.6f}
  φ₀ = {fitted_params['phi_0']['value']:.6f} ± {fitted_params['phi_0']['stderr']:.6f}
"""
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plot:
            filename = f'lomb_scargle_vs_josephson_comparison_{model_type}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Comparison plot saved: {filename}")
        
        plt.show()
        
        return josephson_stats

    # ...existing code...
def analyze_real_simulation_data():
    """Analyze real simulation data from CSV files"""
    print("🔬 Analyzing Real Simulation Data")
    print("="*50)
    
    # Load simulation data and parameters
    try:
        sim_data = load_simulation_data('simulation_results.csv')
        sim_params = load_simulation_parameters('simulation_parameters.csv')
        
        print(f"✅ Loaded simulation data: {len(sim_data)} data points")
        print(f"   Phi_ext range: {sim_data['Phi_ext'].min():.2e} to {sim_data['Phi_ext'].max():.2e}")
        print(f"   I_s range: {sim_data['I_s'].min():.2e} to {sim_data['I_s'].max():.2e}")
        print(f"   True frequency: {sim_params['f'].iloc[0]:.2e}")
        
        # Create analyzer
        analyzer = JosephsonAnalyzer()
        
        # Prepare data
        data = {
            'Phi_ext': sim_data['Phi_ext'].values,
            'I_s': sim_data['I_s'].values,
            'I_s_error': np.full_like(sim_data['I_s'].values, 1e-7),  # Estimated noise level
            'true_params': {
                'Ic': sim_params['Ic'].iloc[0],  # Note: CSV uses 'Ic' not 'I_c'
                'f': sim_params['f'].iloc[0],
                'phi_0': sim_params['phi_0'].iloc[0],
                'T': sim_params['T'].iloc[0],
                'd': sim_params['d'].iloc[0],
                'r': sim_params['r'].iloc[0],
                'C': sim_params['C'].iloc[0]
            }
        }
        
        parameters = {
            'f': sim_params['f'].iloc[0],
            'Ic': sim_params['Ic'].iloc[0],  # Note: CSV uses 'Ic' not 'I_c'
            'phi_0': sim_params['phi_0'].iloc[0],
            'T': sim_params['T'].iloc[0]
        }
        
        # Add to analyzer
        analyzer.add_simulation_data(
            model_type='real_josephson_sim',
            data=data,
            parameters=parameters,
            model_name='Real Josephson Simulation'
        )
        
        # Perform analysis with 1st-order polynomial detrending
        print("\n🔧 Performing Lomb-Scargle analysis with 1st-order detrending...")
        result1 = analyzer.analyze_with_lomb_scargle('real_josephson_sim', detrend_order=1)
        
        if result1:
            # Plot results
            analyzer.plot_analysis_results('real_josephson_sim', save_plot=True)
            
            # 顯示並保存去趨勢化後的數據
            print("\n📊 Generating detrended data visualization and CSV...")
            analyzer.plot_detrended_data_comparison('real_josephson_sim', save_plot=True)
            analyzer.save_detrended_data_to_csv('real_josephson_sim')
            
        # Also analyze sine simulation data
        print("\n" + "="*50)
        print("🔬 Analyzing Sine Simulation Data")
        
        sine_data = load_simulation_data('sine_simulation_results.csv')
        sine_params = load_simulation_parameters('sine_simulation_parameters.csv')
        
        print(f"✅ Loaded sine data: {len(sine_data)} data points")
        print(f"   Phi_ext range: {sine_data['Phi_ext'].min():.2e} to {sine_data['Phi_ext'].max():.2e}")
        print(f"   I_s range: {sine_data['I_s'].min():.2e} to {sine_data['I_s'].max():.2e}")
        print(f"   True frequency: {sine_params['f'].iloc[0]:.2e}")
        
        # Prepare sine data
        sine_data_dict = {
            'Phi_ext': sine_data['Phi_ext'].values,
            'I_s': sine_data['I_s'].values,
            'I_s_error': np.full_like(sine_data['I_s'].values, 1e-7),
            'true_params': {
                'Ic': sine_params['Ic'].iloc[0],  # Note: CSV uses 'Ic' not 'I_c'
                'f': sine_params['f'].iloc[0],
                'phi_0': sine_params['phi_0'].iloc[0],
                'T': 0.5,  # Default value since T is empty in sine_simulation_parameters.csv
                'd': sine_params['d'].iloc[0],
                'r': sine_params['r'].iloc[0],
                'C': sine_params['C'].iloc[0]
            }
        }
        
        sine_parameters = {
            'f': sine_params['f'].iloc[0],
            'Ic': sine_params['Ic'].iloc[0],  # Note: CSV uses 'Ic' not 'I_c'
            'phi_0': sine_params['phi_0'].iloc[0],
            'T': 0.5  # Default value since T is empty in sine_simulation_parameters.csv
        }
        
        # Add to analyzer
        analyzer.add_simulation_data(
            model_type='sine_josephson_sim',
            data=sine_data_dict,
            parameters=sine_parameters,
            model_name='Sine Josephson Simulation'
        )
        
        # Perform analysis with 1st order
        sine_result1 = analyzer.analyze_with_lomb_scargle('sine_josephson_sim', detrend_order=1)
        
        if sine_result1:
            # Plot results
            analyzer.plot_analysis_results('sine_josephson_sim', save_plot=True)
            
        # Generate final comparison report
        analyzer.generate_comparison_report()
        
    except Exception as e:
        print(f"❌ Error analyzing simulation data: {e}")
        return None


def main():
    """Main function - Demonstrate Lomb-Scargle analysis"""
    print("🚀 Josephson Junction Lomb-Scargle Analysis Demo")
    print("="*60)
    
    # First analyze real simulation data
    analyze_real_simulation_data()
    
    print("\n" + "="*60)
    print("🧪 Example Data Analysis")
    print("="*60)
    
    # Create analyzer
    analyzer = JosephsonAnalyzer()
    
    # Generate example data
    print("\n📊 Generating example Josephson junction data...")
    data, parameters = create_example_josephson_data()
    
    # Add to analyzer
    analyzer.add_simulation_data(
        model_type='example_josephson',
        data=data,
        parameters=parameters,
        model_name='Example Josephson Junction'
    )
    
    # Perform Lomb-Scargle analysis with 1st order
    print("\n🔬 Performing Lomb-Scargle analysis with 1st-order detrending...")
    result1 = analyzer.analyze_with_lomb_scargle('example_josephson', detrend_order=1)
    
    # Plot results
    if result1:
        print("\n📈 Generating analysis plots...")
        analyzer.plot_analysis_results('example_josephson', save_plot=True)
    
    # Demonstrate standalone function
    print("\n🧪 Demonstrating standalone analysis function...")
    standalone_result = analyze_with_lomb_scargle_standalone(
        times=data['Phi_ext'],
        values=data['I_s'],
        errors=data['I_s_error'],
        true_frequency=parameters['f'],
        model_name='Standalone Analysis Example'
    )
    
    # Generate comprehensive comparison report
    print("\n📊 Generating comprehensive comparison report...")
    analyzer.generate_comparison_report()
    
    print("\n✅ Analysis completed!")


if __name__ == "__main__":
    main()
