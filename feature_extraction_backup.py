#!/usr/bin/env python3
"""
從實驗數據中提取特徵值：結合圖像分析與卷積/去卷積技術
基於 README.md 中描述的方法實現
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, optimize, ndimage
from scipy.sparse import diags
from scipy.linalg import solve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class SuperconductorAnalyzer:
    """超導體數據分析器"""
    
    def __init__(self, data_path):
        """初始化分析器
        
        Args:
            data_path: 數據文件路徑
        """
        self.data_path = data_path
        self.data = None
        self.y_field_values = None
        self.features = {}
        self.images = {}
        
    def load_data(self):
        """載入和預處理數據"""
        print("=== 步驟1: 數據預處理與清洗 ===")
        
        # 讀取數據
        self.data = pd.read_csv(self.data_path)
        print(f"數據形狀: {self.data.shape}")
        
        # 檢查缺失值
        missing_values = self.data.isnull().sum()
        if missing_values.any():
            print("缺失值:")
            print(missing_values[missing_values > 0])
            # 移除缺失值
            self.data = self.data.dropna()
            print(f"清理後數據形狀: {self.data.shape}")
        
        # 檢查異常值
        self._detect_outliers()
        
        # 獲取y_field的唯一值
        self.y_field_values = sorted(self.data['y_field'].unique())
        print(f"y_field 唯一值數量: {len(self.y_field_values)}")
        print(f"y_field 範圍: {self.y_field_values[0]:.6f} - {self.y_field_values[-1]:.6f}")
        
    def _detect_outliers(self):
        """檢測異常值"""
        print("\n檢測異常值:")
        
        for col in ['meas_voltage_K2', 'dV_dI']:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
            print(f"{col}: {len(outliers)} 個異常值 ({len(outliers)/len(self.data)*100:.2f}%)")
            
            # 可選：移除極端異常值（超過3個標準差）
            mean_val = self.data[col].mean()
            std_val = self.data[col].std()
            extreme_outliers = self.data[abs(self.data[col] - mean_val) > 3 * std_val]
            if len(extreme_outliers) > 0:
                print(f"  極端異常值: {len(extreme_outliers)} 個")
    
    def extract_conventional_features(self):
        """步驟2: 常規特徵提取"""
        print("\n=== 步驟2: 常規特徵提取 ===")
        
        features_list = []
        
        for y_field in self.y_field_values:
            # 獲取當前y_field的數據
            field_data = self.data[self.data['y_field'] == y_field].copy()
            field_data = field_data.sort_values('appl_current')
            
            # 提取特徵
            features = self._extract_features_for_field(field_data, y_field)
            features_list.append(features)
            
            # 每10個y_field值顯示一次進度
            if len(features_list) % 10 == 0:
                print(f"已處理 {len(features_list)}/{len(self.y_field_values)} 個y_field值")
        
        # 轉換為DataFrame
        self.features = pd.DataFrame(features_list)
        print(f"\n提取的特徵維度: {self.features.shape}")
        print("提取的特徵:", list(self.features.columns))
        
        # 顯示統計信息
        self._display_feature_statistics()
        
    def _extract_features_for_field(self, field_data, y_field):
        """為單個y_field值提取特徵"""
        I = field_data['appl_current'].values
        V = field_data['meas_voltage_K2'].values
        dV_dI = field_data['dV_dI'].values
        
        features = {'y_field': y_field}
        
        # 1. 臨界電流 (Ic) - 使用dV/dI峰值方法
        try:
            # 找到dV/dI的峰值位置
            positive_peaks, _ = signal.find_peaks(dV_dI, height=np.percentile(dV_dI, 90))
            negative_peaks, _ = signal.find_peaks(-dV_dI, height=np.percentile(-dV_dI, 90))
            
            if len(positive_peaks) > 0:
                max_peak_idx = positive_peaks[np.argmax(dV_dI[positive_peaks])]
                features['Ic_positive'] = abs(I[max_peak_idx])
                features['dV_dI_max'] = dV_dI[max_peak_idx]
            else:
                features['Ic_positive'] = np.nan
                features['dV_dI_max'] = np.nan
                
            if len(negative_peaks) > 0:
                max_neg_peak_idx = negative_peaks[np.argmax(-dV_dI[negative_peaks])]
                features['Ic_negative'] = abs(I[max_neg_peak_idx])
            else:
                features['Ic_negative'] = np.nan
                
            # 平均臨界電流
            if not np.isnan(features['Ic_positive']) and not np.isnan(features['Ic_negative']):
                features['Ic_average'] = (features['Ic_positive'] + features['Ic_negative']) / 2
            elif not np.isnan(features['Ic_positive']):
                features['Ic_average'] = features['Ic_positive']
            elif not np.isnan(features['Ic_negative']):
                features['Ic_average'] = features['Ic_negative']
            else:
                features['Ic_average'] = np.nan
                
        except Exception as e:
            features.update({
                'Ic_positive': np.nan,
                'Ic_negative': np.nan,
                'Ic_average': np.nan,
                'dV_dI_max': np.nan
            })
        
        # 2. 正常態電阻 (Rn) - 高電流區域的dV/dI平台值
        try:
            high_current_threshold = 0.8 * np.max(np.abs(I))
            high_current_mask = np.abs(I) > high_current_threshold
            if np.sum(high_current_mask) > 5:
                features['Rn'] = np.median(dV_dI[high_current_mask])
            else:
                features['Rn'] = np.median(dV_dI[np.abs(I) > 0.5 * np.max(np.abs(I))])
        except:
            features['Rn'] = np.nan
        
        # 3. n值 - 通過冪律擬合估算
        try:
            # 選擇轉變區域的數據
            transition_mask = (np.abs(I) > 0.1 * np.max(np.abs(I))) & (np.abs(I) < 0.9 * np.max(np.abs(I)))
            if np.sum(transition_mask) > 10:
                I_trans = I[transition_mask]
                V_trans = V[transition_mask]
                
                # 過濾掉零或負電壓
                positive_V_mask = V_trans > 0
                if np.sum(positive_V_mask) > 5:
                    I_fit = np.abs(I_trans[positive_V_mask])
                    V_fit = V_trans[positive_V_mask]
                    
                    # 對數-對數擬合
                    log_I = np.log10(I_fit + 1e-12)
                    log_V = np.log10(V_fit + 1e-12)
                    
                    # 線性擬合
                    coeffs = np.polyfit(log_I, log_V, 1)
                    features['n_value'] = coeffs[0]  # 斜率即為n值
                else:
                    features['n_value'] = np.nan
            else:
                features['n_value'] = np.nan
        except:
            features['n_value'] = np.nan
        
        # 4. 轉變寬度 - dV/dI峰的半峰全寬
        try:
            if not np.isnan(features['dV_dI_max']):
                peak_height = features['dV_dI_max']
                half_height = peak_height / 2
                
                # 找到半峰全寬
                above_half = dV_dI > half_height
                if np.sum(above_half) > 1:
                    indices = np.where(above_half)[0]
                    width_current = I[indices[-1]] - I[indices[0]]
                    features['transition_width'] = abs(width_current)
                else:
                    features['transition_width'] = np.nan
            else:
                features['transition_width'] = np.nan
        except:
            features['transition_width'] = np.nan
        
        # 5. 統計特徵
        features['dV_dI_mean'] = np.mean(dV_dI)
        features['dV_dI_std'] = np.std(dV_dI)
        features['dV_dI_skewness'] = self._calculate_skewness(dV_dI)
        features['dV_dI_kurtosis'] = self._calculate_kurtosis(dV_dI)
        
        # 6. 電壓偏移
        zero_current_idx = np.argmin(np.abs(I))
        features['voltage_offset'] = V[zero_current_idx]
        
        return features
    
    def _calculate_skewness(self, data):
        """計算偏度"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return np.mean(((data - mean) / std) ** 3)
        except:
            return np.nan
    
    def _calculate_kurtosis(self, data):
        """計算峰度"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return np.mean(((data - mean) / std) ** 4) - 3
        except:
            return np.nan
    
    def _display_feature_statistics(self):
        """顯示特徵統計信息"""
        print("\n特徵統計信息:")
        numeric_features = self.features.select_dtypes(include=[np.number])
        
        for col in numeric_features.columns:
            if col != 'y_field':
                valid_data = numeric_features[col].dropna()
                if len(valid_data) > 0:
                    print(f"{col}: 均值={valid_data.mean():.6e}, 標準差={valid_data.std():.6e}, "
                          f"有效值={len(valid_data)}/{len(numeric_features)}")
    
    def create_2d_images(self):
        """步驟3: 用於圖像分析的數據轉換"""
        print("\n=== 步驟3: 用於圖像分析的數據轉換 ===")
        
        # 創建二維網格
        y_field_grid = sorted(self.data['y_field'].unique())
        current_grid = sorted(self.data['appl_current'].unique())
        
        print(f"網格大小: {len(y_field_grid)} x {len(current_grid)}")
        
        # 創建二維圖像
        voltage_image = np.full((len(y_field_grid), len(current_grid)), np.nan)
        dV_dI_image = np.full((len(y_field_grid), len(current_grid)), np.nan)
        
        # 填充圖像數據
        for i, y_field in enumerate(y_field_grid):
            field_data = self.data[self.data['y_field'] == y_field]
            
            for j, current in enumerate(current_grid):
                # 找到最接近的電流值
                current_data = field_data[
                    np.abs(field_data['appl_current'] - current) < 1e-9
                ]
                
                if len(current_data) > 0:
                    voltage_image[i, j] = current_data['meas_voltage_K2'].iloc[0]
                    dV_dI_image[i, j] = current_data['dV_dI'].iloc[0]
        
        # 儲存圖像
        self.images = {
            'voltage': voltage_image,
            'dV_dI': dV_dI_image,
            'y_field_grid': np.array(y_field_grid),
            'current_grid': np.array(current_grid)
        }
        
        print(f"電壓圖像: {voltage_image.shape}, 有效像素: {np.sum(~np.isnan(voltage_image))}")
        print(f"dV/dI圖像: {dV_dI_image.shape}, 有效像素: {np.sum(~np.isnan(dV_dI_image))}")
        
        # 基本圖像處理
        self._apply_image_processing()
        
    def _apply_image_processing(self):
        """應用基本圖像處理技術"""
        print("\n應用圖像處理技術:")
        
        # 處理NaN值 - 使用鄰近像素插值
        for key in ['voltage', 'dV_dI']:
            image = self.images[key].copy()
            
            # 使用中值濾波降噪
            # 首先用線性插值填充NaN
            mask = ~np.isnan(image)
            if np.sum(mask) > 0:
                from scipy.interpolate import griddata
                rows, cols = np.mgrid[0:image.shape[0], 0:image.shape[1]]
                
                # 獲取有效點
                valid_points = np.column_stack((rows[mask], cols[mask]))
                valid_values = image[mask]
                
                # 插值到所有點
                if len(valid_points) > 3:
                    interpolated = griddata(
                        valid_points, valid_values, 
                        (rows, cols), method='linear', fill_value=np.nanmean(valid_values)
                    )
                    
                    # 應用中值濾波
                    filtered = ndimage.median_filter(interpolated, size=3)
                    self.images[f'{key}_filtered'] = filtered
                    
                    print(f"{key} 圖像已處理 (形狀: {filtered.shape})")
    
    def apply_deconvolution(self):
        """步驟4: 去卷積（可選但建議用於細化）"""
        print("\n=== 步驟4: 去卷積處理 ===")
        
        # Tikhonov正則化去卷積
        self._tikhonov_deconvolution()
        
    def _tikhonov_deconvolution(self):
        """Tikhonov正則化方法"""
        print("應用Tikhonov正則化:")
        
        regularization_params = [1e-6, 1e-5, 1e-4, 1e-3]
        self.deconvolved_results = {}
        success_count = 0
        
        # 選擇代表性的y_field值進行去卷積
        sample_fields = self.y_field_values[::100]  # 減少採樣以提高速度
        
        for i, y_field in enumerate(sample_fields):
            field_data = self.data[self.data['y_field'] == y_field].copy()
            field_data = field_data.sort_values('appl_current')
            
            V = field_data['meas_voltage_K2'].values
            I = field_data['appl_current'].values
            dV_dI_measured = field_data['dV_dI'].values
            
            if len(V) > 10:  # 確保有足夠的數據點
                # 平滑處理原始dV/dI數據
                window_size = min(5, len(dV_dI_measured) // 4)
                if window_size >= 3:
                    smoothed_dV_dI = signal.savgol_filter(dV_dI_measured, window_size, 2)
                    
                    # 構建模糊核（儀器響應函數的近似）
                    kernel_size = 5
                    kernel = signal.gaussian(kernel_size, std=1.0)
                    kernel = kernel / np.sum(kernel)
                    
                    # 使用最適合的正則化參數
                    eta = 1e-4
                    try:
                        # Wiener去卷積
                        deconvolved = self._wiener_deconvolution(smoothed_dV_dI, kernel, eta)
                        
                        self.deconvolved_results[y_field] = {
                            'original': dV_dI_measured,
                            'smoothed': smoothed_dV_dI,
                            'deconvolved': deconvolved,
                            'current': I,
                            'voltage': V,
                            'eta': eta
                        }
                        success_count += 1
                        
                        if success_count == 1:  # 報告第一個成功案例
                            print(f"  ✓ y_field={y_field:.6f}, η={eta}")
                        
                    except Exception as e:
                        print(f"  ✗ y_field={y_field:.6f}: {str(e)}")
                        continue
        
        print(f"去卷積處理完成: {success_count}/{len(sample_fields)} 成功")
    
    def _wiener_deconvolution(self, signal_data, kernel, noise_variance):
        """Wiener去卷積方法"""
        # 頻域實現
        signal_fft = np.fft.fft(signal_data)
        kernel_fft = np.fft.fft(kernel, len(signal_data))
        
        # Wiener濾波器
        power_spectrum = np.abs(kernel_fft) ** 2
        wiener_filter = np.conj(kernel_fft) / (power_spectrum + noise_variance)
        
        # 去卷積
        deconvolved_fft = signal_fft * wiener_filter
        deconvolved = np.real(np.fft.ifft(deconvolved_fft))
        
        return deconvolved
    
    def _build_difference_matrix(self, n, order=1):
        """構建差分矩陣"""
        if order == 1:
            return diags([1, -1], [0, 1], shape=(n-1, n)).toarray()
        elif order == 2:
            return diags([1, -2, 1], [0, 1, 2], shape=(n-2, n)).toarray()
        else:
            raise ValueError("只支持1階和2階差分")
    
    def _solve_tikhonov(self, A, b, D, eta):
        """求解Tikhonov正則化問題"""
        # 求解 (A^T A + η D^T D) x = A^T b
        AtA = A.T @ A
        DtD = D.T @ D
        Atb = A.T @ b
        
        return solve(AtA + eta * DtD, Atb)
    
    def visualize_results(self):
        """可視化結果"""
        print("\n=== 結果可視化 ===")
        
        # 創建圖形
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 特徵隨y_field的變化
        plt.subplot(3, 4, 1)
        if 'Ic_average' in self.features.columns:
            valid_data = self.features.dropna(subset=['Ic_average'])
            if len(valid_data) > 0:
                plt.plot(valid_data['y_field'], valid_data['Ic_average'], 'b.-')
                plt.xlabel('y_field')
                plt.ylabel('臨界電流 Ic (A)')
                plt.title('臨界電流隨y_field變化')
                plt.grid(True)
        
        plt.subplot(3, 4, 2)
        if 'Rn' in self.features.columns:
            valid_data = self.features.dropna(subset=['Rn'])
            if len(valid_data) > 0:
                plt.plot(valid_data['y_field'], valid_data['Rn'], 'r.-')
                plt.xlabel('y_field')
                plt.ylabel('正常態電阻 Rn (Ω)')
                plt.title('正常態電阻隨y_field變化')
                plt.grid(True)
        
        plt.subplot(3, 4, 3)
        if 'n_value' in self.features.columns:
            valid_data = self.features.dropna(subset=['n_value'])
            if len(valid_data) > 0:
                plt.plot(valid_data['y_field'], valid_data['n_value'], 'g.-')
                plt.xlabel('y_field')
                plt.ylabel('n值')
                plt.title('n值隨y_field變化')
                plt.grid(True)
        
        plt.subplot(3, 4, 4)
        if 'transition_width' in self.features.columns:
            valid_data = self.features.dropna(subset=['transition_width'])
            if len(valid_data) > 0:
                plt.plot(valid_data['y_field'], valid_data['transition_width'], 'm.-')
                plt.xlabel('y_field')
                plt.ylabel('轉變寬度 (A)')
                plt.title('轉變寬度隨y_field變化')
                plt.grid(True)
        
        # 2. 二維圖像
        if 'dV_dI_filtered' in self.images:
            plt.subplot(3, 4, 5)
            im1 = plt.imshow(self.images['dV_dI_filtered'], 
                           extent=[self.images['current_grid'][0]*1e6, 
                                  self.images['current_grid'][-1]*1e6,
                                  self.images['y_field_grid'][0], 
                                  self.images['y_field_grid'][-1]],
                           aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(im1, label='dV/dI (Ω)')
            plt.xlabel('施加電流 (µA)')
            plt.ylabel('y_field')
            plt.title('dV/dI 二維圖 (濾波後)')
        
        if 'voltage_filtered' in self.images:
            plt.subplot(3, 4, 6)
            im2 = plt.imshow(self.images['voltage_filtered'], 
                           extent=[self.images['current_grid'][0]*1e6, 
                                  self.images['current_grid'][-1]*1e6,
                                  self.images['y_field_grid'][0], 
                                  self.images['y_field_grid'][-1]],
                           aspect='auto', origin='lower', cmap='RdBu_r')
            plt.colorbar(im2, label='電壓 (V)')
            plt.xlabel('施加電流 (µA)')
            plt.ylabel('y_field')
            plt.title('電壓 二維圖 (濾波後)')
        
        # 3. 樣本I-V和dV/dI曲線
        plt.subplot(3, 4, 7)
        sample_y_field = self.y_field_values[len(self.y_field_values)//2]
        sample_data = self.data[self.data['y_field'] == sample_y_field].sort_values('appl_current')
        plt.plot(sample_data['appl_current']*1e6, sample_data['meas_voltage_K2']*1e3, 'b-')
        plt.xlabel('施加電流 (µA)')
        plt.ylabel('測量電壓 (mV)')
        plt.title(f'I-V曲線 (y_field={sample_y_field:.6f})')
        plt.grid(True)
        
        plt.subplot(3, 4, 8)
        plt.plot(sample_data['appl_current']*1e6, sample_data['dV_dI'], 'r-')
        plt.xlabel('施加電流 (µA)')
        plt.ylabel('dV/dI (Ω)')
        plt.title(f'dV/dI曲線 (y_field={sample_y_field:.6f})')
        plt.grid(True)
        
        # 4. 特徵分布
        plt.subplot(3, 4, 9)
        if 'Ic_average' in self.features.columns:
            valid_ic = self.features['Ic_average'].dropna()
            if len(valid_ic) > 0:
                plt.hist(valid_ic*1e6, bins=30, alpha=0.7)
                plt.xlabel('臨界電流 (µA)')
                plt.ylabel('頻次')
                plt.title('臨界電流分布')
        
        plt.subplot(3, 4, 10)
        if 'Rn' in self.features.columns:
            valid_rn = self.features['Rn'].dropna()
            if len(valid_rn) > 0:
                plt.hist(valid_rn, bins=30, alpha=0.7, color='orange')
                plt.xlabel('正常態電阻 (Ω)')
                plt.ylabel('頻次')
                plt.title('正常態電阻分布')
        
        # 5. 去卷積結果（如果有）
        if hasattr(self, 'deconvolved_results') and self.deconvolved_results:
            plt.subplot(3, 4, 11)
            # 取第一個成功的去卷積結果
            first_result = next(iter(self.deconvolved_results.values()))
            
            plt.plot(first_result['current']*1e6, first_result['original'], 'b-', 
                    label='原始dV/dI', alpha=0.7, linewidth=1)
            plt.plot(first_result['current']*1e6, first_result['smoothed'], 'g-', 
                    label='平滑dV/dI', alpha=0.8, linewidth=1.5)
            plt.plot(first_result['current']*1e6, first_result['deconvolved'], 'r-', 
                    label='去卷積dV/dI', alpha=0.9, linewidth=2)
            
            plt.xlabel('施加電流 (µA)')
            plt.ylabel('dV/dI (Ω)')
            plt.title('去卷積比較')
            plt.legend(fontsize=8)
            plt.grid(True)
        else:
            plt.subplot(3, 4, 11)
            plt.text(0.5, 0.5, '去卷積處理\n未完成', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=12)
            plt.title('去卷積結果')
        
        # 6. 機器學習特徵（如果有）
        if hasattr(self, 'ml_features') and self.ml_features.get('pca') is not None:
            plt.subplot(3, 4, 12)
            pca_data = self.ml_features['pca']
            if pca_data.shape[1] >= 2:
                scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], 
                                    c=self.features['y_field'], cmap='viridis', alpha=0.7)
                plt.colorbar(scatter, label='y_field')
                plt.xlabel(f'PC1 ({self.pca_info["explained_variance_ratio"][0]:.1%})')
                                plt.ylabel(f'PC2 ({self.pca_info["explained_variance_ratio"][1]:.1%})')
                plt.title('PCA特徵空間')
                plt.grid(True)
            else:
                plt.text(0.5, 0.5, 'PCA特徵\n維度不足', ha='center', va='center', 
                        transform=plt.gca().transAxes, fontsize=12)
                plt.title('PCA結果')
        else:
            plt.subplot(3, 4, 12)
            plt.text(0.5, 0.5, '機器學習特徵\n提取未完成', ha='center', va='center', 
                    transform=plt.gca().transAxes, fontsize=12)
            plt.title('機器學習特徵')
        
        # 調整布局並保存
        plt.tight_layout()
        plt.savefig('analysis_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("分析結果已保存到 analysis_results.png")
                plt.title('特徵相關性')
        
        plt.tight_layout()
        plt.savefig('/workspaces/IVB/analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("分析結果已保存到 analysis_results.png")
    
    def generate_summary_report(self):
        """生成總結報告"""
        print("\n" + "="*50)
        print("         超導體數據分析總結報告")
        print("="*50)
        
        print(f"\n數據概況:")
        print(f"  總數據點: {len(self.data)}")
        print(f"  y_field範圍: {self.y_field_values[0]:.6f} - {self.y_field_values[-1]:.6f}")
        print(f"  y_field步數: {len(self.y_field_values)}")
        print(f"  電流範圍: {self.data['appl_current'].min()*1e6:.2f} - {self.data['appl_current'].max()*1e6:.2f} µA")
        
        print(f"\n提取特徵統計:")
        if len(self.features) > 0:
            for feature in ['Ic_average', 'Rn', 'n_value', 'transition_width']:
                if feature in self.features.columns:
                    valid_data = self.features[feature].dropna()
                    if len(valid_data) > 0:
                        if feature == 'Ic_average':
                            print(f"  臨界電流: {valid_data.mean()*1e6:.3f} ± {valid_data.std()*1e6:.3f} µA")
                        elif feature == 'Rn':
                            print(f"  正常態電阻: {valid_data.mean():.2f} ± {valid_data.std():.2f} Ω")
                        elif feature == 'n_value':
                            print(f"  n值: {valid_data.mean():.2f} ± {valid_data.std():.2f}")
                        elif feature == 'transition_width':
                            print(f"  轉變寬度: {valid_data.mean()*1e6:.3f} ± {valid_data.std()*1e6:.3f} µA")
        
        print(f"\n圖像分析:")
        if self.images:
            print(f"  生成二維圖像: {list(self.images.keys())}")
            if 'dV_dI' in self.images:
                print(f"  dV/dI圖像大小: {self.images['dV_dI'].shape}")
        
        print(f"\n去卷積處理:")
        if hasattr(self, 'deconvolved_results'):
            print(f"  成功應用Tikhonov正則化")
            print(f"  正則化參數: η = {self.deconvolved_results['eta']}")
        else:
            print(f"  去卷積處理未完成")
        
        print("\n分析完成! 建議:")
        print("1. 檢查特徵隨y_field的變化趨勢")
        print("2. 分析二維圖像中的模式和結構")
        print("3. 考慮進一步的機器學習分析")
        print("4. 驗證物理解釋的合理性")
    
    def extract_ml_features(self):
        """步驟5: 機器學習特徵提取"""
        print("\n=== 步驟5: 機器學習特徵提取 ===")
        
        # 確保 features 是 DataFrame
        if not isinstance(self.features, pd.DataFrame):
            print("特徵數據不是DataFrame格式，跳過機器學習特徵提取")
            return
        
        # 準備機器學習輸入數據
        ml_features = self._prepare_ml_data()
        
        if ml_features.size > 0:
            # PCA特徵提取
            pca_features = self._extract_pca_features(ml_features)
            
            # 統計特徵分析
            stats_features = self._extract_statistical_features()
            
            # 合併所有ML特徵
            self.ml_features = {
                'pca': pca_features,
                'statistics': stats_features,
                'raw_features': ml_features
            }
            
            print(f"機器學習特徵提取完成: PCA({pca_features.shape[1]}維), 統計特徵({len(stats_features)}項)")
        else:
            print("無足夠數據進行機器學習特徵提取")
    
    def _prepare_ml_data(self):
        """準備機器學習數據"""
        # 從特徵數據中提取數值列
        numeric_features = []
        
        for col in self.features.columns:
            if col != 'y_field' and pd.api.types.is_numeric_dtype(self.features[col]):
                valid_data = self.features[col].dropna()
                if len(valid_data) > 0:
                    numeric_features.append(self.features[col].fillna(self.features[col].median()).values)
        
        if numeric_features:
            return np.column_stack(numeric_features)
        else:
            return np.array([])
    
    def _extract_pca_features(self, data):
        """PCA特徵提取"""
        if data.size == 0:
            return np.array([])
        
        # 標準化數據
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # PCA降維
        n_components = min(5, data.shape[1], data.shape[0])
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(data_scaled)
        
        # 儲存PCA信息
        self.pca_info = {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'components': pca.components_
        }
        
        print(f"  PCA累計解釋方差: {self.pca_info['cumulative_variance'][-1]:.3f}")
        
        return pca_features
    
    def _extract_statistical_features(self):
        """統計特徵分析"""
        stats = {}
        
        # 對每個y_field區間進行統計
        y_field_bins = pd.cut(self.features['y_field'], bins=10)
        
        for feature in ['Ic_average', 'Rn', 'n_value', 'dV_dI_max']:
            if feature in self.features.columns:
                trend = self._calculate_trend(self.features[feature])
                variability = self.features[feature].std() / self.features[feature].mean()
                
                stats[f'{feature}_trend'] = trend
                stats[f'{feature}_variability'] = variability
                
                # 分區統計
                try:
                    grouped = self.features.groupby(y_field_bins)[feature]
                    stats[f'{feature}_bin_variation'] = grouped.std().mean()
                except Exception:
                    stats[f'{feature}_bin_variation'] = 0
        
        return stats
    
    def _calculate_trend(self, series):
        """計算趨勢係數"""
        valid_data = series.dropna()
        if len(valid_data) < 2:
            return 0
        
        x = np.arange(len(valid_data))
        try:
            slope, _ = np.polyfit(x, valid_data, 1)
            return slope
        except Exception:
            return 0


def main():
    """主函數"""
    print("超導體數據特徵提取分析")
    print("基於README.md中描述的方法")
    print("-" * 50)
    
    # 初始化分析器
    analyzer = SuperconductorAnalyzer('/workspaces/IVB/164.csv')
    
    try:
        # 執行分析流程
        analyzer.load_data()
        analyzer.extract_conventional_features()
        analyzer.create_2d_images()
        analyzer.apply_deconvolution()
        analyzer.visualize_results()
        analyzer.generate_summary_report()
        analyzer.extract_ml_features()  # 添加機器學習特徵提取步驟
        
    except Exception as e:
        print(f"分析過程中出現錯誤: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
