#!/usr/bin/env python3
"""
從實驗數據中提取特徵值：結合圖像分析與卷積/去卷積技術
基於 README.md 中描述的方法實現 - 500.csv版本
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, ndimage
from ml_models import AutoencoderModel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
import warnings
warnings.filterwarnings('ignore')

# matplotlib 設定確保中文字體顯示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class SuperconductorAnalyzer:
    """超導體數據分析器"""
    
    def __init__(self, data_path):
        """初始化分析器
        
        Args:
            data_path: 數據文件路徑
        """
        self.data_path = data_path
        self.data = pd.DataFrame()  # 初始化為空DataFrame而不是None
        self.y_field_values = []    # 初始化為空list而不是None
        self.features = pd.DataFrame()  # 初始化為空DataFrame而不是dict
        self.images = {}
        
        # 自動檢測電壓列名稱
        self.voltage_column = None
        
    def load_data(self):
        """載入和預處理數據"""
        print("=== Step 1: Data Preprocessing and Cleaning ===")
        
        # 讀取數據
        self.data = pd.read_csv(self.data_path)
        print(f"Data shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        
        # 自動檢測電壓列
        voltage_candidates = [col for col in self.data.columns if 'voltage' in col.lower()]
        if voltage_candidates:
            self.voltage_column = voltage_candidates[0]
            print(f"Detected voltage column: {self.voltage_column}")
        else:
            raise ValueError("No voltage column found in data")
        
        # 檢查缺失值
        missing_values = self.data.isnull().sum()
        if missing_values.any():
            print("Missing values:")
            print(missing_values[missing_values > 0])
            # 移除缺失值
            self.data = self.data.dropna()
            print(f"Shape after cleaning: {self.data.shape}")
        
        # 檢查異常值
        print("\nDetecting outliers:")
        for col in [self.voltage_column, 'dV_dI']:
            if col in self.data.columns:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_mask = (self.data[col] < Q1 - 1.5*IQR) | (self.data[col] > Q3 + 1.5*IQR)
                outlier_count = outlier_mask.sum()
                print(f"{col}: {outlier_count} outliers ({outlier_count/len(self.data)*100:.2f}%)")
                
                # 極端異常值
                extreme_mask = (self.data[col] < Q1 - 3*IQR) | (self.data[col] > Q3 + 3*IQR)
                extreme_count = extreme_mask.sum()
                if extreme_count > 0:
                    print(f"  Extreme outliers: {extreme_count}")
        
        # 獲取y_field值
        self.y_field_values = sorted(self.data['y_field'].unique())
        print(f"y_field unique values: {len(self.y_field_values)}")
        print(f"y_field range: {self.y_field_values[0]:.6f} - {self.y_field_values[-1]:.6f}")
        
    def extract_conventional_features(self):
        """步驟2: 常規特徵提取"""
        print("\n=== Step 2: Conventional Feature Extraction ===")
        
        features_list = []
        
        for i, y_field in enumerate(self.y_field_values):
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{len(self.y_field_values)} y_field values")
            
            field_data = self.data[self.data['y_field'] == y_field].sort_values('appl_current')
            
            if len(field_data) == 0:
                continue
                
            features = self._extract_features_for_field(field_data, y_field)
            features_list.append(features)
        
        # 合併所有特徵
        self.features = pd.DataFrame(features_list)
        
        print(f"\nExtracted features dimensions: {self.features.shape}")
        print(f"Extracted features: {list(self.features.columns)}")
        
        # 顯示特徵統計
        print("\nFeature statistics:")
        numeric_features = self.features.select_dtypes(include=[np.number])
        
        for col in numeric_features.columns:
            if col != 'y_field':
                valid_data = numeric_features[col].dropna()
                if len(valid_data) > 0:
                    print(f"{col}: mean={valid_data.mean():.6e}, std={valid_data.std():.6e}, valid={len(valid_data)}/{len(self.features)}")

    def _extract_features_for_field(self, field_data, y_field):
        """為特定y_field值提取特徵"""
        features = {'y_field': y_field}
        
        try:
            current = field_data['appl_current'].values
            voltage = field_data[self.voltage_column].values
            dV_dI = field_data['dV_dI'].values
            
            # 1. 臨界電流特徵
            positive_mask = current > 0
            negative_mask = current < 0
            
            if np.any(positive_mask):
                dV_dI_pos = dV_dI[positive_mask]
                current_pos = current[positive_mask]
                if len(dV_dI_pos) > 0:
                    max_idx = np.argmax(dV_dI_pos)
                    features['Ic_positive'] = current_pos[max_idx]
                    features['dV_dI_max'] = dV_dI_pos[max_idx]
            
            if np.any(negative_mask):
                dV_dI_neg = dV_dI[negative_mask]
                current_neg = current[negative_mask]
                if len(dV_dI_neg) > 0:
                    max_idx = np.argmax(dV_dI_neg)
                    features['Ic_negative'] = abs(current_neg[max_idx])
            
            # 平均臨界電流
            ic_vals = []
            if 'Ic_positive' in features:
                ic_vals.append(features['Ic_positive'])
            if 'Ic_negative' in features:
                ic_vals.append(features['Ic_negative'])
            if ic_vals:
                features['Ic_average'] = np.mean(ic_vals)
            
            # 2. 正常態電阻
            high_current_mask = np.abs(current) > 0.8 * np.max(np.abs(current))
            if np.any(high_current_mask):
                V_high = voltage[high_current_mask]
                I_high = current[high_current_mask]
                if len(V_high) > 1:
                    slope, _ = np.polyfit(I_high, V_high, 1)
                    features['Rn'] = slope
            
            # 3. n值計算
            features['n_value'] = self._calculate_n_value(current, voltage)
            
            # 4. 轉變寬度
            features['transition_width'] = self._calculate_transition_width(current, dV_dI)
            
            # 5. dV/dI統計特徵
            features['dV_dI_mean'] = np.mean(dV_dI)
            features['dV_dI_std'] = np.std(dV_dI)
            features['dV_dI_skewness'] = self._calculate_skewness(dV_dI)
            features['dV_dI_kurtosis'] = self._calculate_kurtosis(dV_dI)
            
            # 6. 電壓偏移
            zero_current_mask = np.abs(current) < 1e-8
            if np.any(zero_current_mask):
                features['voltage_offset'] = np.mean(voltage[zero_current_mask])
        
        except Exception as e:
            print(f"Error extracting features for y_field {y_field}: {e}")
        
        return features
    
    def _calculate_n_value(self, current, voltage):
        """計算n值"""
        try:
            # 使用10%-90%準則
            max_voltage = np.max(np.abs(voltage))
            v10 = 0.1 * max_voltage
            v90 = 0.9 * max_voltage
            
            # 找到對應的電流值
            mask = (np.abs(voltage) >= v10) & (np.abs(voltage) <= v90)
            if np.sum(mask) > 1:
                V_range = voltage[mask]
                I_range = current[mask]
                
                # 避免log(0)
                V_range = V_range[V_range > 0]
                I_range = I_range[:len(V_range)]
                
                if len(V_range) > 1:
                    # n = d(ln V) / d(ln I)
                    log_V = np.log(V_range)
                    log_I = np.log(np.abs(I_range) + 1e-12)
                    slope, _ = np.polyfit(log_I, log_V, 1)
                    return slope
        except Exception:
            pass
        return np.nan
    
    def _calculate_transition_width(self, current, dV_dI):
        """計算轉變寬度"""
        try:
            max_dV_dI = np.max(dV_dI)
            half_max = max_dV_dI / 2
            
            # 找到半峰寬
            indices = np.where(dV_dI >= half_max)[0]
            if len(indices) > 1:
                width_indices = indices[-1] - indices[0]
                if width_indices < len(current):
                    return abs(current[indices[-1]] - current[indices[0]])
        except Exception:
            pass
        return np.nan
    
    def _calculate_skewness(self, data):
        """計算偏度"""
        try:
            data = data[~np.isnan(data)]
            if len(data) > 2:
                mean_val = np.mean(data)
                std_val = np.std(data)
                if std_val > 0:
                    return np.mean(((data - mean_val) / std_val) ** 3)
        except Exception:
            pass
        return np.nan
    
    def _calculate_kurtosis(self, data):
        """計算峰度"""
        try:
            data = data[~np.isnan(data)]
            if len(data) > 3:
                mean_val = np.mean(data)
                std_val = np.std(data)
                if std_val > 0:
                    return np.mean(((data - mean_val) / std_val) ** 4) - 3
        except Exception:
            pass
        return np.nan
    
    def run_complete_analysis(self):
        """運行完整分析流程"""
        self.load_data()
        self.extract_conventional_features()
        self.create_2d_images()
        self.apply_deconvolution()
        self.extract_ml_features()
        self.visualize_results()
        self.generate_summary_report()
    
    def create_2d_images(self):
        """步驟3: 用於圖像分析的數據轉換"""
        print("\n=== Step 3: Data Transformation for Image Analysis ===")
        
        # 創建二維網格
        y_fields = sorted(self.data['y_field'].unique())
        currents = sorted(self.data['appl_current'].unique())
        
        print(f"Grid size: {len(y_fields)} x {len(currents)}")
        
        # 創建網格
        Y_grid, I_grid = np.meshgrid(y_fields, currents, indexing='ij')
        
        # 初始化圖像數組
        voltage_image = np.full_like(Y_grid, np.nan)
        dV_dI_image = np.full_like(Y_grid, np.nan)
        
        # 填充數據
        valid_pixels = 0
        for i, y_field in enumerate(y_fields):
            for j, current in enumerate(currents):
                data_point = self.data[
                    (self.data['y_field'] == y_field) & 
                    (self.data['appl_current'] == current)
                ]
                if len(data_point) > 0:
                    voltage_image[i, j] = data_point[self.voltage_column].iloc[0]
                    dV_dI_image[i, j] = data_point['dV_dI'].iloc[0]
                    valid_pixels += 1
        
        print(f"Voltage image: {voltage_image.shape}, valid pixels: {valid_pixels}")
        print(f"dV/dI image: {dV_dI_image.shape}, valid pixels: {valid_pixels}")
        
        # 儲存圖像和網格
        self.images = {
            'voltage': voltage_image,
            'dV_dI': dV_dI_image,
            'y_field_grid': Y_grid,
            'current_grid': I_grid
        }
        
        # 應用圖像處理技術
        print("\nApplying image processing techniques:")
        self._apply_image_processing()
    
    def _apply_image_processing(self):
        """應用圖像處理技術"""
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
                    
                    print(f"{key} image processed (shape: {filtered.shape})")
    
    def apply_deconvolution(self):
        """步驟4: 去卷積處理"""
        print("\n=== Step 4: Deconvolution Processing ===")
        
        # 簡化去卷積 - 僅處理前幾個y_field值作為示例
        self.deconvolved_results = {}
        sample_fields = self.y_field_values[:6]
        
        for y_field in sample_fields:
            field_data = self.data[self.data['y_field'] == y_field].sort_values('appl_current')
            
            if len(field_data) > 10:
                try:
                    current = field_data['appl_current'].values
                    dV_dI = field_data['dV_dI'].values
                    
                    # 平滑處理
                    smoothed = ndimage.gaussian_filter1d(dV_dI, sigma=1)
                    
                    # 簡單去卷積 - 使用Wiener濾波
                    kernel = signal.gaussian(5, 1)
                    kernel = kernel / np.sum(kernel)
                    
                    # Wiener去卷積
                    noise_power = 0.1
                    signal_power = np.var(smoothed)
                    wiener_filter = np.conj(kernel) / (np.abs(kernel)**2 + noise_power/signal_power)
                    
                    # 在頻域應用
                    smoothed_fft = np.fft.fft(smoothed)
                    kernel_fft = np.fft.fft(kernel, len(smoothed))
                    wiener_fft = np.fft.fft(wiener_filter, len(smoothed))
                    
                    deconvolved_fft = smoothed_fft * wiener_fft
                    deconvolved = np.real(np.fft.ifft(deconvolved_fft))
                    
                    self.deconvolved_results[y_field] = {
                        'current': current,
                        'original': dV_dI,
                        'smoothed': smoothed,
                        'deconvolved': deconvolved
                    }
                    
                    print(f"  ✓ y_field={y_field:.6f}")
                    
                except Exception as e:
                    print(f"  ✗ y_field={y_field:.6f}: {e}")
        
        print(f"Deconvolution completed: {len(self.deconvolved_results)}/{len(sample_fields)} successful")
    
    def extract_ml_features(self):
        """步驟5: 機器學習特徵提取"""
        print("\n=== Step 5: Machine Learning Feature Extraction ===")
        
        try:
            # 準備數據
            numeric_features = []
            feature_names = []
            
            for col in self.features.columns:
                if col != 'y_field' and pd.api.types.is_numeric_dtype(self.features[col]):
                    valid_data = self.features[col].dropna()
                    if len(valid_data) > 0:
                        filled_data = self.features[col].fillna(self.features[col].median())
                        numeric_features.append(filled_data.values)
                        feature_names.append(col)
            
            if numeric_features:
                feature_matrix = np.column_stack(numeric_features)
                
                # PCA分析
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(feature_matrix)
                
                n_components = min(5, feature_matrix.shape[1], feature_matrix.shape[0])
                pca = PCA(n_components=n_components)
                pca_features = pca.fit_transform(features_scaled)
                
                # 統計特徵
                stats = {}
                for i, feature in enumerate(feature_names):
                    series = self.features[feature]
                    trend = self._calculate_trend(series)
                    variability = series.std() / series.mean() if series.mean() != 0 else 0
                    
                    stats[f'{feature}_trend'] = trend
                    stats[f'{feature}_variability'] = variability
                
                # PCA features
                self.ml_features = {
                    'pca': pca_features,
                    'statistics': stats,
                    'feature_names': feature_names
                }
                
                # Autoencoder deep features
                ae = AutoencoderModel(input_dim=features_scaled.shape[1], latent_dim=min(5, features_scaled.shape[1]))
                ae.train(features_scaled, epochs=30, batch_size=8)
                deep_feats = ae.extract_features(features_scaled)
                self.ml_features['deep'] = deep_feats
                
                # t-SNE and UMAP embedding
                tsne = TSNE(n_components=2, random_state=42)
                tsne_embed = tsne.fit_transform(features_scaled)
                
                self.ml_features['tsne'] = tsne_embed
                
                if UMAP_AVAILABLE:
                    reducer = UMAP(n_components=2, random_state=42)
                    umap_embed = reducer.fit_transform(features_scaled)
                    self.ml_features['umap'] = umap_embed
                
                self.pca_info = {
                    'explained_variance_ratio': pca.explained_variance_ratio_,
                    'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
                }
                
                print("Machine learning feature extraction completed:")
                print(f"  PCA dimensions: {pca_features.shape}")
                print(f"  Cumulative explained variance: {self.pca_info['cumulative_variance'][-1]:.3f}")
                print(f"  Statistical features: {len(stats)}")
            else:
                print("Insufficient data for machine learning feature extraction")
                
        except Exception as e:
            print(f"Machine learning feature extraction failed: {e}")
    
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
    
    def visualize_results(self):
        """可視化結果"""
        print("\n=== Results Visualization ===")
        
        plt.figure(figsize=(16, 12))
        
        # 1. Feature variation with y_field
        plt.subplot(3, 3, 1)
        if 'Ic_average' in self.features.columns:
            plt.plot(self.features['y_field'], self.features['Ic_average']*1e6, 'b.-', markersize=2)
            plt.xlabel('y_field')
            plt.ylabel('Critical Current (µA)')
            plt.title('Critical Current vs y_field')
            plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 2)
        if 'Rn' in self.features.columns:
            plt.plot(self.features['y_field'], self.features['Rn'], 'r.-', markersize=2)
            plt.xlabel('y_field')
            plt.ylabel('Normal Resistance (Ω)')
            plt.title('Normal Resistance vs y_field')
            plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 3, 3)
        if 'n_value' in self.features.columns:
            plt.plot(self.features['y_field'], self.features['n_value'], 'g.-', markersize=2)
            plt.xlabel('y_field')
            plt.ylabel('n-value')
            plt.title('n-value vs y_field')
            plt.grid(True, alpha=0.3)
        
        # 2. 2D Images
        if 'dV_dI_filtered' in self.images:
            plt.subplot(3, 3, 4)
            im = plt.imshow(self.images['dV_dI_filtered'], 
                           aspect='auto', cmap='viridis', origin='lower')
            plt.colorbar(im, label='dV/dI (Ω)')
            plt.xlabel('Current Index')
            plt.ylabel('y_field Index')
            plt.title('dV/dI 2D Image')
        
        # 3. Sample curves
        plt.subplot(3, 3, 5)
        sample_y_field = self.y_field_values[len(self.y_field_values)//2]
        sample_data = self.data[self.data['y_field'] == sample_y_field].sort_values('appl_current')
        if len(sample_data) > 0:
            plt.plot(sample_data['appl_current']*1e6, sample_data['dV_dI'], 'r-', linewidth=1)
            plt.xlabel('Applied Current (µA)')
            plt.ylabel('dV/dI (Ω)')
            plt.title('Sample dV/dI Curve')
            plt.grid(True, alpha=0.3)
        
        # 4. Feature distribution
        plt.subplot(3, 3, 6)
        if 'Ic_average' in self.features.columns:
            plt.hist(self.features['Ic_average']*1e6, bins=30, alpha=0.7, color='blue')
            plt.xlabel('Critical Current (µA)')
            plt.ylabel('Frequency')
            plt.title('Critical Current Distribution')
        
        # 5. PCA結果
        if hasattr(self, 'ml_features') and 'pca' in self.ml_features:
            plt.subplot(3, 3, 7)
            pca_data = self.ml_features['pca']
            if pca_data.shape[1] >= 2:
                scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], 
                                    c=self.features['y_field'], cmap='viridis', 
                                    alpha=0.6, s=10)
                plt.colorbar(scatter, label='y_field')
                plt.xlabel('PCA Dimension 1')
                plt.ylabel('PCA Dimension 2')
                plt.title('PCA Feature Space')
                plt.grid(True, alpha=0.3)
        
        # 6. Deconvolution results
        if hasattr(self, 'deconvolved_results') and self.deconvolved_results:
            plt.subplot(3, 3, 8)
            first_result = next(iter(self.deconvolved_results.values()))
            plt.plot(first_result['current']*1e6, first_result['original'], 'b-', 
                    label='Original', alpha=0.7, linewidth=1)
            plt.plot(first_result['current']*1e6, first_result['deconvolved'], 'r-', 
                    label='Processed', linewidth=2)
            plt.xlabel('Applied Current (µA)')
            plt.ylabel('dV/dI (Ω)')
            plt.title('Deconvolution Comparison')
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)
        
        # 7. Statistical summary
        plt.subplot(3, 3, 9)
        stats_text = f"Feature count: {len(self.features.columns)-1}\n"
        stats_text += f"y_field steps: {len(self.y_field_values)}\n"
        stats_text += f"Data points: {len(self.data)}\n"
        if hasattr(self, 'ml_features'):
            stats_text += f"PCA dimensions: {self.ml_features['pca'].shape[1]}\n"
            if hasattr(self, 'pca_info'):
                stats_text += f"Explained variance: {self.pca_info['cumulative_variance'][-1]:.2%}"
        
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        plt.title('Analysis Summary')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('analysis_results_500.png', dpi=300, bbox_inches='tight')
        print("Analysis results saved to analysis_results_500.png")
    
    def generate_summary_report(self):
        """生成總結報告"""
        print("\n" + "="*50)
        print("         Superconductor Data Analysis Report - 500.csv")
        print("="*50)
        
        print("\nData Overview:")
        print(f"  Total data points: {len(self.data)}")
        print(f"  y_field range: {self.y_field_values[0]:.6f} - {self.y_field_values[-1]:.6f}")
        print(f"  y_field steps: {len(self.y_field_values)}")
        print(f"  Current range: {self.data['appl_current'].min()*1e6:.2f} - {self.data['appl_current'].max()*1e6:.2f} µA")
        print(f"  Voltage column used: {self.voltage_column}")
        
        print("\nExtracted Feature Statistics:")
        if isinstance(self.features, pd.DataFrame):
            for feature in ['Ic_average', 'Rn', 'n_value', 'transition_width']:
                if feature in self.features.columns:
                    valid_data = self.features[feature].dropna()
                    if len(valid_data) > 0:
                        if feature == 'Ic_average':
                            print(f"  Critical current: {valid_data.mean()*1e6:.3f} ± {valid_data.std()*1e6:.3f} µA")
                        elif feature == 'Rn':
                            print(f"  Normal resistance: {valid_data.mean():.2f} ± {valid_data.std():.2f} Ω")
                        elif feature == 'n_value':
                            print(f"  n-value: {valid_data.mean():.3f} ± {valid_data.std():.3f}")
                        elif feature == 'transition_width':
                            print(f"  Transition width: {valid_data.mean()*1e6:.3f} ± {valid_data.std()*1e6:.3f} µA")
        
        print("\nImage Analysis:")
        if self.images:
            print(f"  Generated 2D images: {list(self.images.keys())}")
            
        print("\nMachine Learning Analysis:")
        if hasattr(self, 'ml_features'):
            print(f"  PCA dimensionality reduction: {self.ml_features['pca'].shape}")
            if hasattr(self, 'pca_info'):
                print(f"  Cumulative explained variance: {self.pca_info['cumulative_variance'][-1]:.1%}")
            print(f"  Statistical features: {len(self.ml_features['statistics'])}")
        
        print("\nDeconvolution Processing:")
        if hasattr(self, 'deconvolved_results'):
            print(f"  Successfully processed: {len(self.deconvolved_results)} y_field values")
        
        print("\nAnalysis completed! Recommendations:")
        print("1. Examine feature trends with y_field variation")
        print("2. Analyze patterns and structures in 2D images")
        print("3. Consider further machine learning analysis")
        print("4. Verify physical interpretation reasonableness")
        print("5. Adjust parameters for specific applications")

if __name__ == '__main__':
    # 執行完整分析流程
    analyzer = SuperconductorAnalyzer('500.csv')
    analyzer.run_complete_analysis()
