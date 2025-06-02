#!/usr/bin/env python3
"""
從實驗數據中提取特徵值：結合圖像分析與卷積/去卷積技術
基於 README.md 中描述的方法實現 - 修復版本
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
        print("\n檢測異常值:")
        for col in ['meas_voltage_K2', 'dV_dI']:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
            outlier_count = outliers.sum()
            print(f"{col}: {outlier_count} 個異常值 ({outlier_count/len(self.data)*100:.2f}%)")
            
            # 檢查極端異常值
            extreme_outliers = (self.data[col] < Q1 - 3 * IQR) | (self.data[col] > Q3 + 3 * IQR)
            extreme_count = extreme_outliers.sum()
            if extreme_count > 0:
                print(f"  極端異常值: {extreme_count} 個")
        
        # 獲取y_field唯一值
        self.y_field_values = sorted(self.data['y_field'].unique())
        print(f"y_field 唯一值數量: {len(self.y_field_values)}")
        print(f"y_field 範圍: {self.y_field_values[0]:.6f} - {self.y_field_values[-1]:.6f}")
    
    def extract_conventional_features(self):
        """步驟2: 常規特徵提取"""
        print("\n=== 步驟2: 常規特徵提取 ===")
        
        features_list = []
        
        for i, y_field in enumerate(self.y_field_values):
            if (i + 1) % 10 == 0:
                print(f"已處理 {i+1}/{len(self.y_field_values)} 個y_field值")
            
            # 提取該y_field下的數據
            field_data = self.data[self.data['y_field'] == y_field].copy()
            field_data = field_data.sort_values('appl_current')
            
            if len(field_data) < 10:  # 確保有足夠的數據點
                continue
            
            # 提取基本特徵
            features = self._extract_features_for_field(field_data, y_field)
            features_list.append(features)
        
        # 轉換為DataFrame
        self.features = pd.DataFrame(features_list)
        print(f"\n提取的特徵維度: {self.features.shape}")
        print(f"提取的特徵: {list(self.features.columns)}")
        
        # 統計信息
        print("\n特徵統計信息:")
        for col in self.features.columns:
            if col != 'y_field' and pd.api.types.is_numeric_dtype(self.features[col]):
                valid_data = self.features[col].dropna()
                if len(valid_data) > 0:
                    print(f"{col}: 均值={valid_data.mean():.6e}, 標準差={valid_data.std():.6e}, 有效值={len(valid_data)}/{len(self.features)}")
    
    def _extract_features_for_field(self, field_data, y_field):
        """為特定y_field提取特徵"""
        features = {'y_field': y_field}
        
        current = field_data['appl_current'].values
        voltage = field_data['meas_voltage_K2'].values
        dV_dI = field_data['dV_dI'].values
        
        try:
            # 1. 臨界電流 (Ic) - 基於dV/dI峰值
            dV_dI_smooth = signal.savgol_filter(dV_dI, window_length=min(5, len(dV_dI)//2*2+1), polyorder=2)
            
            # 正向和負向臨界電流
            positive_mask = current > 0
            negative_mask = current < 0
            
            if np.sum(positive_mask) > 0:
                pos_peak_idx = np.argmax(dV_dI_smooth[positive_mask])
                features['Ic_positive'] = current[positive_mask][pos_peak_idx]
                features['dV_dI_max'] = dV_dI_smooth[positive_mask][pos_peak_idx]
            else:
                features['Ic_positive'] = np.nan
                features['dV_dI_max'] = np.nan
            
            if np.sum(negative_mask) > 0:
                neg_peak_idx = np.argmax(dV_dI_smooth[negative_mask])
                features['Ic_negative'] = abs(current[negative_mask][neg_peak_idx])
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
            
            # 2. 正常態電阻 (Rn) - 高電流區域的平台值
            high_current_mask = np.abs(current) > np.percentile(np.abs(current), 80)
            if np.sum(high_current_mask) > 0:
                features['Rn'] = np.median(dV_dI[high_current_mask])
            else:
                features['Rn'] = np.nan
            
            # 3. n值 - 轉變陡峭度
            features['n_value'] = self._calculate_n_value(current, voltage)
            
            # 4. 轉變寬度
            features['transition_width'] = self._calculate_transition_width(current, dV_dI_smooth)
            
            # 5. 統計特徵
            features['dV_dI_mean'] = np.mean(dV_dI)
            features['dV_dI_std'] = np.std(dV_dI)
            features['dV_dI_skewness'] = self._calculate_skewness(dV_dI)
            features['dV_dI_kurtosis'] = self._calculate_kurtosis(dV_dI)
            features['voltage_offset'] = np.mean(voltage[np.abs(current) < np.percentile(np.abs(current), 10)])
            
        except Exception:
            # 如果計算失敗，填入NaN
            for key in ['Ic_positive', 'dV_dI_max', 'Ic_negative', 'Ic_average', 'Rn', 'n_value', 
                       'transition_width', 'dV_dI_mean', 'dV_dI_std', 'dV_dI_skewness', 
                       'dV_dI_kurtosis', 'voltage_offset']:
                if key not in features:
                    features[key] = np.nan
        
        return features
    
    def _calculate_n_value(self, current, voltage):
        """計算n值（超導轉變的陡峭度）"""
        try:
            # 移除零電流點
            nonzero_mask = current != 0
            I_nz = current[nonzero_mask]
            V_nz = voltage[nonzero_mask]
            
            if len(I_nz) < 5:
                return np.nan
            
            # 在對數空間中擬合
            log_I = np.log10(np.abs(I_nz))
            log_V = np.log10(np.abs(V_nz) + 1e-12)  # 避免log(0)
            
            # 線性擬合
            valid_mask = np.isfinite(log_I) & np.isfinite(log_V)
            if np.sum(valid_mask) < 3:
                return np.nan
            
            coeffs = np.polyfit(log_I[valid_mask], log_V[valid_mask], 1)
            return coeffs[0]  # 斜率即為n值
            
        except Exception:
            return np.nan
    
    def _calculate_transition_width(self, current, dV_dI):
        """計算轉變寬度（半峰全寬）"""
        try:
            # 找到峰值
            peak_value = np.max(dV_dI)
            half_max = peak_value / 2
            
            # 找到半峰值點
            above_half = dV_dI >= half_max
            if np.sum(above_half) < 2:
                return np.nan
            
            indices = np.where(above_half)[0]
            width_indices = indices[-1] - indices[0]
            
            if width_indices < len(current):
                return np.abs(current[indices[-1]] - current[indices[0]])
            else:
                return np.nan
                
        except Exception:
            return np.nan
    
    def _calculate_skewness(self, data):
        """計算偏度"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return np.mean(((data - mean) / std) ** 3)
        except Exception:
            return np.nan
    
    def _calculate_kurtosis(self, data):
        """計算峰度"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return np.mean(((data - mean) / std) ** 4) - 3
        except Exception:
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
        print("\n=== 步驟3: 用於圖像分析的數據轉換 ===")
        
        # 創建二維網格
        y_fields = sorted(self.data['y_field'].unique())
        currents = sorted(self.data['appl_current'].unique())
        
        print(f"網格大小: {len(y_fields)} x {len(currents)}")
        
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
                    voltage_image[i, j] = data_point['meas_voltage_K2'].iloc[0]
                    dV_dI_image[i, j] = data_point['dV_dI'].iloc[0]
                    valid_pixels += 1
        
        print(f"電壓圖像: {voltage_image.shape}, 有效像素: {valid_pixels}")
        print(f"dV/dI圖像: {dV_dI_image.shape}, 有效像素: {valid_pixels}")
        
        # 儲存圖像和網格
        self.images = {
            'voltage': voltage_image,
            'dV_dI': dV_dI_image,
            'y_field_grid': Y_grid,
            'current_grid': I_grid
        }
        
        # 應用圖像處理技術
        print("\n應用圖像處理技術:")
        self._apply_image_processing()
    
    def _apply_image_processing(self):
        """應用圖像處理技術"""
        for key in ['voltage', 'dV_dI']:
            image = self.images[key].copy()
            
            # 使用中值濾波降噪
            mask = ~np.isnan(image)
            if np.sum(mask) > 0:
                try:
                    from scipy.interpolate import griddata
                    rows, cols = np.mgrid[0:image.shape[0], 0:image.shape[1]]
                    
                    # 獲取有效點
                    valid_points = np.column_stack((rows[mask], cols[mask]))
                    valid_values = image[mask]
                    
                    # 插值到所有點
                    if len(valid_points) > 3:
                        interpolated = griddata(
                            valid_points, valid_values, 
                            (rows, cols), method='linear', fill_value=float(np.nanmean(valid_values))
                        )
                        
                        # 應用中值濾波
                        filtered = ndimage.median_filter(interpolated, size=3)
                        self.images[f'{key}_filtered'] = filtered
                        
                        print(f"{key} 圖像已處理 (形狀: {filtered.shape})")
                except Exception as e:
                    print(f"{key} 圖像處理失敗: {e}")
    
    def apply_deconvolution(self):
        """步驟4: 去卷積處理"""
        print("\n=== 步驟4: 去卷積處理 ===")
        self.deconvolved_results = {}
        
        try:
            # 簡化的去卷積處理
            sample_fields = self.y_field_values[::100]  # 減少採樣
            success_count = 0
            
            for y_field in sample_fields:
                field_data = self.data[self.data['y_field'] == y_field].copy()
                field_data = field_data.sort_values('appl_current')
                
                if len(field_data) > 10:
                    dV_dI_data = field_data['dV_dI'].values
                    current_data = field_data['appl_current'].values
                    
                    # 簡單的平滑處理作為去卷積的近似
                    smoothed = signal.savgol_filter(dV_dI_data, 
                                                  window_length=min(7, len(dV_dI_data)//2*2+1), 
                                                  polyorder=2)
                    
                    self.deconvolved_results[y_field] = {
                        'original': dV_dI_data,
                        'smoothed': smoothed,
                        'deconvolved': smoothed,  # 簡化版本
                        'current': current_data
                    }
                    success_count += 1
                    if success_count == 1:
                        print(f"  ✓ y_field={y_field:.6f}")
            
            print(f"去卷積處理完成: {success_count}/{len(sample_fields)} 成功")
            
        except Exception as e:
            print(f"去卷積處理失敗: {e}")
    
    def extract_ml_features(self):
        """步驟5: 機器學習特徵提取"""
        print("\n=== 步驟5: 機器學習特徵提取 ===")
        
        try:
            # 準備數值特徵矩陣
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
                
                print("機器學習特徵提取完成:")
                print(f"  PCA維度: {pca_features.shape}")
                print(f"  累計解釋方差: {self.pca_info['cumulative_variance'][-1]:.3f}")
                print(f"  統計特徵: {len(stats)}")
            else:
                print("無足夠數據進行機器學習特徵提取")
                
        except Exception as e:
            print(f"機器學習特徵提取失敗: {e}")
    
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
        print("\n=== 結果可視化 ===")
        
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
        
        # 5. PCA results
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
        plt.savefig('analysis_results_improved.png', dpi=300, bbox_inches='tight')
        print("分析結果已保存到 analysis_results_improved.png")
    
    def generate_summary_report(self):
        """生成總結報告 - 迭代優化版"""
        print("\n" + "="*50)
        print("         超導體數據分析總結報告 - 迭代優化版")
        print("="*50)
        
        print("\n數據概況:")
        print(f"  總數據點: {len(self.data)}")
        # 修正屬性名稱為 y_field_values
        print(f"  y_field範圍: {self.y_field_values[0]:.6f} - {self.y_field_values[-1]:.6f}")
        print(f"  y_field步數: {len(self.y_field_values)}")
        print(f"  電流範圍: {self.data['appl_current'].min()*1e6:.2f} - {self.data['appl_current'].max()*1e6:.2f} µA")
        
        print("\n提取特徵統計:")
        for feature in ['Ic_average', 'Rn', 'n_value', 'transition_width']:
            if feature in self.features.columns:
                valid_data = self.features[feature].dropna()
                if len(valid_data) > 0:
                    if feature == 'Ic_average':
                        print(f"  臨界電流: {valid_data.mean()*1e6:.3f} ± {valid_data.std()*1e6:.3f} µA")
                    elif feature == 'Rn':
                        print(f"  正常態電阻: {valid_data.mean():.2f} ± {valid_data.std():.2f} Ω")
                    elif feature == 'n_value':
                        print(f"  n值: {valid_data.mean():.3f} ± {valid_data.std():.3f}")
                    elif feature == 'transition_width':
                        print(f"  轉變寬度: {valid_data.mean()*1e6:.3f} ± {valid_data.std()*1e6:.3f} µA")
        
        print("\n圖像分析:")
        if self.images:
            print(f"  生成二維圖像: {list(self.images.keys())}")
            
        print("\n機器學習分析:")
        if hasattr(self, 'ml_features'):
            print(f"  PCA降維: {self.ml_features['pca'].shape}")
            print(f"  累計解釋方差: {self.pca_info['cumulative_variance'][-1]:.1%}")
            print(f"  統計特徵: {len(self.ml_features['statistics'])}")
        
        print("\n去卷積處理:")
        if hasattr(self, 'deconvolved_results'):
            print(f"  處理成功: {len(self.deconvolved_results)} 個y_field值")
        
        print("\n分析完成! 建議:")
        print("1. 檢查特徵隨y_field的變化趨勢")
        print("2. 分析二維圖像中的模式和結構")
        print("3. 考慮進一步的機器學習分析")
        print("4. 驗證物理解釋的合理性")
        print("5. 針對特定應用調整參數")

if __name__ == '__main__':
    # 執行完整分析流程
    analyzer = SuperconductorAnalyzer('164.csv')
    analyzer.run_complete_analysis()
