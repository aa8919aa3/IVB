#!/usr/bin/env python3
"""
進階超導體數據分析器 - 基於 feature_extraction_500.py 改進版本
整合了最佳實踐和增強功能的超導體實驗數據分析工具
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, ndimage
from ml_models import AutoencoderModel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# matplotlib 設定確保中文字體顯示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class AdvancedSuperconductorAnalyzer:
    """進階超導體數據分析器"""
    
    def __init__(self, data_path, config=None):
        """初始化分析器
        
        Args:
            data_path: 數據文件路徑
            config: 配置字典，用於自定義分析參數
        """
        self.data_path = data_path
        self.data = pd.DataFrame()
        self.y_field_values = []
        self.features = pd.DataFrame()
        self.images = {}
        self.ml_features = {}
        self.clustering_results = {}
        
        # 預設配置
        self.config = {
            'outlier_threshold': 3.0,  # IQR倍數
            'smoothing_window': 5,     # 平滑窗口大小
            'pca_components': 5,       # PCA組件數
            'clustering_enabled': True, # 是否啟用聚類分析
            'advanced_features': True,  # 是否提取進階特徵
            'image_resolution': (200, 200),  # 圖像解析度
        }
        
        if config:
            self.config.update(config)
        
        # 自動檢測電壓列名稱
        self.voltage_column = None
        
    def load_and_preprocess_data(self):
        """載入和預處理數據 - 增強版"""
        print("=== Step 1: Enhanced Data Preprocessing and Cleaning ===")
        
        # 讀取數據
        self.data = pd.read_csv(self.data_path)
        print(f"📊 Initial data shape: {self.data.shape}")
        print(f"📋 Columns: {list(self.data.columns)}")
        
        # 自動檢測電壓列
        voltage_candidates = [col for col in self.data.columns 
                            if any(keyword in col.lower() for keyword in ['voltage', 'volt', 'v_'])]
        if voltage_candidates:
            self.voltage_column = voltage_candidates[0]
            print(f"🔍 Detected voltage column: {self.voltage_column}")
        else:
            raise ValueError("❌ No voltage column found in data")
        
        # 檢查數據類型
        print("\n📝 Data types:")
        for col in self.data.columns:
            print(f"  {col}: {self.data[col].dtype}")
        
        # 處理缺失值
        missing_before = self.data.isnull().sum().sum()
        if missing_before > 0:
            print(f"\n⚠️  Missing values found: {missing_before}")
            self.data = self.data.dropna()
            print(f"✅ Shape after cleaning: {self.data.shape}")
        
        # 進階異常值檢測和處理
        self._detect_and_handle_outliers()
        
        # 數據平滑（可選）
        if self.config['smoothing_window'] > 1:
            self._apply_data_smoothing()
        
        # 獲取y_field值
        self.y_field_values = sorted(self.data['y_field'].unique())
        print(f"🎯 y_field unique values: {len(self.y_field_values)}")
        print(f"📏 y_field range: {self.y_field_values[0]:.6f} - {self.y_field_values[-1]:.6f}")
        
        # 數據統計摘要
        self._generate_data_summary()
        
    def _detect_and_handle_outliers(self):
        """進階異常值檢測和處理"""
        print(f"\n🔍 Advanced outlier detection (threshold: {self.config['outlier_threshold']} IQR):")
        
        for col in [self.voltage_column, 'dV_dI', 'appl_current']:
            if col in self.data.columns:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                threshold = self.config['outlier_threshold']
                outlier_mask = (self.data[col] < Q1 - threshold*IQR) | (self.data[col] > Q3 + threshold*IQR)
                outlier_count = outlier_mask.sum()
                
                print(f"  {col}: {outlier_count} outliers ({outlier_count/len(self.data)*100:.2f}%)")
                
                # 統計信息
                if outlier_count > 0:
                    print(f"    Range: [{self.data[col].min():.6e}, {self.data[col].max():.6e}]")
                    print(f"    Q1-Q3: [{Q1:.6e}, {Q3:.6e}]")
    
    def _apply_data_smoothing(self):
        """應用數據平滑"""
        window = self.config['smoothing_window']
        print(f"\n🔧 Applying data smoothing (window size: {window})")
        
        for field in self.y_field_values[:5]:  # 示例：只對前5個field應用平滑
            mask = self.data['y_field'] == field
            field_data = self.data[mask].sort_values('appl_current')
            
            if len(field_data) >= window:
                # 對電壓數據應用滑動平均
                smoothed_voltage = field_data[self.voltage_column].rolling(window=window, center=True).mean()
                self.data.loc[mask, f'{self.voltage_column}_smoothed'] = smoothed_voltage.values
    
    def _generate_data_summary(self):
        """生成數據統計摘要"""
        print("\n📈 Data Summary Statistics:")
        
        # 基本統計
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'y_field':
                data_col = self.data[col]
                print(f"  {col}:")
                print(f"    Mean: {data_col.mean():.6e}")
                print(f"    Std:  {data_col.std():.6e}")
                print(f"    Range: [{data_col.min():.6e}, {data_col.max():.6e}]")
        
        # 每個field的數據點數量
        field_counts = self.data['y_field'].value_counts()
        print(f"  Points per field: {field_counts.mean():.1f} ± {field_counts.std():.1f}")
    
    def extract_enhanced_features(self):
        """步驟2: 增強特徵提取"""
        print("\n=== Step 2: Enhanced Feature Extraction ===")
        
        features_list = []
        
        for i, y_field in enumerate(self.y_field_values):
            if (i + 1) % 20 == 0 or i == 0:
                print(f"🔄 Processed {i+1}/{len(self.y_field_values)} y_field values")
            
            field_data = self.data[self.data['y_field'] == y_field].sort_values('appl_current')
            
            if len(field_data) == 0:
                continue
                
            features = self._extract_comprehensive_features_for_field(field_data, y_field)
            features_list.append(features)
        
        # 合併所有特徵
        self.features = pd.DataFrame(features_list)
        
        print(f"\n✅ Extracted features dimensions: {self.features.shape}")
        print(f"📊 Total features: {len(self.features.columns)-1}")  # 減1是因為包含y_field列
        
        # 顯示特徵統計
        self._display_feature_statistics()
    
    def _extract_comprehensive_features_for_field(self, field_data, y_field):
        """為特定y_field值提取全面特徵"""
        features = {'y_field': y_field}
        
        try:
            current = field_data['appl_current'].values
            voltage = field_data[self.voltage_column].values
            dV_dI = field_data['dV_dI'].values
            
            # 1. 基本電特性特徵
            features.update(self._extract_electrical_features(current, voltage, dV_dI))
            
            # 2. 統計特徵
            features.update(self._extract_statistical_features(current, voltage, dV_dI))
            
            # 3. 形態學特徵
            features.update(self._extract_morphological_features(current, voltage, dV_dI))
            
            # 4. 進階物理特徵
            if self.config['advanced_features']:
                features.update(self._extract_advanced_physical_features(current, voltage, dV_dI))
        
        except Exception as e:
            print(f"⚠️  Error extracting features for y_field {y_field}: {e}")
            # 填充默認值
            default_features = {
                'Ic_positive': np.nan, 'Ic_negative': np.nan, 'Ic_average': np.nan,
                'Rn': np.nan, 'n_value': np.nan, 'transition_width': np.nan
            }
            features.update(default_features)
        
        return features
    
    def _extract_electrical_features(self, current, voltage, dV_dI):
        """提取電學特徵"""
        features = {}
        
        # 臨界電流分析
        positive_mask = current > 0
        negative_mask = current < 0
        
        if np.any(positive_mask):
            dV_dI_pos = dV_dI[positive_mask]
            current_pos = current[positive_mask]
            if len(dV_dI_pos) > 0:
                max_idx = np.argmax(dV_dI_pos)
                features['Ic_positive'] = current_pos[max_idx]
                features['dV_dI_max_positive'] = dV_dI_pos[max_idx]
        
        if np.any(negative_mask):
            dV_dI_neg = dV_dI[negative_mask]
            current_neg = current[negative_mask]
            if len(dV_dI_neg) > 0:
                max_idx = np.argmax(dV_dI_neg)
                features['Ic_negative'] = abs(current_neg[max_idx])
                features['dV_dI_max_negative'] = dV_dI_neg[max_idx]
        
        # 平均臨界電流
        ic_vals = []
        if 'Ic_positive' in features:
            ic_vals.append(features['Ic_positive'])
        if 'Ic_negative' in features:
            ic_vals.append(features['Ic_negative'])
        if ic_vals:
            features['Ic_average'] = np.mean(ic_vals)
            features['Ic_asymmetry'] = abs(features.get('Ic_positive', 0) - features.get('Ic_negative', 0))
        
        # 正常態電阻
        high_current_mask = np.abs(current) > 0.8 * np.max(np.abs(current))
        if np.any(high_current_mask):
            V_high = voltage[high_current_mask]
            I_high = current[high_current_mask]
            if len(V_high) > 1:
                slope, intercept = np.polyfit(I_high, V_high, 1)
                features['Rn'] = slope
                features['voltage_offset'] = intercept
        
        # n值計算
        features['n_value'] = self._calculate_n_value(current, voltage)
        
        # 轉變寬度
        features['transition_width'] = self._calculate_transition_width(current, dV_dI)
        
        return features
    
    def _extract_statistical_features(self, current, voltage, dV_dI):
        """提取統計特徵"""
        features = {}
        
        # dV/dI統計特徵
        features['dV_dI_mean'] = np.mean(dV_dI)
        features['dV_dI_std'] = np.std(dV_dI)
        features['dV_dI_skewness'] = self._calculate_skewness(dV_dI)
        features['dV_dI_kurtosis'] = self._calculate_kurtosis(dV_dI)
        features['dV_dI_peak_to_peak'] = np.max(dV_dI) - np.min(dV_dI)
        
        # 電壓統計特徵
        features['voltage_mean'] = np.mean(voltage)
        features['voltage_std'] = np.std(voltage)
        features['voltage_range'] = np.max(voltage) - np.min(voltage)
        
        # 電流範圍特徵
        features['current_range'] = np.max(current) - np.min(current)
        features['current_density'] = len(current) / features['current_range'] if features['current_range'] > 0 else 0
        
        return features
    
    def _extract_morphological_features(self, current, voltage, dV_dI):
        """提取形態學特徵"""
        features = {}
        
        # 峰值特徵
        dV_dI_peaks, _ = signal.find_peaks(dV_dI, height=np.mean(dV_dI) + np.std(dV_dI))
        features['num_peaks'] = len(dV_dI_peaks)
        
        if len(dV_dI_peaks) > 0:
            features['peak_heights_mean'] = np.mean(dV_dI[dV_dI_peaks])
            features['peak_heights_std'] = np.std(dV_dI[dV_dI_peaks])
        
        # 曲率特徵
        if len(voltage) > 2:
            curvature = np.gradient(np.gradient(voltage))
            features['curvature_mean'] = np.mean(curvature)
            features['curvature_std'] = np.std(curvature)
            features['max_curvature'] = np.max(np.abs(curvature))
        
        return features
    
    def _extract_advanced_physical_features(self, current, voltage, dV_dI):
        """提取進階物理特徵"""
        features = {}
        
        # 磁滯特徵
        if len(current) > 10:
            # 計算磁滯環面積（粗略估計）
            features['hysteresis_area'] = np.trapezoid(voltage, current)
        
        # 超導轉變特徵
        transition_mask = (dV_dI > np.percentile(dV_dI, 75, axis=None))
        if np.any(transition_mask):
            features['transition_sharpness'] = np.max(dV_dI) / np.mean(dV_dI)
            features['transition_current_width'] = (
                np.max(current[transition_mask]) - np.min(current[transition_mask])
                if np.sum(transition_mask) > 1 else 0
            )
        
        # 噪聲特徵
        voltage_smooth = ndimage.gaussian_filter1d(voltage, sigma=1)
        noise_level = np.std(voltage - voltage_smooth)
        features['noise_level'] = noise_level
        features['signal_to_noise'] = np.std(voltage) / noise_level if noise_level > 0 else np.inf
        
        return features
    
    def _calculate_n_value(self, current, voltage):
        """計算n值 - 改進版"""
        try:
            # 使用10%-90%準則
            max_voltage = np.max(np.abs(voltage))
            if max_voltage == 0:
                return np.nan
                
            v10 = 0.1 * max_voltage
            v90 = 0.9 * max_voltage
            
            # 找到對應的電流值
            mask = (np.abs(voltage) >= v10) & (np.abs(voltage) <= v90)
            if np.sum(mask) < 2:
                return np.nan
            
            v_range = voltage[mask]
            i_range = current[mask]
            
            # 過濾掉零電流值
            non_zero_mask = i_range != 0
            if np.sum(non_zero_mask) < 2:
                return np.nan
            
            v_range = v_range[non_zero_mask]
            i_range = i_range[non_zero_mask]
            
            # 計算 n = log(V90/V10) / log(I90/I10)
            log_v_ratio = np.log(np.abs(v_range[-1]) / np.abs(v_range[0]))
            log_i_ratio = np.log(np.abs(i_range[-1]) / np.abs(i_range[0]))
            
            if log_i_ratio != 0:
                return log_v_ratio / log_i_ratio
            else:
                return np.nan
                
        except Exception:
            return np.nan
    
    def _calculate_transition_width(self, current, dV_dI):
        """計算轉變寬度 - 改進版"""
        try:
            max_dV_dI = np.max(dV_dI)
            half_max = max_dV_dI / 2
            
            # 找到半高寬
            above_half_max = dV_dI >= half_max
            if np.sum(above_half_max) < 2:
                return np.nan
            
            indices = np.where(above_half_max)[0]
            width = abs(current[indices[-1]] - current[indices[0]])
            return width
            
        except Exception:
            return np.nan
    
    def _calculate_skewness(self, data):
        """計算偏度"""
        if len(data) < 3:
            return np.nan
        mean_val = np.mean(data)
        std_val = np.std(data, ddof=1)
        if std_val == 0:
            return np.nan
        skew = np.mean(((data - mean_val) / std_val) ** 3)
        return skew
    
    def _calculate_kurtosis(self, data):
        """計算峰度"""
        if len(data) < 4:
            return np.nan
        mean_val = np.mean(data)
        std_val = np.std(data, ddof=1)
        if std_val == 0:
            return np.nan
        kurt = np.mean(((data - mean_val) / std_val) ** 4) - 3
        return kurt
    
    def _display_feature_statistics(self):
        """顯示特徵統計"""
        print("\n📊 Feature Statistics:")
        numeric_features = self.features.select_dtypes(include=[np.number])
        
        # 顯示關鍵特徵
        key_features = ['Ic_average', 'Rn', 'n_value', 'transition_width']
        for feature in key_features:
            if feature in numeric_features.columns:
                valid_data = numeric_features[feature].dropna()
                if len(valid_data) > 0:
                    mean_val = valid_data.mean()
                    std_val = valid_data.std()
                    print(f"  {feature}: {mean_val:.6e} ± {std_val:.6e} ({len(valid_data)} valid)")
    
    def create_advanced_images(self):
        """步驟3: 創建進階2D圖像"""
        print("\n=== Step 3: Advanced 2D Image Generation ===")
        
        resolution = self.config['image_resolution']
        
        # 創建網格
        y_grid = np.linspace(min(self.y_field_values), max(self.y_field_values), resolution[0])
        current_range = [self.data['appl_current'].min(), self.data['appl_current'].max()]
        current_grid = np.linspace(current_range[0], current_range[1], resolution[1])
        
        print(f"🎨 Creating images with resolution: {resolution}")
        print(f"📏 y_field grid: {len(y_grid)} points")
        print(f"📏 Current grid: {len(current_grid)} points")
        
        # 創建不同類型的圖像
        image_types = [
            ('voltage', self.voltage_column),
            ('dV_dI', 'dV_dI'),
            ('resistance', 'computed')  # 計算的電阻圖像
        ]
        
        for img_name, data_column in image_types:
            if img_name == 'resistance':
                # 計算電阻圖像
                image = self._create_resistance_image(y_grid, current_grid)
            else:
                image = self._create_interpolated_image(y_grid, current_grid, data_column)
            
            if image is not None:
                self.images[img_name] = image
                
                # 應用進階圖像處理
                processed_image = self._apply_advanced_image_processing(image)
                self.images[f'{img_name}_enhanced'] = processed_image
        
        print(f"✅ Generated {len(self.images)} images")
    
    def _create_interpolated_image(self, y_grid, current_grid, data_column):
        """創建插值圖像"""
        if data_column not in self.data.columns:
            return None
        
        try:
            from scipy.interpolate import griddata
            
            # 準備數據點
            points = self.data[['y_field', 'appl_current']].values
            values = self.data[data_column].values
            
            # 創建網格
            Y_grid, I_grid = np.meshgrid(y_grid, current_grid, indexing='ij')
            
            # 插值
            image = griddata(points, values, (Y_grid, I_grid), method='linear')
            
            return image
            
        except Exception as e:
            print(f"⚠️  Error creating {data_column} image: {e}")
            return None
    
    def _create_resistance_image(self, y_grid, current_grid):
        """創建電阻圖像"""
        try:
            # 計算局部電阻 R = dV/dI
            resistance_image = np.zeros((len(y_grid), len(current_grid)))
            
            for i, y_field in enumerate(y_grid):
                # 找到最接近的y_field值
                closest_field_idx = np.argmin(np.abs(np.array(self.y_field_values) - y_field))
                closest_field = self.y_field_values[closest_field_idx]
                
                field_data = self.data[self.data['y_field'] == closest_field]
                if len(field_data) > 0:
                    # 插值到current_grid
                    from scipy.interpolate import interp1d
                    if 'dV_dI' in field_data.columns:
                        current_data = field_data['appl_current'].values
                        resistance_data = field_data['dV_dI'].values
                        
                        if len(current_data) > 1:
                            f = interp1d(current_data, resistance_data, 
                                       bounds_error=False, fill_value=np.nan)
                            resistance_image[i, :] = f(current_grid)
            
            return resistance_image
            
        except Exception as e:
            print(f"⚠️  Error creating resistance image: {e}")
            return None
    
    def _apply_advanced_image_processing(self, image):
        """應用進階圖像處理"""
        if image is None:
            return None
        
        try:
            # 去噪
            denoised = ndimage.gaussian_filter(image, sigma=1.0)
            
            # 增強對比度
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            enhanced = scaler.fit_transform(denoised.reshape(-1, 1)).reshape(image.shape)
            
            return enhanced
            
        except Exception:
            return image
    
    def perform_machine_learning_analysis(self):
        """步驟4: 機器學習分析"""
        print("\n=== Step 4: Advanced Machine Learning Analysis ===")
        
        # 準備特徵矩陣
        numeric_features = self.features.select_dtypes(include=[np.number])
        feature_matrix = numeric_features.drop('y_field', axis=1, errors='ignore')
        
        # 處理缺失值
        feature_matrix = feature_matrix.fillna(feature_matrix.mean())
        
        if feature_matrix.empty:
            print("⚠️  No valid features for ML analysis")
            return
        
        print(f"🤖 ML analysis with {feature_matrix.shape[1]} features, {feature_matrix.shape[0]} samples")
        
        # 標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(feature_matrix)
        
        # PCA降維
        n_components = min(self.config['pca_components'], X_scaled.shape[1], X_scaled.shape[0])
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        print(f"📉 PCA: {X_scaled.shape[1]} → {X_pca.shape[1]} dimensions")
        print(f"📊 Explained variance ratio: {pca.explained_variance_ratio_}")
        
        self.ml_features = {
            'original': X_scaled,
            'pca': X_pca,
            'feature_names': feature_matrix.columns.tolist(),
            'pca_components': pca.components_,
            'explained_variance': pca.explained_variance_ratio_
        }
        
        # 聚類分析
        if self.config['clustering_enabled']:
            self._perform_clustering_analysis(X_pca)
        
        # 自編碼器特徵提取
        self._perform_autoencoder_analysis(X_scaled)
    
    def _perform_clustering_analysis(self, X_pca):
        """執行聚類分析"""
        print("\n🔍 Performing clustering analysis...")
        
        # K-means聚類
        best_k = 2
        best_score = -1
        
        for k in range(2, min(8, len(X_pca))):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_pca)
                score = silhouette_score(X_pca, labels)
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    
            except Exception:
                continue
        
        # 最佳K-means
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_pca)
        
        # DBSCAN聚類
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_pca)
        
        self.clustering_results = {
            'kmeans_labels': kmeans_labels,
            'kmeans_centers': kmeans.cluster_centers_,
            'best_k': best_k,
            'silhouette_score': best_score,
            'dbscan_labels': dbscan_labels,
            'n_clusters_dbscan': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        }
        
        print(f"🎯 Best K-means: k={best_k}, silhouette={best_score:.3f}")
        print(f"🎯 DBSCAN: {self.clustering_results['n_clusters_dbscan']} clusters")
    
    def _perform_autoencoder_analysis(self, X_scaled):
        """執行自編碼器分析"""
        print("\n🧠 Training autoencoder...")
        
        try:
            # 創建和訓練自編碼器
            autoencoder = AutoencoderModel(X_scaled.shape[1], latent_dim=3)
            autoencoder.train(X_scaled, epochs=50, batch_size=min(16, len(X_scaled)))
            
            # 提取潛在特徵
            latent_features = autoencoder.extract_features(X_scaled)
            
            self.ml_features['autoencoder'] = latent_features
            
            print(f"✅ Autoencoder: {X_scaled.shape[1]} → {latent_features.shape[1]} latent dimensions")
            
        except Exception as e:
            print(f"⚠️  Autoencoder analysis failed: {e}")
    
    def create_comprehensive_visualizations(self):
        """創建綜合可視化"""
        print("\n=== Step 5: Comprehensive Visualization ===")
        
        plt.figure(figsize=(20, 15))
        
        # 1. 數據概覽
        ax1 = plt.subplot(3, 4, 1)
        self._plot_data_overview(ax1)
        
        # 2. 特徵分佈
        ax2 = plt.subplot(3, 4, 2)
        self._plot_feature_distributions(ax2)
        
        # 3. 臨界電流分析
        ax3 = plt.subplot(3, 4, 3)
        self._plot_critical_current_analysis(ax3)
        
        # 4. 2D圖像
        ax4 = plt.subplot(3, 4, 4)
        self._plot_2d_images(ax4)
        
        # 5. PCA結果
        ax5 = plt.subplot(3, 4, 5)
        self._plot_pca_analysis(ax5)
        
        # 6. 聚類結果
        ax6 = plt.subplot(3, 4, 6)
        self._plot_clustering_results(ax6)
        
        # 7. Ic×Rn 分析
        ax7 = plt.subplot(3, 4, 7)
        self._plot_ic_rn_analysis(ax7)
        
        # 8. 相關性熱圖
        ax8 = plt.subplot(3, 4, 8)
        self._plot_correlation_heatmap(ax8)
        
        # 9-12. 詳細分析圖
        axes_detailed = [plt.subplot(3, 4, i) for i in range(9, 13)]
        self._plot_detailed_analysis(axes_detailed)
        
        plt.tight_layout()
        
        # 保存圖片
        output_filename = f"advanced_analysis_{self.data_path.replace('.csv', '')}.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"📊 Advanced visualization saved: {output_filename}")
        
        return output_filename
    
    def _plot_data_overview(self, ax):
        """繪製數據概覽"""
        if not self.data.empty:
            ax.scatter(self.data['y_field'], self.data['appl_current']*1e6, 
                      alpha=0.1, s=1, c=self.data[self.voltage_column])
            ax.set_xlabel('y_field')
            ax.set_ylabel('Current (µA)')
            ax.set_title('Data Distribution')
            plt.colorbar(ax.collections[0], ax=ax, label='Voltage')
    
    def _plot_feature_distributions(self, ax):
        """繪製特徵分佈"""
        key_features = ['Ic_average', 'Rn', 'n_value']
        for feature in key_features:
            if feature in self.features.columns:
                data = self.features[feature].dropna()
                if len(data) > 0:
                    ax.hist(data, alpha=0.7, label=feature, bins=20)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Feature Distributions')
        ax.legend()
        ax.set_yscale('log')
    
    def _plot_critical_current_analysis(self, ax):
        """繪製臨界電流分析"""
        if 'Ic_average' in self.features.columns:
            ic_data = self.features['Ic_average'].dropna() * 1e6
            y_fields = self.features.loc[ic_data.index, 'y_field']
            
            ax.plot(y_fields, ic_data, 'b-', alpha=0.7, linewidth=1)
            ax.scatter(y_fields, ic_data, c='red', s=10, alpha=0.7)
            ax.set_xlabel('y_field')
            ax.set_ylabel('Critical Current (µA)')
            ax.set_title('Critical Current vs Field')
            ax.grid(True, alpha=0.3)
    
    def _plot_2d_images(self, ax):
        """繪製2D圖像"""
        if 'voltage' in self.images:
            im = ax.imshow(self.images['voltage'], aspect='auto', cmap='viridis')
            ax.set_title('Voltage Image')
            ax.set_xlabel('Current Index')
            ax.set_ylabel('Field Index')
            plt.colorbar(im, ax=ax)
    
    def _plot_pca_analysis(self, ax):
        """繪製PCA分析"""
        if 'pca' in self.ml_features:
            pca_data = self.ml_features['pca']
            if pca_data.shape[1] >= 2:
                scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], 
                           c=self.features['y_field'], cmap='viridis', alpha=0.7)
                ax.set_xlabel(f"PC1 ({self.ml_features['explained_variance'][0]:.1%})")
                ax.set_ylabel(f"PC2 ({self.ml_features['explained_variance'][1]:.1%})")
                ax.set_title('PCA Visualization')
                plt.colorbar(scatter, ax=ax)
    
    def _plot_clustering_results(self, ax):
        """繪製聚類結果"""
        if self.clustering_results and 'pca' in self.ml_features:
            pca_data = self.ml_features['pca']
            if pca_data.shape[1] >= 2:
                labels = self.clustering_results['kmeans_labels']
                ax.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, 
                                   cmap='tab10', alpha=0.7)
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.set_title(f"K-means Clustering (k={self.clustering_results['best_k']})")
    
    def _plot_ic_rn_analysis(self, ax):
        """繪製 Ic×Rn 分析圖 (每個磁場下的臨界電流×正常電阻)"""
        # 檢查所需的列是否存在
        required_cols = ['Ic_positive', 'Ic_negative', 'Rn', 'y_field']
        available_cols = [col for col in required_cols if col in self.features.columns]
        
        if len(available_cols) >= 3:  # 至少需要一個Ic值、Rn和y_field
            # 過濾有效數據
            valid_data = self.features.dropna(subset=['Rn', 'y_field'])
            
            # 為每個y_field計算Ic×Rn
            ic_rn_positive = []
            ic_rn_negative = []
            y_field_values = []
            
            for _, row in valid_data.iterrows():
                y_field = row['y_field']
                rn = row['Rn']
                
                # 正向臨界電流 × Rn
                if 'Ic_positive' in row and not pd.isna(row['Ic_positive']):
                    ic_rn_pos = row['Ic_positive'] * rn * 1e6  # 轉換為 µV
                    ic_rn_positive.append(ic_rn_pos)
                    y_field_values.append(y_field)
                
                # 負向臨界電流 × Rn
                if 'Ic_negative' in row and not pd.isna(row['Ic_negative']):
                    ic_rn_neg = row['Ic_negative'] * rn * 1e6  # 轉換為 µV
                    ic_rn_negative.append(ic_rn_neg)
                    # 為負向添加相同的y_field值
                    if 'Ic_positive' not in row or pd.isna(row['Ic_positive']):
                        y_field_values.append(y_field)
            
            if ic_rn_positive or ic_rn_negative:
                # 繪製正向Ic×Rn
                if ic_rn_positive:
                    y_pos = y_field_values[:len(ic_rn_positive)]
                    ax.scatter(ic_rn_positive, y_pos, alpha=0.7, s=40, 
                             c='blue', label='Ic+ × Rn', marker='o')
                
                # 繪製負向Ic×Rn
                if ic_rn_negative:
                    y_neg = y_field_values[-len(ic_rn_negative):] if len(ic_rn_positive) > 0 else y_field_values
                    ax.scatter(ic_rn_negative, y_neg, alpha=0.7, s=40, 
                             c='red', label='Ic- × Rn', marker='^')
                
                ax.set_xlabel('Ic × Rn (µV)')
                ax.set_ylabel('Magnetic Field (T)')
                ax.set_title('Critical Current × Normal Resistance\nvs Magnetic Field')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                
                # 添加趨勢線
                all_ic_rn = ic_rn_positive + ic_rn_negative
                all_y_field = y_field_values
                
                if len(all_ic_rn) > 2:
                    try:
                        # 按y_field排序以便繪製平滑的趨勢線
                        sorted_data = sorted(zip(all_y_field, all_ic_rn))
                        y_sorted, ic_rn_sorted = zip(*sorted_data)
                        
                        z = np.polyfit(ic_rn_sorted, y_sorted, 1)
                        p = np.poly1d(z)
                        x_trend = np.linspace(min(all_ic_rn), max(all_ic_rn), 100)
                        ax.plot(x_trend, p(x_trend), "g--", alpha=0.6, linewidth=1, label='Trend')
                        ax.legend(fontsize=8)
                    except Exception:
                        pass
                
                # 顯示統計信息
                if all_ic_rn:
                    mean_product = np.mean(all_ic_rn)
                    ax.axvline(mean_product, color='orange', linestyle=':', alpha=0.7)
                    ax.text(0.02, 0.98, f'Mean: {mean_product:.1f} µV\nPoints: {len(all_ic_rn)}', 
                           transform=ax.transAxes, va='top', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No valid Ic×Rn data found', transform=ax.transAxes, 
                       ha='center', va='center')
                ax.set_title('Ic × Rn Analysis (No Data)')
        else:
            missing_cols = [col for col in required_cols if col not in self.features.columns]
            ax.text(0.5, 0.5, f'Missing columns: {missing_cols}', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=10)
            ax.set_title('Ic × Rn Analysis (Missing Data)')
    
    def _plot_correlation_heatmap(self, ax):
        """繪製相關性熱圖"""
        numeric_features = self.features.select_dtypes(include=[np.number])
        key_features = ['Ic_average', 'Rn', 'n_value', 'transition_width', 'dV_dI_mean']
        available_features = [f for f in key_features if f in numeric_features.columns]
        
        if len(available_features) > 1:
            corr_matrix = numeric_features[available_features].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Feature Correlations')
    
    def _plot_detailed_analysis(self, axes):
        """繪製詳細分析圖"""
        # 1. I-V 特性曲線 (多個 y_field 值)
        if len(axes) >= 1:
            self._plot_iv_characteristics(axes[0])
            
        # 2. 樣本 dV/dI 曲線比較
        if len(axes) >= 2:
            self._plot_sample_dvdi_curves(axes[1])
            
        # 3. 轉變寬度分析
        if len(axes) >= 3:
            self._plot_transition_analysis(axes[2])
            
        # 4. 噪聲分析 / 數據質量評估
        if len(axes) >= 4:
            self._plot_data_quality_analysis(axes[3])
    
    def _plot_iv_characteristics(self, ax):
        """繪製 I-V 特性曲線"""
        try:
            # 選擇幾個代表性的 y_field 值進行繪製
            n_curves = min(5, len(self.y_field_values))
            field_indices = np.linspace(0, len(self.y_field_values)-1, n_curves, dtype=int)
            
            colors = ['blue', 'green', 'red', 'orange', 'purple'][:n_curves]
            
            for i, field_idx in enumerate(field_indices):
                y_field = self.y_field_values[field_idx]
                field_data = self.data[self.data['y_field'] == y_field].sort_values('appl_current')
                
                if len(field_data) > 0:
                    current = field_data['appl_current'].values * 1e6  # Convert to µA
                    voltage = field_data[self.voltage_column].values * 1e6  # Convert to µV
                    
                    # 繪製 I-V 曲線
                    ax.plot(current, voltage, 'o-', color=colors[i], 
                           markersize=2, linewidth=1, alpha=0.8,
                           label=f'y_field={y_field:.6f}')
            
            ax.set_xlabel('Applied Current (µA)')
            ax.set_ylabel('Measured Voltage (µV)')
            ax.set_title('I-V Characteristics')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='best')
            
            # 設置坐標軸範圍
            ax.set_xlim(self.data['appl_current'].min()*1e6*1.1, 
                       self.data['appl_current'].max()*1e6*1.1)
            
        except Exception as e:
            print(f"⚠️  Warning: Could not plot I-V characteristics: {e}")
            ax.text(0.5, 0.5, 'I-V Characteristics\n(Error in plotting)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('I-V Characteristics')
    
    def _plot_sample_dvdi_curves(self, ax):
        """繪製樣本 dV/dI 曲線"""
        try:
            # 選擇中間的 y_field 值作為示例
            mid_idx = len(self.y_field_values) // 2
            sample_y_field = self.y_field_values[mid_idx]
            sample_data = self.data[self.data['y_field'] == sample_y_field].sort_values('appl_current')
            
            if len(sample_data) > 0 and 'dV_dI' in sample_data.columns:
                current = sample_data['appl_current'].values * 1e6  # µA
                dV_dI = sample_data['dV_dI'].values  # Ω
                
                # 繪製原始 dV/dI 曲線
                ax.plot(current, dV_dI, 'b-', linewidth=1.5, alpha=0.7, label='dV/dI')
                
                # 標記臨界電流位置
                if 'Ic_average' in self.features.columns:
                    feature_row = self.features[self.features['y_field'] == sample_y_field]
                    if len(feature_row) > 0:
                        ic_avg = feature_row['Ic_average'].iloc[0] * 1e6  # µA
                        ax.axvline(x=ic_avg, color='red', linestyle='--', alpha=0.8, 
                                  label=f'Ic = {ic_avg:.2f} µA')
                        ax.axvline(x=-ic_avg, color='red', linestyle='--', alpha=0.8)
                
                ax.set_xlabel('Applied Current (µA)')
                ax.set_ylabel('dV/dI (Ω)')
                ax.set_title(f'Sample dV/dI Curve (y_field={sample_y_field:.6f})')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                
                # 設置 y 軸範圍，避免極值
                dV_dI_clean = dV_dI[~np.isinf(dV_dI)]
                if len(dV_dI_clean) > 0:
                    y_min, y_max = np.percentile(dV_dI_clean, [5, 95], axis=None)
                    ax.set_ylim(y_min, y_max)
            else:
                ax.text(0.5, 0.5, 'Sample dV/dI Curve\n(No dV/dI data)', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Sample dV/dI Curve')
                
        except Exception as e:
            print(f"⚠️  Warning: Could not plot dV/dI curves: {e}")
            ax.text(0.5, 0.5, 'Sample dV/dI Curve\n(Error in plotting)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Sample dV/dI Curve')
    
    def _plot_transition_analysis(self, ax):
        """繪製超導轉變分析"""
        try:
            if 'transition_width' in self.features.columns:
                # 繪製轉變寬度隨 y_field 的變化
                tw_data = self.features['transition_width'].dropna() * 1e6  # Convert to µA
                y_fields = self.features.loc[tw_data.index, 'y_field']
                
                ax.plot(y_fields, tw_data, 'g-o', markersize=3, linewidth=1.5, alpha=0.8)
                ax.set_xlabel('y_field')
                ax.set_ylabel('Transition Width (µA)')
                ax.set_title('Superconducting Transition Width')
                ax.grid(True, alpha=0.3)
                
                # 添加統計信息
                mean_tw = tw_data.mean()
                std_tw = tw_data.std()
                ax.text(0.05, 0.95, f'Mean: {mean_tw:.2f} µA\nStd: {std_tw:.2f} µA', 
                       transform=ax.transAxes, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'Transition Analysis\n(No transition width data)', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Transition Analysis')
                
        except Exception as e:
            print(f"⚠️  Warning: Could not plot transition analysis: {e}")
            ax.text(0.5, 0.5, 'Transition Analysis\n(Error in plotting)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Transition Analysis')
    
    def _plot_data_quality_analysis(self, ax):
        """繪製數據質量分析"""
        try:
            # 計算每個 y_field 的數據點數量
            data_counts = self.data.groupby('y_field').size()
            
            ax.plot(data_counts.index, data_counts.values, 'purple', marker='o', 
                   markersize=2, linewidth=1, alpha=0.7)
            ax.set_xlabel('y_field')
            ax.set_ylabel('Number of Data Points')
            ax.set_title('Data Coverage Analysis')
            ax.grid(True, alpha=0.3)
            
            # 添加統計信息
            mean_points = data_counts.mean()
            min_points = data_counts.min()
            max_points = data_counts.max()
            
            info_text = f'Mean: {mean_points:.1f}\nMin: {min_points}\nMax: {max_points}'
            ax.text(0.05, 0.95, info_text, transform=ax.transAxes, 
                   verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # 標記數據密度較低的區域
            threshold = float(mean_points * 0.5)
            low_density_mask = np.array(data_counts.values) < threshold
            low_density_indices = data_counts.index[low_density_mask]
            low_density_values = data_counts.values[low_density_mask]
            if len(low_density_values) > 0:
                ax.scatter(low_density_indices, low_density_values, 
                          color='red', s=20, alpha=0.8, label='Low Density')
                ax.legend(fontsize=8)
                
        except Exception as e:
            print(f"⚠️  Warning: Could not plot data quality analysis: {e}")
            ax.text(0.5, 0.5, 'Data Quality Analysis\n(Error in plotting)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Data Quality Analysis')
    
    def generate_comprehensive_report(self):
        """生成綜合報告"""
        print("\n" + "="*80)
        print("           ADVANCED SUPERCONDUCTOR DATA ANALYSIS REPORT")
        print("="*80)
        
        # 基本信息
        print(f"\n📄 Dataset: {self.data_path}")
        print(f"📊 Data Points: {len(self.data):,}")
        print(f"🧭 y_field Range: {self.y_field_values[0]:.6f} - {self.y_field_values[-1]:.6f}")
        print(f"🔢 Field Steps: {len(self.y_field_values)}")
        print(f"⚡ Current Range: {self.data['appl_current'].min()*1e6:.2f} - {self.data['appl_current'].max()*1e6:.2f} µA")
        
        # 特徵統計
        print("\n🎯 FEATURE ANALYSIS:")
        print(f"📈 Total Features Extracted: {len(self.features.columns)-1}")
        
        key_features = {
            'Ic_average': ('Critical Current', 'µA', 1e6),
            'Rn': ('Normal Resistance', 'Ω', 1),
            'n_value': ('n-value', '', 1),
            'transition_width': ('Transition Width', 'µA', 1e6)
        }
        
        for feature, (name, unit, scale) in key_features.items():
            if feature in self.features.columns:
                data = self.features[feature].dropna()
                if len(data) > 0:
                    mean_val = data.mean() * scale
                    std_val = data.std() * scale
                    print(f"  {name}: {mean_val:.3f} ± {std_val:.3f} {unit}")
        
        # 機器學習結果
        if self.ml_features:
            print("\n🤖 MACHINE LEARNING ANALYSIS:")
            if 'pca' in self.ml_features:
                print(f"  PCA Dimensions: {self.ml_features['original'].shape[1]} → {self.ml_features['pca'].shape[1]}")
                total_variance = sum(self.ml_features['explained_variance'])
                print(f"  Cumulative Variance Explained: {total_variance:.1%}")
            
            if self.clustering_results:
                print(f"  Optimal Clusters (K-means): {self.clustering_results['best_k']}")
                print(f"  Silhouette Score: {self.clustering_results['silhouette_score']:.3f}")
                print(f"  DBSCAN Clusters: {self.clustering_results['n_clusters_dbscan']}")
        
        # 圖像分析
        if self.images:
            print("\n🖼️  IMAGE ANALYSIS:")
            print(f"  Generated Images: {len(self.images)}")
            print(f"  Image Types: {list(self.images.keys())}")
        
        # 數據質量評估
        print("\n📊 DATA QUALITY ASSESSMENT:")
        missing_rate = self.features.isnull().sum().sum() / (len(self.features) * len(self.features.columns))
        print(f"  Missing Data Rate: {missing_rate:.1%}")
        
        # 物理解釋
        print("\n🔬 PHYSICAL INTERPRETATION:")
        if 'Ic_average' in self.features.columns:
            ic_data = self.features['Ic_average'].dropna()
            if len(ic_data) > 0:
                ic_variability = ic_data.std() / ic_data.mean()
                print(f"  Critical Current Variability: {ic_variability:.1%}")
                
                if ic_variability < 0.1:
                    quality = "🟢 Excellent"
                elif ic_variability < 0.2:
                    quality = "🟡 Good"
                else:
                    quality = "🔴 Poor"
                print(f"  Sample Quality: {quality}")
        
        # 建議
        print("\n💡 RECOMMENDATIONS:")
        print("  1. ✅ Comprehensive feature extraction completed")
        print("  2. ✅ Machine learning analysis provides dimensional insights")
        print("  3. ✅ Clustering reveals data structure patterns")
        print("  4. 🔍 Consider physics-based model validation")
        print("  5. 🎯 Optimize measurement parameters based on critical current trends")
        
        print("\n" + "="*80)
        print("                    🎉 ANALYSIS COMPLETED SUCCESSFULLY! 🎉")
        print("="*80)
    
    def run_complete_analysis(self):
        """執行完整分析流程"""
        print("🚀 Starting Advanced Superconductor Analysis...")
        
        try:
            # 步驟1: 數據預處理
            self.load_and_preprocess_data()
            
            # 步驟2: 特徵提取
            self.extract_enhanced_features()
            
            # 步驟3: 圖像生成
            self.create_advanced_images()
            
            # 步驟4: 機器學習分析
            self.perform_machine_learning_analysis()
            
            # 步驟5: 可視化
            output_file = self.create_comprehensive_visualizations()
            
            # 步驟6: 報告生成
            self.generate_comprehensive_report()
            
            print("\n✅ Analysis completed successfully!")
            print(f"📊 Results saved as: {output_file}")
            
            return {
                'features': self.features,
                'images': self.images,
                'ml_features': self.ml_features,
                'clustering': self.clustering_results,
                'output_file': output_file
            }
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """主函數 - 演示如何使用分析器"""
    
    # 配置參數
    config = {
        'outlier_threshold': 3.0,
        'smoothing_window': 3,
        'pca_components': 5,
        'clustering_enabled': True,
        'advanced_features': True,
        'image_resolution': (150, 200)
    }
    
    # 分析不同數據集
    datasets = ['317.csv', '500.csv']  # 可以根據需要修改
    
    for dataset in datasets:
        try:
            print(f"\n{'='*60}")
            print(f"🔬 Analyzing dataset: {dataset}")
            print(f"{'='*60}")
            
            # 創建分析器
            analyzer = AdvancedSuperconductorAnalyzer(dataset, config)
            
            # 執行分析
            results = analyzer.run_complete_analysis()
            
            if results:
                print(f"✅ Successfully analyzed {dataset}")
            else:
                print(f"❌ Failed to analyze {dataset}")
                
        except FileNotFoundError:
            print(f"⚠️  Dataset {dataset} not found, skipping...")
        except Exception as e:
            print(f"❌ Error analyzing {dataset}: {e}")

if __name__ == '__main__':
    main()
