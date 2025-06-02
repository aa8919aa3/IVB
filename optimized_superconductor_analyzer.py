#!/usr/bin/env python3
"""
優化版進階超導體分析器 - 並行處理實現
實現並行特徵提取和性能優化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial
import time
import os
from advanced_superconductor_analyzer import AdvancedSuperconductorAnalyzer

class OptimizedSuperconductorAnalyzer(AdvancedSuperconductorAnalyzer):
    """優化版超導體分析器，支持並行處理"""
    
    def __init__(self, filename, use_parallel=True, max_workers=None):
        """初始化優化版分析器"""
        super().__init__(filename)
        self.use_parallel = use_parallel
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.chunk_size = 1000  # 每個處理塊的大小
        
    def extract_enhanced_features_parallel(self):
        """並行版本的增強特徵提取"""
        print("\n=== Step 2: Parallel Enhanced Feature Extraction ===")
        
        if not self.use_parallel or len(self.y_field_values) < 4:
            # 如果不使用並行或數據量太小，使用原始方法
            return self.extract_enhanced_features()
        
        start_time = time.time()
        
        # 將y_field值分組，準備並行處理
        n_groups = min(self.max_workers, len(self.y_field_values))
        group_size = len(self.y_field_values) // n_groups
        
        y_field_groups = []
        for i in range(n_groups):
            start_idx = i * group_size
            if i == n_groups - 1:  # 最後一組包含剩餘的所有值
                end_idx = len(self.y_field_values)
            else:
                end_idx = (i + 1) * group_size
            y_field_groups.append(self.y_field_values[start_idx:end_idx])
        
        print(f"🔄 Parallel processing with {n_groups} workers")
        print(f"🔄 Group sizes: {[len(group) for group in y_field_groups]}")
        
        # 使用進程池進行並行處理
        with ProcessPoolExecutor(max_workers=n_groups) as executor:
            # 為每個進程準備數據和處理函數
            process_func = partial(self._process_y_field_group, 
                                 data=self.data.copy())
            
            # 提交所有任務
            futures = [executor.submit(process_func, group) for group in y_field_groups]
            
            # 收集結果
            all_features = []
            for i, future in enumerate(futures):
                try:
                    group_features = future.result(timeout=300)  # 5分鐘超時
                    all_features.extend(group_features)
                    print(f"✅ Group {i+1}/{n_groups} completed ({len(group_features)} features)")
                except Exception as e:
                    print(f"❌ Group {i+1} failed: {e}")
                    # 回退到序列處理
                    return self.extract_enhanced_features()
        
        # 合併所有特徵
        if all_features:
            self.features = pd.concat(all_features, ignore_index=True)
            print(f"✅ Parallel feature extraction completed")
            print(f"✅ Extracted features dimensions: {self.features.shape}")
            print(f"📊 Total features: {len(self.features.columns)-1}")
        else:
            print("❌ Parallel processing failed, falling back to sequential")
            return self.extract_enhanced_features()
        
        processing_time = time.time() - start_time
        print(f"⏱️  Parallel processing time: {processing_time:.2f} seconds")
        
        return self.features
    
    @staticmethod
    def _process_y_field_group(y_field_group, data):
        """處理單個y_field組的靜態方法（用於並行處理）"""
        features_list = []
        
        for y_field in y_field_group:
            # 提取該y_field的數據
            field_data = data[data['y_field'] == y_field].copy()
            
            if len(field_data) < 5:  # 數據點太少，跳過
                continue
            
            # 確保數據按電流排序
            field_data = field_data.sort_values('appl_current')
            
            current = field_data['appl_current'].values
            voltage_col = [col for col in field_data.columns if 'voltage' in col][0]
            voltage = field_data[voltage_col].values
            dv_di = field_data['dV_dI'].values
            
            # 計算特徵
            features = OptimizedSuperconductorAnalyzer._extract_features_for_curve(
                current, voltage, dv_di, y_field
            )
            
            if features:
                features_list.append(features)
        
        # 轉換為DataFrame
        if features_list:
            return [pd.DataFrame(features_list)]
        else:
            return []
    
    @staticmethod
    def _extract_features_for_curve(current, voltage, dv_di, y_field):
        """為單條曲線提取特徵的靜態方法"""
        try:
            features = {'y_field': y_field}
            
            # 1. 基本統計特徵
            features['V_mean'] = np.mean(voltage)
            features['V_std'] = np.std(voltage)
            features['V_max'] = np.max(voltage)
            features['V_min'] = np.min(voltage)
            features['V_range'] = features['V_max'] - features['V_min']
            
            features['dVdI_mean'] = np.mean(dv_di)
            features['dVdI_std'] = np.std(dv_di)
            features['dVdI_max'] = np.max(dv_di)
            features['dVdI_min'] = np.min(dv_di)
            
            # 2. 臨界電流估算（簡化版）
            try:
                # 找到dV/dI的最大值位置作為臨界電流的近似
                ic_idx = np.argmax(dv_di)
                features['Ic_from_max_dVdI'] = abs(current[ic_idx])
                
                # 使用閾值方法
                voltage_threshold = features['V_max'] * 0.1
                threshold_indices = np.where(np.abs(voltage) > voltage_threshold)[0]
                if len(threshold_indices) > 0:
                    features['Ic_threshold'] = abs(current[threshold_indices[0]])
                else:
                    features['Ic_threshold'] = features['Ic_from_max_dVdI']
                
                # 平均臨界電流
                features['Ic_average'] = (features['Ic_from_max_dVdI'] + features['Ic_threshold']) / 2
            except:
                features['Ic_from_max_dVdI'] = np.nan
                features['Ic_threshold'] = np.nan
                features['Ic_average'] = np.nan
            
            # 3. 法向電阻（簡化版）
            try:
                high_current_idx = np.where(np.abs(current) > features['Ic_average'] * 2)[0]
                if len(high_current_idx) > 0:
                    features['Rn'] = np.mean(np.abs(voltage[high_current_idx] / current[high_current_idx]))
                else:
                    features['Rn'] = np.abs(voltage[-1] / current[-1]) if current[-1] != 0 else np.nan
            except:
                features['Rn'] = np.nan
            
            # 4. 轉換寬度
            try:
                ic_10 = features['Ic_average'] * 0.1
                ic_90 = features['Ic_average'] * 0.9
                features['transition_width'] = ic_90 - ic_10
            except:
                features['transition_width'] = np.nan
            
            # 5. 功率特徵
            power = voltage * current
            features['power_max'] = np.max(np.abs(power))
            features['power_mean'] = np.mean(np.abs(power))
            
            # 6. 頻譜特徵（簡化版）
            try:
                voltage_fft = np.fft.fft(voltage)
                power_spectrum = np.abs(voltage_fft)
                features['spectral_centroid'] = np.sum(np.arange(len(power_spectrum)) * power_spectrum) / np.sum(power_spectrum)
                features['spectral_spread'] = np.sqrt(np.sum((np.arange(len(power_spectrum)) - features['spectral_centroid'])**2 * power_spectrum) / np.sum(power_spectrum))
            except:
                features['spectral_centroid'] = np.nan
                features['spectral_spread'] = np.nan
            
            return features
            
        except Exception as e:
            print(f"Warning: Feature extraction failed for y_field {y_field}: {e}")
            return None
    
    def create_performance_comparison(self, original_analyzer=None):
        """創建性能比較報告"""
        if original_analyzer is None:
            return
        
        print("\n=== Performance Comparison ===")
        
        # 比較處理時間
        if hasattr(self, 'processing_times') and hasattr(original_analyzer, 'processing_times'):
            orig_time = sum(original_analyzer.processing_times.values())
            opt_time = sum(self.processing_times.values())
            speedup = orig_time / opt_time if opt_time > 0 else 0
            
            print(f"Original processing time: {orig_time:.2f}s")
            print(f"Optimized processing time: {opt_time:.2f}s") 
            print(f"Speedup: {speedup:.2f}x")
            
            # 創建性能比較圖
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 處理時間比較
            methods = ['Original', 'Optimized']
            times = [orig_time, opt_time]
            ax1.bar(methods, times, color=['lightcoral', 'lightgreen'])
            ax1.set_ylabel('Processing Time (seconds)')
            ax1.set_title('Processing Time Comparison')
            
            # 加速比
            ax2.bar(['Speedup'], [speedup], color='gold')
            ax2.set_ylabel('Speedup Factor')
            ax2.set_title(f'Performance Improvement: {speedup:.2f}x')
            
            plt.tight_layout()
            plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
            print("📊 Performance comparison saved: performance_comparison.png")

def test_optimization():
    """測試優化效果"""
    print("🚀 TESTING OPTIMIZATION PERFORMANCE")
    print("="*50)
    
    dataset = "500.csv"  # 使用中等大小的數據集
    
    # 測試原始版本
    print("\n📊 Testing Original Analyzer...")
    start_time = time.time()
    original = AdvancedSuperconductorAnalyzer(dataset)
    original.load_and_preprocess_data()
    original.extract_enhanced_features()
    original_time = time.time() - start_time
    
    # 測試優化版本
    print("\n⚡ Testing Optimized Analyzer...")
    start_time = time.time()
    optimized = OptimizedSuperconductorAnalyzer(dataset, use_parallel=True)
    optimized.load_and_preprocess_data()
    optimized.extract_enhanced_features_parallel()
    optimized_time = time.time() - start_time
    
    # 比較結果
    print("\n" + "="*50)
    print("📈 PERFORMANCE COMPARISON RESULTS")
    print("="*50)
    print(f"Original time: {original_time:.2f} seconds")
    print(f"Optimized time: {optimized_time:.2f} seconds")
    
    if optimized_time > 0:
        speedup = original_time / optimized_time
        print(f"Speedup: {speedup:.2f}x")
        
        if speedup > 1:
            print("✅ Optimization successful!")
        else:
            print("⚠️  No significant speedup (overhead may dominate for this dataset size)")
    else:
        print("❌ Optimization test failed")
    
    # 驗證結果一致性
    if len(original.features) == len(optimized.features):
        print("✅ Feature count consistency verified")
    else:
        print(f"⚠️  Feature count mismatch: {len(original.features)} vs {len(optimized.features)}")
    
    return original_time, optimized_time

if __name__ == "__main__":
    test_optimization()
