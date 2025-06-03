#!/usr/bin/env python3
"""
測試原始數據 Josephson 擬合修正
============================

驗證完整 Josephson 擬合確實使用原始數據而非去趨勢化數據。

作者：GitHub Copilot
日期：2025年6月3日
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# 添加當前目錄到 Python 路徑
sys.path.append('/Users/albert-mac/Code/GitHub/IVB/Sim')
from Fit import JosephsonAnalyzer

def load_test_data(csv_path):
    """載入測試數據"""
    try:
        data = pd.read_csv(csv_path)
        print(f"✅ 成功載入測試數據: {csv_path}")
        print(f"   數據點數: {len(data)}")
        print(f"   y_field 範圍: {data['y_field'].min():.6f} 到 {data['y_field'].max():.6f}")
        print(f"   Ic 範圍: {data['Ic'].min():.2e} 到 {data['Ic'].max():.2e}")
        return data
    except Exception as e:
        print(f"❌ 載入數據失敗: {e}")
        return None

def test_original_data_fitting():
    """測試原始數據擬合"""
    print("🔬 測試原始數據 Josephson 擬合修正")
    print("="*60)
    
    # 載入測試數據（使用較小的數據集進行快速測試）
    csv_path = '/Users/albert-mac/Code/GitHub/IVB/Ic/kay164Ic+.csv'
    data = load_test_data(csv_path)
    
    if data is None:
        print("❌ 無法載入測試數據，退出測試")
        return
    
    # 準備數據
    phi_ext = data['y_field'].values
    I_s = data['Ic'].values
    
    # 估計誤差（10% 的數據變異）
    I_s_error = np.abs(I_s) * 0.1
    
    # 創建數據字典
    data_dict = {
        'Phi_ext': phi_ext,
        'I_s': I_s,
        'I_s_error': I_s_error
    }
    
    # 估計基本參數
    estimated_params = {
        'f': 1.0 / (2 * np.pi),
        'Ic': np.max(I_s),
        'phi_0': 0.0,
        'T': 0.5
    }
    
    # 創建分析器
    analyzer = JosephsonAnalyzer()
    model_type = 'test_original_data'
    model_name = 'Test Original Data Fitting'
    
    # 添加數據到分析器
    analyzer.add_simulation_data(
        model_type=model_type,
        data=data_dict,
        parameters=estimated_params,
        model_name=model_name
    )
    
    print(f"\n📊 數據統計:")
    print(f"   原始數據平均值: {np.mean(I_s):.6e}")
    print(f"   原始數據標準差: {np.std(I_s):.6e}")
    print(f"   原始數據範圍: {np.max(I_s) - np.min(I_s):.6e}")
    
    # 執行 Lomb-Scargle 分析（使用去趨勢化進行頻率檢測）
    print(f"\n🔧 執行 Lomb-Scargle 分析（去趨勢化用於頻率檢測）...")
    ls_result = analyzer.analyze_with_lomb_scargle(model_type, detrend_order=1)
    
    if ls_result:
        print(f"\n📈 Lomb-Scargle 結果:")
        print(f"   檢測頻率: {ls_result['best_frequency']:.6e}")
        print(f"   檢測振幅: {ls_result['amplitude']:.6e}")
        print(f"   去趨勢化偏移: {ls_result['offset']:.6e}")
        
        # 執行完整 Josephson 擬合（使用原始數據）
        print(f"\n🚀 執行完整 Josephson 方程式擬合（使用原始數據）...")
        fitter = analyzer.fit_complete_josephson_equation(
            model_type=model_type,
            use_lbfgsb=True,
            save_results=True
        )
        
        if fitter:
            # 獲取擬合參數
            fitted_params = fitter.get_fitted_parameters()
            
            print(f"\n📋 關鍵參數比較:")
            print("-"*50)
            print(f"原始數據統計:")
            print(f"   平均值 (真實基線): {np.mean(I_s):.6e}")
            print(f"   標準差: {np.std(I_s):.6e}")
            
            print(f"\nLomb-Scargle 結果 (基於去趨勢化數據):")
            print(f"   振幅: {ls_result['amplitude']:.6e}")
            print(f"   偏移: {ls_result['offset']:.6e}")
            
            print(f"\n完整 Josephson 擬合結果 (基於原始數據):")
            print(f"   I_c: {fitted_params['I_c']['value']:.6e} ± {fitted_params['I_c']['stderr']:.6e}")
            print(f"   C (常數項): {fitted_params['C']['value']:.6e} ± {fitted_params['C']['stderr']:.6e}")
            print(f"   r (線性項): {fitted_params['r']['value']:.6e} ± {fitted_params['r']['stderr']:.6e}")
            print(f"   f (頻率): {fitted_params['f']['value']:.6e} ± {fitted_params['f']['stderr']:.6e}")
            
            # 驗證擬合曲線包含常數項
            fitted_curve = fitter.calculate_fitted_curve(phi_ext)
            print(f"\n🔍 擬合曲線驗證:")
            print(f"   擬合曲線平均值: {np.mean(fitted_curve):.6e}")
            print(f"   擬合曲線範圍: {np.max(fitted_curve) - np.min(fitted_curve):.6e}")
            print(f"   與原始數據平均值差異: {abs(np.mean(fitted_curve) - np.mean(I_s)):.6e}")
            
            # 計算 R²
            from sklearn.metrics import r2_score
            r2 = r2_score(I_s, fitted_curve)
            print(f"   R²: {r2:.6f}")
            
            # 檢查常數項是否合理
            expected_baseline = np.mean(I_s)
            fitted_baseline = fitted_params['C']['value']
            baseline_diff = abs(fitted_baseline - expected_baseline)
            baseline_ratio = baseline_diff / expected_baseline
            
            print(f"\n✅ 常數項 C 驗證:")
            print(f"   預期基線 (原始數據平均): {expected_baseline:.6e}")
            print(f"   擬合基線 (C 參數): {fitted_baseline:.6e}")
            print(f"   差異: {baseline_diff:.6e}")
            print(f"   相對差異: {baseline_ratio*100:.2f}%")
            
            if baseline_ratio < 0.1:  # 10% 以內認為合理
                print("   ✅ 常數項 C 設定合理，使用了原始數據的真實基線")
            else:
                print("   ⚠️  常數項 C 可能不太合理，需要進一步檢查")
            
            # 生成比較圖
            print(f"\n📊 生成比較分析圖...")
            analyzer.compare_lomb_scargle_vs_josephson_fit(model_type, save_plot=True)
            
            print(f"\n✅ 測試完成！")
            print("="*60)
            
            return True
        else:
            print("❌ 完整 Josephson 擬合失敗")
            return False
    else:
        print("❌ Lomb-Scargle 分析失敗")
        return False

if __name__ == "__main__":
    test_original_data_fitting()
