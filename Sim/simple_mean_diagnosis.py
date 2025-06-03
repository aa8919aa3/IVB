#!/usr/bin/env python3
"""
簡化的 Josephson 擬合平均值差異診斷工具
=====================================

專注於分析為什麼擬合曲線的平均值與原始資料平均值差異很大。

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
from Fit import JosephsonFitter, complete_josephson_equation

def load_and_analyze_dataset(dataset_name, csv_path):
    """載入並分析數據集"""
    
    print(f"\n🔬 分析 {dataset_name}")
    print("="*60)
    
    # 1. 載入數據
    try:
        data = pd.read_csv(csv_path)
        phi_ext = np.array(data.iloc[:, 0].values, dtype=float)  # 確保是 numpy array
        I_s = np.array(data.iloc[:, 1].values, dtype=float)       # 確保是 numpy array
        
        print(f"✅ 成功載入數據: {csv_path}")
        print(f"   數據點數: {len(phi_ext)}")
        print(f"   phi_ext 範圍: {phi_ext.min():.6f} 到 {phi_ext.max():.6f}")
        print(f"   I_s 範圍: {I_s.min():.2e} 到 {I_s.max():.2e}")
        print(f"   I_s 平均值: {np.mean(I_s):.6e}")
        print(f"   I_s 標準差: {np.std(I_s):.6e}")
        
    except Exception as e:
        print(f"❌ 載入數據失敗: {str(e)}")
        return
    
    original_mean = np.mean(I_s)
    
    # 2. 執行 Josephson 擬合
    try:
        print(f"\n🔧 執行 Josephson 擬合...")
        fitter = JosephsonFitter()
        
        # 創建模型
        fitter.create_model()
        
        # 執行擬合 (不使用 Lomb-Scargle 結果，因為可能有問題)
        fit_result = fitter.fit(
            phi_ext=phi_ext,
            I_s=I_s,
            I_s_error=None,
            lomb_scargle_result=None,  # 不使用可能有問題的 Lomb-Scargle 結果
            method='lbfgsb'
        )
        
        if fit_result is None:
            print(f"❌ 擬合失敗")
            return
            
        print(f"✅ 擬合成功")
        
    except Exception as e:
        print(f"❌ 擬合過程出錯: {str(e)}")
        return
    
    # 3. 分析擬合結果
    fitted_params = fitter.get_fitted_parameters()
    fitted_curve = fitter.calculate_fitted_curve(phi_ext)
    fitted_mean = np.mean(fitted_curve)
    
    print(f"\n📋 擬合參數:")
    for param_name, param_info in fitted_params.items():
        print(f"   {param_name}: {param_info['value']:.6e}")
    
    print(f"\n📊 平均值比較:")
    print(f"   原始資料平均值: {original_mean:.6e}")
    print(f"   擬合曲線平均值: {fitted_mean:.6e}")
    print(f"   絕對差異: {abs(fitted_mean - original_mean):.6e}")
    
    if original_mean != 0:
        relative_diff = abs(fitted_mean - original_mean) / abs(original_mean) * 100
        print(f"   相對差異: {relative_diff:.2f}%")
        
        # 如果相對差異很大，進行詳細分析
        if relative_diff > 5.0:
            print(f"\n🚨 平均值差異過大！進行詳細分析...")
            analyze_components(phi_ext, fitted_params, original_mean, fitted_mean)
    
    return {
        'dataset': dataset_name,
        'original_mean': original_mean,
        'fitted_mean': fitted_mean,
        'fitted_params': fitted_params,
        'phi_ext': phi_ext,
        'I_s': I_s,
        'fitted_curve': fitted_curve
    }

def analyze_components(phi_ext, fitted_params, original_mean, fitted_mean):
    """分析 Josephson 方程式各組成部分"""
    
    # 提取參數
    I_c = fitted_params['I_c']['value']
    f = fitted_params['f']['value']  
    d = fitted_params['d']['value']
    phi_0 = fitted_params['phi_0']['value']
    T = fitted_params['T']['value']
    r = fitted_params['r']['value']
    C = fitted_params['C']['value']
    
    print(f"\n🔍 組成部分分析:")
    
    # 計算相位
    phase = 2 * np.pi * f * (phi_ext - d) - phi_0
    
    # 計算各組成部分
    sin_half_phase = np.sin(phase / 2)
    denominator = np.sqrt(1 - T * sin_half_phase**2)
    josephson_term = I_c * np.sin(phase) / denominator
    linear_term = r * (phi_ext - d)
    
    # 計算平均值
    josephson_mean = np.mean(josephson_term)
    linear_mean = np.mean(linear_term)
    
    print(f"   Josephson 項平均值: {josephson_mean:.6e}")
    print(f"   線性項平均值: {linear_mean:.6e}")
    print(f"   常數項 C: {C:.6e}")
    print(f"   理論總和: {josephson_mean + linear_mean + C:.6e}")
    print(f"   實際擬合平均值: {fitted_mean:.6e}")
    
    # 診斷問題
    print(f"\n🔍 問題診斷:")
    
    # 檢查常數項
    c_diff = abs(C - original_mean)
    if c_diff < 1e-10:
        print(f"   ✅ 常數項 C 正確設置 (差異: {c_diff:.2e})")
    else:
        print(f"   ❌ 常數項 C 設置不正確 (差異: {c_diff:.2e})")
    
    # 檢查線性項貢獻
    linear_contribution = abs(linear_mean)
    if linear_contribution > abs(original_mean) * 0.01:  # 如果線性項貢獻超過原始平均值的1%
        print(f"   ⚠️  線性項貢獻過大: {linear_mean:.6e}")
        print(f"      r 參數: {r:.6e}")
        print(f"      (phi_ext - d) 平均值: {np.mean(phi_ext - d):.6e}")
        print(f"      建議: 確保 phi_ext 資料正確中心化")
    else:
        print(f"   ✅ 線性項貢獻合理: {linear_mean:.6e}")
    
    # 檢查 Josephson 項貢獻
    if abs(josephson_mean) > abs(original_mean) * 0.01:
        print(f"   ⚠️  Josephson 項平均值非零: {josephson_mean:.6e}")
        print(f"      這可能表示相位或頻率參數有問題")
    else:
        print(f"   ✅ Josephson 項平均值接近零: {josephson_mean:.6e}")
    
    # 資料中心化檢查
    phi_center = np.mean(phi_ext)
    print(f"\n📊 資料中心化檢查:")
    print(f"   phi_ext 平均值: {phi_center:.6e}")
    print(f"   d 參數值: {d:.6e}")
    print(f"   (phi_ext - d) 平均值: {np.mean(phi_ext - d):.6e}")
    
    if abs(np.mean(phi_ext - d)) > 1e-6:
        print(f"   ⚠️  資料未正確中心化！")
        print(f"      建議: 調整初始參數估計方法")

def main():
    """主程序"""
    
    print("🔍 Josephson 擬合平均值差異診斷工具")
    print("="*60)
    
    # 要分析的數據集
    datasets = [
        ("Kay164 Ic+", "/Users/albert-mac/Code/GitHub/IVB/Ic/kay164Ic+.csv"),
        ("511 Ic+", "/Users/albert-mac/Code/GitHub/IVB/Ic/511Ic+.csv")
    ]
    
    results = []
    
    for dataset_name, csv_path in datasets:
        try:
            result = load_and_analyze_dataset(dataset_name, csv_path)
            if result:
                results.append(result)
        except Exception as e:
            print(f"❌ 分析 {dataset_name} 時出錯: {str(e)}")
    
    # 總結
    print(f"\n📋 診斷總結")
    print("="*60)
    
    for result in results:
        rel_diff = abs(result['fitted_mean'] - result['original_mean']) / abs(result['original_mean']) * 100
        print(f"{result['dataset']}: 相對差異 {rel_diff:.2f}%")
    
    print(f"\n🎯 建議修正方向:")
    print(f"1. 檢查線性項係數 r 是否過大")
    print(f"2. 確保 phi_ext 資料正確中心化 (phi_ext - d 的平均值應接近 0)")
    print(f"3. 檢查 Josephson 項的平均值是否接近零")
    print(f"4. 驗證常數項 C 是否正確設置為原始資料平均值")

if __name__ == "__main__":
    main()
