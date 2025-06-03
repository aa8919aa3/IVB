#!/usr/bin/env python3
"""
診斷 Josephson 擬合平均值差異問題
================================

分析為什麼擬合曲線的平均值與原始資料平均值差異很大。

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
from Fit import JosephsonAnalyzer, complete_josephson_equation

def load_experimental_data(csv_path):
    """載入實驗數據"""
    try:
        data = pd.read_csv(csv_path)
        phi_ext = data.iloc[:, 0].values  # 第一列為外部磁通
        I_s = data.iloc[:, 1].values      # 第二列為超導電流
        
        print(f"✅ 成功載入數據: {csv_path}")
        print(f"   數據點數: {len(phi_ext)}")
        print(f"   phi_ext 範圍: {phi_ext.min():.6f} 到 {phi_ext.max():.6f}")
        print(f"   I_s 範圍: {I_s.min():.2e} 到 {I_s.max():.2e}")
        print(f"   I_s 平均值: {np.mean(I_s):.2e}")
        print(f"   I_s 標準差: {np.std(I_s):.2e}")
        
        return phi_ext, I_s
        
    except Exception as e:
        print(f"❌ 載入數據失敗: {str(e)}")
        return None, None

def analyze_josephson_components(phi_ext, fitted_params):
    """分析 Josephson 方程式各組成部分的貢獻"""
    
    # 提取參數
    I_c = fitted_params['I_c']['value']
    f = fitted_params['f']['value']
    d = fitted_params['d']['value']
    phi_0 = fitted_params['phi_0']['value']
    T = fitted_params['T']['value']
    r = fitted_params['r']['value']
    C = fitted_params['C']['value']
    
    print(f"\n🔍 分析 Josephson 方程式各組成部分:")
    print(f"   I_c = {I_c:.6e}")
    print(f"   f = {f:.6e}")
    print(f"   d = {d:.6e}")
    print(f"   phi_0 = {phi_0:.6f}")
    print(f"   T = {T:.6f}")
    print(f"   r = {r:.6e}")
    print(f"   C = {C:.6e}")
    
    # 計算各組成部分
    phase = 2 * np.pi * f * (phi_ext - d) - phi_0
    sin_half_phase = np.sin(phase / 2)
    denominator = np.sqrt(1 - T * sin_half_phase**2)
    
    # Josephson 項
    josephson_term = I_c * np.sin(phase) / denominator
    
    # 線性項
    linear_term = r * (phi_ext - d)
    
    # 計算各項的統計特性
    josephson_mean = np.mean(josephson_term)
    josephson_std = np.std(josephson_term)
    linear_mean = np.mean(linear_term)
    linear_std = np.std(linear_term)
    
    print(f"\n📊 各組成部分的統計特性:")
    print(f"   Josephson 項:")
    print(f"      平均值: {josephson_mean:.6e}")
    print(f"      標準差: {josephson_std:.6e}")
    print(f"      範圍: {josephson_term.min():.6e} 到 {josephson_term.max():.6e}")
    
    print(f"   線性項 r*(phi_ext - d):")
    print(f"      平均值: {linear_mean:.6e}")
    print(f"      標準差: {linear_std:.6e}")
    print(f"      範圍: {linear_term.min():.6e} 到 {linear_term.max():.6e}")
    
    print(f"   常數項 C:")
    print(f"      值: {C:.6e}")
    
    # 總和
    total_theoretical = josephson_mean + linear_mean + C
    total_actual = complete_josephson_equation(phi_ext, I_c, f, d, phi_0, T, r, C)
    total_actual_mean = np.mean(total_actual)
    
    print(f"\n🔍 總和分析:")
    print(f"   理論平均值 (各項平均值相加): {total_theoretical:.6e}")
    print(f"   實際擬合曲線平均值: {total_actual_mean:.6e}")
    print(f"   差異: {abs(total_theoretical - total_actual_mean):.6e}")
    
    return {
        'josephson_term': josephson_term,
        'linear_term': linear_term,
        'constant_term': C,
        'total_fit': total_actual,
        'josephson_mean': josephson_mean,
        'linear_mean': linear_mean,
        'total_mean': total_actual_mean
    }

def diagnose_mean_difference(dataset_name, csv_path):
    """診斷特定數據集的平均值差異問題"""
    
    print(f"\n🔬 診斷 {dataset_name} 的平均值差異問題")
    print("="*60)
    
    # 1. 載入數據
    phi_ext, I_s = load_experimental_data(csv_path)
    if phi_ext is None:
        return
    
    original_mean = np.mean(I_s)
    
    # 2. 執行擬合分析
    analyzer = JosephsonAnalyzer()
    analyzer.add_simulation_data(
        'test_data', 
        {'Phi_ext': phi_ext, 'I_s': I_s, 'parameters': {}}, 
        {}, 
        dataset_name
    )
    
    # Lomb-Scargle 分析（處理可能的 NaN 問題）
    try:
        ls_result = analyzer.analyze_with_lomb_scargle('test_data', detrend_order=1)
        if ls_result is None:
            print("❌ Lomb-Scargle 分析失敗")
            return
    except Exception as e:
        print(f"❌ Lomb-Scargle 分析出錯: {str(e)}")
        print("⚠️  將使用默認參數進行 Josephson 擬合...")
        # 創建默認的 Lomb-Scargle 結果
        ls_result = {
            'best_frequency': 1.0 / (2 * np.pi),
            'phase': 0.0,
            'amplitude': np.std(I_s) * 2,
            'baseline': np.mean(I_s)
        }
    
    # 完整 Josephson 擬合
    fitter = analyzer.fit_complete_josephson_equation(
        'test_data',
        use_lbfgsb=True,
        save_results=False
    )
    
    if fitter is None:
        print("❌ Josephson 擬合失敗")
        return
    
    # 3. 分析擬合結果
    fitted_params = fitter.get_fitted_parameters()
    fitted_curve = fitter.calculate_fitted_curve(phi_ext)
    fitted_mean = np.mean(fitted_curve)
    
    print(f"\n📋 基本統計比較:")
    print(f"   原始資料平均值: {original_mean:.6e}")
    print(f"   擬合曲線平均值: {fitted_mean:.6e}")
    print(f"   絕對差異: {abs(fitted_mean - original_mean):.6e}")
    print(f"   相對差異: {abs(fitted_mean - original_mean) / original_mean * 100:.2f}%")
    
    # 4. 詳細分析各組成部分
    components = analyze_josephson_components(phi_ext, fitted_params)
    
    # 5. 檢查問題來源
    print(f"\n🚨 問題診斷:")
    
    # 檢查線性項是否為主要問題
    linear_contribution = abs(components['linear_mean'])
    josephson_contribution = abs(components['josephson_mean'])
    
    if linear_contribution > josephson_contribution:
        print(f"   ⚠️  線性項貢獻過大！")
        print(f"      線性項平均值: {components['linear_mean']:.6e}")
        print(f"      Josephson項平均值: {components['josephson_mean']:.6e}")
        print(f"      線性項/Josephson項比值: {linear_contribution/josephson_contribution:.2f}")
    
    # 檢查 phi_ext 的範圍和 d 參數
    phi_range = phi_ext.max() - phi_ext.min()
    phi_center = (phi_ext.max() + phi_ext.min()) / 2
    d_value = fitted_params['d']['value']
    
    print(f"   📊 phi_ext 分析:")
    print(f"      phi_ext 範圍: {phi_ext.min():.6f} 到 {phi_ext.max():.6f}")
    print(f"      phi_ext 中心: {phi_center:.6f}")
    print(f"      phi_ext 範圍寬度: {phi_range:.6f}")
    print(f"      d 參數值: {d_value:.6f}")
    print(f"      (phi_ext - d) 平均值: {np.mean(phi_ext - d_value):.6f}")
    
    # 6. 建議修正方法
    print(f"\n💡 建議修正方法:")
    
    if abs(np.mean(phi_ext - d_value)) > 1e-3:
        print(f"   1. phi_ext 數據未正確中心化")
        print(f"      建議: 調整 d 參數使 (phi_ext - d) 平均值接近 0")
        
    if linear_contribution > 0.1 * original_mean:
        print(f"   2. 線性項係數 r 過大")
        print(f"      當前 r = {fitted_params['r']['value']:.6e}")
        print(f"      建議: 限制 r 的範圍或重新檢查去趨勢化")
    
    # 7. 生成診斷圖表
    plot_diagnostic_analysis(phi_ext, I_s, components, fitted_params, dataset_name)
    
    return components

def plot_diagnostic_analysis(phi_ext, I_s, components, fitted_params, dataset_name):
    """生成診斷分析圖表"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Josephson 擬合診斷分析 - {dataset_name}', fontsize=16)
    
    # 1. 原始數據 vs 擬合結果
    ax1 = axes[0, 0]
    ax1.plot(phi_ext, I_s, 'b.', alpha=0.6, label='原始數據', markersize=3)
    ax1.plot(phi_ext, components['total_fit'], 'r-', linewidth=2, label='擬合結果')
    ax1.axhline(y=np.mean(I_s), color='b', linestyle='--', alpha=0.7, label=f'原始平均值 = {np.mean(I_s):.2e}')
    ax1.axhline(y=np.mean(components['total_fit']), color='r', linestyle='--', alpha=0.7, 
                label=f'擬合平均值 = {np.mean(components["total_fit"]):.2e}')
    ax1.set_xlabel('External Flux (φ_ext)')
    ax1.set_ylabel('Supercurrent (I_s)')
    ax1.set_title('數據對比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Josephson 項分析
    ax2 = axes[0, 1]
    ax2.plot(phi_ext, components['josephson_term'], 'g-', linewidth=2, label='Josephson 項')
    ax2.axhline(y=components['josephson_mean'], color='g', linestyle='--', alpha=0.7,
                label=f'平均值 = {components["josephson_mean"]:.2e}')
    ax2.set_xlabel('External Flux (φ_ext)')
    ax2.set_ylabel('Josephson Term')
    ax2.set_title('Josephson 項分析')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 線性項分析
    ax3 = axes[0, 2]
    ax3.plot(phi_ext, components['linear_term'], 'm-', linewidth=2, label='線性項')
    ax3.axhline(y=components['linear_mean'], color='m', linestyle='--', alpha=0.7,
                label=f'平均值 = {components["linear_mean"]:.2e}')
    ax3.set_xlabel('External Flux (φ_ext)')
    ax3.set_ylabel('Linear Term')
    ax3.set_title('線性項分析')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 各項貢獻比較
    ax4 = axes[1, 0]
    contributions = ['Josephson', '線性項', '常數項 C']
    values = [components['josephson_mean'], components['linear_mean'], components['constant_term']]
    colors = ['green', 'magenta', 'orange']
    
    bars = ax4.bar(contributions, values, color=colors, alpha=0.7)
    ax4.set_ylabel('平均值貢獻')
    ax4.set_title('各項平均值貢獻')
    ax4.grid(True, alpha=0.3)
    
    # 在柱狀圖上添加數值標籤
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2e}', ha='center', va='bottom')
    
    # 5. 殘差分析
    ax5 = axes[1, 1]
    residuals = I_s - components['total_fit']
    ax5.plot(phi_ext, residuals, 'k.', alpha=0.6, markersize=3)
    ax5.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax5.axhline(y=np.mean(residuals), color='orange', linestyle='--', alpha=0.7,
                label=f'殘差平均值 = {np.mean(residuals):.2e}')
    ax5.set_xlabel('External Flux (φ_ext)')
    ax5.set_ylabel('Residuals')
    ax5.set_title('殘差分析')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 參數信息
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    param_text = f"""
擬合參數:

I_c = {fitted_params['I_c']['value']:.3e}
f = {fitted_params['f']['value']:.3e}
d = {fitted_params['d']['value']:.6f}
φ₀ = {fitted_params['phi_0']['value']:.6f}
T = {fitted_params['T']['value']:.6f}
r = {fitted_params['r']['value']:.3e}
C = {fitted_params['C']['value']:.3e}

診斷結果:
原始平均值: {np.mean(I_s):.3e}
擬合平均值: {components['total_mean']:.3e}
差異: {abs(components['total_mean'] - np.mean(I_s)):.3e}
相對差異: {abs(components['total_mean'] - np.mean(I_s))/np.mean(I_s)*100:.1f}%
"""
    
    ax6.text(0.05, 0.95, param_text, transform=ax6.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存圖片
    filename = f'josephson_diagnostic_analysis_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 診斷分析圖已保存: {filename}")
    
    plt.show()

def main():
    """主函數"""
    print("🔍 Josephson 擬合平均值差異診斷工具")
    print("="*60)
    
    # 分析兩個數據集
    datasets = [
        ("Kay164 Ic+", "/Users/albert-mac/Code/GitHub/IVB/Ic/kay164Ic+.csv"),
        ("511 Ic+", "/Users/albert-mac/Code/GitHub/IVB/Ic/511Ic+.csv")
    ]
    
    results = {}
    
    for dataset_name, csv_path in datasets:
        try:
            result = diagnose_mean_difference(dataset_name, csv_path)
            if result is not None:
                results[dataset_name] = result
        except Exception as e:
            print(f"❌ 診斷 {dataset_name} 時出錯: {str(e)}")
    
    # 總結分析
    print(f"\n📋 診斷總結")
    print("="*60)
    
    for dataset_name, result in results.items():
        print(f"\n{dataset_name}:")
        print(f"   Josephson項平均貢獻: {result['josephson_mean']:.3e}")
        print(f"   線性項平均貢獻: {result['linear_mean']:.3e}")
        print(f"   常數項貢獻: {result['constant_term']:.3e}")
        print(f"   總擬合平均值: {result['total_mean']:.3e}")
        
        if abs(result['linear_mean']) > abs(result['josephson_mean']):
            print(f"   ⚠️  問題：線性項貢獻過大！")

if __name__ == "__main__":
    main()
