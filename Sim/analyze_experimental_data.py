#!/usr/bin/env python3
"""
實驗數據 Josephson 擬合分析
===========================

對 kay164Ic+.csv 和 511Ic+.csv 進行完整的 Josephson 方程式擬合分析。

作者：GitHub Copilot
日期：2025年6月3日
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# 添加當前目錄到 Python 路徑
sys.path.append('/Users/albert-mac/Code/GitHub/IVB/Sim')
from Fit import JosephsonFitter, JosephsonAnalyzer, ModelStatistics

def load_experimental_data(csv_path):
    """
    載入實驗數據 CSV 文件
    
    Args:
        csv_path: CSV 文件路徑
        
    Returns:
        DataFrame: 包含 y_field 和 Ic 的數據
    """
    try:
        data = pd.read_csv(csv_path)
        print(f"✅ 成功載入數據: {csv_path}")
        print(f"   數據點數: {len(data)}")
        print(f"   y_field 範圍: {data['y_field'].min():.6f} 到 {data['y_field'].max():.6f}")
        print(f"   Ic 範圍: {data['Ic'].min():.2e} 到 {data['Ic'].max():.2e}")
        print(f"   Ic 平均值: {data['Ic'].mean():.2e}")
        print(f"   Ic 標準差: {data['Ic'].std():.2e}")
        return data
    except Exception as e:
        print(f"❌ 載入數據失敗: {e}")
        return None

def analyze_experimental_data(csv_path, model_name):
    """
    分析單個實驗數據文件
    
    Args:
        csv_path: CSV 文件路徑
        model_name: 模型名稱
        
    Returns:
        JosephsonAnalyzer: 分析器對象
    """
    print(f"\n🔬 分析 {model_name}")
    print("="*60)
    
    # 載入數據
    data = load_experimental_data(csv_path)
    if data is None:
        return None
    
    # 準備數據格式
    phi_ext = data['y_field'].values
    I_s = data['Ic'].values
    
    # 估計誤差（假設為信號的 1%）
    I_s_error = np.full_like(I_s, np.std(I_s) * 0.01)
    
    # 創建分析器
    analyzer = JosephsonAnalyzer()
    
    # 準備數據字典
    data_dict = {
        'Phi_ext': phi_ext,
        'I_s': I_s,
        'I_s_error': I_s_error
    }
    
    # 估計基本參數（用於參考）
    estimated_params = {
        'f': 1.0 / (2 * np.pi),  # 默認頻率
        'Ic': np.max(I_s),  # 峰值作為臨界電流估計
        'phi_0': 0.0,
        'T': 0.5
    }
    
    # 創建模型類型名稱
    model_type = model_name.lower().replace(' ', '_').replace('+', 'plus').replace('-', 'minus')
    
    # 添加到分析器
    analyzer.add_simulation_data(
        model_type=model_type,
        data=data_dict,
        parameters=estimated_params,
        model_name=model_name
    )
    
    # 執行 Lomb-Scargle 分析
    print(f"\n🔧 執行 Lomb-Scargle 分析...")
    ls_result = analyzer.analyze_with_lomb_scargle(model_type, detrend_order=1)
    
    if ls_result:
        # 繪制 Lomb-Scargle 結果
        print(f"\n📈 生成 Lomb-Scargle 分析圖...")
        analyzer.plot_analysis_results(model_type, save_plot=True)
        
        # 執行完整 Josephson 擬合
        print(f"\n🚀 執行完整 Josephson 方程式擬合...")
        fitter = analyzer.fit_complete_josephson_equation(
            model_type=model_type,
            use_lbfgsb=True,
            save_results=True
        )
        
        if fitter:
            # 比較 Lomb-Scargle 與完整擬合結果
            print(f"\n📊 生成比較分析圖...")
            analyzer.compare_lomb_scargle_vs_josephson_fit(model_type, save_plot=True)
            
            # 保存去趨勢化數據
            print(f"\n💾 保存去趨勢化數據...")
            analyzer.plot_detrended_data_comparison(model_type, save_plot=True)
            analyzer.save_detrended_data_to_csv(model_type)
            
            # 打印擬合參數摘要
            fitted_params = fitter.get_fitted_parameters()
            if fitted_params:
                print(f"\n📋 {model_name} 擬合參數摘要:")
                print("-"*40)
                for param_name, param_info in fitted_params.items():
                    print(f"   {param_name}: {param_info['value']:.6f} ± {param_info['stderr']:.6f}")
    
    return analyzer

def compare_multiple_datasets(analyzers_dict):
    """
    比較多個數據集的分析結果
    
    Args:
        analyzers_dict: 包含分析器的字典 {name: analyzer}
    """
    print(f"\n📊 多數據集比較分析")
    print("="*60)
    
    # 創建比較圖
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('多數據集 Josephson 擬合比較', fontsize=16)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # 1. 原始數據比較
    ax1 = axes[0, 0]
    for i, (name, analyzer) in enumerate(analyzers_dict.items()):
        for model_type, data in analyzer.simulation_results.items():
            ax1.plot(data['Phi_ext'], data['I_s'], '.', 
                    alpha=0.6, label=name, color=colors[i % len(colors)], markersize=2)
    ax1.set_xlabel('y_field')
    ax1.set_ylabel('Ic (A)')
    ax1.set_title('原始數據比較')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Lomb-Scargle 頻率比較
    ax2 = axes[0, 1]
    dataset_names = []
    best_frequencies = []
    
    for name, analyzer in analyzers_dict.items():
        for model_type, result in analyzer.analysis_results.items():
            dataset_names.append(name)
            best_frequencies.append(result['best_frequency'])
    
    ax2.bar(range(len(dataset_names)), best_frequencies, color=colors[:len(dataset_names)])
    ax2.set_xlabel('數據集')
    ax2.set_ylabel('檢測頻率')
    ax2.set_title('Lomb-Scargle 檢測頻率比較')
    ax2.set_xticks(range(len(dataset_names)))
    ax2.set_xticklabels(dataset_names, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. 擬合參數比較（如果有完整擬合結果）
    ax3 = axes[1, 0]
    param_data = {'I_c': [], 'f': [], 'T': []}
    labels = []
    
    for name, analyzer in analyzers_dict.items():
        if hasattr(analyzer, 'josephson_fitters'):
            for model_type, fitter in analyzer.josephson_fitters.items():
                fitted_params = fitter.get_fitted_parameters()
                if fitted_params:
                    param_data['I_c'].append(fitted_params['I_c']['value'])
                    param_data['f'].append(fitted_params['f']['value'])
                    param_data['T'].append(fitted_params['T']['value'])
                    labels.append(name)
    
    if param_data['I_c']:
        x_pos = np.arange(len(labels))
        width = 0.25
        
        ax3.bar(x_pos - width, param_data['I_c'], width, label='I_c', alpha=0.8)
        ax3_twin = ax3.twinx()
        ax3_twin.bar(x_pos, param_data['f'], width, label='f', alpha=0.8, color='orange')
        ax3_twin.bar(x_pos + width, param_data['T'], width, label='T', alpha=0.8, color='green')
        
        ax3.set_xlabel('數據集')
        ax3.set_ylabel('I_c (A)', color='blue')
        ax3_twin.set_ylabel('f, T', color='orange')
        ax3.set_title('擬合參數比較')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(labels, rotation=45)
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, '暫無完整擬合結果', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('擬合參數比較')
    
    # 4. 統計摘要
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = "數據集統計摘要:\n\n"
    for name, analyzer in analyzers_dict.items():
        for model_type, data in analyzer.simulation_results.items():
            I_s = data['I_s']
            summary_text += f"{name}:\n"
            summary_text += f"  數據點: {len(I_s)}\n"
            summary_text += f"  Ic 範圍: {I_s.min():.2e} - {I_s.max():.2e}\n"
            summary_text += f"  Ic 平均: {I_s.mean():.2e}\n"
            summary_text += f"  Ic 標準差: {I_s.std():.2e}\n\n"
    
    if hasattr(analyzer, 'analysis_results'):
        for name, analyzer in analyzers_dict.items():
            for model_type, result in analyzer.analysis_results.items():
                summary_text += f"{name} LS 分析:\n"
                summary_text += f"  最佳頻率: {result['best_frequency']:.6f}\n"
                summary_text += f"  R²: {result['statistics'].r_squared:.6f}\n\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存比較圖
    filename = 'experimental_data_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ 比較分析圖已保存: {filename}")
    
    plt.show()

def main():
    """主函數"""
    print("🚀 實驗數據 Josephson 擬合分析")
    print("="*70)
    
    # 定義要分析的數據文件
    datasets = [
        {
            'path': '/Users/albert-mac/Code/GitHub/IVB/Ic/kay164Ic+.csv',
            'name': 'Kay164 Ic+'
        },
        {
            'path': '/Users/albert-mac/Code/GitHub/IVB/Ic/511Ic+.csv',
            'name': '511 Ic+'
        }
    ]
    
    # 分析器字典
    analyzers = {}
    
    # 分析每個數據集
    for dataset in datasets:
        if os.path.exists(dataset['path']):
            analyzer = analyze_experimental_data(dataset['path'], dataset['name'])
            if analyzer:
                analyzers[dataset['name']] = analyzer
        else:
            print(f"❌ 文件不存在: {dataset['path']}")
    
    # 生成比較報告
    if analyzers:
        print(f"\n📊 生成綜合比較報告...")
        
        # 為每個分析器生成報告
        for name, analyzer in analyzers.items():
            print(f"\n📋 {name} 詳細報告:")
            analyzer.generate_comparison_report()
        
        # 多數據集比較
        compare_multiple_datasets(analyzers)
        
        print(f"\n✅ 所有分析完成！")
        print(f"   已分析 {len(analyzers)} 個數據集")
        print(f"   結果文件已保存到當前目錄")
        
    else:
        print("❌ 沒有成功分析任何數據集")

if __name__ == "__main__":
    # 切換到 Sim 目錄以確保路徑正確
    os.chdir('/Users/albert-mac/Code/GitHub/IVB/Sim')
    main()
