#!/usr/bin/env python3
"""
測試 Ic×Rn 分析修復效果
"""

import matplotlib.pyplot as plt
from advanced_superconductor_analyzer import AdvancedSuperconductorAnalyzer
import pandas as pd
import numpy as np

def test_ic_rn_analysis(csv_file):
    """測試指定 CSV 文件的 Ic×Rn 分析"""
    print(f"\n🔍 測試 {csv_file}...")
    
    try:
        # 初始化分析器
        analyzer = AdvancedSuperconductorAnalyzer(csv_file)
        analyzer.load_and_preprocess_data()
        analyzer.extract_enhanced_features()
        
        if hasattr(analyzer, 'features') and not analyzer.features.empty:
            print('✅ 特徵提取成功')
            
            # 檢查數據範圍
            valid_data = analyzer.features.dropna(subset=['Rn', 'y_field'])
            ic_rn_values = []
            field_values = []
            
            for _, row in valid_data.iterrows():
                y_field = row['y_field']
                rn = row['Rn']
                if 'Ic_positive' in row and not pd.isna(row['Ic_positive']):
                    ic_rn_pos = row['Ic_positive'] * rn * 1e6
                    ic_rn_values.append(ic_rn_pos)
                    field_values.append(y_field * 1000)  # 轉換為 mT
                if 'Ic_negative' in row and not pd.isna(row['Ic_negative']):
                    ic_rn_neg = row['Ic_negative'] * rn * 1e6
                    ic_rn_values.append(ic_rn_neg)
                    field_values.append(y_field * 1000)  # 轉換為 mT
            
            if ic_rn_values:
                print(f'📊 IcRn 範圍: {min(ic_rn_values):.1f} 到 {max(ic_rn_values):.1f} µV')
                print(f'🧲 磁場範圍: {min(field_values):.3f} 到 {max(field_values):.3f} mT')
                print(f'📈 數據點數量: {len(ic_rn_values)}')
                print(f'🎯 唯一磁場值: {len(set(field_values))}')
                
                # 測試繪圖功能
                fig, ax = plt.subplots(figsize=(10, 6))
                analyzer._plot_ic_rn_analysis(ax)
                
                output_file = f'test_ic_rn_{csv_file.replace(".csv", "")}.png'
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                print(f'✅ 測試圖片已保存為 {output_file}')
                plt.close()
                
                return True
            else:
                print('❌ 沒有找到有效的 IcRn 數據')
                return False
        else:
            print('❌ 特徵提取失敗')
            return False
            
    except Exception as e:
        print(f'❌ 測試失敗: {e}')
        return False

if __name__ == "__main__":
    # 測試所有數據文件
    test_files = ['500.csv', '317.csv', '164.csv']
    
    print("🚀 開始測試 Ic×Rn 分析修復效果...")
    
    results = {}
    for csv_file in test_files:
        results[csv_file] = test_ic_rn_analysis(csv_file)
    
    print("\n📋 測試結果總結:")
    for file, success in results.items():
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"  {file}: {status}")
    
    successful_tests = sum(results.values())
    total_tests = len(results)
    print(f"\n🎯 總體結果: {successful_tests}/{total_tests} 個測試成功")
    
    if successful_tests == total_tests:
        print("🎉 所有測試都成功！Ic×Rn 分析修復完成")
    else:
        print("⚠️ 仍有部分測試失敗，需要進一步檢查")
