#!/usr/bin/env python3
"""
簡化版通用臨界電流分析示例
Simplified Universal Critical Current Analysis Examples

這個腳本展示了如何使用 UniversalIcAnalyzer 來增加您的臨界電流分析的通用性

Usage:
    python simple_ic_examples.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from universal_ic_analyzer import UniversalIcAnalyzer, quick_plot, quick_compare

def example_1_single_dataset():
    """示例 1: 分析單個數據集 (原始代碼的改進版)"""
    print("=" * 60)
    print("📊 示例 1: 分析單個數據集")
    print("=" * 60)
    
    # 原始代碼
    print("\n🔹 原始代碼風格:")
    dataid = 93
    df = pd.read_csv(f'Ic/{dataid}Ic+.csv')
    print(f"📊 Columns: {list(df.columns)}")
    print(f"📈 Shape: {df.shape}")
    
    # 改進版 - 使用 UniversalIcAnalyzer
    print("\n🔹 改進版 - 使用通用分析器:")
    analyzer = UniversalIcAnalyzer("Ic")
    
    # 檢查可用的文件類型
    if str(dataid) in analyzer.available_files:
        available_types = list(analyzer.available_files[str(dataid)].keys())
        print(f"📊 Dataset {dataid} 可用類型: {available_types}")
        
        # 繪製所有可用類型
        fig = analyzer.plot_single_dataset(str(dataid), save_plot=True)
        plt.show()
    else:
        print(f"❌ 數據集 {dataid} 不存在")

def example_2_quick_functions():
    """示例 2: 使用快速函數"""
    print("\n" + "=" * 60)
    print("📊 示例 2: 使用快速函數")
    print("=" * 60)
    
    # 快速繪製單個數據集
    print("\n🔹 快速繪製:")
    quick_plot("93", save_plot=True)
    
    # 快速比較多個數據集 
    print("\n🔹 快速比較:")
    compare_ids = ["93", "164", "317"]  # 移除 500 因為它沒有 standard 類型
    quick_compare(compare_ids, file_type='standard', save_plot=True)

def example_3_flexible_analysis():
    """示例 3: 靈活的數據分析"""
    print("\n" + "=" * 60)
    print("📊 示例 3: 靈活的數據分析")
    print("=" * 60)
    
    analyzer = UniversalIcAnalyzer("Ic")
    
    # 自動檢測雙極性數據集
    print("\n🔹 檢測雙極性數據集:")
    bipolar_datasets = []
    for data_id in analyzer.get_available_ids():
        files = analyzer.available_files[data_id]
        if 'positive' in files and 'negative' in files:
            bipolar_datasets.append(data_id)
    
    print(f"發現 {len(bipolar_datasets)} 個雙極性數據集")
    print(f"前 10 個: {bipolar_datasets[:10]}")
    
    # 分析第一個雙極性數據集
    if bipolar_datasets:
        sample_id = bipolar_datasets[0]
        print(f"\n🔹 分析雙極性數據集 {sample_id}:")
        
        # 載入並比較正負向數據
        data_dict = analyzer.load_data(sample_id, ['positive', 'negative'])
        
        for file_type, df in data_dict.items():
            ic_mean = df['Ic'].mean() * 1e6
            ic_max = df['Ic'].max() * 1e6
            print(f"  {file_type}: 平均 Ic = {ic_mean:.2f} µA, 最大 Ic = {ic_max:.2f} µA")
        
        # 繪製比較圖
        fig = analyzer.plot_single_dataset(sample_id, ['positive', 'negative'], save_plot=True)
        if fig:
            plt.show()

def example_4_batch_processing():
    """示例 4: 批量處理"""
    print("\n" + "=" * 60)
    print("📊 示例 4: 批量處理範例")
    print("=" * 60)
    
    analyzer = UniversalIcAnalyzer("Ic")
    
    # 選擇一些有代表性的數據集進行批量分析
    sample_ids = ["93", "164", "317", "65", "500"]
    
    print(f"\n🔹 批量分析 {len(sample_ids)} 個數據集:")
    
    for data_id in sample_ids:
        if data_id in analyzer.available_files:
            analysis = analyzer.analyze_dataset(data_id)
            if analysis:
                print(f"\n📈 數據集 {data_id}:")
                print(f"  可用類型: {', '.join(analysis['available_types'])}")
                
                for file_type, stats in analysis['statistics'].items():
                    print(f"  {file_type}: {stats['count']} 點, "
                          f"平均 Ic = {stats['mean_ic']:.2f} µA")

def example_5_custom_comparison():
    """示例 5: 自定義比較"""
    print("\n" + "=" * 60)
    print("📊 示例 5: 自定義比較")
    print("=" * 60)
    
    analyzer = UniversalIcAnalyzer("Ic")
    
    # 比較具有正向數據的數據集
    print("\n🔹 比較正向臨界電流:")
    positive_datasets = []
    for data_id in analyzer.get_available_ids():
        if 'positive' in analyzer.available_files[data_id]:
            positive_datasets.append(data_id)
    
    # 選擇前幾個進行比較
    sample_positive = positive_datasets[:5]
    print(f"比較數據集: {sample_positive}")
    
    fig = analyzer.plot_comparison(sample_positive, file_type='positive', save_plot=True)
    if fig:
        plt.show()

def main():
    """主函數"""
    print("🎯 通用臨界電流分析器使用示例")
    print("Universal Critical Current Analyzer Examples")
    print("=" * 60)
    
    try:
        # 運行所有示例
        example_1_single_dataset()
        example_2_quick_functions() 
        example_3_flexible_analysis()
        example_4_batch_processing()
        example_5_custom_comparison()
        
        print("\n" + "=" * 60)
        print("✅ 所有示例運行完成！")
        print("📊 生成的圖片文件:")
        import glob
        png_files = glob.glob("ic_*.png")
        for file in png_files:
            print(f"  📈 {file}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⚠️  用戶中止執行")
    except Exception as e:
        print(f"\n❌ 運行錯誤: {e}")

if __name__ == "__main__":
    main()
