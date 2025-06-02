#!/usr/bin/env python3
"""
代碼升級指南：從簡單到通用
Code Upgrade Guide: From Simple to Universal

這個腳本展示如何將您現有的簡單臨界電流分析代碼升級為更通用的版本

Original Code (您的原始代碼):
```python
import pandas as pd
import matplotlib.pyplot as plt
dataid = 93
df = pd.read_csv(f'Ic/{dataid}Ic+.csv')
print(f"📊 Columns: {list(df.columns)}")
print(f"📈 Shape: {df.shape}")
df.plot(x='y_field', y='Ic', figsize=(16, 9), 
        title='y_field vs Ic', marker='o', linewidth=2)
plt.grid(True, alpha=0.3)
plt.show()
```

Upgraded Versions (升級版本):
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob
import re

# 方法 1: 最小修改 - 增加文件檢測功能
def upgrade_v1_minimal(dataid=93):
    """升級版本 1: 最小修改，增加通用性"""
    print("🔧 升級版本 1: 最小修改")
    print("-" * 40)
    
    data_dir = Path("Ic")
    
    # 檢測可用的文件類型
    file_patterns = {
        'positive': f'{dataid}Ic+.csv',
        'negative': f'{dataid}Ic-.csv', 
        'standard': f'{dataid}Ic.csv'
    }
    
    available_files = {}
    for file_type, filename in file_patterns.items():
        file_path = data_dir / filename
        if file_path.exists():
            available_files[file_type] = file_path
    
    if not available_files:
        print(f"❌ 數據集 {dataid} 不存在")
        return
    
    print(f"📊 數據集 {dataid} 可用文件: {list(available_files.keys())}")
    
    # 繪製所有可用文件
    plt.figure(figsize=(16, 9))
    
    colors = {'positive': 'red', 'negative': 'blue', 'standard': 'green'}
    
    for file_type, file_path in available_files.items():
        df = pd.read_csv(file_path)
        print(f"📈 {file_type}: Columns: {list(df.columns)}, Shape: {df.shape}")
        
        # 確保按 y_field 排序
        df_sorted = df.sort_values('y_field')
        
        plt.plot(df_sorted['y_field'], df_sorted['Ic'] * 1e6,  # 轉換為 µA
                label=f'{file_type.title()} Ic',
                marker='o', linewidth=2, 
                color=colors.get(file_type, 'black'))
    
    plt.xlabel('y_field')
    plt.ylabel('Critical Current (µA)')
    plt.title(f'y_field vs Ic (Dataset {dataid})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

# 方法 2: 函數化 - 支持多個數據集
def upgrade_v2_functional(dataids=[93, 164, 317]):
    """升級版本 2: 函數化，支持多個數據集"""
    print("\n🔧 升級版本 2: 函數化")
    print("-" * 40)
    
    def load_dataset(dataid, file_type='standard'):
        """載入指定數據集和類型"""
        data_dir = Path("Ic")
        
        if file_type == 'standard':
            filename = f'{dataid}Ic.csv'
        elif file_type == 'positive':
            filename = f'{dataid}Ic+.csv'
        elif file_type == 'negative':
            filename = f'{dataid}Ic-.csv'
        else:
            return None
        
        file_path = data_dir / filename
        if file_path.exists():
            df = pd.read_csv(file_path)
            df = df.sort_values('y_field')  # 確保排序
            return df
        return None
    
    def plot_comparison(dataids, file_type='standard'):
        """比較多個數據集"""
        plt.figure(figsize=(16, 9))
        
        for i, dataid in enumerate(dataids):
            df = load_dataset(dataid, file_type)
            if df is not None:
                plt.plot(df['y_field'], df['Ic'] * 1e6,
                        label=f'Dataset {dataid}',
                        marker='o', linewidth=2)
                print(f"✅ 載入數據集 {dataid}: {df.shape[0]} 點")
            else:
                print(f"❌ 數據集 {dataid} ({file_type}) 不存在")
        
        plt.xlabel('y_field')
        plt.ylabel('Critical Current (µA)')
        plt.title(f'Critical Current Comparison ({file_type.title()})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
    
    # 執行比較
    plot_comparison(dataids, 'standard')

# 方法 3: 自動檢測 - 掃描所有可用數據
def upgrade_v3_auto_detect():
    """升級版本 3: 自動檢測所有可用數據"""
    print("\n🔧 升級版本 3: 自動檢測")
    print("-" * 40)
    
    def scan_all_datasets():
        """掃描所有可用數據集"""
        data_dir = Path("Ic")
        datasets = {}
        
        for csv_file in data_dir.glob("*.csv"):
            # 解析文件名
            match = re.match(r'(\d+)Ic([+-]?)\.csv', csv_file.name)
            if match:
                dataid = int(match.group(1))
                suffix = match.group(2)
                
                if suffix == '+':
                    file_type = 'positive'
                elif suffix == '-':
                    file_type = 'negative'
                else:
                    file_type = 'standard'
                
                if dataid not in datasets:
                    datasets[dataid] = {}
                datasets[dataid][file_type] = csv_file
        
        return datasets
    
    # 掃描數據
    all_datasets = scan_all_datasets()
    print(f"🔍 發現 {len(all_datasets)} 個數據集")
    
    # 展示前幾個
    sample_ids = sorted(list(all_datasets.keys()))[:5]
    print(f"📊 展示前 5 個數據集: {sample_ids}")
    
    plt.figure(figsize=(16, 9))
    
    for i, dataid in enumerate(sample_ids):
        # 優先使用 standard，然後 positive
        file_types = ['standard', 'positive', 'negative']
        df = None
        
        for ft in file_types:
            if ft in all_datasets[dataid]:
                df = pd.read_csv(all_datasets[dataid][ft])
                df = df.sort_values('y_field')
                break
        
        if df is not None:
            plt.plot(df['y_field'], df['Ic'] * 1e6,
                    label=f'Dataset {dataid}',
                    marker='o', linewidth=2)
    
    plt.xlabel('y_field')
    plt.ylabel('Critical Current (µA)')
    plt.title('Multi-Dataset Critical Current Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

# 方法 4: 完整解決方案 - 使用 UniversalIcAnalyzer
def upgrade_v4_complete():
    """升級版本 4: 完整解決方案"""
    print("\n🔧 升級版本 4: 完整解決方案")
    print("-" * 40)
    
    try:
        from universal_ic_analyzer import UniversalIcAnalyzer
        
        # 一行代碼替換原始功能
        analyzer = UniversalIcAnalyzer("Ic")
        
        # 原始代碼的等效操作
        dataid = "93"
        fig = analyzer.plot_single_dataset(dataid, save_plot=True)
        if fig:
            plt.show()
        
        print("✅ 使用 UniversalIcAnalyzer 完成升級")
        
    except ImportError:
        print("❌ UniversalIcAnalyzer 模組不可用，請確保 universal_ic_analyzer.py 在同一目錄")

def main():
    """主函數 - 演示所有升級方法"""
    print("🎯 代碼升級指南：從簡單到通用")
    print("=" * 60)
    print("原始代碼功能：繪製單個數據集的臨界電流")
    print("升級目標：支持多種文件格式和多數據集比較")
    print("=" * 60)
    
    # 方法 1: 最小修改
    upgrade_v1_minimal(93)
    
    # 方法 2: 函數化
    upgrade_v2_functional([93, 164, 317])
    
    # 方法 3: 自動檢測
    upgrade_v3_auto_detect()
    
    # 方法 4: 完整解決方案
    upgrade_v4_complete()
    
    print("\n" + "=" * 60)
    print("📋 升級總結:")
    print("✅ V1: 最小修改 - 支持多種文件類型")
    print("✅ V2: 函數化 - 支持多數據集比較")  
    print("✅ V3: 自動檢測 - 掃描所有可用數據")
    print("✅ V4: 完整解決方案 - 使用專業分析器")
    print("=" * 60)

if __name__ == "__main__":
    main()
