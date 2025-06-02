#!/usr/bin/env python3
"""
通用臨界電流分析器
Universal Critical Current Analyzer

支持多種文件格式：
- {id}Ic.csv (標準臨界電流)
- {id}Ic+.csv (正向臨界電流) 
- {id}Ic-.csv (負向臨界電流)

Author: GitHub Copilot
Date: 2025-06-02
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import seaborn as sns

class UniversalIcAnalyzer:
    """通用臨界電流分析器"""
    
    def __init__(self, data_dir: str = "Ic"):
        """
        初始化分析器
        
        Args:
            data_dir: 數據文件目錄路徑
        """
        self.data_dir = Path(data_dir)
        self.available_files = {}
        self.data_cache = {}
        
        # 設置繪圖樣式
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 掃描可用文件
        self._scan_available_files()
    
    def _scan_available_files(self):
        """掃描可用的 Ic 文件"""
        if not self.data_dir.exists():
            print(f"❌ 數據目錄不存在: {self.data_dir}")
            return
        
        # 文件模式
        patterns = {
            'standard': r'(\d+)Ic\.csv$',
            'positive': r'(\d+)Ic\+\.csv$', 
            'negative': r'(\d+)Ic-\.csv$',
            'special': r'([a-zA-Z]+\d+)Ic[+-]?\.csv$'  # 支持 kay164Ic+.csv 等特殊格式
        }
        
        for file_path in self.data_dir.glob("*.csv"):
            filename = file_path.name
            
            for file_type, pattern in patterns.items():
                match = re.search(pattern, filename)
                if match:
                    data_id = match.group(1)
                    
                    if data_id not in self.available_files:
                        self.available_files[data_id] = {}
                    
                    self.available_files[data_id][file_type] = file_path
                    break
        
        print(f"🔍 掃描到 {len(self.available_files)} 個數據集:")
        for data_id in sorted(self.available_files.keys(), key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x)):
            types = list(self.available_files[data_id].keys())
            print(f"  📊 {data_id}: {', '.join(types)}")
    
    def get_available_ids(self) -> List[str]:
        """獲取所有可用的數據 ID"""
        return sorted(self.available_files.keys(), 
                     key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x))
    
    def load_data(self, data_id: str, file_types: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        載入指定 ID 的數據
        
        Args:
            data_id: 數據 ID
            file_types: 要載入的文件類型列表，None 表示載入所有可用類型
            
        Returns:
            包含不同文件類型數據的字典
        """
        if data_id not in self.available_files:
            print(f"❌ 數據 ID '{data_id}' 不存在")
            return {}
        
        if file_types is None:
            file_types = list(self.available_files[data_id].keys())
        
        loaded_data = {}
        
        for file_type in file_types:
            if file_type not in self.available_files[data_id]:
                print(f"⚠️  {data_id} 沒有 {file_type} 類型的文件")
                continue
            
            file_path = self.available_files[data_id][file_type]
            cache_key = f"{data_id}_{file_type}"
            
            # 檢查緩存
            if cache_key in self.data_cache:
                loaded_data[file_type] = self.data_cache[cache_key]
                continue
            
            try:
                df = pd.read_csv(file_path)
                
                # 驗證數據格式
                if not self._validate_data_format(df):
                    print(f"⚠️  {file_path.name} 數據格式不正確")
                    continue
                
                # 數據清理
                df = self._clean_data(df)
                
                # 緩存數據
                self.data_cache[cache_key] = df
                loaded_data[file_type] = df
                
                print(f"✅ 載入 {file_path.name}: {df.shape[0]} 數據點")
                
            except Exception as e:
                print(f"❌ 載入 {file_path.name} 失敗: {e}")
        
        return loaded_data
    
    def _validate_data_format(self, df: pd.DataFrame) -> bool:
        """驗證數據格式"""
        required_columns = ['y_field', 'Ic']
        return all(col in df.columns for col in required_columns)
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理數據"""
        # 移除 NaN 值
        df = df.dropna()
        
        # 按 y_field 排序
        df = df.sort_values('y_field').reset_index(drop=True)
        
        # 移除異常值（Ic 為 0 或負值）
        df = df[df['Ic'] > 0]
        
        return df
    
    def plot_single_dataset(self, data_id: str, file_types: Optional[List[str]] = None, 
                           figsize: Tuple[int, int] = (12, 8), save_plot: bool = False) -> plt.Figure:
        """
        繪製單個數據集的圖表
        
        Args:
            data_id: 數據 ID
            file_types: 要繪製的文件類型列表
            figsize: 圖表大小
            save_plot: 是否保存圖片
            
        Returns:
            matplotlib Figure 對象
        """
        data_dict = self.load_data(data_id, file_types)
        
        if not data_dict:
            print(f"❌ 無法載入數據集 {data_id}")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 顏色和標記映射
        style_map = {
            'standard': {'color': 'blue', 'marker': 'o', 'label': 'Standard Ic'},
            'positive': {'color': 'red', 'marker': '^', 'label': 'Positive Ic'}, 
            'negative': {'color': 'green', 'marker': 'v', 'label': 'Negative Ic'},
            'special': {'color': 'purple', 'marker': 's', 'label': 'Special Ic'}
        }
        
        for file_type, df in data_dict.items():
            style = style_map.get(file_type, {'color': 'black', 'marker': 'o', 'label': file_type})
            
            # 確保按 y_field 排序繪製
            df_sorted = df.sort_values('y_field')
            
            ax.plot(df_sorted['y_field'], df_sorted['Ic'] * 1e6,  # 轉換為 µA
                   color=style['color'], 
                   marker=style['marker'],
                   label=style['label'],
                   linewidth=2,
                   markersize=4,
                   alpha=0.8)
        
        # 圖表設置
        ax.set_xlabel('Magnetic Field (y_field)', fontsize=12)
        ax.set_ylabel('Critical Current (µA)', fontsize=12)
        ax.set_title(f'Critical Current vs Magnetic Field (Dataset {data_id})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # 添加統計信息
        self._add_statistics_text(ax, data_dict)
        
        plt.tight_layout()
        
        # 保存圖片
        if save_plot:
            filename = f"ic_analysis_{data_id}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 圖片已保存: {filename}")
        
        return fig
    
    def plot_comparison(self, data_ids: List[str], file_type: str = 'standard',
                       figsize: Tuple[int, int] = (16, 10), save_plot: bool = False) -> plt.Figure:
        """
        比較多個數據集
        
        Args:
            data_ids: 要比較的數據 ID 列表
            file_type: 文件類型
            figsize: 圖表大小  
            save_plot: 是否保存圖片
            
        Returns:
            matplotlib Figure 對象
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(data_ids)))
        
        valid_datasets = 0
        
        for i, data_id in enumerate(data_ids):
            data_dict = self.load_data(data_id, [file_type])
            
            if file_type not in data_dict:
                print(f"⚠️  數據集 {data_id} 沒有 {file_type} 類型文件")
                continue
            
            df = data_dict[file_type]
            df_sorted = df.sort_values('y_field')
            
            ax.plot(df_sorted['y_field'], df_sorted['Ic'] * 1e6,
                   color=colors[i],
                   label=f'Dataset {data_id}',
                   linewidth=2,
                   marker='o',
                   markersize=3,
                   alpha=0.8)
            
            valid_datasets += 1
        
        if valid_datasets == 0:
            print("❌ 沒有有效的數據集用於比較")
            return None
        
        # 圖表設置
        ax.set_xlabel('Magnetic Field (y_field)', fontsize=12)
        ax.set_ylabel('Critical Current (µA)', fontsize=12)
        ax.set_title(f'Critical Current Comparison ({file_type.title()})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
        plt.tight_layout()
        
        # 保存圖片
        if save_plot:
            filename = f"ic_comparison_{file_type}_{'_'.join(data_ids[:5])}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 比較圖已保存: {filename}")
        
        return fig
    
    def _add_statistics_text(self, ax, data_dict: Dict[str, pd.DataFrame]):
        """添加統計信息文本"""
        stats_text = []
        
        for file_type, df in data_dict.items():
            ic_values = df['Ic'] * 1e6  # µA
            stats_text.append(f"{file_type.title()}:")
            stats_text.append(f"  Mean: {ic_values.mean():.2f} µA")
            stats_text.append(f"  Max: {ic_values.max():.2f} µA")
            stats_text.append(f"  Points: {len(ic_values)}")
            stats_text.append("")
        
        # 添加文本框
        textstr = '\n'.join(stats_text[:-1])  # 移除最後的空行
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=props)
    
    def analyze_dataset(self, data_id: str) -> Dict:
        """
        分析單個數據集的詳細統計
        
        Args:
            data_id: 數據 ID
            
        Returns:
            分析結果字典
        """
        data_dict = self.load_data(data_id)
        
        if not data_dict:
            return {}
        
        analysis = {
            'data_id': data_id,
            'available_types': list(data_dict.keys()),
            'statistics': {}
        }
        
        for file_type, df in data_dict.items():
            ic_values = df['Ic'] * 1e6  # µA
            field_range = df['y_field'].max() - df['y_field'].min()
            
            stats = {
                'count': len(df),
                'mean_ic': ic_values.mean(),
                'std_ic': ic_values.std(),
                'max_ic': ic_values.max(),
                'min_ic': ic_values.min(),
                'field_range': field_range,
                'field_min': df['y_field'].min(),
                'field_max': df['y_field'].max()
            }
            
            analysis['statistics'][file_type] = stats
        
        return analysis
    
    def batch_analyze(self, data_ids: Optional[List[str]] = None, 
                     output_file: str = "ic_analysis_report.txt") -> Dict:
        """
        批量分析多個數據集
        
        Args:
            data_ids: 要分析的數據 ID 列表，None 表示分析所有
            output_file: 報告輸出文件名
            
        Returns:
            分析結果字典
        """
        if data_ids is None:
            data_ids = self.get_available_ids()
        
        batch_results = {}
        
        print(f"📊 開始批量分析 {len(data_ids)} 個數據集...")
        
        for data_id in data_ids:
            print(f"📈 分析數據集 {data_id}...")
            analysis = self.analyze_dataset(data_id)
            if analysis:
                batch_results[data_id] = analysis
        
        # 生成報告
        self._generate_analysis_report(batch_results, output_file)
        
        return batch_results
    
    def _generate_analysis_report(self, results: Dict, output_file: str):
        """生成分析報告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("臨界電流數據分析報告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"分析時間: {pd.Timestamp.now()}\n")
            f.write(f"分析數據集數量: {len(results)}\n\n")
            
            for data_id, analysis in results.items():
                f.write(f"數據集 ID: {data_id}\n")
                f.write("-" * 30 + "\n")
                f.write(f"可用文件類型: {', '.join(analysis['available_types'])}\n\n")
                
                for file_type, stats in analysis['statistics'].items():
                    f.write(f"  {file_type.title()} 類型:\n")
                    f.write(f"    數據點數: {stats['count']}\n")
                    f.write(f"    平均臨界電流: {stats['mean_ic']:.3f} µA\n")
                    f.write(f"    臨界電流標準差: {stats['std_ic']:.3f} µA\n")
                    f.write(f"    最大臨界電流: {stats['max_ic']:.3f} µA\n")
                    f.write(f"    最小臨界電流: {stats['min_ic']:.3f} µA\n")
                    f.write(f"    磁場範圍: {stats['field_min']:.6f} - {stats['field_max']:.6f}\n")
                    f.write(f"    磁場跨度: {stats['field_range']:.6f}\n\n")
                
                f.write("\n")
        
        print(f"📄 分析報告已保存: {output_file}")


# 便捷函數
def quick_plot(data_id: str, data_dir: str = "Ic", save_plot: bool = False):
    """快速繪製單個數據集"""
    analyzer = UniversalIcAnalyzer(data_dir)
    fig = analyzer.plot_single_dataset(data_id, save_plot=save_plot)
    if fig:
        plt.show()
    return fig

def quick_compare(data_ids: List[str], file_type: str = 'standard', 
                 data_dir: str = "Ic", save_plot: bool = False):
    """快速比較多個數據集"""
    analyzer = UniversalIcAnalyzer(data_dir)
    fig = analyzer.plot_comparison(data_ids, file_type, save_plot=save_plot)
    if fig:
        plt.show()
    return fig


# 主程序示例
if __name__ == "__main__":
    # 創建分析器
    analyzer = UniversalIcAnalyzer("Ic")
    
    # 示例 1: 分析單個數據集 (原始代碼的改進版)
    print("\n🎯 示例 1: 分析數據集 93")
    fig1 = analyzer.plot_single_dataset("93", save_plot=True)
    if fig1:
        plt.show()
    
    # 示例 2: 比較多個數據集
    print("\n🎯 示例 2: 比較數據集 93, 164, 317, 500")
    compare_ids = ["93", "164", "317", "500"]
    fig2 = analyzer.plot_comparison(compare_ids, file_type='standard', save_plot=True)
    if fig2:
        plt.show()
    
    # 示例 3: 分析所有可用數據集的統計信息
    print("\n🎯 示例 3: 批量統計分析")
    # analyzer.batch_analyze()  # 注釋掉以避免過長的分析時間
    
    # 示例 4: 展示具有正負向數據的數據集
    print("\n🎯 示例 4: 具有正負向數據的數據集")
    bipolar_ids = []
    for data_id in analyzer.get_available_ids():
        files = analyzer.available_files[data_id]
        if 'positive' in files and 'negative' in files:
            bipolar_ids.append(data_id)
    
    print(f"發現 {len(bipolar_ids)} 個具有正負向數據的數據集: {bipolar_ids[:10]}...")
    
    if bipolar_ids:
        # 展示第一個雙極性數據集
        sample_id = bipolar_ids[0]
        print(f"\n展示雙極性數據集: {sample_id}")
        fig3 = analyzer.plot_single_dataset(sample_id, ['positive', 'negative'], save_plot=True)
        if fig3:
            plt.show()
