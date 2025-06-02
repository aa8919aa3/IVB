#!/usr/bin/env python3
"""
全面性能基準測試 - 優化效果評估
測試並行優化在所有數據集上的表現
"""

import time
import pandas as pd
import matplotlib.pyplot as plt
from advanced_superconductor_analyzer import AdvancedSuperconductorAnalyzer
from optimized_superconductor_analyzer import OptimizedSuperconductorAnalyzer
import numpy as np

def benchmark_single_dataset(dataset_name, use_full_analysis=False):
    """對單個數據集進行基準測試"""
    print(f"\n🔬 BENCHMARKING: {dataset_name}")
    print("="*50)
    
    results = {
        'dataset': dataset_name,
        'data_points': 0,
        'y_field_count': 0,
        'original_time': 0,
        'optimized_time': 0,
        'speedup': 0,
        'original_features': 0,
        'optimized_features': 0,
        'success': False
    }
    
    try:
        # 測試原始版本
        print("📊 Testing Original Analyzer...")
        start_time = time.time()
        
        original = AdvancedSuperconductorAnalyzer(dataset_name)
        original.load_and_preprocess_data()
        
        results['data_points'] = len(original.data)
        results['y_field_count'] = len(original.y_field_values)
        
        if use_full_analysis:
            original.extract_enhanced_features()
            original.create_advanced_images()
        else:
            original.extract_enhanced_features()
        
        original_time = time.time() - start_time
        results['original_time'] = original_time
        results['original_features'] = len(original.features) if hasattr(original, 'features') else 0
        
        # 測試優化版本
        print("⚡ Testing Optimized Analyzer...")
        start_time = time.time()
        
        optimized = OptimizedSuperconductorAnalyzer(dataset_name, use_parallel=True)
        optimized.load_and_preprocess_data()
        
        if use_full_analysis:
            optimized.extract_enhanced_features_parallel()
            optimized.create_advanced_images()
        else:
            optimized.extract_enhanced_features_parallel()
        
        optimized_time = time.time() - start_time
        results['optimized_time'] = optimized_time
        results['optimized_features'] = len(optimized.features) if hasattr(optimized, 'features') else 0
        
        # 計算性能提升
        if optimized_time > 0:
            results['speedup'] = original_time / optimized_time
        
        results['success'] = True
        
        print(f"✅ Original: {original_time:.2f}s, Optimized: {optimized_time:.2f}s")
        print(f"⚡ Speedup: {results['speedup']:.2f}x")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        results['error'] = str(e)
    
    return results

def run_comprehensive_benchmark():
    """運行全面性能基準測試"""
    print("🚀 COMPREHENSIVE PERFORMANCE BENCHMARK")
    print("="*70)
    print("Testing optimization performance across all datasets")
    
    datasets = ['164.csv', '317.csv', '500.csv']
    
    # 快速基準測試（僅特徵提取）
    print("\n📊 PHASE 1: FEATURE EXTRACTION BENCHMARK")
    print("-"*50)
    
    quick_results = []
    for dataset in datasets:
        result = benchmark_single_dataset(dataset, use_full_analysis=False)
        quick_results.append(result)
    
    # 完整基準測試（包含圖像生成）
    print("\n📊 PHASE 2: FULL ANALYSIS BENCHMARK (500.csv only)")
    print("-"*50)
    
    full_result = benchmark_single_dataset('500.csv', use_full_analysis=True)
    
    # 生成詳細報告
    print("\n" + "="*70)
    print("📈 COMPREHENSIVE BENCHMARK RESULTS")
    print("="*70)
    
    # 創建結果表格
    results_df = pd.DataFrame(quick_results)
    
    print("\n🏃‍♂️ FEATURE EXTRACTION PERFORMANCE:")
    print("-"*50)
    print(f"{'Dataset':<12} {'Points':<8} {'Fields':<8} {'Orig(s)':<8} {'Opt(s)':<8} {'Speedup':<8}")
    print("-"*50)
    
    for _, row in results_df.iterrows():
        if row['success']:
            print(f"{row['dataset']:<12} {row['data_points']:<8,} {row['y_field_count']:<8} "
                  f"{row['original_time']:<8.2f} {row['optimized_time']:<8.2f} {row['speedup']:<8.2f}x")
    
    # 統計分析
    successful_results = results_df[results_df['success']]
    if len(successful_results) > 0:
        avg_speedup = successful_results['speedup'].mean()
        max_speedup = successful_results['speedup'].max()
        min_speedup = successful_results['speedup'].min()
        total_time_saved = (successful_results['original_time'] - successful_results['optimized_time']).sum()
        
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"Average Speedup: {avg_speedup:.2f}x")
        print(f"Best Speedup: {max_speedup:.2f}x")
        print(f"Worst Speedup: {min_speedup:.2f}x")
        print(f"Total Time Saved: {total_time_saved:.2f} seconds")
    
    # 完整分析結果
    if full_result['success']:
        print(f"\n🔬 FULL ANALYSIS PERFORMANCE (500.csv):")
        print(f"Original Time: {full_result['original_time']:.2f}s")
        print(f"Optimized Time: {full_result['optimized_time']:.2f}s")
        print(f"Full Analysis Speedup: {full_result['speedup']:.2f}x")
    
    # 創建性能圖表
    create_performance_charts(quick_results, full_result)
    
    # 生成優化建議
    generate_optimization_recommendations(quick_results)
    
    return quick_results, full_result

def create_performance_charts(quick_results, full_result):
    """創建性能比較圖表"""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 處理時間比較
        datasets = [r['dataset'] for r in quick_results if r['success']]
        original_times = [r['original_time'] for r in quick_results if r['success']]
        optimized_times = [r['optimized_time'] for r in quick_results if r['success']]
        
        x = np.arange(len(datasets))
        width = 0.35
        
        ax1.bar(x - width/2, original_times, width, label='Original', color='lightcoral')
        ax1.bar(x + width/2, optimized_times, width, label='Optimized', color='lightgreen')
        ax1.set_xlabel('Dataset')
        ax1.set_ylabel('Processing Time (seconds)')
        ax1.set_title('Processing Time Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(datasets)
        ax1.legend()
        
        # 2. 加速比
        speedups = [r['speedup'] for r in quick_results if r['success']]
        colors = ['gold' if s > 1 else 'lightcoral' for s in speedups]
        
        bars = ax2.bar(datasets, speedups, color=colors)
        ax2.set_ylabel('Speedup Factor')
        ax2.set_title('Performance Speedup')
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        
        # 添加數值標籤
        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{speedup:.2f}x', ha='center', va='bottom')
        
        # 3. 數據規模 vs 性能
        data_points = [r['data_points'] for r in quick_results if r['success']]
        
        ax3.scatter(data_points, speedups, color='blue', s=100)
        ax3.set_xlabel('Data Points')
        ax3.set_ylabel('Speedup Factor')
        ax3.set_title('Speedup vs Dataset Size')
        
        # 添加趨勢線
        if len(data_points) > 1:
            z = np.polyfit(data_points, speedups, 1)
            p = np.poly1d(z)
            ax3.plot(data_points, p(data_points), "r--", alpha=0.8)
        
        # 4. 處理率比較 (points/second)
        original_rates = [r['data_points']/r['original_time'] for r in quick_results if r['success']]
        optimized_rates = [r['data_points']/r['optimized_time'] for r in quick_results if r['success']]
        
        ax4.bar(x - width/2, original_rates, width, label='Original', color='lightcoral')
        ax4.bar(x + width/2, optimized_rates, width, label='Optimized', color='lightgreen')
        ax4.set_xlabel('Dataset')
        ax4.set_ylabel('Processing Rate (points/second)')
        ax4.set_title('Processing Rate Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(datasets)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('performance_benchmark_results.png', dpi=300, bbox_inches='tight')
        print("\n📊 Performance charts saved: performance_benchmark_results.png")
        
    except Exception as e:
        print(f"⚠️  Chart generation failed: {e}")

def generate_optimization_recommendations(results):
    """基於基準測試結果生成優化建議"""
    print("\n💡 OPTIMIZATION RECOMMENDATIONS")
    print("="*50)
    
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("❌ No successful results to analyze")
        return
    
    # 分析性能模式
    speedups = [r['speedup'] for r in successful_results]
    avg_speedup = np.mean(speedups)
    
    print(f"📊 Current Optimization Status:")
    print(f"   Average Speedup: {avg_speedup:.2f}x")
    
    if avg_speedup > 1.5:
        print("✅ Parallel optimization is working well!")
        print("🎯 Recommendations for further improvement:")
        print("   • Consider GPU acceleration for large datasets")
        print("   • Implement memory pooling for frequent allocations")
        print("   • Add adaptive chunking based on system resources")
    elif avg_speedup > 1.0:
        print("🟡 Moderate improvement achieved")
        print("🔧 Recommendations:")
        print("   • Increase parallelization granularity")
        print("   • Profile CPU-bound operations")
        print("   • Consider vectorization optimizations")
    else:
        print("🔴 Optimization overhead may be too high")
        print("⚠️  Recommendations:")
        print("   • Increase minimum dataset size for parallel processing")
        print("   • Optimize inter-process communication")
        print("   • Consider sequential processing for small datasets")
    
    # 數據集特定建議
    for result in successful_results:
        dataset = result['dataset']
        speedup = result['speedup']
        size = result['data_points']
        
        if speedup < 1.0:
            print(f"⚠️  {dataset}: Consider sequential processing (overhead too high)")
        elif size > 40000 and speedup < 2.0:
            print(f"🔧 {dataset}: Large dataset with low speedup - investigate bottlenecks")

if __name__ == "__main__":
    quick_results, full_result = run_comprehensive_benchmark()
