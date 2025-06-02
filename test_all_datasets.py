#!/usr/bin/env python3
"""
全數據集兼容性測試 - 進階超導體分析器
測試所有可用數據集的兼容性和性能
"""

import sys
import time
import traceback
import os
from advanced_superconductor_analyzer import AdvancedSuperconductorAnalyzer

def test_dataset(dataset_name):
    """測試單個數據集"""
    print(f"\n🔬 TESTING DATASET: {dataset_name}")
    print("="*50)
    
    start_time = time.time()
    
    try:
        # 創建分析器實例
        analyzer = AdvancedSuperconductorAnalyzer(dataset_name)
        print("✅ Analyzer initialized")
        
        # 1. 數據載入和預處理
        print("🔄 Loading and preprocessing data...")
        analyzer.load_and_preprocess_data()
        data_shape = analyzer.data.shape
        y_field_count = len(analyzer.y_field_values)
        print(f"✅ Data: {data_shape[0]:,} points, {y_field_count} y_field values")
        
        # 2. 特徵提取
        print("🔄 Extracting enhanced features...")
        analyzer.extract_enhanced_features()
        feature_count = analyzer.features.shape[1] - 1  # 減去y_field列
        print(f"✅ Features: {feature_count} extracted")
        
        # 3. 圖像生成
        print("🔄 Creating advanced images...")
        analyzer.create_advanced_images()
        image_count = len(analyzer.images)
        print(f"✅ Images: {image_count} created")
        
        # 4. 機器學習分析
        print("🔄 Performing ML analysis...")
        analyzer.perform_machine_learning_analysis()
        ml_feature_count = len(analyzer.ml_features)
        print(f"✅ ML features: {ml_feature_count} extracted")
        
        # 5. 可視化生成
        print("🔄 Creating visualizations...")
        output_file = analyzer.create_comprehensive_visualizations()
        print(f"✅ Visualization: {output_file}")
        
        # 6. 報告生成
        print("🔄 Generating report...")
        analyzer.generate_comprehensive_report()
        print("✅ Report generated")
        
        # 計算處理時間
        processing_time = time.time() - start_time
        
        # 返回測試結果
        results = {
            'dataset': dataset_name,
            'success': True,
            'data_points': data_shape[0],
            'y_field_values': y_field_count,
            'features_extracted': feature_count,
            'images_created': image_count,
            'ml_features': ml_feature_count,
            'processing_time': processing_time,
            'output_file': output_file,
            'error': None
        }
        
        print(f"✅ {dataset_name} processed in {processing_time:.2f} seconds")
        return results
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = str(e)
        
        print(f"❌ ERROR in {dataset_name}: {error_msg}")
        
        results = {
            'dataset': dataset_name,
            'success': False,
            'processing_time': processing_time,
            'error': error_msg,
            'traceback': traceback.format_exc()
        }
        
        return results

def run_comprehensive_dataset_test():
    """運行所有數據集的全面測試"""
    print("🚀 COMPREHENSIVE DATASET COMPATIBILITY TEST")
    print("="*70)
    print("Testing all available datasets for compatibility and performance")
    
    # 定義要測試的數據集
    datasets = ['164.csv', '317.csv', '500.csv']
    
    # 檢查數據集是否存在
    available_datasets = []
    for dataset in datasets:
        if os.path.exists(dataset):
            available_datasets.append(dataset)
            print(f"✅ Found: {dataset}")
        else:
            print(f"❌ Missing: {dataset}")
    
    if not available_datasets:
        print("❌ No datasets found! Please ensure CSV files are in the current directory.")
        return False
    
    print(f"\n📊 Testing {len(available_datasets)} datasets...")
    
    # 測試所有可用數據集
    results = []
    total_start_time = time.time()
    
    for dataset in available_datasets:
        result = test_dataset(dataset)
        results.append(result)
    
    total_time = time.time() - total_start_time
    
    # 生成總結報告
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
    print("="*70)
    
    successful_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]
    
    print(f"✅ Successful tests: {len(successful_tests)}/{len(results)}")
    print(f"❌ Failed tests: {len(failed_tests)}")
    print(f"⏱️  Total processing time: {total_time:.2f} seconds")
    
    if successful_tests:
        print("\n📈 SUCCESSFUL DATASETS:")
        print("-" * 50)
        for result in successful_tests:
            print(f"🔬 {result['dataset']}:")
            print(f"   📊 Data points: {result['data_points']:,}")
            print(f"   🎯 Features: {result['features_extracted']}")
            print(f"   🖼️  Images: {result['images_created']}")
            print(f"   🤖 ML features: {result['ml_features']}")
            print(f"   ⏱️  Time: {result['processing_time']:.2f}s")
            print(f"   📁 Output: {result['output_file']}")
    
    if failed_tests:
        print("\n❌ FAILED DATASETS:")
        print("-" * 50)
        for result in failed_tests:
            print(f"🔬 {result['dataset']}:")
            print(f"   ❌ Error: {result['error']}")
            print(f"   ⏱️  Time: {result['processing_time']:.2f}s")
    
    # 性能分析
    if successful_tests:
        print("\n⚡ PERFORMANCE ANALYSIS:")
        print("-" * 50)
        
        # 計算每個指標的統計數據
        data_points = [r['data_points'] for r in successful_tests]
        processing_times = [r['processing_time'] for r in successful_tests]
        features = [r['features_extracted'] for r in successful_tests]
        
        avg_time = sum(processing_times) / len(processing_times)
        max_time = max(processing_times)
        min_time = min(processing_times)
        
        max_dataset = successful_tests[processing_times.index(max_time)]['dataset']
        min_dataset = successful_tests[processing_times.index(min_time)]['dataset']
        
        print(f"📊 Dataset size range: {min(data_points):,} - {max(data_points):,} points")
        print(f"⏱️  Processing time: {min_time:.2f}s - {max_time:.2f}s (avg: {avg_time:.2f}s)")
        print(f"🚀 Fastest: {min_dataset} ({min_time:.2f}s)")
        print(f"🐌 Slowest: {max_dataset} ({max_time:.2f}s)")
        print(f"🎯 Feature extraction: {min(features)} - {max(features)} features")
    
    # 兼容性報告
    print("\n🔗 COMPATIBILITY STATUS:")
    print("-" * 50)
    if len(successful_tests) == len(results):
        print("✅ FULL COMPATIBILITY: All datasets processed successfully!")
        print("🎉 The analyzer is production-ready for all available datasets.")
    else:
        print(f"⚠️  PARTIAL COMPATIBILITY: {len(successful_tests)}/{len(results)} datasets successful")
        print("🔧 Some datasets may require additional handling or debugging.")
    
    return len(failed_tests) == 0

if __name__ == "__main__":
    success = run_comprehensive_dataset_test()
    
    print("\n" + "="*70)
    if success:
        print("🎉 ALL DATASET TESTS PASSED!")
        print("✅ Ready for production deployment")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("🔧 Review errors and implement fixes")
    
    print("="*70)
    sys.exit(0 if success else 1)
