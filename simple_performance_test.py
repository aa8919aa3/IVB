#!/usr/bin/env python3
"""
簡化性能測試 - 快速驗證優化效果
"""

import time
import sys
import os

def simple_performance_test():
    """簡單的性能對比測試"""
    print("🚀 SIMPLE PERFORMANCE TEST")
    print("="*40)
    
    # 確保模塊可以導入
    try:
        from advanced_superconductor_analyzer import AdvancedSuperconductorAnalyzer
        from optimized_superconductor_analyzer import OptimizedSuperconductorAnalyzer
        print("✅ Modules imported successfully")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return
    
    dataset = "500.csv"
    if not os.path.exists(dataset):
        print(f"❌ Dataset {dataset} not found")
        return
    
    # 測試原始版本 - 只測試關鍵部分
    print(f"\n📊 Testing Original Analyzer ({dataset})...")
    try:
        start_time = time.time()
        original = AdvancedSuperconductorAnalyzer(dataset)
        original.load_and_preprocess_data()
        
        # 只測試前20個y_field值以加快測試
        original.y_field_values = original.y_field_values[:20]
        original.extract_enhanced_features()
        
        original_time = time.time() - start_time
        original_features = len(original.features)
        print(f"✅ Original: {original_time:.2f}s, {original_features} features")
    except Exception as e:
        print(f"❌ Original test failed: {e}")
        return
    
    # 測試優化版本
    print(f"\n⚡ Testing Optimized Analyzer ({dataset})...")
    try:
        start_time = time.time()
        optimized = OptimizedSuperconductorAnalyzer(dataset, use_parallel=True)
        optimized.load_and_preprocess_data()
        
        # 只測試前20個y_field值
        optimized.y_field_values = optimized.y_field_values[:20]
        optimized.extract_enhanced_features_parallel()
        
        optimized_time = time.time() - start_time
        optimized_features = len(optimized.features)
        print(f"✅ Optimized: {optimized_time:.2f}s, {optimized_features} features")
    except Exception as e:
        print(f"❌ Optimized test failed: {e}")
        return
    
    # 計算結果
    if optimized_time > 0:
        speedup = original_time / optimized_time
        print(f"\n📈 RESULTS:")
        print(f"Original time: {original_time:.2f}s")
        print(f"Optimized time: {optimized_time:.2f}s")
        print(f"Speedup: {speedup:.2f}x")
        
        if speedup > 1.0:
            print("✅ Optimization successful!")
        else:
            print("⚠️  No speedup (possibly due to overhead)")
        
        if original_features == optimized_features:
            print("✅ Feature consistency verified")
        else:
            print(f"⚠️  Feature count differs: {original_features} vs {optimized_features}")
    
    print("\n🎉 Test completed!")

if __name__ == "__main__":
    simple_performance_test()
