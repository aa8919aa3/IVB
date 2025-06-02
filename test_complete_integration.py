#!/usr/bin/env python3
"""
完整集成測試 - 進階超導體分析器
"""

import sys
import traceback
from advanced_superconductor_analyzer import AdvancedSuperconductorAnalyzer

def test_complete_integration():
    """運行完整集成測試"""
    print("🚀 ADVANCED SUPERCONDUCTOR ANALYZER - COMPLETE INTEGRATION TEST")
    print("="*70)
    
    try:
        # 測試 500.csv 數據集
        print("\n📊 Testing with dataset: 500.csv")
        
        # 創建分析器實例
        analyzer = AdvancedSuperconductorAnalyzer('500.csv')
        print("✅ Analyzer initialized successfully")
        
        # 測試數據載入
        print("\n🔄 Step 1: Loading and preprocessing data...")
        analyzer.load_and_preprocess_data()
        print(f"✅ Data loaded: {analyzer.data.shape}")
        print(f"✅ y_field values: {len(analyzer.y_field_values)}")
        
        # 測試特徵提取
        print("\n🔄 Step 2: Extracting enhanced features...")
        analyzer.extract_enhanced_features()
        print(f"✅ Features extracted: {analyzer.features.shape}")
        
        # 測試圖像生成
        print("\n🔄 Step 3: Creating advanced images...")
        analyzer.create_advanced_images()
        print(f"✅ Images created: {len(analyzer.images)}")
        
        # 測試機器學習分析
        print("\n🔄 Step 4: Performing ML analysis...")
        analyzer.perform_machine_learning_analysis()
        print(f"✅ ML features: {len(analyzer.ml_features)}")
        
        # 測試可視化
        print("\n🔄 Step 5: Creating comprehensive visualizations...")
        output_file = analyzer.create_comprehensive_visualizations()
        print(f"✅ Visualization saved: {output_file}")
        
        # 測試報告生成
        print("\n🔄 Step 6: Generating comprehensive report...")
        analyzer.generate_comprehensive_report()
        print("✅ Report generated successfully")
        
        # 測試摘要
        print("\n" + "="*70)
        print("🎉 INTEGRATION TEST RESULTS:")
        print("="*70)
        print(f"📊 Dataset: 500.csv")
        print(f"📈 Data points: {len(analyzer.data):,}")
        print(f"🎯 Features extracted: {len(analyzer.features.columns)-1}")
        print(f"🖼️  Images generated: {len(analyzer.images)}")
        print(f"🤖 ML features: {len(analyzer.ml_features)}")
        print(f"🔬 Clustering results: {'Yes' if analyzer.clustering_results else 'No'}")
        print(f"📊 Output file: {output_file}")
        print("\n✅ ALL TESTS PASSED! 🎉")
        
        return True
        
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED!")
        print(f"Error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_integration()
    sys.exit(0 if success else 1)
