#!/usr/bin/env python3
"""
測試 I-V 特性曲線繪製功能
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from advanced_superconductor_analyzer import AdvancedSuperconductorAnalyzer

def test_iv_characteristics():
    """測試 I-V 特性曲線繪製功能"""
    print("🧪 Testing I-V Characteristics Plotting...")
    
    try:
        # 創建分析器
        analyzer = AdvancedSuperconductorAnalyzer('500.csv')
        print("✅ Analyzer created successfully")
        
        # 載入數據
        analyzer.load_and_preprocess_data()
        print("✅ Data loaded successfully")
        
        # 提取特徵
        analyzer.extract_enhanced_features()
        print("✅ Features extracted successfully")
        
        # 測試單獨的 I-V 繪製功能
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 測試新增的 I-V 特性曲線方法
        analyzer._plot_iv_characteristics(axes[0, 0])
        analyzer._plot_sample_dvdi_curves(axes[0, 1])
        analyzer._plot_transition_analysis(axes[1, 0])
        analyzer._plot_data_quality_analysis(axes[1, 1])
        
        plt.tight_layout()
        
        # 保存測試圖像
        output_file = "test_iv_characteristics.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ I-V characteristics test plot saved as: {output_file}")
        
        # 顯示一些統計信息
        print(f"\n📊 Dataset Statistics:")
        print(f"   • Total data points: {len(analyzer.data):,}")
        print(f"   • Unique y_field values: {len(analyzer.y_field_values)}")
        print(f"   • Current range: {analyzer.data['appl_current'].min()*1e6:.2f} to {analyzer.data['appl_current'].max()*1e6:.2f} µA")
        print(f"   • Voltage column used: {analyzer.voltage_column}")
        
        if hasattr(analyzer, 'features') and len(analyzer.features) > 0:
            print(f"   • Features extracted: {len(analyzer.features.columns)-1}")
            
            # 顯示一些關鍵特徵統計
            key_features = ['Ic_average', 'Rn', 'n_value', 'transition_width']
            for feature in key_features:
                if feature in analyzer.features.columns:
                    data = analyzer.features[feature].dropna()
                    if len(data) > 0:
                        if feature == 'Ic_average':
                            print(f"   • {feature}: {data.mean()*1e6:.3f} ± {data.std()*1e6:.3f} µA")
                        elif feature == 'transition_width':
                            print(f"   • {feature}: {data.mean()*1e6:.3f} ± {data.std()*1e6:.3f} µA")
                        else:
                            print(f"   • {feature}: {data.mean():.3f} ± {data.std():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_iv_characteristics()
    if success:
        print("\n🎉 I-V characteristics plotting test completed successfully!")
    else:
        print("\n💥 Test failed!")
