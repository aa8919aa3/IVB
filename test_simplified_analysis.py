#!/usr/bin/env python3
"""
簡化的高級分析器測試 - 專注於 I-V 特性曲線
"""

from advanced_superconductor_analyzer import AdvancedSuperconductorAnalyzer

def main():
    """運行簡化的分析測試"""
    print("🚀 Starting simplified I-V analysis test...")
    
    try:
        # 創建分析器
        analyzer = AdvancedSuperconductorAnalyzer('500.csv')
        
        # 步驟1: 載入數據
        print("\n📊 Step 1: Loading data...")
        analyzer.load_and_preprocess_data()
        
        # 步驟2: 提取特徵
        print("\n🔬 Step 2: Extracting features...")
        analyzer.extract_enhanced_features()
        
        # 步驟3: 創建可視化 (主要測試 I-V 特性曲線)
        print("\n📈 Step 3: Creating visualizations...")
        output_file = analyzer.create_comprehensive_visualizations()
        
        # 步驟4: 生成報告
        print("\n📋 Step 4: Generating report...")
        analyzer.generate_comprehensive_report()
        
        print(f"\n✅ Analysis completed! Output saved as: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Simplified analysis test completed successfully!")
    else:
        print("\n💥 Analysis test failed!")
