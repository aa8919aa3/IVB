#!/usr/bin/env python3
"""
測試 Lomb-Scargle → Complete Josephson Equation 擬合工作流程
"""

from Fit import *

def test_complete_workflow():
    """測試完整工作流程"""
    print("🚀 測試 Lomb-Scargle → Josephson 擬合工作流程")
    print("="*60)
    
    try:
        # 1. 創建分析器
        analyzer = JosephsonAnalyzer()
        
        # 2. 載入真實模擬數據
        print("\n📊 載入真實模擬數據...")
        sim_data = load_simulation_data('simulation_results.csv')
        sim_params = load_simulation_parameters('simulation_parameters.csv')
        
        print(f"✅ 數據載入成功:")
        print(f"   數據點數: {len(sim_data)}")
        print(f"   Phi_ext 範圍: {sim_data['Phi_ext'].min():.2e} 到 {sim_data['Phi_ext'].max():.2e}")
        print(f"   I_s 範圍: {sim_data['I_s'].min():.2e} 到 {sim_data['I_s'].max():.2e}")
        print(f"   真實頻率: {sim_params['f'].iloc[0]:.6f}")
        print(f"   真實臨界電流: {sim_params['Ic'].iloc[0]:.2e}")
        
        # 3. 準備數據
        data = {
            'Phi_ext': sim_data['Phi_ext'].values,
            'I_s': sim_data['I_s'].values,
            'I_s_error': np.full_like(sim_data['I_s'].values, 1e-7),
            'true_params': {
                'Ic': sim_params['Ic'].iloc[0],
                'f': sim_params['f'].iloc[0],
                'phi_0': sim_params['phi_0'].iloc[0],
                'T': sim_params['T'].iloc[0],
                'd': sim_params['d'].iloc[0],
                'r': sim_params['r'].iloc[0],
                'C': sim_params['C'].iloc[0]
            }
        }
        
        parameters = {
            'f': sim_params['f'].iloc[0],
            'Ic': sim_params['Ic'].iloc[0],
            'phi_0': sim_params['phi_0'].iloc[0],
            'T': sim_params['T'].iloc[0]
        }
        
        # 4. 添加到分析器
        analyzer.add_simulation_data(
            model_type='real_josephson_sim',
            data=data,
            parameters=parameters,
            model_name='Real Josephson Simulation'
        )
        
        # 5. 執行 Lomb-Scargle 分析
        print("\n🔬 執行 Lomb-Scargle 分析...")
        ls_result = analyzer.analyze_with_lomb_scargle('real_josephson_sim', detrend_order=1)
        
        if ls_result is None:
            print("❌ Lomb-Scargle 分析失敗")
            return
        
        print("✅ Lomb-Scargle 分析完成")
        
        # 6. 執行完整 Josephson 方程式擬合
        print("\n🚀 執行完整 Josephson 方程式擬合...")
        fitter = analyzer.fit_complete_josephson_equation(
            model_type='real_josephson_sim',
            use_lbfgsb=True,
            save_results=True
        )
        
        if fitter is None:
            print("❌ Josephson 擬合失敗")
            return
        
        print("✅ Josephson 擬合完成")
        
        # 7. 執行比較分析
        print("\n📊 執行比較分析...")
        comparison_stats = analyzer.compare_lomb_scargle_vs_josephson_fit(
            model_type='real_josephson_sim',
            save_plot=True
        )
        
        if comparison_stats is not None:
            print("✅ 比較分析完成")
        
        # 8. 生成最終報告
        print("\n📋 生成最終分析報告...")
        analyzer.generate_comparison_report()
        
        print("\n🎉 完整工作流程測試成功！")
        
    except Exception as e:
        print(f"❌ 測試過程中發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_complete_workflow()
