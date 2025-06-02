#!/usr/bin/env python3
"""
Critical Current Analysis Method Comparison
Compare the exact implementation between 500.py and 317.py methods
"""

import pandas as pd
import numpy as np

def analyze_500_method(df, voltage_col):
    """Replicate the exact 500.py critical current calculation method"""
    print("🔬 500.py Method Analysis:")
    
    # Group by y_field (exact same as 500.py)
    field_groups = df.groupby('y_field')
    all_features = []
    
    for field, group_data in field_groups:
        features = {'y_field': field}
        
        current = group_data['appl_current'].values
        voltage = group_data[voltage_col].values
        
        if 'dV_dI' in group_data.columns:
            dV_dI = group_data['dV_dI'].values
            
            # 1. Critical current features (exact 500.py implementation)
            positive_mask = current > 0
            negative_mask = current < 0
            
            if np.any(positive_mask):
                dV_dI_pos = dV_dI[positive_mask]
                current_pos = current[positive_mask]
                if len(dV_dI_pos) > 0:
                    max_idx = np.argmax(dV_dI_pos)
                    features['Ic_positive'] = current_pos[max_idx]
                    features['dV_dI_max'] = dV_dI_pos[max_idx]
            
            if np.any(negative_mask):
                dV_dI_neg = dV_dI[negative_mask]
                current_neg = current[negative_mask]
                if len(dV_dI_neg) > 0:
                    max_idx = np.argmax(dV_dI_neg)
                    features['Ic_negative'] = abs(current_neg[max_idx])
            
            # Average critical current
            ic_vals = []
            if 'Ic_positive' in features:
                ic_vals.append(features['Ic_positive'])
            if 'Ic_negative' in features:
                ic_vals.append(features['Ic_negative'])
            if ic_vals:
                features['Ic_average'] = np.mean(ic_vals)
        
        all_features.append(features)
    
    # Calculate statistics across all fields
    ic_positive_values = [f['Ic_positive'] for f in all_features if 'Ic_positive' in f]
    ic_negative_values = [f['Ic_negative'] for f in all_features if 'Ic_negative' in f]
    ic_average_values = [f['Ic_average'] for f in all_features if 'Ic_average' in f]
    
    print(f"   • Fields processed: {len(all_features)}")
    print(f"   • Positive Ic values: {len(ic_positive_values)}")
    print(f"   • Negative Ic values: {len(ic_negative_values)}")
    print(f"   • Average Ic values: {len(ic_average_values)}")
    
    if ic_average_values:
        print(f"   • Ic mean: {np.mean(ic_average_values)*1e6:.3f} µA")
        print(f"   • Ic std: {np.std(ic_average_values)*1e6:.3f} µA")
        
    return ic_average_values, ic_positive_values, ic_negative_values

def analyze_317_current_method(df, voltage_col):
    """Current 317.py critical current calculation method"""
    print("\n🔧 Current 317.py Method Analysis:")
    
    # Current implementation from 317.py
    field_groups = df.groupby('y_field')
    critical_currents = []
    ic_positive_values = []
    ic_negative_values = []
    
    for field, group in field_groups:
        group_sorted = group.sort_values('appl_current')
        voltage_values = group_sorted[voltage_col].values
        current_values = group_sorted['appl_current'].values
        
        if 'dV_dI' in group_sorted.columns:
            dV_dI_values = group_sorted['dV_dI'].values
            
            positive_mask = current_values > 0
            negative_mask = current_values < 0
            
            # The issue: accumulating values globally instead of per-field
            if np.any(positive_mask):
                dV_dI_pos = dV_dI_values[positive_mask]
                current_pos = current_values[positive_mask]
                if len(dV_dI_pos) > 0:
                    max_idx = np.argmax(dV_dI_pos)
                    ic_pos = current_pos[max_idx]
                    ic_positive_values.append(ic_pos)  # Global accumulation
            
            if np.any(negative_mask):
                dV_dI_neg = dV_dI_values[negative_mask]
                current_neg = current_values[negative_mask]
                if len(dV_dI_neg) > 0:
                    max_idx = np.argmax(dV_dI_neg)
                    ic_neg = abs(current_neg[max_idx])
                    ic_negative_values.append(ic_neg)  # Global accumulation
            
            # Problem: using last appended values instead of current field values
            field_ic_values = []
            if ic_positive_values:
                field_ic_values.append(ic_positive_values[-1])  # Wrong: using last global value
            if ic_negative_values:
                field_ic_values.append(ic_negative_values[-1])  # Wrong: using last global value
            
            if field_ic_values:
                critical_currents.append(np.mean(field_ic_values))
    
    print(f"   • Fields processed: {len(critical_currents)}")
    print(f"   • Critical currents calculated: {len(critical_currents)}")
    
    if critical_currents:
        print(f"   • Ic mean: {np.mean(critical_currents)*1e6:.3f} µA")
        print(f"   • Ic std: {np.std(critical_currents)*1e6:.3f} µA")
        
    return critical_currents, ic_positive_values, ic_negative_values

def analyze_corrected_317_method(df, voltage_col):
    """Corrected 317.py method that exactly follows 500.py approach"""
    print("\n✅ Corrected 317.py Method Analysis:")
    
    field_groups = df.groupby('y_field')
    critical_currents = []
    all_ic_positive = []
    all_ic_negative = []
    
    for field, group in field_groups:
        group_sorted = group.sort_values('appl_current')
        voltage_values = group_sorted[voltage_col].values
        current_values = group_sorted['appl_current'].values
        
        if 'dV_dI' in group_sorted.columns:
            dV_dI_values = group_sorted['dV_dI'].values
            
            # Per-field calculation (like 500.py)
            field_ic_positive = None
            field_ic_negative = None
            
            positive_mask = current_values > 0
            negative_mask = current_values < 0
            
            if np.any(positive_mask):
                dV_dI_pos = dV_dI_values[positive_mask]
                current_pos = current_values[positive_mask]
                if len(dV_dI_pos) > 0:
                    max_idx = np.argmax(dV_dI_pos)
                    field_ic_positive = current_pos[max_idx]
                    all_ic_positive.append(field_ic_positive)
            
            if np.any(negative_mask):
                dV_dI_neg = dV_dI_values[negative_mask]
                current_neg = current_values[negative_mask]
                if len(dV_dI_neg) > 0:
                    max_idx = np.argmax(dV_dI_neg)
                    field_ic_negative = abs(current_neg[max_idx])
                    all_ic_negative.append(field_ic_negative)
            
            # Calculate field-specific average
            field_ic_values = []
            if field_ic_positive is not None:
                field_ic_values.append(field_ic_positive)
            if field_ic_negative is not None:
                field_ic_values.append(field_ic_negative)
            
            if field_ic_values:
                critical_currents.append(np.mean(field_ic_values))
    
    print(f"   • Fields processed: {len(critical_currents)}")
    print(f"   • Critical currents calculated: {len(critical_currents)}")
    
    if critical_currents:
        print(f"   • Ic mean: {np.mean(critical_currents)*1e6:.3f} µA")
        print(f"   • Ic std: {np.std(critical_currents)*1e6:.3f} µA")
        
    return critical_currents, all_ic_positive, all_ic_negative

def main():
    """Compare critical current analysis methods"""
    print("="*80)
    print("🔍 CRITICAL CURRENT ANALYSIS METHOD COMPARISON")
    print("="*80)
    
    # Test on 317.csv first
    print("\n📊 Testing on 317.csv dataset:")
    df = pd.read_csv("317.csv")
    voltage_col = 'meas_voltage_K1'
    
    print(f"   • Dataset shape: {df.shape}")
    print(f"   • Unique y_field values: {len(df['y_field'].unique())}")
    print(f"   • Has dV_dI column: {'dV_dI' in df.columns}")
    
    # Compare methods
    ic_500, _, _ = analyze_500_method(df, voltage_col)
    ic_317_current, _, _ = analyze_317_current_method(df, voltage_col)
    ic_317_corrected, _, _ = analyze_corrected_317_method(df, voltage_col)
    
    print("\n" + "="*80)
    print("📈 COMPARISON SUMMARY:")
    print("="*80)
    
    methods = [
        ("500.py Method (Reference)", ic_500),
        ("317.py Current Method", ic_317_current),
        ("317.py Corrected Method", ic_317_corrected)
    ]
    
    for method_name, values in methods:
        if values:
            mean_val = np.mean(values) * 1e6
            std_val = np.std(values) * 1e6
            print(f"{method_name:25} | {mean_val:8.3f} ± {std_val:6.3f} µA | {len(values):3d} values")
        else:
            print(f"{method_name:25} | {'N/A':>18} | {'0':>3} values")
    
    print("\n💡 ANALYSIS:")
    if ic_500 and ic_317_corrected:
        diff_corrected = abs(np.mean(ic_500) - np.mean(ic_317_corrected)) / np.mean(ic_500) * 100
        print(f"   • Difference between 500.py and corrected 317.py: {diff_corrected:.1f}%")
        
    if ic_317_current and ic_317_corrected:
        diff_current = abs(np.mean(ic_317_current) - np.mean(ic_317_corrected)) / np.mean(ic_317_corrected) * 100
        print(f"   • Difference between current and corrected 317.py: {diff_current:.1f}%")
    
    print(f"   • The 500.py method is more accurate because it properly calculates")
    print(f"     critical current per magnetic field value, then takes statistics")
    print(f"   • The current 317.py method has a bug in global accumulation")

if __name__ == "__main__":
    main()
