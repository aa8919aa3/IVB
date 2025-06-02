#!/usr/bin/env python3
"""
測試critical current修復效果
"""

import pandas as pd
import numpy as np

def test_critical_current_calculation():
    """測試修復後的critical current計算"""
    
    # Load data
    df = pd.read_csv('317.csv')
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Find voltage column
    voltage_col = None
    for col in df.columns:
        if 'volt' in col.lower() or 'v_' in col.lower():
            voltage_col = col
            break
    
    if voltage_col is None:
        print("No voltage column found!")
        return
    
    print(f"Using voltage column: {voltage_col}")
    
    # Calculate dV/dI if not present
    if 'dV_dI' not in df.columns:
        print("Calculating dV/dI...")
        df = df.sort_values(['y_field', 'appl_current'])
        dV_dI_values = []
        
        for field in df['y_field'].unique():
            field_data = df[df['y_field'] == field].sort_values('appl_current')
            if len(field_data) > 1:
                current = field_data['appl_current'].values
                voltage = field_data[voltage_col].values
                dV_dI = np.gradient(voltage, current)
                dV_dI_values.extend(dV_dI)
            else:
                dV_dI_values.append(0)
        
        df['dV_dI'] = dV_dI_values
    
    # Test the fixed critical current calculation
    print("\n=== Testing Fixed Critical Current Calculation ===")
    
    field_groups = df.groupby('y_field')
    critical_currents = []
    ic_positive_values = []
    ic_negative_values = []
    
    for field, group in field_groups:
        print(f"\nAnalyzing field: {field}")
        
        # Sort by current
        group_sorted = group.sort_values('appl_current')
        voltage_values = group_sorted[voltage_col].values
        current_values = group_sorted['appl_current'].values
        dV_dI_values = group_sorted['dV_dI'].values
        
        print(f"  Data points: {len(current_values)}")
        print(f"  Current range: {current_values.min():.6f} to {current_values.max():.6f}")
        
        # Analyze positive and negative current regions separately
        positive_mask = current_values > 0
        negative_mask = current_values < 0
        
        # Initialize field-specific critical current values
        field_ic_positive = None
        field_ic_negative = None
        
        # Positive critical current (from dV/dI peak)
        if np.any(positive_mask):
            dV_dI_pos = dV_dI_values[positive_mask]
            current_pos = current_values[positive_mask]
            if len(dV_dI_pos) > 0:
                max_idx = np.argmax(dV_dI_pos)
                field_ic_positive = current_pos[max_idx]
                ic_positive_values.append(field_ic_positive)
                print(f"  Positive Ic: {field_ic_positive*1e6:.3f} µA")
        
        # Negative critical current (from dV/dI peak)
        if np.any(negative_mask):
            dV_dI_neg = dV_dI_values[negative_mask]
            current_neg = current_values[negative_mask]
            if len(dV_dI_neg) > 0:
                max_idx = np.argmax(dV_dI_neg)
                field_ic_negative = abs(current_neg[max_idx])
                ic_negative_values.append(field_ic_negative)
                print(f"  Negative Ic: {field_ic_negative*1e6:.3f} µA")
        
        # Calculate average critical current for this field only
        field_ic_values = []
        if field_ic_positive is not None:
            field_ic_values.append(field_ic_positive)
        if field_ic_negative is not None:
            field_ic_values.append(field_ic_negative)
        
        if field_ic_values:
            field_avg = np.mean(field_ic_values)
            critical_currents.append(field_avg)
            print(f"  Field average Ic: {field_avg*1e6:.3f} µA")
    
    # Final results
    print("\n=== Final Results ===")
    if critical_currents:
        overall_mean = np.mean(critical_currents)
        overall_std = np.std(critical_currents)
        print(f"Overall critical current: {overall_mean*1e6:.3f} ± {overall_std*1e6:.3f} µA")
        print(f"Number of field measurements: {len(critical_currents)}")
    
    if ic_positive_values:
        pos_mean = np.mean(ic_positive_values)
        pos_std = np.std(ic_positive_values)
        print(f"Positive critical current: {pos_mean*1e6:.3f} ± {pos_std*1e6:.3f} µA")
    
    if ic_negative_values:
        neg_mean = np.mean(ic_negative_values)
        neg_std = np.std(ic_negative_values)
        print(f"Negative critical current: {neg_mean*1e6:.3f} ± {neg_std*1e6:.3f} µA")

if __name__ == "__main__":
    test_critical_current_calculation()
