#!/usr/bin/env python3
"""
Comparative Analysis of Multiple Superconductor Datasets
Comparing results from 164.csv, 500.csv, and 317.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def load_dataset(filename):
    """Load and preprocess a dataset"""
    print(f"Loading {filename}...")
    df = pd.read_csv(filename)
    
    # Automatically detect voltage column
    voltage_col = None
    for col in ['meas_voltage_K2', 'meas_voltage_K1', 'meas_voltage']:
        if col in df.columns:
            voltage_col = col
            break
    
    if voltage_col is None:
        raise ValueError(f"No voltage column found in {filename}!")
    
    # Clean data
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    return df, voltage_col

def extract_key_features(df, voltage_col, dataset_name):
    """Extract key features for comparison"""
    features = {
        'dataset': dataset_name,
        'n_samples': len(df),
        'voltage_col': voltage_col,
        'field_range': df['y_field'].max() - df['y_field'].min(),
        'field_steps': len(df['y_field'].unique()),
        'current_range': df['appl_current'].max() - df['appl_current'].min(),
        'current_steps': len(df['appl_current'].unique()),
        'voltage_mean': df[voltage_col].mean(),
        'voltage_std': df[voltage_col].std(),
        'voltage_range': df[voltage_col].max() - df[voltage_col].min(),
        'current_mean': df['appl_current'].mean(),
        'current_std': df['appl_current'].std(),
    }
    
    # Add differential resistance info if available
    if 'dV_dI' in df.columns:
        features['has_diff_resistance'] = True
        features['diff_resistance_mean'] = df['dV_dI'].mean()
        features['diff_resistance_std'] = df['dV_dI'].std()
    else:
        features['has_diff_resistance'] = False
    
    # Calculate critical current estimates
    try:
        field_groups = df.groupby('y_field')
        critical_currents = []
        
        for field, group in field_groups:
            group_sorted = group.sort_values('appl_current')
            voltage_values = group_sorted[voltage_col].values
            current_values = group_sorted['appl_current'].values
            
            # Find zero crossings
            zero_crossings = np.where(np.diff(np.sign(voltage_values)))[0]
            if len(zero_crossings) > 0:
                idx = zero_crossings[0]
                if idx < len(current_values) - 1:
                    x1, x2 = current_values[idx], current_values[idx + 1]
                    y1, y2 = voltage_values[idx], voltage_values[idx + 1]
                    if y2 != y1:
                        critical_current = x1 - y1 * (x2 - x1) / (y2 - y1)
                        critical_currents.append(abs(critical_current))
        
        if critical_currents:
            features['critical_current_mean'] = np.mean(critical_currents)
            features['critical_current_std'] = np.std(critical_currents)
            features['critical_current_max'] = np.max(critical_currents)
        else:
            features['critical_current_mean'] = 0
            features['critical_current_std'] = 0
            features['critical_current_max'] = 0
    except:
        features['critical_current_mean'] = 0
        features['critical_current_std'] = 0
        features['critical_current_max'] = 0
    
    return features

def create_comparison_plots(datasets_info, datasets_data):
    """Create comprehensive comparison plots"""
    print("Creating comparison visualizations...")
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Dataset Overview Comparison
    plt.subplot(3, 4, 1)
    dataset_names = [info['dataset'] for info in datasets_info]
    sample_counts = [info['n_samples'] for info in datasets_info]
    
    bars = plt.bar(dataset_names, sample_counts, color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.title('Dataset Sizes Comparison')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    for bar, count in zip(bars, sample_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(sample_counts),
                f'{count:,}', ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    # 2. Field Range Comparison
    plt.subplot(3, 4, 2)
    field_ranges = [info['field_range'] for info in datasets_info]
    bars = plt.bar(dataset_names, field_ranges, color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.title('Magnetic Field Range Comparison')
    plt.ylabel('Field Range (T)')
    plt.xticks(rotation=45)
    for bar, range_val in zip(bars, field_ranges):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(field_ranges),
                f'{range_val:.4f}', ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    # 3. Current Range Comparison
    plt.subplot(3, 4, 3)
    current_ranges = [info['current_range']*1e6 for info in datasets_info]  # Convert to µA
    bars = plt.bar(dataset_names, current_ranges, color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.title('Current Range Comparison')
    plt.ylabel('Current Range (µA)')
    plt.xticks(rotation=45)
    for bar, range_val in zip(bars, current_ranges):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(current_ranges),
                f'{range_val:.1f}', ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    # 4. Critical Current Comparison
    plt.subplot(3, 4, 4)
    critical_currents = [info['critical_current_mean']*1e6 for info in datasets_info]  # Convert to µA
    bars = plt.bar(dataset_names, critical_currents, color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.title('Average Critical Current Comparison')
    plt.ylabel('Critical Current (µA)')
    plt.xticks(rotation=45)
    for bar, ic_val in zip(bars, critical_currents):
        if ic_val > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(critical_currents),
                    f'{ic_val:.1f}', ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    # 5-7. I-V Characteristics for each dataset
    colors = ['blue', 'red', 'green']
    for i, (dataset_name, (df, voltage_col)) in enumerate(zip(dataset_names, datasets_data)):
        plt.subplot(3, 4, 5 + i)
        
        # Sample some field values for visualization
        unique_fields = sorted(df['y_field'].unique())
        sample_fields = unique_fields[::max(1, len(unique_fields)//5)][:5]
        
        for j, field in enumerate(sample_fields):
            field_data = df[df['y_field'] == field]
            plt.plot(field_data['appl_current'] * 1e6, field_data[voltage_col] * 1e6,
                    'o-', alpha=0.7, markersize=1, linewidth=0.5,
                    label=f'B={field:.3f}T' if j < 3 else "")
        
        plt.xlabel('Applied Current (µA)')
        plt.ylabel('Measured Voltage (µV)')
        plt.title(f'I-V Characteristics - {dataset_name}')
        if i == 0:
            plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
    
    # 8. Voltage Distribution Comparison
    plt.subplot(3, 4, 8)
    for i, (dataset_name, (df, voltage_col)) in enumerate(zip(dataset_names, datasets_data)):
        plt.hist(df[voltage_col] * 1e6, bins=50, alpha=0.5, label=dataset_name, 
                color=colors[i], density=True)
    plt.xlabel('Voltage (µV)')
    plt.ylabel('Density')
    plt.title('Voltage Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. Current Distribution Comparison
    plt.subplot(3, 4, 9)
    for i, (dataset_name, (df, voltage_col)) in enumerate(zip(dataset_names, datasets_data)):
        plt.hist(df['appl_current'] * 1e6, bins=50, alpha=0.5, label=dataset_name,
                color=colors[i], density=True)
    plt.xlabel('Current (µA)')
    plt.ylabel('Density')
    plt.title('Current Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 10. Field Distribution Comparison
    plt.subplot(3, 4, 10)
    for i, (dataset_name, (df, voltage_col)) in enumerate(zip(dataset_names, datasets_data)):
        plt.hist(df['y_field'], bins=30, alpha=0.5, label=dataset_name,
                color=colors[i], density=True)
    plt.xlabel('Magnetic Field (T)')
    plt.ylabel('Density')
    plt.title('Field Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 11. Data Quality Metrics
    plt.subplot(3, 4, 11)
    metrics = ['Field Steps', 'Current Steps']
    dataset_metrics = []
    for info in datasets_info:
        dataset_metrics.append([info['field_steps'], info['current_steps']])
    
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, (dataset_name, metric_vals) in enumerate(zip(dataset_names, dataset_metrics)):
        plt.bar(x + i*width, metric_vals, width, label=dataset_name, color=colors[i], alpha=0.7)
    
    plt.xlabel('Metrics')
    plt.ylabel('Count')
    plt.title('Measurement Resolution Comparison')
    plt.xticks(x + width, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 12. Summary Statistics Table
    plt.subplot(3, 4, 12)
    plt.axis('off')
    
    # Create summary table
    table_data = []
    for info in datasets_info:
        row = [
            info['dataset'],
            f"{info['n_samples']:,}",
            f"{info['field_range']:.4f}",
            f"{info['current_range']*1e6:.1f}",
            f"{info['critical_current_mean']*1e6:.1f}" if info['critical_current_mean'] > 0 else "N/A",
            info['voltage_col']
        ]
        table_data.append(row)
    
    col_labels = ['Dataset', 'Samples', 'Field Range\n(T)', 'Current Range\n(µA)', 
                  'Critical Current\n(µA)', 'Voltage Column']
    
    table = plt.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    plt.title('Dataset Summary Table', pad=20)
    
    plt.tight_layout()
    plt.savefig('comparative_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Comparison visualization saved as 'comparative_analysis.png'")

def generate_comparison_report(datasets_info):
    """Generate a comparative analysis report"""
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS REPORT - SUPERCONDUCTOR DATASETS")
    print("="*80)
    
    print(f"\n📊 DATASETS OVERVIEW:")
    for info in datasets_info:
        print(f"   • {info['dataset']}: {info['n_samples']:,} samples, using {info['voltage_col']}")
    
    print(f"\n🔬 MEASUREMENT RANGES:")
    print(f"   Dataset    | Field Range (T) | Current Range (µA) | Samples")
    print(f"   -----------|----------------|-------------------|----------")
    for info in datasets_info:
        print(f"   {info['dataset']:<10} | {info['field_range']:<14.4f} | {info['current_range']*1e6:<17.1f} | {info['n_samples']:>8,}")
    
    print(f"\n⚡ SUPERCONDUCTOR CHARACTERISTICS:")
    print(f"   Dataset    | Critical Current (µA) | Voltage Sensor")
    print(f"   -----------|----------------------|----------------")
    for info in datasets_info:
        ic_str = f"{info['critical_current_mean']*1e6:.1f}" if info['critical_current_mean'] > 0 else "N/A"
        print(f"   {info['dataset']:<10} | {ic_str:<20} | {info['voltage_col']}")
    
    print(f"\n📈 MEASUREMENT RESOLUTION:")
    print(f"   Dataset    | Field Steps | Current Steps | Resolution Ratio")
    print(f"   -----------|-------------|---------------|------------------")
    for info in datasets_info:
        resolution_ratio = info['field_steps'] * info['current_steps'] / info['n_samples']
        print(f"   {info['dataset']:<10} | {info['field_steps']:<11} | {info['current_steps']:<13} | {resolution_ratio:<16.3f}")
    
    print(f"\n🔍 DATA QUALITY INDICATORS:")
    for info in datasets_info:
        voltage_snr = abs(info['voltage_mean'] / info['voltage_std']) if info['voltage_std'] > 0 else float('inf')
        current_snr = abs(info['current_mean'] / info['current_std']) if info['current_std'] > 0 else float('inf')
        
        print(f"   {info['dataset']}:")
        print(f"     • Voltage signal-to-noise: {voltage_snr:.1f}")
        print(f"     • Current signal-to-noise: {current_snr:.1f}")
        print(f"     • Has differential resistance: {'Yes' if info['has_diff_resistance'] else 'No'}")
        if info['has_diff_resistance']:
            print(f"     • Diff. resistance mean: {info['diff_resistance_mean']:.2e} Ω")
    
    print(f"\n🎯 KEY DIFFERENCES:")
    # Find the dataset with highest critical current
    max_ic_dataset = max(datasets_info, key=lambda x: x['critical_current_mean'])
    min_ic_dataset = min(datasets_info, key=lambda x: x['critical_current_mean'])
    
    print(f"   • Highest critical current: {max_ic_dataset['dataset']} ({max_ic_dataset['critical_current_mean']*1e6:.1f} µA)")
    print(f"   • Lowest critical current: {min_ic_dataset['dataset']} ({min_ic_dataset['critical_current_mean']*1e6:.1f} µA)")
    
    # Find dataset with most samples
    max_samples_dataset = max(datasets_info, key=lambda x: x['n_samples'])
    min_samples_dataset = min(datasets_info, key=lambda x: x['n_samples'])
    
    print(f"   • Most comprehensive: {max_samples_dataset['dataset']} ({max_samples_dataset['n_samples']:,} samples)")
    print(f"   • Most focused: {min_samples_dataset['dataset']} ({min_samples_dataset['n_samples']:,} samples)")
    
    # Voltage sensor differences
    voltage_sensors = set(info['voltage_col'] for info in datasets_info)
    if len(voltage_sensors) > 1:
        print(f"   • Different voltage sensors used: {', '.join(voltage_sensors)}")
        k1_datasets = [info['dataset'] for info in datasets_info if 'K1' in info['voltage_col']]
        k2_datasets = [info['dataset'] for info in datasets_info if 'K2' in info['voltage_col']]
        if k1_datasets:
            print(f"     - K1 sensor: {', '.join(k1_datasets)}")
        if k2_datasets:
            print(f"     - K2 sensor: {', '.join(k2_datasets)}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    
    # Check for measurement consistency
    field_ranges = [info['field_range'] for info in datasets_info]
    if max(field_ranges) / min(field_ranges) > 2:
        print(f"   • Field ranges vary significantly - consider normalizing for comparison")
    
    current_ranges = [info['current_range'] for info in datasets_info]
    if max(current_ranges) / min(current_ranges) > 2:
        print(f"   • Current ranges vary significantly - different experimental conditions")
    
    if len(voltage_sensors) > 1:
        print(f"   • Multiple voltage sensors used - calibration may be needed for direct comparison")
    
    # Sample size recommendations
    total_samples = sum(info['n_samples'] for info in datasets_info)
    print(f"   • Total samples available: {total_samples:,}")
    print(f"   • Consider combining datasets for enhanced machine learning analysis")
    
    print("\n" + "="*80)

def main():
    """Main comparative analysis pipeline"""
    filenames = ["164.csv", "500.csv", "317.csv"]
    
    datasets_info = []
    datasets_data = []
    
    print("Starting comparative analysis of superconductor datasets...")
    
    for filename in filenames:
        try:
            df, voltage_col = load_dataset(filename)
            dataset_name = filename.replace('.csv', '')
            
            # Extract features
            features = extract_key_features(df, voltage_col, dataset_name)
            datasets_info.append(features)
            datasets_data.append((df, voltage_col))
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    if len(datasets_info) >= 2:
        # Create comparison visualizations
        create_comparison_plots(datasets_info, datasets_data)
        
        # Generate comparison report
        generate_comparison_report(datasets_info)
        
        print(f"\n✅ Comparative analysis completed successfully!")
        print(f"📊 Results saved as: comparative_analysis.png")
    else:
        print("❌ Need at least 2 datasets for comparison")

if __name__ == "__main__":
    main()
