#!/usr/bin/env python3
"""
Enhanced Feature Extraction and Analysis for Superconductor Data (317.csv)
Adapted for different column structures with English matplotlib labels
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try to import UMAP, but continue without it if not available
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    print("UMAP not available. Some visualizations will be skipped.")
    UMAP_AVAILABLE = False

def load_and_examine_data(filename):
    """Load data and examine its structure"""
    print(f"Loading and examining {filename}...")
    df = pd.read_csv(filename)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Basic statistics:\n{df.describe()}")
    
    # Check for missing values
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Automatically detect voltage column
    voltage_col = None
    for col in ['meas_voltage_K2', 'meas_voltage_K1', 'meas_voltage']:
        if col in df.columns:
            voltage_col = col
            break
    
    if voltage_col is None:
        raise ValueError("No voltage column found!")
    
    print(f"Using voltage column: {voltage_col}")
    
    return df, voltage_col

def clean_data(df, voltage_col):
    """Clean and preprocess the data"""
    print("Cleaning data...")
    
    # Remove infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Remove rows with NaN values
    initial_shape = df.shape[0]
    df = df.dropna()
    final_shape = df.shape[0]
    
    print(f"Removed {initial_shape - final_shape} rows with missing/infinite values")
    
    # Remove extreme outliers (beyond 5 standard deviations)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        z_scores = np.abs(stats.zscore(df[col]))
        df = df[z_scores < 5]
    
    print(f"Final dataset shape after cleaning: {df.shape}")
    return df

def extract_features(df, voltage_col):
    """Extract comprehensive features from the data"""
    print("Extracting features...")
    
    features = {}
    
    # Basic statistical features
    for col in df.columns:
        if col != 'y_field':  # Exclude field as it's the independent variable
            features[f'{col}_mean'] = df[col].mean()
            features[f'{col}_std'] = df[col].std()
            features[f'{col}_min'] = df[col].min()
            features[f'{col}_max'] = df[col].max()
            features[f'{col}_range'] = df[col].max() - df[col].min()
            features[f'{col}_skew'] = df[col].skew()
            features[f'{col}_kurtosis'] = df[col].kurtosis()
    
    # Enhanced critical current calculation using dV/dI analysis
    try:
        # Group by y_field and find critical current for each field value
        field_groups = df.groupby('y_field')
        critical_currents = []
        ic_positive_values = []
        ic_negative_values = []
        normal_resistances = []
        n_values = []
        transition_widths = []
        
        for field, group in field_groups:
            # Sort by current and analyze dV/dI characteristics
            group_sorted = group.sort_values('appl_current')
            voltage_values = group_sorted[voltage_col].values
            current_values = group_sorted['appl_current'].values
            
            # Use dV/dI if available for better critical current determination
            if 'dV_dI' in group_sorted.columns:
                dV_dI_values = group_sorted['dV_dI'].values
                
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
                
                # Negative critical current (from dV/dI peak)
                if np.any(negative_mask):
                    dV_dI_neg = dV_dI_values[negative_mask]
                    current_neg = current_values[negative_mask]
                    if len(dV_dI_neg) > 0:
                        max_idx = np.argmax(dV_dI_neg)
                        field_ic_negative = abs(current_neg[max_idx])
                        ic_negative_values.append(field_ic_negative)
                
                # Calculate average critical current for this field only
                field_ic_values = []
                if field_ic_positive is not None:
                    field_ic_values.append(field_ic_positive)
                if field_ic_negative is not None:
                    field_ic_values.append(field_ic_negative)
                
                if field_ic_values:
                    critical_currents.append(np.mean(field_ic_values))
                
                # Calculate normal resistance (high current region)
                high_current_mask = np.abs(current_values) > 0.8 * np.max(np.abs(current_values))
                if np.any(high_current_mask):
                    V_high = voltage_values[high_current_mask]
                    I_high = current_values[high_current_mask]
                    if len(V_high) > 1:
                        try:
                            slope, _ = np.polyfit(I_high, V_high, 1)
                            normal_resistances.append(slope)
                        except Exception:
                            pass
                
                # Calculate n-value (transition sharpness)
                try:
                    n_val = calculate_n_value(current_values, voltage_values)
                    if not np.isnan(n_val):
                        n_values.append(n_val)
                except Exception:
                    pass
                
                # Calculate transition width (FWHM of dV/dI)
                try:
                    max_dV_dI = np.max(dV_dI_values)
                    half_max = max_dV_dI / 2
                    indices = np.where(dV_dI_values >= half_max)[0]
                    if len(indices) > 1:
                        width = abs(current_values[indices[-1]] - current_values[indices[0]])
                        transition_widths.append(width)
                except Exception:
                    pass
            
            else:
                # Fallback to voltage zero crossing method
                zero_crossings = np.where(np.diff(np.sign(voltage_values)))[0]
                if len(zero_crossings) > 0:
                    idx = zero_crossings[0]
                    if idx < len(current_values) - 1:
                        x1, x2 = current_values[idx], current_values[idx + 1]
                        y1, y2 = voltage_values[idx], voltage_values[idx + 1]
                        if y2 != y1:
                            critical_current = x1 - y1 * (x2 - x1) / (y2 - y1)
                            critical_currents.append(abs(critical_current))
        
        # Store enhanced critical current features
        if critical_currents:
            features['critical_current_mean'] = np.mean(critical_currents)
            features['critical_current_std'] = np.std(critical_currents)
            features['critical_current_max'] = np.max(critical_currents)
            features['critical_current_min'] = np.min(critical_currents)
        
        if ic_positive_values:
            features['critical_current_positive_mean'] = np.mean(ic_positive_values)
            features['critical_current_positive_std'] = np.std(ic_positive_values)
        
        if ic_negative_values:
            features['critical_current_negative_mean'] = np.mean(ic_negative_values)
            features['critical_current_negative_std'] = np.std(ic_negative_values)
        
        if normal_resistances:
            features['normal_resistance_mean'] = np.mean(normal_resistances)
            features['normal_resistance_std'] = np.std(normal_resistances)
        
        if n_values:
            features['n_value_mean'] = np.mean(n_values)
            features['n_value_std'] = np.std(n_values)
        
        if transition_widths:
            features['transition_width_mean'] = np.mean(transition_widths)
            features['transition_width_std'] = np.std(transition_widths)
            
    except Exception as e:
        print(f"Could not calculate critical current: {e}")
    
    # Field range analysis
    features['field_range'] = df['y_field'].max() - df['y_field'].min()
    features['field_steps'] = len(df['y_field'].unique())
    
    # Current range analysis
    features['current_range'] = df['appl_current'].max() - df['appl_current'].min()
    features['current_steps'] = len(df['appl_current'].unique())
    
    # Resistance features (V/I)
    df_nonzero_current = df[df['appl_current'] != 0]
    if len(df_nonzero_current) > 0:
        resistance = df_nonzero_current[voltage_col] / df_nonzero_current['appl_current']
        features['resistance_mean'] = resistance.mean()
        features['resistance_std'] = resistance.std()
        features['resistance_min'] = resistance.min()
        features['resistance_max'] = resistance.max()
    
    # Differential resistance features (if dV_dI column exists)
    if 'dV_dI' in df.columns:
        features['diff_resistance_mean'] = df['dV_dI'].mean()
        features['diff_resistance_std'] = df['dV_dI'].std()
        features['diff_resistance_transition_points'] = len(df[abs(df['dV_dI'] - df['dV_dI'].mean()) > 2 * df['dV_dI'].std()])
    
    return features

def advanced_analysis(df, voltage_col):
    """Perform advanced analysis including ML techniques"""
    print("Performing advanced analysis...")
    
    # Prepare data for ML
    feature_columns = ['y_field', 'appl_current', voltage_col]
    if 'dV_dI' in df.columns:
        feature_columns.append('dV_dI')
    
    X = df[feature_columns].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    # PCA Analysis
    pca = PCA()
    pca.fit_transform(X_scaled)
    
    # Store PCA results
    results['pca_explained_variance'] = pca.explained_variance_ratio_
    results['pca_cumulative_variance'] = np.cumsum(pca.explained_variance_ratio_)
    results['pca_components'] = pca.components_
    
    # Clustering Analysis
    # K-means clustering
    silhouette_scores = []
    k_range = range(2, min(8, len(X_scaled) // 10))
    
    for k in k_range:
        if len(X_scaled) > k:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(score)
    
    if silhouette_scores:
        optimal_k = k_range[np.argmax(silhouette_scores)]
        results['optimal_clusters'] = optimal_k
        results['max_silhouette_score'] = max(silhouette_scores)
        
        # Final clustering with optimal k
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
        cluster_labels = kmeans_final.fit_predict(X_scaled)
        results['cluster_labels'] = cluster_labels
    
    # Anomaly Detection
    isolation_forest = IsolationForest(contamination=0.1, random_state=42)
    anomaly_labels = isolation_forest.fit_predict(X_scaled)
    results['anomaly_labels'] = anomaly_labels
    results['anomaly_ratio'] = np.sum(anomaly_labels == -1) / len(anomaly_labels)
    
    # UMAP Dimensionality Reduction (if available)
    if UMAP_AVAILABLE:
        umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        X_umap = umap_reducer.fit_transform(X_scaled)
        results['umap_embedding'] = X_umap
    
    return results, X_scaled, feature_columns

def create_visualizations(df, voltage_col, features, ml_results, X_scaled, feature_columns, output_filename):
    """Create comprehensive visualizations"""
    print("Creating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a large figure with multiple subplots
    plt.figure(figsize=(20, 24))
    
    # 1. Data Distribution Overview
    plt.subplot(4, 3, 1)
    for i, col in enumerate(feature_columns):
        plt.subplot(4, 3, i+1)
        plt.hist(df[col] if col in df.columns else X_scaled[:, i], bins=50, alpha=0.7, edgecolor='black')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
    
    # 2. Correlation Matrix
    plt.subplot(4, 3, 4)
    corr_data = df[feature_columns] if all(col in df.columns for col in feature_columns) else pd.DataFrame(X_scaled, columns=feature_columns)
    correlation_matrix = corr_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    
    # 3. PCA Explained Variance
    plt.subplot(4, 3, 5)
    plt.plot(range(1, len(ml_results['pca_explained_variance']) + 1), 
             ml_results['pca_explained_variance'], 'bo-', label='Individual')
    plt.plot(range(1, len(ml_results['pca_cumulative_variance']) + 1), 
             ml_results['pca_cumulative_variance'], 'ro-', label='Cumulative')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. I-V Characteristics
    plt.subplot(4, 3, 6)
    # Sample some data points for clearer visualization
    sample_fields = df['y_field'].unique()[::max(1, len(df['y_field'].unique())//10)]
    for field in sample_fields[:5]:  # Show only first 5 field values
        field_data = df[df['y_field'] == field]
        plt.plot(field_data['appl_current'] * 1e6, field_data[voltage_col] * 1e6, 
                'o-', alpha=0.7, markersize=2, label=f'Field: {field:.3f} T')
    plt.xlabel('Applied Current (µA)')
    plt.ylabel('Measured Voltage (µV)')
    plt.title('I-V Characteristics')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 5. Clustering Results
    if 'cluster_labels' in ml_results:
        plt.subplot(4, 3, 7)
        scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                            c=ml_results['cluster_labels'], cmap='viridis', alpha=0.6)
        plt.xlabel(feature_columns[0])
        plt.ylabel(feature_columns[1])
        plt.title(f'K-means Clustering (k={ml_results.get("optimal_clusters", "N/A")})')
        plt.colorbar(scatter)
        plt.grid(True, alpha=0.3)
    
    # 6. Anomaly Detection
    plt.subplot(4, 3, 8)
    colors = ['red' if x == -1 else 'blue' for x in ml_results['anomaly_labels']]
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=colors, alpha=0.6)
    plt.xlabel(feature_columns[0])
    plt.ylabel(feature_columns[1])
    plt.title(f'Anomaly Detection ({ml_results["anomaly_ratio"]:.1%} anomalies)')
    plt.grid(True, alpha=0.3)
    
    # 7. UMAP Visualization (if available)
    if 'umap_embedding' in ml_results:
        plt.subplot(4, 3, 9)
        plt.scatter(ml_results['umap_embedding'][:, 0], 
                   ml_results['umap_embedding'][:, 1], 
                   alpha=0.6, s=1)
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.title('UMAP Dimensionality Reduction')
        plt.grid(True, alpha=0.3)
    
    # 8. Critical Current vs Field (if available)
    if 'critical_current_mean' in features:
        plt.subplot(4, 3, 10)
        field_groups = df.groupby('y_field')
        critical_currents = []
        field_values = []
        
        for field, group in field_groups:
            group_sorted = group.sort_values('appl_current')
            voltage_values = group_sorted[voltage_col].values
            current_values = group_sorted['appl_current'].values
            
            zero_crossings = np.where(np.diff(np.sign(voltage_values)))[0]
            if len(zero_crossings) > 0:
                idx = zero_crossings[0]
                if idx < len(current_values) - 1:
                    x1, x2 = current_values[idx], current_values[idx + 1]
                    y1, y2 = voltage_values[idx], voltage_values[idx + 1]
                    if y2 != y1:
                        critical_current = x1 - y1 * (x2 - x1) / (y2 - y1)
                        critical_currents.append(abs(critical_current) * 1e6)  # Convert to µA
                        field_values.append(field)
        
        if critical_currents:
            plt.plot(field_values, critical_currents, 'bo-', markersize=4)
            plt.xlabel('Magnetic Field (T)')
            plt.ylabel('Critical Current (µA)')
            plt.title('Critical Current vs Magnetic Field')
            plt.grid(True, alpha=0.3)
    
    # 9. Differential Resistance (if available)
    if 'dV_dI' in df.columns:
        plt.subplot(4, 3, 11)
        plt.scatter(df['appl_current'] * 1e6, df['dV_dI'], alpha=0.5, s=1)
        plt.xlabel('Applied Current (µA)')
        plt.ylabel('dV/dI (Ω)')
        plt.title('Differential Resistance')
        plt.grid(True, alpha=0.3)
    
    # 10. Feature Importance (PCA Components)
    plt.subplot(4, 3, 12)
    components_to_show = min(3, len(ml_results['pca_components']))
    x_pos = np.arange(len(feature_columns))
    width = 0.25
    
    for i in range(components_to_show):
        plt.bar(x_pos + i*width, ml_results['pca_components'][i], 
                width, label=f'PC{i+1}', alpha=0.8)
    
    plt.xlabel('Features')
    plt.ylabel('Component Weight')
    plt.title('PCA Component Weights')
    plt.xticks(x_pos + width, feature_columns, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved as {output_filename}")

def generate_report(filename, features, ml_results):
    """Generate a comprehensive analysis report"""
    print("\n" + "="*80)
    print(f"COMPREHENSIVE ANALYSIS REPORT - {filename}")
    print("="*80)
    
    print("\n📊 DATASET SUMMARY:")
    print(f"   • File: {filename}")
    print(f"   • Features extracted: {len(features)}")
    
    print("\n🔬 KEY MEASUREMENTS:")
    voltage_cols = [k for k in features.keys() if 'meas_voltage' in k and '_mean' in k]
    if voltage_cols:
        voltage_key = voltage_cols[0]
        print(f"   • Average voltage: {features[voltage_key]:.2e} V")
        voltage_std_key = voltage_key.replace('_mean', '_std')
        if voltage_std_key in features:
            print(f"   • Voltage std deviation: {features[voltage_std_key]:.2e} V")
    
    current_mean_key = 'appl_current_mean'
    if current_mean_key in features:
        print(f"   • Average applied current: {features[current_mean_key]*1e6:.2f} µA")
        current_range_key = 'appl_current_range'
        if current_range_key in features:
            print(f"   • Current range: {features[current_range_key]*1e6:.2f} µA")
    
    if 'field_range' in features:
        print(f"   • Magnetic field range: {features['field_range']:.4f} T")
        print(f"   • Number of field steps: {features['field_steps']}")
    
    print("\n⚡ ENHANCED SUPERCONDUCTOR CHARACTERISTICS:")
    
    # Enhanced critical current analysis
    if 'critical_current_mean' in features:
        print("   📈 Critical Current Analysis:")
        print(f"     • Average critical current: {features['critical_current_mean']*1e6:.3f} µA")
        if 'critical_current_std' in features:
            print(f"     • Critical current std dev: {features['critical_current_std']*1e6:.3f} µA")
        if 'critical_current_max' in features:
            print(f"     • Maximum critical current: {features['critical_current_max']*1e6:.3f} µA")
        if 'critical_current_min' in features:
            print(f"     • Minimum critical current: {features['critical_current_min']*1e6:.3f} µA")
    
    # Positive and negative critical currents
    if 'critical_current_positive_mean' in features:
        print(f"     • Positive Ic (average): {features['critical_current_positive_mean']*1e6:.3f} µA")
        if 'critical_current_positive_std' in features:
            print(f"     • Positive Ic (std dev): {features['critical_current_positive_std']*1e6:.3f} µA")
    
    if 'critical_current_negative_mean' in features:
        print(f"     • Negative Ic (average): {features['critical_current_negative_mean']*1e6:.3f} µA")
        if 'critical_current_negative_std' in features:
            print(f"     • Negative Ic (std dev): {features['critical_current_negative_std']*1e6:.3f} µA")
    
    # Normal resistance analysis
    if 'normal_resistance_mean' in features:
        print("   🔧 Normal Resistance Analysis:")
        print(f"     • Average normal resistance: {features['normal_resistance_mean']:.3e} Ω")
        if 'normal_resistance_std' in features:
            print(f"     • Normal resistance std dev: {features['normal_resistance_std']:.3e} Ω")
    
    # n-value analysis  
    if 'n_value_mean' in features:
        print("   📊 Transition Sharpness (n-value):")
        print(f"     • Average n-value: {features['n_value_mean']:.3f}")
        if 'n_value_std' in features:
            print(f"     • n-value std dev: {features['n_value_std']:.3f}")
        
        # Interpret n-value
        avg_n = features['n_value_mean']
        if avg_n > 20:
            quality = "Excellent (Sharp transition)"
        elif avg_n > 10:
            quality = "Good (Moderately sharp)"
        elif avg_n > 5:
            quality = "Fair (Broad transition)"
        else:
            quality = "Poor (Very broad transition)"
        print(f"     • Transition quality: {quality}")
    
    # Transition width analysis
    if 'transition_width_mean' in features:
        print("   📏 Transition Width Analysis:")
        print(f"     • Average transition width: {features['transition_width_mean']*1e6:.3f} µA")
        if 'transition_width_std' in features:
            print(f"     • Transition width std dev: {features['transition_width_std']*1e6:.3f} µA")
    
    # Legacy resistance features
    resistance_keys = [k for k in features.keys() if 'resistance_mean' in k and 'normal' not in k]
    if resistance_keys:
        resistance_key = resistance_keys[0]
        print(f"   • Average resistance: {features[resistance_key]:.2e} Ω")
    
    if 'diff_resistance_mean' in features:
        print(f"   • Average differential resistance: {features['diff_resistance_mean']:.2e} Ω")
        if 'diff_resistance_transition_points' in features:
            print(f"   • Transition points detected: {features['diff_resistance_transition_points']}")
    
    print("\n🤖 MACHINE LEARNING INSIGHTS:")
    if 'pca_explained_variance' in ml_results:
        print(f"   • First PC explains: {ml_results['pca_explained_variance'][0]:.1%} of variance")
        print(f"   • First 2 PCs explain: {ml_results['pca_cumulative_variance'][1]:.1%} of variance")
    
    if 'optimal_clusters' in ml_results:
        print(f"   • Optimal number of clusters: {ml_results['optimal_clusters']}")
        print(f"   • Clustering quality (silhouette): {ml_results['max_silhouette_score']:.3f}")
    
    if 'anomaly_ratio' in ml_results:
        print(f"   • Anomalous data points: {ml_results['anomaly_ratio']:.1%}")
    
    print("\n📈 STATISTICAL SUMMARY:")
    print(f"   • Total features analyzed: {len(features)}")
    
    # Show top 5 most variable features
    variance_features = {k: v for k, v in features.items() if '_std' in k}
    if variance_features:
        sorted_variance = sorted(variance_features.items(), key=lambda x: abs(x[1]), reverse=True)
        print("   • Most variable measurements:")
        for i, (feat, val) in enumerate(sorted_variance[:5]):
            clean_name = feat.replace('_std', '').replace('_', ' ')
            print(f"     {i+1}. {clean_name}: {val:.2e}")
    
    print("\n💡 PHYSICAL INTERPRETATION:")
    
    # Critical current uniformity
    if 'critical_current_mean' in features and 'critical_current_std' in features:
        ic_cv = features['critical_current_std'] / features['critical_current_mean'] if features['critical_current_mean'] != 0 else float('inf')
        print(f"   • Critical current uniformity (CV): {ic_cv:.3f}")
        if ic_cv < 0.1:
            uniformity = "Excellent uniformity"
        elif ic_cv < 0.2:
            uniformity = "Good uniformity"
        elif ic_cv < 0.5:
            uniformity = "Moderate uniformity"
        else:
            uniformity = "Poor uniformity"
        print(f"     → {uniformity}")
    
    # Asymmetry analysis
    if 'critical_current_positive_mean' in features and 'critical_current_negative_mean' in features:
        ic_pos = features['critical_current_positive_mean']
        ic_neg = features['critical_current_negative_mean']
        asymmetry = abs(ic_pos - ic_neg) / max(ic_pos, ic_neg) if max(ic_pos, ic_neg) != 0 else 0
        print(f"   • Critical current asymmetry: {asymmetry:.3f}")
        if asymmetry < 0.05:
            symmetry = "Highly symmetric"
        elif asymmetry < 0.1:
            symmetry = "Moderately symmetric"
        else:
            symmetry = "Asymmetric"
        print(f"     → {symmetry}")
    
    print("\n🎯 DEVICE QUALITY ASSESSMENT:")
    
    # Overall quality score
    quality_factors = []
    
    if 'n_value_mean' in features:
        n_score = min(features['n_value_mean'] / 20, 1.0)  # Normalize to 1.0
        quality_factors.append(('Transition sharpness', n_score))
    
    if 'critical_current_mean' in features and 'critical_current_std' in features:
        ic_uniformity = 1.0 - min(ic_cv, 1.0)  # Higher uniformity = higher score
        quality_factors.append(('Current uniformity', ic_uniformity))
    
    if 'critical_current_positive_mean' in features and 'critical_current_negative_mean' in features:
        symmetry_score = 1.0 - min(asymmetry * 10, 1.0)  # Higher symmetry = higher score
        quality_factors.append(('Current symmetry', symmetry_score))
    
    if quality_factors:
        overall_score = np.mean([score for _, score in quality_factors])
        print(f"   • Overall device quality score: {overall_score:.3f}/1.0")
        
        for factor_name, score in quality_factors:
            print(f"     - {factor_name}: {score:.3f}")
        
        if overall_score > 0.8:
            grade = "A (Excellent)"
        elif overall_score > 0.6:
            grade = "B (Good)"
        elif overall_score > 0.4:
            grade = "C (Fair)"
        else:
            grade = "D (Poor)"
        print(f"   • Device grade: {grade}")
    
    print("\n" + "="*80)

def calculate_n_value(current, voltage):
    """Calculate n-value using 10%-90% criterion"""
    try:
        # Use 10%-90% criterion
        max_voltage = np.max(np.abs(voltage))
        v10 = 0.1 * max_voltage
        v90 = 0.9 * max_voltage
        
        # Find corresponding current values
        mask = (np.abs(voltage) >= v10) & (np.abs(voltage) <= v90)
        if np.sum(mask) > 1:
            V_range = voltage[mask]
            I_range = current[mask]
            
            # Avoid log(0)
            V_range = V_range[V_range > 0]
            I_range = I_range[:len(V_range)]
            
            if len(V_range) > 1:
                # n = d(ln V) / d(ln I)
                log_V = np.log(V_range)
                log_I = np.log(np.abs(I_range) + 1e-12)
                slope, _ = np.polyfit(log_I, log_V, 1)
                return slope
    except Exception:
        pass
    return np.nan

def calculate_skewness(data):
    """Calculate skewness"""
    try:
        data = data[~np.isnan(data)]
        if len(data) > 2:
            mean_val = np.mean(data)
            std_val = np.std(data)
            if std_val > 0:
                return np.mean(((data - mean_val) / std_val) ** 3)
    except Exception:
        pass
    return np.nan

def calculate_kurtosis(data):
    """Calculate kurtosis"""
    try:
        data = data[~np.isnan(data)]
        if len(data) > 3:
            mean_val = np.mean(data)
            std_val = np.std(data)
            if std_val > 0:
                return np.mean(((data - mean_val) / std_val) ** 4) - 3
    except Exception:
        pass
    return np.nan

def main():
    """Main analysis pipeline"""
    filename = "317.csv"
    output_filename = "analysis_results_317.png"
    
    try:
        # Load and examine data
        df, voltage_col = load_and_examine_data(filename)
        
        # Clean data
        df_clean = clean_data(df, voltage_col)
        
        # Extract features
        features = extract_features(df_clean, voltage_col)
        
        # Perform advanced analysis
        ml_results, X_scaled, feature_columns = advanced_analysis(df_clean, voltage_col)
        
        # Create visualizations
        create_visualizations(df_clean, voltage_col, features, ml_results, X_scaled, feature_columns, output_filename)
        
        # Generate report
        generate_report(filename, features, ml_results)
        
        print("\n✅ Analysis completed successfully!")
        print(f"📊 Results saved as: {output_filename}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
