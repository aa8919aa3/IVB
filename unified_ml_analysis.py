#!/usr/bin/env python3
"""
Unified Machine Learning Analysis for Multiple Superconductor Datasets
Advanced analysis combining 164.csv, 500.csv, and 317.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Try to import UMAP
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    print("UMAP not available. Using PCA and t-SNE for dimensionality reduction.")
    UMAP_AVAILABLE = False

def load_all_datasets():
    """Load and combine all datasets with proper labeling"""
    print("Loading all datasets...")
    
    datasets = []
    filenames = ["164.csv", "500.csv", "317.csv"]
    
    for filename in filenames:
        try:
            df = pd.read_csv(filename)
            dataset_name = filename.replace('.csv', '')
            
            # Detect voltage column
            voltage_col = None
            for col in ['meas_voltage_K2', 'meas_voltage_K1', 'meas_voltage']:
                if col in df.columns:
                    voltage_col = col
                    break
            
            if voltage_col is None:
                print(f"Warning: No voltage column found in {filename}")
                continue
            
            # Standardize column names
            df_standardized = df.copy()
            df_standardized['voltage'] = df[voltage_col]
            df_standardized['dataset'] = dataset_name
            df_standardized['voltage_sensor'] = voltage_col
            
            # Clean data
            df_standardized = df_standardized.replace([np.inf, -np.inf], np.nan).dropna()
            
            datasets.append(df_standardized)
            print(f"  • {filename}: {len(df_standardized)} samples, voltage column: {voltage_col}")
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    if not datasets:
        raise ValueError("No datasets could be loaded!")
    
    # Combine all datasets
    combined_df = pd.concat(datasets, ignore_index=True)
    print(f"\nCombined dataset: {len(combined_df)} total samples from {len(datasets)} files")
    
    return combined_df, datasets

def extract_unified_features(df):
    """Extract features that work across all datasets"""
    print("Extracting unified features...")
    
    # Group by dataset for feature extraction
    feature_data = []
    
    for dataset_name in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset_name].copy()
        
        features = {
            'dataset': dataset_name,
            'n_samples': len(dataset_df),
        }
        
        # Basic statistical features
        for col in ['y_field', 'appl_current', 'voltage']:
            if col in dataset_df.columns:
                features[f'{col}_mean'] = dataset_df[col].mean()
                features[f'{col}_std'] = dataset_df[col].std()
                features[f'{col}_min'] = dataset_df[col].min()
                features[f'{col}_max'] = dataset_df[col].max()
                features[f'{col}_range'] = dataset_df[col].max() - dataset_df[col].min()
                features[f'{col}_skew'] = dataset_df[col].skew()
                features[f'{col}_kurtosis'] = dataset_df[col].kurtosis()
        
        # Calculate critical current for this dataset
        try:
            field_groups = dataset_df.groupby('y_field')
            critical_currents = []
            
            for field, group in field_groups:
                group_sorted = group.sort_values('appl_current')
                voltage_values = group_sorted['voltage'].values
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
        
        # Add voltage sensor type
        features['voltage_sensor'] = dataset_df['voltage_sensor'].iloc[0]
        
        feature_data.append(features)
    
    feature_df = pd.DataFrame(feature_data)
    return feature_df

def advanced_ml_analysis(combined_df, feature_df):
    """Perform advanced machine learning analysis"""
    print("Performing advanced ML analysis...")
    
    results = {}
    
    # Prepare data for ML (sample-level analysis)
    # Select common features across all datasets
    ml_features = ['y_field', 'appl_current', 'voltage']
    X = combined_df[ml_features].values
    y_dataset = combined_df['dataset'].values
    
    # Encode dataset labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_dataset)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA Analysis
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    results['pca_explained_variance'] = pca.explained_variance_ratio_
    results['pca_cumulative_variance'] = np.cumsum(pca.explained_variance_ratio_)
    results['X_pca'] = X_pca
    
    # t-SNE for visualization
    print("Computing t-SNE embedding...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)//4))
    X_tsne = tsne.fit_transform(X_scaled[:10000] if len(X_scaled) > 10000 else X_scaled)  # Subsample for speed
    results['X_tsne'] = X_tsne
    results['y_tsne'] = y_encoded[:len(X_tsne)]
    
    # UMAP (if available)
    if UMAP_AVAILABLE:
        print("Computing UMAP embedding...")
        umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        X_umap = umap_reducer.fit_transform(X_scaled[:10000] if len(X_scaled) > 10000 else X_scaled)
        results['X_umap'] = X_umap
        results['y_umap'] = y_encoded[:len(X_umap)]
    
    # Classification Analysis
    print("Training dataset classifier...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    
    y_pred = rf_classifier.predict(X_test)
    
    results['classification_report'] = classification_report(y_test, y_pred, 
                                                           target_names=label_encoder.classes_)
    results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
    results['feature_importance'] = rf_classifier.feature_importances_
    results['feature_names'] = ml_features
    results['label_encoder'] = label_encoder
    
    # Clustering Analysis on combined data
    print("Performing clustering analysis...")
    optimal_k = min(len(np.unique(y_encoded)) + 2, 8)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    results['cluster_labels'] = cluster_labels
    results['optimal_k'] = optimal_k
    
    # Anomaly Detection
    print("Detecting anomalies...")
    isolation_forest = IsolationForest(contamination=0.1, random_state=42)
    anomaly_labels = isolation_forest.fit_predict(X_scaled)
    
    results['anomaly_labels'] = anomaly_labels
    results['anomaly_ratio'] = np.sum(anomaly_labels == -1) / len(anomaly_labels)
    
    # Dataset-level feature analysis
    print("Analyzing dataset-level features...")
    feature_cols = [col for col in feature_df.columns if col not in ['dataset', 'voltage_sensor']]
    X_features = feature_df[feature_cols].values
    
    # PCA on dataset features
    if len(X_features) > 1:
        feature_scaler = StandardScaler()
        X_features_scaled = feature_scaler.fit_transform(X_features)
        
        feature_pca = PCA()
        X_features_pca = feature_pca.fit_transform(X_features_scaled)
        
        results['dataset_pca_explained_variance'] = feature_pca.explained_variance_ratio_
        results['X_features_pca'] = X_features_pca
        results['feature_columns'] = feature_cols
    
    return results

def create_unified_visualizations(combined_df, feature_df, ml_results):
    """Create comprehensive unified visualizations"""
    print("Creating unified visualizations...")
    
    fig = plt.figure(figsize=(24, 20))
    
    # Color scheme for datasets
    dataset_colors = {'164': 'blue', '500': 'red', '317': 'green'}
    
    # 1. Combined Data Distribution
    plt.subplot(4, 4, 1)
    for dataset in combined_df['dataset'].unique():
        data = combined_df[combined_df['dataset'] == dataset]['voltage']
        plt.hist(data * 1e6, bins=30, alpha=0.6, label=f'Dataset {dataset}', 
                color=dataset_colors.get(dataset, 'gray'), density=True)
    plt.xlabel('Voltage (µV)')
    plt.ylabel('Density')
    plt.title('Voltage Distribution Across All Datasets')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. PCA Explained Variance
    plt.subplot(4, 4, 2)
    plt.plot(range(1, len(ml_results['pca_explained_variance']) + 1), 
             ml_results['pca_explained_variance'], 'bo-', label='Individual')
    plt.plot(range(1, len(ml_results['pca_cumulative_variance']) + 1), 
             ml_results['pca_cumulative_variance'], 'ro-', label='Cumulative')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance (Combined Data)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Dataset Classification Results
    plt.subplot(4, 4, 3)
    cm = ml_results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=ml_results['label_encoder'].classes_,
                yticklabels=ml_results['label_encoder'].classes_)
    plt.title('Dataset Classification Confusion Matrix')
    plt.xlabel('Predicted Dataset')
    plt.ylabel('True Dataset')
    
    # 4. Feature Importance
    plt.subplot(4, 4, 4)
    importance = ml_results['feature_importance']
    feature_names = ml_results['feature_names']
    
    bars = plt.bar(feature_names, importance, color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.title('Feature Importance for Dataset Classification')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45)
    for bar, imp in zip(bars, importance):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{imp:.3f}', ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    # 5. PCA Visualization (First 2 components)
    plt.subplot(4, 4, 5)
    for i, dataset in enumerate(combined_df['dataset'].unique()):
        mask = combined_df['dataset'] == dataset
        if len(ml_results['X_pca']) > len(mask):
            # Handle subsampling case
            mask = mask[:len(ml_results['X_pca'])]
        plt.scatter(ml_results['X_pca'][mask, 0], ml_results['X_pca'][mask, 1], 
                   alpha=0.6, label=f'Dataset {dataset}', s=1,
                   color=dataset_colors.get(dataset, 'gray'))
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. t-SNE Visualization
    plt.subplot(4, 4, 6)
    tsne_colors = [dataset_colors.get(ml_results['label_encoder'].classes_[label], 'gray') 
                   for label in ml_results['y_tsne']]
    plt.scatter(ml_results['X_tsne'][:, 0], ml_results['X_tsne'][:, 1], 
               c=tsne_colors, alpha=0.6, s=1)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE Visualization')
    
    # Create custom legend for t-SNE
    for dataset, color in dataset_colors.items():
        if dataset in ml_results['label_encoder'].classes_:
            plt.scatter([], [], c=color, label=f'Dataset {dataset}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. UMAP Visualization (if available)
    if 'X_umap' in ml_results:
        plt.subplot(4, 4, 7)
        umap_colors = [dataset_colors.get(ml_results['label_encoder'].classes_[label], 'gray') 
                       for label in ml_results['y_umap']]
        plt.scatter(ml_results['X_umap'][:, 0], ml_results['X_umap'][:, 1], 
                   c=umap_colors, alpha=0.6, s=1)
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.title('UMAP Visualization')
        
        # Create custom legend for UMAP
        for dataset, color in dataset_colors.items():
            if dataset in ml_results['label_encoder'].classes_:
                plt.scatter([], [], c=color, label=f'Dataset {dataset}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 8. Clustering Results
    plt.subplot(4, 4, 8)
    plt.scatter(ml_results['X_pca'][:, 0], ml_results['X_pca'][:, 1], 
               c=ml_results['cluster_labels'], cmap='viridis', alpha=0.6, s=1)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title(f'K-means Clustering (k={ml_results["optimal_k"]})')
    plt.colorbar()
    plt.grid(True, alpha=0.3)
    
    # 9. Anomaly Detection
    plt.subplot(4, 4, 9)
    colors = ['red' if x == -1 else 'blue' for x in ml_results['anomaly_labels']]
    plt.scatter(ml_results['X_pca'][:, 0], ml_results['X_pca'][:, 1], 
               c=colors, alpha=0.6, s=1)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title(f'Anomaly Detection ({ml_results["anomaly_ratio"]:.1%} anomalies)')
    plt.grid(True, alpha=0.3)
    
    # 10. Dataset-level Features Comparison
    plt.subplot(4, 4, 10)
    if 'X_features_pca' in ml_results:
        for i, dataset in enumerate(feature_df['dataset']):
            plt.scatter(ml_results['X_features_pca'][i, 0], 
                       ml_results['X_features_pca'][i, 1] if ml_results['X_features_pca'].shape[1] > 1 else 0,
                       s=200, label=f'Dataset {dataset}', 
                       color=dataset_colors.get(dataset, 'gray'))
        plt.xlabel('First PC (Dataset Features)')
        plt.ylabel('Second PC (Dataset Features)')
        plt.title('Dataset-level Feature Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 11-12. Critical Current Comparison
    plt.subplot(4, 4, 11)
    datasets = feature_df['dataset'].tolist()
    critical_currents = [feature_df[feature_df['dataset'] == d]['critical_current_mean'].iloc[0] * 1e6 
                        for d in datasets]
    
    bars = plt.bar(datasets, critical_currents, 
                  color=[dataset_colors.get(d, 'gray') for d in datasets])
    plt.title('Critical Current Comparison')
    plt.ylabel('Critical Current (µA)')
    plt.xlabel('Dataset')
    for bar, ic in zip(bars, critical_currents):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(critical_currents),
                f'{ic:.1f}', ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    # 12. Sample Distribution
    plt.subplot(4, 4, 12)
    sample_counts = [feature_df[feature_df['dataset'] == d]['n_samples'].iloc[0] for d in datasets]
    
    bars = plt.bar(datasets, sample_counts, 
                  color=[dataset_colors.get(d, 'gray') for d in datasets])
    plt.title('Sample Count Distribution')
    plt.ylabel('Number of Samples')
    plt.xlabel('Dataset')
    for bar, count in zip(bars, sample_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(sample_counts),
                f'{count:,}', ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    # 13. Voltage Range Comparison
    plt.subplot(4, 4, 13)
    voltage_ranges = [feature_df[feature_df['dataset'] == d]['voltage_range'].iloc[0] * 1e6 
                     for d in datasets]
    
    bars = plt.bar(datasets, voltage_ranges, 
                  color=[dataset_colors.get(d, 'gray') for d in datasets])
    plt.title('Voltage Range Comparison')
    plt.ylabel('Voltage Range (µV)')
    plt.xlabel('Dataset')
    for bar, vrange in zip(bars, voltage_ranges):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(voltage_ranges),
                f'{vrange:.1f}', ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    # 14. Current Range Comparison
    plt.subplot(4, 4, 14)
    current_ranges = [feature_df[feature_df['dataset'] == d]['appl_current_range'].iloc[0] * 1e6 
                     for d in datasets]
    
    bars = plt.bar(datasets, current_ranges, 
                  color=[dataset_colors.get(d, 'gray') for d in datasets])
    plt.title('Current Range Comparison')
    plt.ylabel('Current Range (µA)')
    plt.xlabel('Dataset')
    for bar, crange in zip(bars, current_ranges):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(current_ranges),
                f'{crange:.1f}', ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    # 15. Field Range Comparison
    plt.subplot(4, 4, 15)
    field_ranges = [feature_df[feature_df['dataset'] == d]['y_field_range'].iloc[0] 
                   for d in datasets]
    
    bars = plt.bar(datasets, field_ranges, 
                  color=[dataset_colors.get(d, 'gray') for d in datasets])
    plt.title('Magnetic Field Range Comparison')
    plt.ylabel('Field Range (T)')
    plt.xlabel('Dataset')
    for bar, frange in zip(bars, field_ranges):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(field_ranges),
                f'{frange:.4f}', ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    # 16. Combined I-V Characteristics Sample
    plt.subplot(4, 4, 16)
    for dataset in combined_df['dataset'].unique()[:3]:  # Show first 3 datasets
        dataset_data = combined_df[combined_df['dataset'] == dataset]
        # Sample data for visualization
        sample_data = dataset_data.sample(min(1000, len(dataset_data)), random_state=42)
        plt.scatter(sample_data['appl_current'] * 1e6, sample_data['voltage'] * 1e6,
                   alpha=0.5, s=0.5, label=f'Dataset {dataset}',
                   color=dataset_colors.get(dataset, 'gray'))
    
    plt.xlabel('Applied Current (µA)')
    plt.ylabel('Measured Voltage (µV)')
    plt.title('Combined I-V Characteristics (Sample)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('unified_ml_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Unified ML visualization saved as 'unified_ml_analysis.png'")

def generate_unified_report(combined_df, feature_df, ml_results):
    """Generate comprehensive unified analysis report"""
    print("\n" + "="*90)
    print("UNIFIED MACHINE LEARNING ANALYSIS REPORT - SUPERCONDUCTOR DATASETS")
    print("="*90)
    
    print(f"\n📊 COMBINED DATASET OVERVIEW:")
    print(f"   • Total samples: {len(combined_df):,}")
    print(f"   • Number of datasets: {combined_df['dataset'].nunique()}")
    print(f"   • Datasets included: {', '.join(combined_df['dataset'].unique())}")
    
    print(f"\n🔬 DATASET BREAKDOWN:")
    for dataset in combined_df['dataset'].unique():
        count = len(combined_df[combined_df['dataset'] == dataset])
        percentage = count / len(combined_df) * 100
        voltage_sensor = combined_df[combined_df['dataset'] == dataset]['voltage_sensor'].iloc[0]
        print(f"   • Dataset {dataset}: {count:,} samples ({percentage:.1f}%) - {voltage_sensor}")
    
    print(f"\n🤖 MACHINE LEARNING RESULTS:")
    print(f"   • PCA: First component explains {ml_results['pca_explained_variance'][0]:.1%} of variance")
    print(f"   • PCA: First 3 components explain {ml_results['pca_cumulative_variance'][2]:.1%} of variance")
    print(f"   • Clustering: Optimal k = {ml_results['optimal_k']}")
    print(f"   • Anomaly detection: {ml_results['anomaly_ratio']:.1%} of samples flagged as anomalous")
    
    print(f"\n🎯 CLASSIFICATION ANALYSIS:")
    print("   Dataset Classification Performance:")
    print("   " + ml_results['classification_report'].replace('\n', '\n   '))
    
    print(f"\n📈 FEATURE IMPORTANCE RANKING:")
    feature_names = ml_results['feature_names']
    feature_importance = ml_results['feature_importance']
    sorted_features = sorted(zip(feature_names, feature_importance), 
                           key=lambda x: x[1], reverse=True)
    
    for i, (feature, importance) in enumerate(sorted_features, 1):
        print(f"   {i}. {feature}: {importance:.3f}")
    
    print(f"\n⚡ SUPERCONDUCTOR CHARACTERISTICS SUMMARY:")
    critical_currents = []
    for dataset in feature_df['dataset']:
        ic = feature_df[feature_df['dataset'] == dataset]['critical_current_mean'].iloc[0]
        critical_currents.append((dataset, ic * 1e6))
    
    critical_currents.sort(key=lambda x: x[1], reverse=True)
    
    print(f"   Critical Current Rankings:")
    for i, (dataset, ic) in enumerate(critical_currents, 1):
        print(f"   {i}. Dataset {dataset}: {ic:.1f} µA")
    
    print(f"\n🔍 DATA QUALITY ASSESSMENT:")
    
    # Voltage sensor analysis
    sensor_types = combined_df['voltage_sensor'].unique()
    print(f"   • Voltage sensors used: {', '.join(sensor_types)}")
    
    for sensor in sensor_types:
        count = len(combined_df[combined_df['voltage_sensor'] == sensor])
        datasets_using = combined_df[combined_df['voltage_sensor'] == sensor]['dataset'].unique()
        print(f"     - {sensor}: {count:,} samples from datasets {', '.join(datasets_using)}")
    
    # Data range analysis
    print(f"\n   • Combined data ranges:")
    print(f"     - Magnetic field: {combined_df['y_field'].min():.4f} to {combined_df['y_field'].max():.4f} T")
    print(f"     - Applied current: {combined_df['appl_current'].min()*1e6:.1f} to {combined_df['appl_current'].max()*1e6:.1f} µA")
    print(f"     - Measured voltage: {combined_df['voltage'].min()*1e6:.1f} to {combined_df['voltage'].max()*1e6:.1f} µV")
    
    print(f"\n💡 KEY INSIGHTS:")
    
    # Most distinguishing feature
    most_important_feature = sorted_features[0][0]
    print(f"   • Most distinguishing feature between datasets: {most_important_feature}")
    
    # Dataset clustering insights
    if len(ml_results['cluster_labels']) > 0:
        unique_clusters = len(np.unique(ml_results['cluster_labels']))
        print(f"   • Data naturally clusters into {unique_clusters} groups")
    
    # Voltage sensor impact
    if len(sensor_types) > 1:
        print(f"   • Multiple voltage sensors may introduce systematic differences")
        print(f"   • Consider sensor calibration for cross-dataset comparisons")
    
    # Sample size recommendations
    total_samples = len(combined_df)
    if total_samples > 50000:
        print(f"   • Large dataset ({total_samples:,} samples) suitable for deep learning approaches")
    elif total_samples > 10000:
        print(f"   • Medium dataset ({total_samples:,} samples) good for comprehensive ML analysis")
    else:
        print(f"   • Small dataset ({total_samples:,} samples) - focus on feature engineering")
    
    print(f"\n🚀 RECOMMENDATIONS:")
    print(f"   • Combined dataset provides {total_samples:,} samples for robust ML training")
    print(f"   • Consider ensemble methods leveraging dataset diversity")
    print(f"   • Voltage sensor differences suggest need for domain adaptation techniques")
    print(f"   • {ml_results['anomaly_ratio']:.1%} anomalous samples identified - investigate for data quality")
    
    if 'X_umap' in ml_results:
        print(f"   • UMAP visualization available for high-dimensional data exploration")
    
    print(f"   • Feature importance suggests {most_important_feature} is key for dataset identification")
    
    print("\n" + "="*90)

def main():
    """Main unified analysis pipeline"""
    try:
        # Load all datasets
        combined_df, datasets = load_all_datasets()
        
        # Extract unified features
        feature_df = extract_unified_features(combined_df)
        
        # Perform advanced ML analysis
        ml_results = advanced_ml_analysis(combined_df, feature_df)
        
        # Create visualizations
        create_unified_visualizations(combined_df, feature_df, ml_results)
        
        # Generate comprehensive report
        generate_unified_report(combined_df, feature_df, ml_results)
        
        print(f"\n✅ Unified ML analysis completed successfully!")
        print(f"📊 Results saved as: unified_ml_analysis.png")
        
    except Exception as e:
        print(f"❌ Error during unified analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
