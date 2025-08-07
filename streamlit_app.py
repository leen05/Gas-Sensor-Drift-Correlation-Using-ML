import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

st.set_page_config(layout="wide")
st.title("📊 Gas Sensor Drift & Correlation Dashboard")

# Upload dataset folder
data_folder = st.sidebar.text_input("Enter path to Dataset folder:", "./data")

@st.cache_data
def load_batches(folder):
    if not os.path.exists(folder):
        st.error(f"Folder {folder} does not exist!")
        return []
    
    try:
        data_files = sorted([
            f for f in os.listdir(folder)
            if f.endswith('.txt') or f.endswith('.csv') or f.endswith('.dat')
        ])
    except PermissionError:
        st.error(f"Permission denied accessing folder: {folder}")
        return []
    
    if not data_files:
        st.warning(f"No .txt, .csv, or .dat files found in {folder}")
        return []
    
    data_files_paths = [os.path.join(folder, file) for file in data_files]
    batches = []
    
    for path in data_files_paths:
        try:
            # Try different reading methods
            df = None
            
            # Method 1: Standard CSV with different separators and encodings
            for sep in [None, ' ', '\t', ',', ';']:
                for encoding in ['utf-8', 'latin1', 'cp1252', 'ascii']:
                    try:
                        if sep is None:
                            df = pd.read_csv(path, delim_whitespace=True, header=None, encoding=encoding)
                        else:
                            df = pd.read_csv(path, sep=sep, header=None, encoding=encoding)
                        
                        if df is not None and not df.empty:
                            break
                    except Exception as e:
                        continue
                if df is not None and not df.empty:
                    break
            
            # Method 2: Read as text lines if CSV methods fail
            if df is None or df.empty:
                try:
                    df = pd.read_csv(path, sep='\s+', header=None, encoding='utf-8', engine='python')
                except Exception as e:
                    pass
            
            if df is not None and not df.empty:
                batches.append(df)
                st.sidebar.success(f" Loaded: {os.path.basename(path)} (Shape: {df.shape})")
            else:
                st.sidebar.error(f" Failed to load: {os.path.basename(path)} - File is empty or unreadable")
                
        except Exception as e:
            st.sidebar.error(f" Failed to load {os.path.basename(path)}: {str(e)}")
    
    return batches

def transform_data_row(row):
    """Transform a single row of data"""
    gas_type = np.nan
    concentration = np.nan
    features = {}
    
    # Handle the first column (gas type and concentration)
    if len(row) > 0 and pd.notna(row.iloc[0]):
        first_val = str(row.iloc[0]).strip()
        if ';' in first_val:
            try:
                parts = first_val.split(';', 1)
                if len(parts) == 2:
                    gas_type = int(parts[0].strip())
                    concentration = float(parts[1].strip())
            except (ValueError, IndexError) as e:
                pass
        else:
            # Maybe it's just a number (gas type only)
            try:
                gas_type = int(first_val)
            except ValueError:
                pass
    
    # Handle feature columns
    for i in range(1, len(row)):
        if pd.notna(row.iloc[i]):
            val = str(row.iloc[i]).strip()
            if ':' in val:
                try:
                    parts = val.split(':', 1)
                    if len(parts) == 2:
                        idx = int(parts[0].strip())
                        feature_val = float(parts[1].strip())
                        features[f"feature_{idx}"] = feature_val
                except (ValueError, IndexError):
                    continue
            else:
                # Maybe it's just a feature value without index
                try:
                    feature_val = float(val)
                    features[f"feature_{i}"] = feature_val
                except ValueError:
                    continue
    
    result = {"gas_type": gas_type, "concentration": concentration}
    result.update(features)
    return pd.Series(result)

# Main application logic
if data_folder and os.path.exists(data_folder):
    batches = load_batches(data_folder)

    if len(batches) < 1:
        st.error(" No valid batch files found in this folder.")
        st.info("Debug: Check the file format. Expected format examples:")
        st.code("""
        Example 1 (Gas sensor data):
        1;20.0 1:0.5 2:0.3 3:0.8
        2;15.5 1:0.6 2:0.4 3:0.9
        
        Example 2 (Simple format):
        1 20.0 0.5 0.3 0.8
        2 15.5 0.6 0.4 0.9
        """)
    else:
        # Transform data with error handling
        transformed_batches = []
        for i, df in enumerate(batches):
            try:
                transformed_df = df.apply(transform_data_row, axis=1)
                
                # Check if transformation was successful
                if not transformed_df.empty:
                    # Check if we have any actual data (not all NaN)
                    non_nan_cols = transformed_df.count()
                    if non_nan_cols.sum() > 0:
                        transformed_batches.append(transformed_df)
                    else:
                        st.error(f" Batch {i+1} transformation resulted in all NaN values")
                        st.write("This might mean the data format doesn't match expected format.")
                else:
                    st.error(f" Batch {i+1} transformation resulted in empty data")
                    
            except Exception as e:
                st.error(f"Error transforming batch {i+1}: {str(e)}")
                st.write("Raw data that caused error:")
                st.dataframe(df.head(5))
        
        if not transformed_batches:
            st.error(" No batches could be successfully transformed.")
            st.info("""
            **Troubleshooting Tips:**
            1. Check if your data files have the correct format
            2. Make sure files are not empty
            3. Verify the file encoding (try UTF-8, Latin-1)
            4. Check if the data separator is correct (space, tab, comma, semicolon)
            
            **Expected data format:**
            - First column: gas_type;concentration (e.g., "1;20.0")
            - Other columns: feature_index:value (e.g., "1:0.5", "2:0.3")
            
            **Alternative format:**
            - Columns separated by spaces/tabs with numeric values
            """)
            st.stop()
        
        # Combine all batches
        try:
            combined_df = pd.concat(transformed_batches, ignore_index=True)
        except Exception as e:
            st.error(f"Error combining batches: {str(e)}")
            st.stop()

        st.success(f" Successfully loaded {len(transformed_batches)} batches with combined shape: {combined_df.shape}")

        # Sidebar slider: only if more than 1 batch
        if len(transformed_batches) > 1:
            selected_batch = st.sidebar.slider("Select Batch #", 1, len(transformed_batches), 1)
        else:
            selected_batch = 1
            st.sidebar.info("Only 1 valid batch found.")

        # Get selected batch data
        df = transformed_batches[selected_batch - 1].copy()
        feature_options = [col for col in df.columns if col.startswith('feature_')]

        # Display basic info
        st.subheader(f"🔍 Dataset Overview - Batch {selected_batch}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", df.shape[0])
        with col2:
            st.metric("Total Features", len(feature_options))
        with col3:
            if 'gas_type' in df.columns:
                st.metric("Gas Types", len(df['gas_type'].dropna().unique()))
            else:
                st.metric("Gas Types", "N/A")
        
        # Show data preview (only if user wants to see it)
        with st.expander("📋 View Sample Data"):
            if not df.empty:
                st.dataframe(df.head(10))
            else:
                st.warning("Selected batch is empty.")
                st.stop()

        # === DASHBOARD VISUALIZATIONS ===
        
        # Gas Type Distribution (Pie Chart)
        if 'gas_type' in df.columns and not df['gas_type'].isna().all():
            st.subheader("Gas Type Distribution")
            gas_counts = df['gas_type'].value_counts()
            
            if not gas_counts.empty:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig1, ax1 = plt.subplots(figsize=(10, 8))
                    colors = plt.cm.Set3(np.linspace(0, 1, len(gas_counts)))
                    wedges, texts, autotexts = ax1.pie(
                        gas_counts.values, 
                        labels=[f'Gas Type {int(x)}' for x in gas_counts.index], 
                        autopct='%1.1f%%', 
                        startangle=90,
                        colors=colors,
                        textprops={'fontsize': 12}
                    )
                    ax1.set_title('Gas Type Distribution', fontsize=16, fontweight='bold', pad=20)
                    plt.tight_layout()
                    st.pyplot(fig1)
                    plt.close()
                
                with col2:
                    st.write("### Distribution Summary")
                    for gas_type, count in gas_counts.items():
                        percentage = (count / gas_counts.sum()) * 100
                        st.metric(f"Gas Type {int(gas_type)}", f"{count} samples", f"{percentage:.1f}%")
            else:
                st.info("No gas type data available for visualization.")
        
        # Feature Analysis - All Features Overview
        if feature_options:
            st.subheader("📊 Feature Analysis Overview")
            
            # Feature statistics table (visual representation)
            feature_stats = df[feature_options].describe()
            
            # Create a cleaner statistics heatmap
            fig_stats, ax_stats = plt.subplots(figsize=(16, max(8, len(feature_options) * 0.3)))
            
            # Limit features for better visualization
            if len(feature_options) > 20:
                selected_stats_features = feature_options[:20]
                st.info(f"Showing statistics for first 20 features (out of {len(feature_options)})")
                stats_subset = feature_stats[selected_stats_features]
            else:
                stats_subset = feature_stats
                selected_stats_features = feature_options
            
            # Create heatmap with better formatting
            sns.heatmap(stats_subset.T, annot=True, fmt='.2f', cmap='YlOrRd', 
                       ax=ax_stats, cbar_kws={"shrink": .8}, linewidths=0.5)
            ax_stats.set_title('Feature Statistics Overview', fontsize=18, fontweight='bold', pad=20)
            ax_stats.set_xlabel('Statistical Measures', fontsize=14, fontweight='bold')
            ax_stats.set_ylabel('Features', fontsize=14, fontweight='bold')
            
            # Improve label formatting
            ax_stats.set_yticklabels([f.replace('feature_', 'F') for f in selected_stats_features], 
                                   rotation=0, fontsize=10)
            ax_stats.set_xticklabels(['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'], 
                                   rotation=45, ha='right', fontsize=11)
            
            plt.tight_layout()
            st.pyplot(fig_stats)
            plt.close()
            
            # Feature Distribution Plots (Grid Layout)
            st.subheader("Feature Distributions")
            
            # Calculate grid size - limit to 12 features for clean visualization
            n_features = min(len(feature_options), 12)
            if len(feature_options) > 12:
                st.info(f"Showing distribution plots for first 12 features (out of {len(feature_options)})")
            
            n_cols = 4
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig_dist, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
            
            # Handle single row case
            if n_rows == 1:
                axes = axes if n_features > 1 else [axes]
            else:
                axes = axes.flatten()
            
            # Create color palette
            colors = plt.cm.Set3(np.linspace(0, 1, n_features))
            
            for i, feature in enumerate(feature_options[:n_features]):
                valid_data = df[feature].dropna()
                if len(valid_data) > 0:
                    # Create histogram with KDE
                    sns.histplot(valid_data, kde=True, ax=axes[i], 
                               color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)
                    axes[i].set_title(f'{feature.replace("feature_", "Feature ")}', 
                                    fontsize=12, fontweight='bold', pad=10)
                    axes[i].grid(True, alpha=0.3)
                    axes[i].set_xlabel('Value', fontsize=10)
                    axes[i].set_ylabel('Frequency', fontsize=10)
                    
                    # Add statistics text
                    mean_val = valid_data.mean()
                    std_val = valid_data.std()
                    axes[i].axvline(mean_val, color='red', linestyle='--', alpha=0.8, 
                                  label=f'Mean: {mean_val:.2f}')
                    axes[i].legend(fontsize=8)
                else:
                    axes[i].text(0.5, 0.5, 'No Data Available', 
                               transform=axes[i].transAxes, ha='center', va='center', 
                               fontsize=14, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7))
                    axes[i].set_title(f'{feature.replace("feature_", "Feature ")} (No Data)', 
                                    fontsize=12, color='red')
            
            # Hide unused subplots
            for i in range(n_features, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig_dist)
            plt.close()
            
            # Feature Boxplots by Gas Type
            if 'gas_type' in df.columns and not df['gas_type'].isna().all():
                st.subheader("Feature Distributions by Gas Type")
                
                # Select features for boxplot (limit to 6 for clean visualization)
                selected_features = feature_options[:6]
                if len(feature_options) > 6:
                    st.info(f"Showing boxplots for first 6 features (out of {len(feature_options)})")
                
                if len(selected_features) > 0:
                    n_cols_box = 3
                    n_rows_box = (len(selected_features) + n_cols_box - 1) // n_cols_box
                    
                    fig_box, axes_box = plt.subplots(n_rows_box, n_cols_box, figsize=(18, 6*n_rows_box))
                    
                    # Handle single row case
                    if n_rows_box == 1:
                        axes_box = axes_box if len(selected_features) > 1 else [axes_box]
                    else:
                        axes_box = axes_box.flatten()
                    
                    # Get unique gas types for consistent coloring
                    gas_types = sorted(df['gas_type'].dropna().unique())
                    palette = sns.color_palette("Set2", len(gas_types))
                    
                    for i, feature in enumerate(selected_features):
                        plot_data = df[['gas_type', feature]].dropna()
                        if not plot_data.empty:
                            sns.boxplot(data=plot_data, x='gas_type', y=feature, 
                                      ax=axes_box[i], palette=palette)
                            axes_box[i].set_title(f'{feature.replace("feature_", "Feature ")} by Gas Type', 
                                                 fontsize=12, fontweight='bold', pad=10)
                            axes_box[i].grid(True, alpha=0.3)
                            axes_box[i].set_xlabel('Gas Type', fontsize=10, fontweight='bold')
                            axes_box[i].set_ylabel('Feature Value', fontsize=10, fontweight='bold')
                        else:
                            axes_box[i].text(0.5, 0.5, 'No Data Available', 
                                           transform=axes_box[i].transAxes, ha='center', va='center',
                                           fontsize=14, fontweight='bold',
                                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7))
                            axes_box[i].set_title(f'{feature.replace("feature_", "Feature ")} (No Data)', 
                                                fontsize=12, color='red')
                    
                    # Hide unused subplots
                    for i in range(len(selected_features), len(axes_box)):
                        axes_box[i].set_visible(False)
                    
                    plt.tight_layout()
                    st.pyplot(fig_box)
                    plt.close()
        
        # Correlation Analysis
        if len(feature_options) > 1:
            st.subheader("Feature Correlation Analysis")
            feature_data = df[feature_options].select_dtypes(include=[np.number])
            
            if not feature_data.empty and feature_data.shape[1] > 1:
                try:
                    corr = feature_data.corr()
                    
                    # Limit features for better visualization (top 20 most important)
                    if len(feature_options) > 20:
                        # Select features with highest variance for better visualization
                        feature_variance = feature_data.var().sort_values(ascending=False)
                        top_features = feature_variance.head(20).index.tolist()
                        corr_subset = corr.loc[top_features, top_features]
                        st.info(f"Showing top 20 features (out of {len(feature_options)}) based on variance for better visualization")
                    else:
                        corr_subset = corr
                        top_features = feature_options
                    
                    # Create improved correlation heatmap
                    fig_corr, ax_corr = plt.subplots(figsize=(16, 14))
                    
                    # Create a mask for the upper triangle
                    mask = np.triu(np.ones_like(corr_subset, dtype=bool))
                    
                    # Generate custom colormap
                    cmap = sns.diverging_palette(250, 10, as_cmap=True)
                    
                    # Create the heatmap with better formatting
                    sns.heatmap(corr_subset, mask=mask, cmap=cmap, center=0,
                              square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                              annot=len(top_features) <= 15,  # Only show annotations if not too many features
                              fmt='.2f', annot_kws={'size': 8}, ax=ax_corr)
                    
                    # Improve labels and title
                    ax_corr.set_title('Feature Correlation Matrix\n(Lower Triangle Only)', 
                                    fontsize=18, fontweight='bold', pad=25)
                    ax_corr.set_xlabel('Features', fontsize=14, fontweight='bold')
                    ax_corr.set_ylabel('Features', fontsize=14, fontweight='bold')
                    
                    # Rotate labels for better readability
                    ax_corr.set_xticklabels(ax_corr.get_xticklabels(), rotation=45, ha='right', fontsize=10)
                    ax_corr.set_yticklabels(ax_corr.get_yticklabels(), rotation=0, fontsize=10)
                    
                    plt.tight_layout()
                    st.pyplot(fig_corr)
                    plt.close()
                    
                    # Enhanced correlation strength analysis
                    st.subheader("🎯 Strong Correlations Analysis")
                    
                    # Find strong correlations
                    high_corr_pairs = []
                    for i in range(len(corr.columns)):
                        for j in range(i+1, len(corr.columns)):
                            corr_value = corr.iloc[i, j]
                            if abs(corr_value) > 0.7:  # Strong correlation threshold
                                high_corr_pairs.append({
                                    'Feature 1': corr.columns[i],
                                    'Feature 2': corr.columns[j],
                                    'Correlation': corr_value,
                                    'Abs_Correlation': abs(corr_value)
                                })
                    
                    if high_corr_pairs:
                        # Sort by absolute correlation value
                        corr_df = pd.DataFrame(high_corr_pairs).sort_values('Abs_Correlation', ascending=True)
                        
                        # Limit to top 15 correlations for better visualization
                        if len(corr_df) > 15:
                            corr_df = corr_df.tail(15)
                            st.info(f"Showing top 15 strongest correlations (out of {len(high_corr_pairs)} found)")
                        
                        fig_high_corr, ax_high_corr = plt.subplots(figsize=(14, max(8, len(corr_df) * 0.6)))
                        
                        # Create gradient colors based on correlation strength
                        colors = []
                        for val in corr_df['Correlation']:
                            if val < -0.8:
                                colors.append('#d62728')  # Strong negative - dark red
                            elif val < -0.7:
                                colors.append('#ff7f0e')  # Moderate negative - orange
                            elif val > 0.8:
                                colors.append('#2ca02c')  # Strong positive - dark green
                            else:
                                colors.append('#98df8a')  # Moderate positive - light green
                        
                        # Create horizontal bar chart
                        bars = ax_high_corr.barh(range(len(corr_df)), corr_df['Correlation'], 
                                               color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)
                        
                        # Customize labels with cleaner formatting
                        feature_labels = []
                        for _, row in corr_df.iterrows():
                            f1 = row['Feature 1'].replace('feature_', 'F')
                            f2 = row['Feature 2'].replace('feature_', 'F')
                            feature_labels.append(f"{f1} ↔ {f2}")
                        
                        ax_high_corr.set_yticks(range(len(corr_df)))
                        ax_high_corr.set_yticklabels(feature_labels, fontsize=11)
                        ax_high_corr.set_xlabel('Correlation Coefficient', fontsize=14, fontweight='bold')
                        ax_high_corr.set_title('Strong Feature Correlations (|r| > 0.7)', 
                                             fontsize=16, fontweight='bold', pad=20)
                        
                        # Add reference lines and grid
                        ax_high_corr.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
                        ax_high_corr.axvline(x=0.7, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Positive threshold')
                        ax_high_corr.axvline(x=-0.7, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Negative threshold')
                        ax_high_corr.grid(True, alpha=0.3, axis='x')
                        ax_high_corr.legend()
                        
                        # Add correlation values on bars
                        for i, (bar, val) in enumerate(zip(bars, corr_df['Correlation'])):
                            width = bar.get_width()
                            offset = 0.02 if width > 0 else -0.02
                            ax_high_corr.text(width + offset, bar.get_y() + bar.get_height()/2,
                                            f'{width:.3f}', 
                                            ha='left' if width > 0 else 'right', 
                                            va='center', fontweight='bold', fontsize=10,
                                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                        
                        # Set x-axis limits for better visualization
                        ax_high_corr.set_xlim(-1.1, 1.1)
                        
                        plt.tight_layout()
                        st.pyplot(fig_high_corr)
                        plt.close()
                        
                        # Add summary statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Strong Correlations", len(high_corr_pairs))
                        with col2:
                            positive_corr = len([x for x in high_corr_pairs if x['Correlation'] > 0.7])
                            st.metric("Strong Positive", positive_corr)
                        with col3:
                            negative_corr = len([x for x in high_corr_pairs if x['Correlation'] < -0.7])
                            st.metric("Strong Negative", negative_corr)
                            
                    else:
                        # Create a nice "no correlations" visualization
                        fig_no_corr, ax_no_corr = plt.subplots(figsize=(10, 6))
                        ax_no_corr.text(0.5, 0.5, 'No Strong Correlations Found\n(|r| > 0.7)', 
                                      ha='center', va='center', fontsize=20, fontweight='bold',
                                      transform=ax_no_corr.transAxes,
                                      bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
                        ax_no_corr.set_xlim(0, 1)
                        ax_no_corr.set_ylim(0, 1)
                        ax_no_corr.axis('off')
                        st.pyplot(fig_no_corr)
                        plt.close()
                        
                        st.info("This suggests that features are relatively independent, which can be good for machine learning models.")
                        
                except Exception as e:
                    st.warning(f"Could not generate correlation analysis: {str(e)}")
            else:
                st.info("Not enough numeric feature data for correlation analysis.")
        
        # Machine Learning Model Performance
        if 'concentration' in df.columns and feature_options:
            st.subheader("🤖 XGBoost Model Performance")
            
            # Prepare data for modeling
            model_data = df[['concentration'] + feature_options].dropna()
            
            if len(model_data) > 10:  # Need minimum samples for meaningful split
                X = model_data[feature_options]
                y = model_data['concentration']
                
                try:
                    # Split data
                    test_size = min(0.3, max(0.1, len(model_data) * 0.2 / len(model_data)))
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    
                    # Train model
                    model = XGBRegressor(random_state=42, n_estimators=100)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Calculate metrics
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                    # Model Performance Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # R² Scatter Plot
                        fig_r2, ax_r2 = plt.subplots(figsize=(10, 8))
                        scatter = ax_r2.scatter(y_test, y_pred, alpha=0.6, c=range(len(y_test)), 
                                              cmap='viridis', s=60, edgecolors='black', linewidth=0.5)
                        
                        # Perfect prediction line
                        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
                        ax_r2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, alpha=0.8, label='Perfect Prediction')
                        
                        ax_r2.set_xlabel("True Concentration", fontsize=12, fontweight='bold')
                        ax_r2.set_ylabel("Predicted Concentration", fontsize=12, fontweight='bold')
                        ax_r2.set_title(f'Model Predictions vs True Values\nR² = {r2:.4f}', 
                                       fontsize=14, fontweight='bold')
                        ax_r2.legend()
                        ax_r2.grid(True, alpha=0.3)
                        
                        # Add colorbar
                        cbar = plt.colorbar(scatter, ax=ax_r2)
                        cbar.set_label('Sample Index', fontsize=10)
                        
                        plt.tight_layout()
                        st.pyplot(fig_r2)
                        plt.close()

                    with col2:
                        # Residuals Analysis
                        residuals = y_test - y_pred
                        
                        fig_residuals, (ax_res1, ax_res2) = plt.subplots(2, 1, figsize=(10, 10))
                        
                        # Residuals histogram
                        sns.histplot(residuals, kde=True, ax=ax_res1, color='skyblue', alpha=0.7)
                        ax_res1.set_xlabel("Residuals", fontsize=12, fontweight='bold')
                        ax_res1.set_ylabel("Frequency", fontsize=12, fontweight='bold')
                        ax_res1.set_title(f'Residuals Distribution\nRMSE = {rmse:.4f}', 
                                         fontsize=14, fontweight='bold')
                        ax_res1.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
                        ax_res1.grid(True, alpha=0.3)
                        
                        # Residuals vs Predicted
                        ax_res2.scatter(y_pred, residuals, alpha=0.6, c='coral', s=60, 
                                      edgecolors='black', linewidth=0.5)
                        ax_res2.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
                        ax_res2.set_xlabel("Predicted Values", fontsize=12, fontweight='bold')
                        ax_res2.set_ylabel("Residuals", fontsize=12, fontweight='bold')
                        ax_res2.set_title('Residuals vs Predicted Values', fontsize=14, fontweight='bold')
                        ax_res2.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig_residuals)
                        plt.close()
                    
                    # Model Performance Metrics
                    st.subheader("📋 Model Performance Summary")
                    
                    # Calculate additional metrics
                    mae = np.mean(np.abs(residuals))
                    max_error = np.max(np.abs(residuals))
                    
                    # Create metrics visualization
                    col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
                    
                    with col_metrics1:
                        st.metric("R² Score", f"{r2:.4f}", 
                                 delta=f"{'Excellent' if r2 > 0.9 else 'Good' if r2 > 0.7 else 'Fair' if r2 > 0.5 else 'Poor'}")
                    
                    with col_metrics2:
                        st.metric("RMSE", f"{rmse:.4f}")
                    
                    with col_metrics3:
                        st.metric("Mean Abs Error", f"{mae:.4f}")
                    
                    with col_metrics4:
                        st.metric("Max Error", f"{max_error:.4f}")
                    
                    # Feature Importance
                    feature_importance = model.feature_importances_
                    importance_df = pd.DataFrame({
                        'Feature': feature_options,
                        'Importance': feature_importance
                    }).sort_values('Importance', ascending=True)
                    
                    if len(importance_df) > 0:
                        st.subheader("Feature Importance")
                        
                        fig_importance, ax_importance = plt.subplots(figsize=(12, max(6, len(importance_df) * 0.4)))
                        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(importance_df)))
                        bars = ax_importance.barh(importance_df['Feature'], importance_df['Importance'], 
                                                color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
                        
                        ax_importance.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
                        ax_importance.set_title('XGBoost Feature Importance', fontsize=14, fontweight='bold')
                        ax_importance.grid(True, alpha=0.3, axis='x')
                        
                        # Add importance values on bars
                        for i, bar in enumerate(bars):
                            width = bar.get_width()
                            ax_importance.text(width + max(importance_df['Importance']) * 0.01, 
                                             bar.get_y() + bar.get_height()/2,
                                             f'{width:.3f}', ha='left', va='center', fontweight='bold')
                        
                        plt.tight_layout()
                        st.pyplot(fig_importance)
                        plt.close()
                        
                except Exception as e:
                    st.error(f"Error in model training: {str(e)}")
            else:
                st.warning("Not enough valid data for modeling (minimum 10 samples required).")
        
        # Data Quality Overview
        st.subheader("🔍 Data Quality Overview")
        
        # Missing data analysis
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        quality_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing Percentage': missing_percent.values
        }).sort_values('Missing Percentage', ascending=False)
        
        if quality_df['Missing Percentage'].max() > 0:
            fig_quality, ax_quality = plt.subplots(figsize=(12, max(6, len(quality_df) * 0.3)))
            colors = ['red' if x > 50 else 'orange' if x > 20 else 'yellow' if x > 5 else 'green' 
                     for x in quality_df['Missing Percentage']]
            
            bars = ax_quality.barh(quality_df['Column'], quality_df['Missing Percentage'], 
                                 color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            ax_quality.set_xlabel('Missing Data Percentage (%)', fontsize=12, fontweight='bold')
            ax_quality.set_title('Data Completeness by Column', fontsize=14, fontweight='bold')
            ax_quality.grid(True, alpha=0.3, axis='x')
            
            # Add percentage values on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                if width > 0:
                    ax_quality.text(width + 1, bar.get_y() + bar.get_height()/2,
                                   f'{width:.1f}%', ha='left', va='center', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig_quality)
            plt.close()
        else:
            st.success("No missing data found in the dataset!")
            
            # Create a simple completeness visualization
            fig_complete, ax_complete = plt.subplots(figsize=(10, 6))
            ax_complete.bar(['Dataset'], [100], color='green', alpha=0.7, width=0.5)
            ax_complete.set_ylabel('Data Completeness (%)', fontsize=12, fontweight='bold')
            ax_complete.set_title('Dataset Completeness', fontsize=14, fontweight='bold')
            ax_complete.set_ylim([0, 110])
            ax_complete.text(0, 105, '100% Complete', ha='center', va='center', 
                           fontsize=16, fontweight='bold', color='green')
            plt.tight_layout()
            st.pyplot(fig_complete)
            plt.close()

else:
    st.warning("Please enter a valid dataset folder path that exists.")
    st.info(" Make sure the folder contains .txt, .csv, or .dat files with your gas sensor data.")
    
   