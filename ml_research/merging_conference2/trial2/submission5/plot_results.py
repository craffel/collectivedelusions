import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    print("--- Generating Results Plots ---")
    
    # Load results
    try:
        df = pd.read_csv('results_summary.txt', keep_default_na=False)
    except Exception as e:
        print("Could not read results_summary.txt:", e)
        return
        
    print(df)
    
    # Filter for Task Arithmetic
    ta_df = df[df['Method'] == 'Task Arithmetic'].copy()
    if ta_df.empty:
        print("No Task Arithmetic results found to plot.")
        return
        
    # Convert types
    ta_df['Scale'] = pd.to_numeric(ta_df['Scale'])
    ta_df['Average'] = pd.to_numeric(ta_df['Average'])
    
    # Group by Calibration method
    plt.figure(figsize=(9, 6))
    
    unique_cals = ta_df['Calibration'].unique()
    # Create a nice set of colors for dynamic methods
    color_cycle = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan', 'magenta']
    
    colors = {'None': 'gray', 'TCAC (Diagonal)': 'tab:orange', 'M-CAC (Multivariate)': 'tab:red'}
    markers = {'None': 'o', 'TCAC (Diagonal)': 's', 'M-CAC (Multivariate)': 'x'}
    linestyles = {'None': '--', 'TCAC (Diagonal)': ':', 'M-CAC (Multivariate)': '-.'}
    
    color_idx = 0
    for cal in unique_cals:
        if cal not in colors:
            colors[cal] = color_cycle[color_idx % len(color_cycle)]
            color_idx += 1
            if 'ODC' in cal or 'Optimal' in cal:
                markers[cal] = 'D'
                linestyles[cal] = '-'
            elif 'R-MCAC' in cal:
                markers[cal] = '^'
                linestyles[cal] = '-'
            else:
                markers[cal] = 'o'
                linestyles[cal] = '-'
    
    for cal, group in ta_df.groupby('Calibration'):
        # Sort by Scale
        group = group.sort_values('Scale')
        plt.plot(group['Scale'], group['Average'], label=cal, 
                 color=colors.get(cal, 'black'), 
                 marker=markers.get(cal, 'o'), 
                 linestyle=linestyles.get(cal, '-'), 
                 linewidth=2, markersize=8)
                 
    # Plot Expert Upper Bound
    expert_row = df[df['Method'] == 'Individual Experts']
    if not expert_row.empty:
        expert_avg = float(expert_row['Average'].values[0])
        plt.axhline(y=expert_avg, color='tab:red', linestyle='-.', label='Individual Expert Average', linewidth=1.5)
        
    plt.xlabel('Task Vector Scaling Factor ($\\lambda$)', fontsize=12)
    plt.ylabel('Average Multi-Task Accuracy (%)', fontsize=12)
    plt.title('Task Arithmetic Merging Accuracy with Activation Calibration', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('calibration_comparison.png', dpi=300)
    print("Plot successfully saved as calibration_comparison.png.")
    
    # Let's also plot the Covariance Distortion Metric
    plt.figure(figsize=(8, 5))
    
    # Filter for rows that have valid Distortion metrics (not -1.0 or '-')
    dist_df = ta_df[ta_df['DistAfter'] != -1.0].copy()
    dist_df['DistAfter'] = pd.to_numeric(dist_df['DistAfter'], errors='coerce')
    
    for cal, group in dist_df.groupby('Calibration'):
        group = group.sort_values('Scale')
        plt.plot(group['Scale'], group['DistAfter'], label=f"{cal} (After)", 
                 color=colors.get(cal, 'black'), 
                 marker=markers.get(cal, 'o'), 
                 linestyle=linestyles.get(cal, '-'), 
                 linewidth=2, markersize=8)
                 
    plt.xlabel('Task Vector Scaling Factor ($\\lambda$)', fontsize=12)
    plt.ylabel('Relative Covariance Distortion (Frobenius)', fontsize=12)
    plt.title('Representational Misalignment (Covariance Distortion) After Calibration', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    
    plt.savefig('covariance_distortion.png', dpi=300)
    print("Plot successfully saved as covariance_distortion.png.")

if __name__ == '__main__':
    main()
