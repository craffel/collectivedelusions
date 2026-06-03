import pandas as pd

def main():
    print("--- Generating LaTeX Table rows from results_summary.txt ---")
    try:
        df = pd.read_csv('results_summary.txt', keep_default_na=False)
    except Exception as e:
        print("Could not read results_summary.txt:", e)
        return
        
    print(df)
    
    # Let's print each row in LaTeX format
    # We want: Method & Scale & Calibration & MNIST & F-MNIST & CIFAR10 & Average & Distortion-Before & Distortion-After \\
    for idx, row in df.iterrows():
        method = row['Method']
        scale = row['Scale']
        cal = row['Calibration']
        mnist = float(row['MNIST'])
        fmnist = float(row['FashionMNIST'])
        cifar = float(row['CIFAR10'])
        avg = float(row['Average'])
        
        # Handle scaling factor formatting
        scale_str = f"{float(scale):.1f}" if method == 'Task Arithmetic' else '-'
        
        # Handle distortion formatting
        bef = row['DistBefore']
        aft = row['DistAfter']
        
        try:
            bef_val = float(bef)
            bef_str = f"{bef_val:.4f}" if bef_val >= 0 else '-'
        except:
            bef_str = '-'
            
        try:
            aft_val = float(aft)
            aft_str = f"{aft_val:.4f}" if aft_val >= 0 else '-'
        except:
            aft_str = '-'
            
        # Bold proposed method
        if 'M-CAC' in cal:
            cal_str = f"\\textbf{{{cal}}}"
            row_str = f"\\textbf{{{method}}} & {scale_str} & {cal_str} & \\textbf{{{mnist:.2f}}} & \\textbf{{{fmnist:.2f}}} & \\textbf{{{cifar:.2f}}} & \\textbf{{{avg:.2f}}} & {bef_str} & \\textbf{{{aft_str}}} \\\\"
        elif 'TCAC' in cal:
            row_str = f"{method} & {scale_str} & {cal} & {mnist:.2f} & {fmnist:.2f} & {cifar:.2f} & {avg:.2f} & {bef_str} & {aft_str} \\\\"
        else:
            row_str = f"{method} & {scale_str} & {cal} & {mnist:.2f} & {fmnist:.2f} & {cifar:.2f} & {avg:.2f} & {bef_str} & {aft_str} \\\\"
            
        print(row_str)

if __name__ == '__main__':
    main()
