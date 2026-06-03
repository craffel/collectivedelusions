import os
import json

def main():
    if not os.path.exists('results/compiled_summary.json'):
        print("No compiled summary found. Cannot populate results.")
        return
        
    with open('results/compiled_summary.json', 'r') as f:
        summary = json.load(f)
        
    # Read LaTeX file
    with open('submission.tex', 'r') as f:
        latex = f.read()
        
    # Define oracle values from training
    oracle = {
        'mnist': 97.30,
        'fashion': 85.23,
        'cifar': 67.13,
        'avg': 83.22
    }
    
    # 1. Populate Oracle
    latex = latex.replace('[OR_M]', f"{oracle['mnist']:.2f}")
    latex = latex.replace('[OR_F]', f"{oracle['fashion']:.2f}")
    latex = latex.replace('[OR_C]', f"{oracle['cifar']:.2f}")
    latex = latex.replace('[OR_A]', f"{oracle['avg']:.2f}")
    
    # Extract WA results
    main_bench = summary.get('main_benchmark', {})
    wa = main_bench.get('WA', {})
    
    # 2. Populate Weight Averaging (WA)
    # None
    latex = latex.replace('[W_N_M]', f"{wa.get('none', {}).get('mnist', 0.0):.2f}")
    latex = latex.replace('[W_N_F]', f"{wa.get('none', {}).get('fashion', 0.0):.2f}")
    latex = latex.replace('[W_N_C]', f"{wa.get('none', {}).get('cifar', 0.0):.2f}")
    latex = latex.replace('[W_N_A]', f"{wa.get('none', {}).get('avg', 0.0):.2f}")
    # TCAC
    latex = latex.replace('[W_T_M]', f"{wa.get('tcac', {}).get('mnist', 0.0):.2f}")
    latex = latex.replace('[W_T_F]', f"{wa.get('tcac', {}).get('fashion', 0.0):.2f}")
    latex = latex.replace('[W_T_C]', f"{wa.get('tcac', {}).get('cifar', 0.0):.2f}")
    latex = latex.replace('[W_T_A]', f"{wa.get('tcac', {}).get('avg', 0.0):.2f}")
    # TAAC
    latex = latex.replace('[W_TA_M]', f"{wa.get('taac', {}).get('mnist', 0.0):.2f}")
    latex = latex.replace('[W_TA_F]', f"{wa.get('taac', {}).get('fashion', 0.0):.2f}")
    latex = latex.replace('[W_TA_C]', f"{wa.get('taac', {}).get('cifar', 0.0):.2f}")
    latex = latex.replace('[W_TA_A]', f"{wa.get('taac', {}).get('avg', 0.0):.2f}")
    # LSC
    latex = latex.replace('[W_L_M]', f"{wa.get('lsc', {}).get('mnist', 0.0):.2f}")
    latex = latex.replace('[W_L_F]', f"{wa.get('lsc', {}).get('fashion', 0.0):.2f}")
    latex = latex.replace('[W_L_C]', f"{wa.get('lsc', {}).get('cifar', 0.0):.2f}")
    latex = latex.replace('[W_L_A]', f"{wa.get('lsc', {}).get('avg', 0.0):.2f}")
    # SP-TAAC (Ours)
    latex = latex.replace('[W_S_M]', f"{wa.get('sp_taac', {}).get('mnist', 0.0):.2f}")
    latex = latex.replace('[W_S_F]', f"{wa.get('sp_taac', {}).get('fashion', 0.0):.2f}")
    latex = latex.replace('[W_S_C]', f"{wa.get('sp_taac', {}).get('cifar', 0.0):.2f}")
    latex = latex.replace('[W_S_A]', f"{wa.get('sp_taac', {}).get('avg', 0.0):.2f}")
    
    # Extract TA lambda = 0.2 results
    ta = main_bench.get('TA (λ=0.2)', {})
    
    # 3. Populate Task Arithmetic (TA) λ=0.2
    # None
    latex = latex.replace('[T_N_M]', f"{ta.get('none', {}).get('mnist', 0.0):.2f}")
    latex = latex.replace('[T_N_F]', f"{ta.get('none', {}).get('fashion', 0.0):.2f}")
    latex = latex.replace('[T_N_C]', f"{ta.get('none', {}).get('cifar', 0.0):.2f}")
    latex = latex.replace('[T_N_A]', f"{ta.get('none', {}).get('avg', 0.0):.2f}")
    # TCAC
    latex = latex.replace('[T_T_M]', f"{ta.get('tcac', {}).get('mnist', 0.0):.2f}")
    latex = latex.replace('[T_T_F]', f"{ta.get('tcac', {}).get('fashion', 0.0):.2f}")
    latex = latex.replace('[T_T_C]', f"{ta.get('tcac', {}).get('cifar', 0.0):.2f}")
    latex = latex.replace('[T_T_A]', f"{ta.get('tcac', {}).get('avg', 0.0):.2f}")
    # TAAC
    latex = latex.replace('[T_TA_M]', f"{ta.get('taac', {}).get('mnist', 0.0):.2f}")
    latex = latex.replace('[T_TA_F]', f"{ta.get('taac', {}).get('fashion', 0.0):.2f}")
    latex = latex.replace('[T_TA_C]', f"{ta.get('taac', {}).get('cifar', 0.0):.2f}")
    latex = latex.replace('[T_TA_A]', f"{ta.get('taac', {}).get('avg', 0.0):.2f}")
    # LSC
    latex = latex.replace('[T_L_M]', f"{ta.get('lsc', {}).get('mnist', 0.0):.2f}")
    latex = latex.replace('[T_L_F]', f"{ta.get('lsc', {}).get('fashion', 0.0):.2f}")
    latex = latex.replace('[T_L_C]', f"{ta.get('lsc', {}).get('cifar', 0.0):.2f}")
    latex = latex.replace('[T_L_A]', f"{ta.get('lsc', {}).get('avg', 0.0):.2f}")
    # SP-TAAC (Ours)
    latex = latex.replace('[T_S_M]', f"{ta.get('sp_taac', {}).get('mnist', 0.0):.2f}")
    latex = latex.replace('[T_S_F]', f"{ta.get('sp_taac', {}).get('fashion', 0.0):.2f}")
    latex = latex.replace('[T_S_C]', f"{ta.get('sp_taac', {}).get('cifar', 0.0):.2f}")
    latex = latex.replace('[T_S_A]', f"{ta.get('sp_taac', {}).get('avg', 0.0):.2f}")
    
    # Save the populated LaTeX file
    with open('submission.tex', 'w') as f:
        f.write(latex)
        
    print("Successfully populated results inside submission.tex!")

if __name__ == '__main__':
    main()
