import json
import os
import subprocess

def load_results():
    if not os.path.exists('results.json'):
        raise FileNotFoundError("results.json not found! Experiments must complete first.")
    with open('results.json', 'r') as f:
        return json.load(f)

def format_acc(val):
    return f"{val:.2f}"

def main():
    print("Loading results from results.json...")
    res = load_results()
    
    # Extract results
    expert_mnist = res['expert_accs']['mnist']
    expert_fmnist = res['expert_accs']['fmnist']
    expert_cifar10 = res['expert_accs']['cifar10']
    expert_avg = (expert_mnist + expert_fmnist + expert_cifar10) / 3.0
    
    wa_mnist = res['wa_results']['mnist']
    wa_fmnist = res['wa_results']['fmnist']
    wa_cifar10 = res['wa_results']['cifar10']
    wa_avg = res['wa_results']['avg']
    
    uipr_mnist = res['uipr_results']['mnist']
    uipr_fmnist = res['uipr_results']['fmnist']
    uipr_cifar10 = res['uipr_results']['cifar10']
    uipr_avg = res['uipr_results']['avg']
    
    hns_mnist = res['hns_results']['mnist']
    hns_fmnist = res['hns_results']['fmnist']
    hns_cifar10 = res['hns_results']['cifar10']
    hns_avg = res['hns_results']['avg']
    
    # Best TA
    best_ta_lambda = res['best_ta']['lambda']
    best_ta_avg = res['best_ta']['avg_acc']
    # Find matching TA results
    best_ta_dict = next(r for r in res['ta_results'] if abs(r['lambda'] - best_ta_lambda) < 1e-5)
    best_ta_mnist = best_ta_dict['mnist']
    best_ta_fmnist = best_ta_dict['fmnist']
    best_ta_cifar10 = best_ta_dict['cifar10']
    
    # CPR 1.732
    cpr_1732_dict = next(r for r in res['cpr_results'] if abs(r['c'] - 1.732) < 1e-3)
    cpr_1732_mnist = cpr_1732_dict['mnist']
    cpr_1732_fmnist = cpr_1732_dict['fmnist']
    cpr_1732_cifar10 = cpr_1732_dict['cifar10']
    cpr_1732_avg = cpr_1732_dict['avg']
    
    # Best CPR
    best_cpr_c = res['best_cpr']['c']
    best_cpr_avg = res['best_cpr']['avg_acc']
    best_cpr_dict = next(r for r in res['cpr_results'] if abs(r['c'] - best_cpr_c) < 1e-5)
    best_cpr_mnist = best_cpr_dict['mnist']
    best_cpr_fmnist = best_cpr_dict['fmnist']
    best_cpr_cifar10 = best_cpr_dict['cifar10']
    
    # Best TIES
    best_ties_p = res['best_ties']['p_trim']
    best_ties_lambda = res['best_ties']['lambda']
    best_ties_mnist = res['best_ties']['mnist']
    best_ties_fmnist = res['best_ties']['fmnist']
    best_ties_cifar10 = res['best_ties']['cifar10']
    best_ties_avg = res['best_ties']['avg_acc']
    
    # Best DARE
    best_dare_p = res['best_dare']['p_drop']
    best_dare_lambda = res['best_dare']['lambda']
    best_dare_mnist = res['best_dare']['mnist']
    best_dare_fmnist = res['best_dare']['fmnist']
    best_dare_cifar10 = res['best_dare']['cifar10']
    best_dare_avg = res['best_dare']['avg_acc']
    
    # Load K=2 results
    k2_mf_1414 = 0.0
    k2_mc_1414 = 0.0
    k2_fc_1414 = 0.0
    best_mf_c = 0.0
    best_mf_acc = 0.0
    best_mc_c = 0.0
    best_mc_acc = 0.0
    best_fc_c = 0.0
    best_fc_acc = 0.0
    
    if os.path.exists('results_k2.json'):
        print("Loading results from results_k2.json...")
        with open('results_k2.json', 'r') as f:
            res_k2 = json.load(f)
        
        # Extract K=2 values at c = 1.414
        k2_mf_1414_dict = next(r for r in res_k2['results']['mnist_fmnist'] if abs(r['c'] - 1.414) < 1e-3)
        k2_mf_1414 = k2_mf_1414_dict['avg']
        
        k2_mc_1414_dict = next(r for r in res_k2['results']['mnist_cifar10'] if abs(r['c'] - 1.414) < 1e-3)
        k2_mc_1414 = k2_mc_1414_dict['avg']
        
        k2_fc_1414_dict = next(r for r in res_k2['results']['fmnist_cifar10'] if abs(r['c'] - 1.414) < 1e-3)
        k2_fc_1414 = k2_fc_1414_dict['avg']
        
        # Extract best results
        best_mf_c = res_k2['best_results']['mnist_fmnist']['c']
        best_mf_acc = res_k2['best_results']['mnist_fmnist']['avg']
        
        best_mc_c = res_k2['best_results']['mnist_cifar10']['c']
        best_mc_acc = res_k2['best_results']['mnist_cifar10']['avg']
        
        best_fc_c = res_k2['best_results']['fmnist_cifar10']['c']
        best_fc_acc = res_k2['best_results']['fmnist_cifar10']['avg']
    
    # Replacement map
    replacements = {
        '@EXPERT_MNIST@': format_acc(expert_mnist),
        '@EXPERT_FMNIST@': format_acc(expert_fmnist),
        '@EXPERT_CIFAR10@': format_acc(expert_cifar10),
        '@EXPERT_AVG@': format_acc(expert_avg),
        
        '@WA_MNIST@': format_acc(wa_mnist),
        '@WA_FMNIST@': format_acc(wa_fmnist),
        '@WA_CIFAR10@': format_acc(wa_cifar10),
        '@WA_AVG@': format_acc(wa_avg),
        
        '@UIPR_MNIST@': format_acc(uipr_mnist),
        '@UIPR_FMNIST@': format_acc(uipr_fmnist),
        '@UIPR_CIFAR10@': format_acc(uipr_cifar10),
        '@UIPR_AVG@': format_acc(uipr_avg),
        
        '@HNS_MNIST@': format_acc(hns_mnist),
        '@HNS_FMNIST@': format_acc(hns_fmnist),
        '@HNS_CIFAR10@': format_acc(hns_cifar10),
        '@HNS_AVG@': format_acc(hns_avg),
        
        '@BEST_TA_LAMBDA@': f"{best_ta_lambda:.2f}",
        '@BEST_TA_MNIST@': format_acc(best_ta_mnist),
        '@BEST_TA_FMNIST@': format_acc(best_ta_fmnist),
        '@BEST_TA_CIFAR10@': format_acc(best_ta_cifar10),
        '@BEST_TA_AVG@': format_acc(best_ta_avg),
        
        '@CPR_1.732_MNIST@': format_acc(cpr_1732_mnist),
        '@CPR_1.732_FMNIST@': format_acc(cpr_1732_fmnist),
        '@CPR_1.732_CIFAR10@': format_acc(cpr_1732_cifar10),
        '@CPR_1.732_AVG@': format_acc(cpr_1732_avg),
        '@CPR_GAIN@': format_acc(cpr_1732_avg - wa_avg),
        
        '@BEST_CPR_C@': f"{best_cpr_c:.3f}" if abs(best_cpr_c - 1.732) < 0.01 else f"{best_cpr_c:.2f}",
        '@BEST_CPR_MNIST@': format_acc(best_cpr_mnist),
        '@BEST_CPR_FMNIST@': format_acc(best_cpr_fmnist),
        '@BEST_CPR_CIFAR10@': format_acc(best_cpr_cifar10),
        '@BEST_CPR_AVG@': format_acc(best_cpr_avg),
        
        '@BEST_TIES_P@': f"{best_ties_p}",
        '@BEST_TIES_LAMBDA@': f"{best_ties_lambda:.3f}" if abs(best_ties_lambda - 1.732) < 0.01 else f"{best_ties_lambda:.2f}",
        '@BEST_TIES_MNIST@': format_acc(best_ties_mnist),
        '@BEST_TIES_FMNIST@': format_acc(best_ties_fmnist),
        '@BEST_TIES_CIFAR10@': format_acc(best_ties_cifar10),
        '@BEST_TIES_AVG@': format_acc(best_ties_avg),
        
        '@BEST_DARE_P@': f"{best_dare_p}",
        '@BEST_DARE_LAMBDA@': f"{best_dare_lambda:.3f}" if abs(best_dare_lambda - 1.732) < 0.01 else f"{best_dare_lambda:.2f}",
        '@BEST_DARE_MNIST@': format_acc(best_dare_mnist),
        '@BEST_DARE_FMNIST@': format_acc(best_dare_fmnist),
        '@BEST_DARE_CIFAR10@': format_acc(best_dare_cifar10),
        '@BEST_DARE_AVG@': format_acc(best_dare_avg),
        
        '@K2_MF_1414@': format_acc(k2_mf_1414),
        '@K2_MC_1414@': format_acc(k2_mc_1414),
        '@K2_FC_1414@': format_acc(k2_fc_1414),
        '@K2_MF_BEST_C@': f"{best_mf_c:.3f}" if abs(best_mf_c - 1.414) < 0.01 else f"{best_mf_c:.2f}",
        '@K2_MF_BEST_ACC@': format_acc(best_mf_acc),
        '@K2_MC_BEST_C@': f"{best_mc_c:.3f}" if abs(best_mc_c - 1.414) < 0.01 else f"{best_mc_c:.2f}",
        '@K2_MC_BEST_ACC@': format_acc(best_mc_acc),
        '@K2_FC_BEST_C@': f"{best_fc_c:.3f}" if abs(best_fc_c - 1.414) < 0.01 else f"{best_fc_c:.2f}",
        '@K2_FC_BEST_ACC@': format_acc(best_fc_acc)
    }
    
    print("\nCalculated Replacements:")
    for k, v in replacements.items():
        print(f"  {k} -> {v}")
        
    # Read submission.tex
    print("\nReading submission.tex...")
    with open('submission.tex', 'r') as f:
        tex_content = f.read()
        
    # Replace all placeholders
    for placeholder, val in replacements.items():
        tex_content = tex_content.replace(placeholder, val)
        
    # Write to compiled output
    print("Writing compiled LaTeX document to submission_compiled.tex...")
    with open('submission_compiled.tex', 'w') as f:
        f.write(tex_content)
        
    # Ensure tectonic can find styles in template/
    # We copy them or run with an environment or run inside the directory
    # Copying is the most robust and simplest!
    styles = ['algorithm.sty', 'algorithmic.sty', 'fancyhdr.sty', 'icml2026.sty', 'icml2026.bst', 'example_paper.bib']
    for style in styles:
        src = os.path.join('template', style)
        if os.path.exists(src) and not os.path.exists(style):
            print(f"Copying {style} from template/ to root directory...")
            with open(src, 'rb') as f_in:
                with open(style, 'wb') as f_out:
                    f_out.write(f_in.read())
                    
    # Compile with tectonic
    print("\nCompiling submission_compiled.tex with Tectonic...")
    cmd = ["./tectonic", "submission_compiled.tex"]
    res_proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    print("Tectonic stdout:")
    print(res_proc.stdout)
    if res_proc.returncode != 0:
        print("Tectonic stderr (Error occurred!):")
        print(res_proc.stderr)
        exit(res_proc.returncode)
        
    # Verify PDF was created and copy to submission.pdf
    if os.path.exists('submission_compiled.pdf'):
        os.rename('submission_compiled.pdf', 'submission.pdf')
        print("\nSUCCESS! submission.pdf has been compiled and saved to the root directory.")
    else:
        print("\nERROR: Compiled PDF file not found!")
        exit(1)

if __name__ == '__main__':
    main()
