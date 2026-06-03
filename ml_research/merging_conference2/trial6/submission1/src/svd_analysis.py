import os
import copy
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def load_expert_model(dataset_name, device='cpu'):
    try:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception:
        model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.1),
        model.fc
    )
    checkpoint_path = f"checkpoints/{dataset_name}_expert.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Expert model not found at {checkpoint_path}.")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def analyze_svd():
    print("==================================================")
    print("📈 RUNNING EXPERT WEIGHT CORRECTIONS SVD DECAY ANALYSIS 📈")
    print("==================================================")
    
    tasks = ['mnist', 'fmnist', 'cifar10']
    expert_state_dicts = {}
    for t in tasks:
        model = load_expert_model(t, device='cpu')
        expert_state_dicts[t] = model.state_dict()
        
    # Compute Weight Averaged (WA) state dict
    wa_state_dict = {}
    keys = list(expert_state_dicts['mnist'].keys())
    for key in keys:
        if 'fc' in key:
            continue
        tensors = [expert_state_dicts[t][key].float() for t in tasks]
        wa_state_dict[key] = torch.stack(tensors).mean(dim=0)
        
    # We want to analyze convolutional layers in layer3 and layer4
    # Let's find all conv layers in layer3 and layer4
    try:
        model_example = resnet18()
    except Exception:
        model_example = resnet18(pretrained=False)
        
    conv_layers = []
    for name, module in model_example.named_modules():
        if isinstance(module, nn.Conv2d) and any(name.startswith(p) for p in ['layer3', 'layer4']):
            conv_layers.append((name, module))
            
    print(f"Found {len(conv_layers)} Conv2d layers in layer3 and layer4:")
    for name, _ in conv_layers:
        print(f"  - {name}")
        
    # Results dictionary to hold: {layer_name: {task: {rank: percentage_explained}}}
    svd_results = {}
    
    for name, module in conv_layers:
        weight_key = f"{name}.weight"
        svd_results[name] = {}
        
        for t in tasks:
            W_expert = expert_state_dicts[t][weight_key].float()
            W_wa = wa_state_dict[weight_key]
            
            # Compute weight correction Delta W
            delta_W = W_expert - W_wa
            
            # Flatten to 2D: C_out x d_in
            C_out = delta_W.size(0)
            d_in = delta_W.view(C_out, -1).size(1)
            delta_W_2D = delta_W.view(C_out, -1)
            
            # SVD
            U, S, Vh = torch.linalg.svd(delta_W_2D, full_matrices=False)
            
            # Explained variance (sum of squared singular values)
            S_sq = S ** 2
            total_var = S_sq.sum().item()
            
            # Percentage explained by ranks 1, 2, 4, 8
            variance_explained = {}
            for r in [1, 2, 4, 8]:
                r_clamped = min(r, len(S_sq))
                explained = S_sq[:r_clamped].sum().item()
                percentage = (explained / total_var) * 100.0 if total_var > 1e-10 else 100.0
                variance_explained[r] = percentage
                
            svd_results[name][t] = {
                'variance_explained': variance_explained,
                'singular_values': S[:15].tolist(), # top 15 singular values for visualization/logging
                'full_dim': len(S)
            }
            
            print(f"Layer: {name:20s} | Task: {t:8s} | Dim: {len(S):3d} | Rank 1: {variance_explained[1]:5.1f}% | Rank 2: {variance_explained[2]:5.1f}% | Rank 4: {variance_explained[4]:5.1f}% | Rank 8: {variance_explained[8]:5.1f}%")

    # Generate LaTeX table source code
    print("\n--- LaTeX Table Output ---")
    latex_code = []
    latex_code.append("\\begin{table*}[t]")
    latex_code.append("\\caption{Cumulative percentage of explained variance (squared singular values) in weight corrections $\\Delta W = W_{\\text{expert}} - W_{\\text{WA}}$ across various SVD ranks $r \\in \\{1, 2, 4, 8\\}$. The rapid decay of singular values empirically confirms that task-specific updates are highly low-rank.}")
    latex_code.append("\\label{tab:svd_decay_analysis}")
    latex_code.append("\\vskip 0.15in")
    latex_code.append("\\begin{center}")
    latex_code.append("\\begin{small}")
    latex_code.append("\\begin{tabular}{llccccc}")
    latex_code.append("\\toprule")
    latex_code.append("Layer Name & Task Domain & Full Dim & Rank 1 (\\%) & Rank 2 (\\%) & Rank 4 (\\%) & Rank 8 (\\%) \\\\")
    latex_code.append("\\midrule")
    
    for name, _ in conv_layers:
        latex_code.append(f"\\multirow{{3}}{{*}}{{\\texttt{{{name}}}}} ")
        for i, t in enumerate(tasks):
            info = svd_results[name][t]
            ve = info['variance_explained']
            full_dim = info['full_dim']
            
            task_label = t.upper() if t != 'fmnist' else 'F-MNIST'
            if i == 0:
                latex_code.append(f"& {task_label} & {full_dim} & {ve[1]:.1f}\\% & {ve[2]:.1f}\\% & {ve[4]:.1f}\\% & {ve[8]:.1f}\\% \\\\")
            elif i == 1:
                latex_code.append(f"& {task_label} & {full_dim} & {ve[1]:.1f}\\% & {ve[2]:.1f}\\% & {ve[4]:.1f}\\% & {ve[8]:.1f}\\% \\\\")
            else:
                latex_code.append(f"& {task_label} & {full_dim} & {ve[1]:.1f}\\% & {ve[2]:.1f}\\% & {ve[4]:.1f}\\% & {ve[8]:.1f}\\% \\\\")
        latex_code.append("\\hline")
        
    # Remove the last \\hline and replace with bottomrule
    if latex_code[-1] == "\\hline":
        latex_code.pop()
    latex_code.append("\\bottomrule")
    latex_code.append("\\end{tabular}")
    latex_code.append("\\end{small}")
    latex_code.append("\\end{center}")
    latex_code.append("\\end{table*}")
    
    latex_str = "\n".join(latex_code)
    print(latex_str)
    
    # Save results to json
    import json
    with open("svd_decay_results.json", "w") as f:
        json.dump(svd_results, f, indent=4)
    print("\n✅ SVD analysis completed and saved to svd_decay_results.json!")

if __name__ == '__main__':
    analyze_svd()
