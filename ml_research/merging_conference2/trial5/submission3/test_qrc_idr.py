import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
import numpy as np

from src.evaluate import load_base_model, load_expert, get_datasets, merge_models, get_batchnorm_modules, evaluate_backbone

def run_qrc_idr(merged_backbone, experts, cal_data, threshold=1e-4, device='cpu'):
    m_model = load_base_model()
    mb_state = copy.deepcopy(merged_backbone)
    mb_state['fc.weight'] = experts[0].state_dict()['fc.weight']
    mb_state['fc.bias'] = experts[0].state_dict()['fc.bias']
    m_model.load_state_dict(mb_state)
    m_model = m_model.to(device)
    m_model.eval()

    expert_loaders = [
        DataLoader(cal_data['mnist_cal'], batch_size=128, shuffle=False),
        DataLoader(cal_data['fmnist_cal'], batch_size=128, shuffle=False),
        DataLoader(cal_data['cifar10_cal'], batch_size=128, shuffle=False)
    ]
    joint_loader = DataLoader(cal_data['joint_cal'], batch_size=128, shuffle=False)

    bn_modules_m = get_batchnorm_modules(m_model)
    bn_modules_e = [get_batchnorm_modules(exp) for exp in experts]

    activations = {}
    def get_hook(name):
        def hook(module, inp, out):
            activations[name] = out.detach()
        return hook

    for l_idx in range(len(bn_modules_m)):
        name_m, module_m = bn_modules_m[l_idx]
        
        target_medians = []
        target_idrs = []
        target_stds = []
        target_means = []
        
        for m_idx, exp in enumerate(experts):
            name_e, module_e = bn_modules_e[m_idx][l_idx]
            handle = module_e.register_forward_hook(get_hook('exp'))
            collected_acts = []
            with torch.no_grad():
                for images, _ in expert_loaders[m_idx]:
                    images = images.to(device)
                    exp(images)
                    collected_acts.append(activations['exp'].cpu())
            handle.remove()
            
            Y_exp = torch.cat(collected_acts, dim=0)
            C = Y_exp.shape[1]
            Y_exp_flat = Y_exp.transpose(0, 1).contiguous().view(C, -1)
            
            ch_medians = torch.median(Y_exp_flat, dim=1).values
            ch_q10 = torch.quantile(Y_exp_flat, 0.10, dim=1)
            ch_q90 = torch.quantile(Y_exp_flat, 0.90, dim=1)
            ch_stds = torch.std(Y_exp_flat, dim=1)
            ch_means = torch.mean(Y_exp_flat, dim=1)
            
            target_medians.append(ch_medians)
            target_idrs.append(ch_q90 - ch_q10)
            target_stds.append(ch_stds)
            target_means.append(ch_means)

        median_target = torch.stack(target_medians).mean(dim=0).to(device)
        idr_target = torch.stack(target_idrs).mean(dim=0).to(device)
        std_target = torch.stack(target_stds).mean(dim=0).to(device)
        mean_target = torch.stack(target_means).mean(dim=0).to(device)

        handle = module_m.register_forward_hook(get_hook('merged'))
        collected_acts_m = []
        with torch.no_grad():
            for images, _ in joint_loader:
                images = images.to(device)
                m_model(images)
                collected_acts_m.append(activations['merged'].cpu())
        handle.remove()
        
        Y_merged = torch.cat(collected_acts_m, dim=0)
        C_m = Y_merged.shape[1]
        Y_merged_flat = Y_merged.transpose(0, 1).contiguous().view(C_m, -1)
        
        median_merged = torch.median(Y_merged_flat, dim=1).values.to(device)
        q10_merged = torch.quantile(Y_merged_flat, 0.10, dim=1).to(device)
        q90_merged = torch.quantile(Y_merged_flat, 0.90, dim=1).to(device)
        idr_merged = q90_merged - q10_merged
        std_merged = torch.std(Y_merged_flat, dim=1).to(device)
        mean_merged = torch.mean(Y_merged_flat, dim=1).to(device)

        # Compute scaling s and shift bcal
        eps = 1e-6
        s = torch.zeros(C_m, device=device)
        bcal = torch.zeros(C_m, device=device)
        
        for c in range(C_m):
            if idr_target[c] < threshold or idr_merged[c] < threshold:
                # Fallback to standard deviation (TAAC style)
                s[c] = std_target[c] / (std_merged[c] + eps)
                bcal[c] = mean_target[c] - s[c] * mean_merged[c]
            else:
                # Use robust IDR-based scaling
                s[c] = idr_target[c] / idr_merged[c]
                bcal[c] = median_target[c] - s[c] * median_merged[c]

        with torch.no_grad():
            module_m.weight.copy_(s * module_m.weight)
            module_m.bias.copy_(s * module_m.bias + bcal)

    return {k: v.cpu() for k, v in m_model.state_dict().items() if 'fc.' not in k}

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    
    # Load models
    base_model = load_base_model().to(device)
    experts = [load_expert(task, device) for task in ['mnist', 'fmnist', 'cifar10']]
    
    # Get calibration data under p=0.2 corruption
    cal_data = get_datasets(N=128, p_corruption=0.2)
    test_loaders = {}
    test_loaders['mnist'] = DataLoader(cal_data['mnist_test'], batch_size=128, shuffle=False)
    test_loaders['fmnist'] = DataLoader(cal_data['fmnist_test'], batch_size=128, shuffle=False)
    test_loaders['cifar10'] = DataLoader(cal_data['cifar10_test'], batch_size=128, shuffle=False)
    
    merged_wa_uncal = merge_models(base_model, experts, mode='wa')
    
    print("\n--- Testing QRC-IDR (p=0.2) ---")
    cal_backbone = run_qrc_idr(merged_wa_uncal, experts, cal_data, threshold=1e-3, device=device)
    
    accs = []
    for i, task in enumerate(['mnist', 'fmnist', 'cifar10']):
        acc = evaluate_backbone(cal_backbone, experts[i], test_loaders[task], device)
        accs.append(acc)
        print(f"  {task.upper()} Accuracy: {acc:.2f}%")
    print(f"Average Accuracy: {sum(accs)/3:.2f}%")

if __name__ == '__main__':
    test()
