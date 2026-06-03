import os
import copy
import torch
from run_benchmark import get_resnet18_model, get_dataloaders, evaluate_model, copy_bn_and_fc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def apply_hns_clamped(merged_model, expert_model, progenitor_model, min_val, max_val):
    merged_state = merged_model.state_dict()
    expert_state = expert_model.state_dict()
    prog_state = progenitor_model.state_dict()
    
    with torch.no_grad():
        for key in merged_state.keys():
            if 'fc' in key or 'classifier' in key:
                continue
            param_m = merged_state[key]
            param_e = expert_state[key]
            param_p = prog_state[key]
            
            if len(param_m.shape) >= 2:
                device = param_m.device
                param_e_dev = param_e.to(device).float()
                param_p_dev = param_p.to(device).float()
                param_m_float = param_m.float()
                
                tv_e = param_e_dev - param_p_dev
                tv_m = param_m_float - param_p_dev
                
                dim = tuple(range(1, len(param_m.shape)))
                norm_e = torch.norm(tv_e, p=2, dim=dim, keepdim=True)
                norm_m = torch.norm(tv_m, p=2, dim=dim, keepdim=True)
                
                scale = norm_e / (norm_m + 1e-8)
                scale = torch.clamp(scale, min=min_val, max=max_val)
                
                tv_m_scaled = tv_m * scale.view(-1, *([1]*(len(param_m.shape)-1)))
                param_m.copy_((param_p_dev + tv_m_scaled).to(param_m.dtype))

def main():
    print("Loading dataloaders...")
    loaders = get_dataloaders()
    
    progenitor = get_resnet18_model().to(DEVICE)
    progenitor.load_state_dict(torch.load('checkpoints/progenitor.pt', map_location=DEVICE))
    
    tasks = ['mnist', 'fmnist', 'cifar']
    experts = {}
    for task in tasks:
        model = get_resnet18_model().to(DEVICE)
        model.load_state_dict(torch.load(f'checkpoints/{task}_expert.pt', map_location=DEVICE))
        experts[task] = model
        
    # Standard WA backbone
    wa_model = get_resnet18_model().to(DEVICE)
    wa_state = wa_model.state_dict()
    keys = [k for k in wa_state.keys() if 'fc' not in k]
    for key in keys:
        temp = torch.zeros_like(wa_state[key], dtype=torch.float32)
        for name, m in experts.items():
            temp += m.state_dict()[key].cpu().float()
        wa_state[key].copy_(temp / 3.0)
    wa_model.load_state_dict(wa_state)
    
    clamp_options = [
        (0.1, 2.0),
        (0.1, 5.0),
        (0.1, 10.0),
        (0.1, 20.0),
        (0.01, 100.0)
    ]
    
    for min_val, max_val in clamp_options:
        print(f"\n--- Sweeping HNS clamp to [{min_val}, {max_val}] ---")
        hns_accs = {}
        for task in tasks:
            temp_model = copy.deepcopy(wa_model)
            apply_hns_clamped(temp_model, experts[task], progenitor, min_val, max_val)
            copy_bn_and_fc(temp_model, experts[task])
            _, acc = evaluate_model(temp_model, loaders[task]['test'])
            hns_accs[task] = acc
            print(f"  HNS Accuracy on {task.upper()}: {acc:.2f}%")
        print(f"  Average HNS Accuracy: {sum(hns_accs.values())/3:.2f}%")

if __name__ == '__main__':
    main()
