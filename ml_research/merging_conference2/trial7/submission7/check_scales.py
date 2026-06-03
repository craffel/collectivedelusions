import torch
from run_benchmark import get_resnet18_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_scales():
    progenitor = get_resnet18_model().to(DEVICE)
    progenitor.load_state_dict(torch.load('checkpoints/progenitor.pt', map_location=DEVICE))
    
    tasks = ['mnist', 'fmnist', 'cifar']
    experts = {}
    for task in tasks:
        model = get_resnet18_model().to(DEVICE)
        model.load_state_dict(torch.load(f'checkpoints/{task}_expert.pt', map_location=DEVICE))
        experts[task] = model
        
    # Standard Weight Averaging of the backbone
    merged_state = get_resnet18_model().state_dict()
    keys = [k for k in merged_state.keys() if 'fc' not in k]
    
    for key in keys:
        temp = torch.zeros_like(merged_state[key], dtype=torch.float32)
        for name, m in experts.items():
            temp += m.state_dict()[key].cpu().float()
        merged_state[key].copy_(temp / 3.0)
        
    for task in tasks:
        print(f"\n=== Scale Factors for {task.upper()} ===")
        expert_state = experts[task].state_dict()
        prog_state = progenitor.state_dict()
        
        scales_list = []
        for key in keys:
            if len(merged_state[key].shape) >= 2:
                param_m = merged_state[key].to(DEVICE).float()
                param_e = expert_state[key].to(DEVICE).float()
                param_p = prog_state[key].to(DEVICE).float()
                
                tv_e = param_e - param_p
                tv_m = param_m - param_p
                
                dim = tuple(range(1, len(param_m.shape)))
                norm_e = torch.norm(tv_e, p=2, dim=dim)
                norm_m = torch.norm(tv_m, p=2, dim=dim)
                
                scale = norm_e / (norm_m + 1e-8)
                scales_list.append((key, scale.mean().item(), scale.std().item(), scale.min().item(), scale.max().item()))
                
        for name, mean, std, mn, mx in scales_list:
            print(f"{name:30s} | Mean: {mean:6.2f} | Std: {std:6.2f} | Min: {mn:6.2f} | Max: {mx:6.2f}")

if __name__ == '__main__':
    check_scales()
