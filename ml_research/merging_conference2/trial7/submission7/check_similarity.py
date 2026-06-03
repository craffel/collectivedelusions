import torch
from run_benchmark import get_resnet18_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_similarity():
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
        print(f"\n=== Cosine Similarities for {task.upper()} ===")
        expert_state = experts[task].state_dict()
        prog_state = progenitor.state_dict()
        
        sims_raw = []
        sims_tv = []
        for key in keys:
            if len(merged_state[key].shape) >= 2:
                param_m = merged_state[key].to(DEVICE).float()
                param_e = expert_state[key].to(DEVICE).float()
                param_p = prog_state[key].to(DEVICE).float()
                
                tv_e = param_e - param_p
                tv_m = param_m - param_p
                
                # Global cosine sim
                cos_raw = torch.nn.functional.cosine_similarity(param_e.flatten().unsqueeze(0), param_m.flatten().unsqueeze(0)).item()
                cos_tv = torch.nn.functional.cosine_similarity(tv_e.flatten().unsqueeze(0), tv_m.flatten().unsqueeze(0)).item()
                
                sims_raw.append(cos_raw)
                sims_tv.append(cos_tv)
                
        print(f"Average weight cosine similarity: {sum(sims_raw)/len(sims_raw):.4f}")
        print(f"Average task vector cosine similarity: {sum(sims_tv)/len(sims_tv):.4f}")

if __name__ == '__main__':
    check_similarity()
