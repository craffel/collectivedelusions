import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED errors
torch.backends.cudnn.enabled = False

# Define transforms
transform_gray = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_color = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class ExpertModel(nn.Module):
    def __init__(self, task_name):
        super().__init__()
        self.backbone = models.resnet18()
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(in_features, 10)
        self.task_name = task_name
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

class MultiTaskMergedModel(nn.Module):
    def __init__(self, tasks=['mnist', 'fashion', 'cifar10']):
        super().__init__()
        self.backbone = models.resnet18()
        self.backbone.fc = nn.Identity()
        self.heads = nn.ModuleDict({
            task: nn.Linear(512, 10) for task in tasks
        })

def get_datasets(task):
    if task == 'mnist':
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_gray)
    elif task == 'fashion':
        test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_gray)
    elif task == 'cifar10':
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_color)
    else:
        raise ValueError(f"Unknown task {task}")
    return test_set

def get_merged_backbone_state_dict(expert_state_dicts, pretrained_state_dict, method="WA", lambda_coeff=0.3):
    merged_state_dict = {}
    keys = list(expert_state_dicts[0].keys())
    for key in keys:
        if key.startswith('backbone.'):
            params = [state_dict[key] for state_dict in expert_state_dicts]
            is_running_stat = any(x in key for x in ['running_mean', 'running_var', 'num_batches_tracked'])
            if is_running_stat:
                if 'num_batches_tracked' in key:
                    merged_state_dict[key] = torch.stack([p.float() for p in params]).mean(dim=0).long()
                else:
                    merged_state_dict[key] = torch.stack(params).mean(dim=0)
            else:
                if method == "WA":
                    merged_state_dict[key] = torch.stack(params).mean(dim=0)
                elif method == "TA":
                    pre_key = key.replace('backbone.', '')
                    pre_param = pretrained_state_dict[pre_key]
                    task_vectors = [p - pre_param for p in params]
                    merged_state_dict[key] = pre_param + lambda_coeff * torch.stack(task_vectors).sum(dim=0)
    return merged_state_dict

# Linear CKA computation
def linear_cka(X, Y):
    # X: [B, D1], Y: [B, D2]
    # Center columns
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    
    # Compute Gram matrices
    K = torch.matmul(X, X.t())
    L = torch.matmul(Y, Y.t())
    
    def center_gram(G):
        n = G.size(0)
        means = G.mean(dim=0, keepdim=True)
        col_means = G.mean(dim=1, keepdim=True)
        grand_mean = G.mean()
        return G - means - col_means + grand_mean
        
    K_c = center_gram(K)
    L_c = center_gram(L)
    
    hsic_kl = torch.trace(torch.matmul(K_c, L_c))
    hsic_kk = torch.trace(torch.matmul(K_c, K_c))
    hsic_ll = torch.trace(torch.matmul(L_c, L_c))
    
    if hsic_kk == 0 or hsic_ll == 0:
        return 0.0
        
    cka = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-8)
    return cka.item()

class ActivationCapturer:
    def __init__(self):
        self.activations = {}
        self.hooks = []
        
    def register(self, name, module):
        def hook(module, input, output):
            self.activations[name] = output.detach().cpu()
        self.hooks.append(module.register_forward_hook(hook))
        
    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running CKA evaluation on device: {device}")
    
    tasks = ['mnist', 'fashion', 'cifar10']
    expert_state_dicts = []
    expert_models = {}
    
    # Load experts
    for task in tasks:
        checkpoint_path = f"expert_{task}.pth"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        expert_state_dicts.append(checkpoint['model_state_dict'])
        
        exp_model = ExpertModel(task)
        exp_model.load_state_dict(checkpoint['model_state_dict'])
        exp_model.to(device)
        exp_model.eval()
        expert_models[task] = exp_model
        
    # Load pre-trained backbone for Task Arithmetic merging
    pretrained_backbone_state_dict = torch.load("resnet18_pretrained.pth", map_location='cpu')
    
    # Create merged models
    merged_wa = MultiTaskMergedModel(tasks)
    wa_sd = get_merged_backbone_state_dict(expert_state_dicts, pretrained_backbone_state_dict, "WA", 0.333)
    merged_wa.backbone.load_state_dict({k.replace('backbone.', ''): v for k, v in wa_sd.items() if k.startswith('backbone.')})
    merged_wa.to(device)
    merged_wa.eval()
    
    merged_ta = MultiTaskMergedModel(tasks)
    ta_sd = get_merged_backbone_state_dict(expert_state_dicts, pretrained_backbone_state_dict, "TA", 0.4)
    merged_ta.backbone.load_state_dict({k.replace('backbone.', ''): v for k, v in ta_sd.items() if k.startswith('backbone.')})
    merged_ta.to(device)
    merged_ta.eval()
    
    # Create evaluation dataset
    eval_datasets = [get_datasets(t) for t in tasks]
    joint_eval_dataset = ConcatDataset(eval_datasets)
    
    # We take a subset of 512 samples for fast CKA computation
    g = torch.Generator()
    g.manual_seed(42)
    indices = torch.randperm(len(joint_eval_dataset), generator=g)[:512].tolist()
    eval_subset = torch.utils.data.Subset(joint_eval_dataset, indices)
    eval_loader = DataLoader(eval_subset, batch_size=512, shuffle=False)
    
    # Get a batch of data
    for inputs, _ in eval_loader:
        inputs = inputs.to(device)
        break
        
    # We want to hook the output of different blocks
    target_blocks = {
        'conv1': 'conv1',
        'layer1': 'layer1',
        'layer2': 'layer2',
        'layer3': 'layer3',
        'layer4': 'layer4'
    }
    
    # Capture activations for each model
    # 1. Experts
    expert_activations = {task: {} for task in tasks}
    for task in tasks:
        capturer = ActivationCapturer()
        for b_name, b_attr in target_blocks.items():
            module = getattr(expert_models[task].backbone, b_attr)
            capturer.register(b_name, module)
        with torch.no_grad():
            _ = expert_models[task](inputs)
        for b_name in target_blocks.keys():
            # Flatten activations to [B, D]
            act = capturer.activations[b_name]
            expert_activations[task][b_name] = act.reshape(act.size(0), -1)
        capturer.remove()
        
    # 2. Merged WA
    wa_activations = {}
    capturer = ActivationCapturer()
    for b_name, b_attr in target_blocks.items():
        module = getattr(merged_wa.backbone, b_attr)
        capturer.register(b_name, module)
    with torch.no_grad():
        _ = merged_wa.backbone(inputs)
    for b_name in target_blocks.keys():
        act = capturer.activations[b_name]
        wa_activations[b_name] = act.reshape(act.size(0), -1)
    capturer.remove()
    
    # 3. Merged TA
    ta_activations = {}
    capturer = ActivationCapturer()
    for b_name, b_attr in target_blocks.items():
        module = getattr(merged_ta.backbone, b_attr)
        capturer.register(b_name, module)
    with torch.no_grad():
        _ = merged_ta.backbone(inputs)
    for b_name in target_blocks.keys():
        act = capturer.activations[b_name]
        ta_activations[b_name] = act.reshape(act.size(0), -1)
    capturer.remove()
    
    # Compute CKAs
    print("\n" + "="*50)
    print("REPRESENTATIONAL CKA SIMILARITIES")
    print("="*50)
    
    results = {
        'WA': {b: [] for b in target_blocks.keys()},
        'TA': {b: [] for b in target_blocks.keys()},
        'Inter-Expert': {b: [] for b in target_blocks.keys()}
    }
    
    for b_name in target_blocks.keys():
        print(f"\nBlock: {b_name}")
        print("-"*30)
        
        # Expert vs Merged WA
        wa_ckas = []
        for task in tasks:
            cka_val = linear_cka(expert_activations[task][b_name], wa_activations[b_name])
            wa_ckas.append(cka_val)
            print(f"  WA vs Expert {task:<8}: {cka_val:.4f}")
        avg_wa_cka = sum(wa_ckas) / len(wa_ckas)
        results['WA'][b_name] = wa_ckas
        print(f"  WA vs Expert (Avg)  : {avg_wa_cka:.4f}")
        
        # Expert vs Merged TA
        ta_ckas = []
        for task in tasks:
            cka_val = linear_cka(expert_activations[task][b_name], ta_activations[b_name])
            ta_ckas.append(cka_val)
            print(f"  TA vs Expert {task:<8}: {cka_val:.4f}")
        avg_ta_cka = sum(ta_ckas) / len(ta_ckas)
        results['TA'][b_name] = ta_ckas
        print(f"  TA vs Expert (Avg)  : {avg_ta_cka:.4f}")
        
        # Inter-Expert similarity (divergence check)
        ie_ckas = []
        for i in range(len(tasks)):
            for j in range(i+1, len(tasks)):
                t1, t2 = tasks[i], tasks[j]
                cka_val = linear_cka(expert_activations[t1][b_name], expert_activations[t2][b_name])
                ie_ckas.append(cka_val)
        avg_ie_cka = sum(ie_ckas) / len(ie_ckas)
        results['Inter-Expert'][b_name] = ie_ckas
        print(f"  Inter-Expert (Avg)  : {avg_ie_cka:.4f}")
        
    # Save results to a file
    torch.save(results, "cka_results.pt")
    print("\nSuccessfully computed and saved CKA results to cka_results.pt")

if __name__ == "__main__":
    main()
