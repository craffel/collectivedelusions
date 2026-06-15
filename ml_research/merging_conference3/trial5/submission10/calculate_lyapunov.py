import os
import torch
import timm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

class GrayscaleToRGB(object):
    def __call__(self, img):
        return img.convert('RGB')

def get_datasets(data_dir='./data'):
    transform_rgb = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_gray = transforms.Compose([
        GrayscaleToRGB(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    mnist_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform_gray)
    fmnist_train = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform_gray)
    cifar_train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_rgb)
    svhn_train = datasets.SVHN(root=data_dir, split='train', download=True, transform=transform_rgb)
    
    return {
        'MNIST': (mnist_train, None),
        'FashionMNIST': (fmnist_train, None),
        'CIFAR10': (cifar_train, None),
        'SVHN': (svhn_train, None)
    }

def get_layer_group_idx(key):
    if 'patch_embed' in key or 'cls_token' in key or 'pos_embed' in key:
        return 0
    elif 'blocks.' in key:
        parts = key.split('.')
        return int(parts[1]) + 1
    elif 'norm.' in key:
        return 13
    else:
        return None

def merge_weights(base_state_dict, task_vectors, alpha_bar, device='cpu'):
    merged_state_dict = {}
    for key in base_state_dict:
        group_idx = get_layer_group_idx(key)
        if group_idx is not None:
            merged_val = base_state_dict[key].clone().to(device)
            for k in range(len(task_vectors)):
                coeff = alpha_bar[group_idx, k].to(device)
                merged_val = merged_val + coeff * task_vectors[k][key].to(device)
            merged_state_dict[key] = merged_val
        else:
            merged_state_dict[key] = base_state_dict[key].clone().to(device)
    return merged_state_dict

class GatedChaosMergeModel(torch.nn.Module):
    def __init__(self, d=4, K=4, L=14, init_val=-2.0):
        super().__init__()
        self.d = d
        self.K = K
        self.L = L
        self.W_init = torch.nn.Parameter(torch.randn(K, d) * 0.02)
        self.b_init = torch.nn.Parameter(torch.zeros(K))
        self.gamma_raw = torch.nn.Parameter(torch.zeros(L))
        self.R_raw = torch.nn.Parameter(torch.ones(L, K) * 0.3)
        self.Phi = torch.nn.Parameter(torch.randn(L, K, d) * 0.02)
        self.phi = torch.nn.Parameter(torch.zeros(L, K))
        self.gate_raw = torch.nn.Parameter(torch.ones(L) * init_val)
        self.register_buffer('P', torch.randn(192, d))
        self.P.data = torch.nn.functional.normalize(self.P.data, dim=0)

    def forward_coefficients(self, x_features, average_batch=True):
        B = x_features.shape[0]
        psi_tilde = torch.matmul(x_features, self.P)
        psi = torch.nn.functional.normalize(psi_tilde, p=2, dim=-1, eps=1e-8)
        s = torch.sigmoid(torch.matmul(psi, self.W_init.t()) + self.b_init)
        all_states = []
        for l in range(self.L):
            gamma_l = torch.sigmoid(self.gamma_raw[l])
            def f(u):
                return 4.0 * u * (1.0 - u)
            f_s = f(s)
            sum_f_s = torch.sum(f_s, dim=-1, keepdim=True)
            s_bar = (1.0 - gamma_l) * f_s + (gamma_l / self.K) * sum_f_s
            delta = 1e-5
            s_bar_clipped = torch.clamp(s_bar, delta, 1.0 - delta)
            logits = torch.log(s_bar_clipped / (1.0 - s_bar_clipped))
            perturbation = torch.matmul(psi, self.Phi[l].t()) + self.phi[l]
            s_candidate = torch.sigmoid(logits + perturbation)
            gate_l = torch.sigmoid(self.gate_raw[l])
            s = (1.0 - gate_l) * s + gate_l * s_candidate
            all_states.append(s)
        all_states = torch.stack(all_states, dim=0)
        R = self.R_raw.unsqueeze(1)
        alpha = R * all_states
        if average_batch:
            return torch.mean(alpha, dim=1)
        return alpha

    def compute_lyapunov_exponents(self, x_features, eps=1e-5):
        # Propagate unperturbed and perturbed states layer-by-layer to measure divergence growth
        B = x_features.shape[0]
        psi_tilde = torch.matmul(x_features, self.P)
        psi = torch.nn.functional.normalize(psi_tilde, p=2, dim=-1, eps=1e-8)
        
        # Base initial state
        s0 = torch.sigmoid(torch.matmul(psi, self.W_init.t()) + self.b_init)
        
        # Perturbed initial state
        pert = torch.randn_like(s0)
        pert = pert / torch.norm(pert, p=2, dim=-1, keepdim=True) * eps
        s0_pert = torch.clamp(s0 + pert, 1e-6, 1.0 - 1e-6)
        
        s = s0.clone()
        s_p = s0_pert.clone()
        
        lyapunovs = []
        
        for l in range(self.L):
            gamma_l = torch.sigmoid(self.gamma_raw[l])
            gate_l = torch.sigmoid(self.gate_raw[l])
            
            def f(u):
                return 4.0 * u * (1.0 - u)
            
            # 1. Unperturbed step
            f_s = f(s)
            s_bar = (1.0 - gamma_l) * f_s + (gamma_l / self.K) * torch.sum(f_s, dim=-1, keepdim=True)
            logits = torch.log(torch.clamp(s_bar, 1e-5, 1.0 - 1e-5) / (1.0 - torch.clamp(s_bar, 1e-5, 1.0 - 1e-5)))
            perturbation = torch.matmul(psi, self.Phi[l].t()) + self.phi[l]
            s_candidate = torch.sigmoid(logits + perturbation)
            s_next = (1.0 - gate_l) * s + gate_l * s_candidate
            
            # 2. Perturbed step
            f_s_p = f(s_p)
            s_bar_p = (1.0 - gamma_l) * f_s_p + (gamma_l / self.K) * torch.sum(f_s_p, dim=-1, keepdim=True)
            logits_p = torch.log(torch.clamp(s_bar_p, 1e-5, 1.0 - 1e-5) / (1.0 - torch.clamp(s_bar_p, 1e-5, 1.0 - 1e-5)))
            s_candidate_p = torch.sigmoid(logits_p + perturbation)
            s_next_p = (1.0 - gate_l) * s_p + gate_l * s_candidate_p
            
            # Measure local distance
            d_in = torch.norm(s_p - s, p=2, dim=-1)
            d_out = torch.norm(s_next_p - s_next, p=2, dim=-1)
            
            # Local expansion factor growth rate
            local_growth = d_out / (d_in + 1e-20)
            local_lyapunov = torch.log(local_growth + 1e-20).mean().item()
            lyapunovs.append(local_lyapunov)
            
            # Re-normalize perturbation for the next layer (standard Benettin algorithm step)
            s = s_next
            diff = s_next_p - s_next
            s_p = s + (diff / torch.norm(diff, p=2, dim=-1, keepdim=True)) * eps
            s_p = torch.clamp(s_p, 1e-6, 1.0 - 1e-6)
            
        return lyapunovs

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    task_names = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
    data_dict = get_datasets()
    
    train_size = 2000
    calib_size = 64
    
    calib_loaders = {}
    calib_features_dict = {}
    calib_labels_dict = {}
    
    base_model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
    base_model.head = torch.nn.Linear(192, 10)
    base_model.load_state_dict(torch.load('checkpoints/base_model.pt', weights_only=True))
    base_model.to(device)
    
    pretrained_state = torch.load('checkpoints/base_model.pt', weights_only=True)
    
    expert_models = {}
    task_heads = []
    
    for task_idx, name in enumerate(task_names):
        train_ds, _ = data_dict[name]
        calib_idx = list(range(min(train_size, train_size + calib_size)))
        calib_sub = Subset(train_ds, calib_idx)
        calib_loaders[name] = DataLoader(calib_sub, batch_size=16, shuffle=False)
        
        expert_path = f'checkpoints/expert_{name}.pt'
        expert = timm.create_model('vit_tiny_patch16_224', pretrained=False)
        expert.head = torch.nn.Linear(192, 10)
        expert.load_state_dict(torch.load(expert_path, weights_only=True))
        expert_models[name] = expert
        
        task_heads.append({
            'weight': expert.head.weight.data.clone(),
            'bias': expert.head.bias.data.clone()
        })
        
    base_model.eval()
    with torch.no_grad():
        for name in task_names:
            feats_list = []
            labels_list = []
            for x, y in calib_loaders[name]:
                x = x.to(device)
                z = base_model.patch_embed(x).mean(dim=1)
                feats_list.append(z.cpu())
                labels_list.append(y)
            calib_features_dict[name] = torch.cat(feats_list, dim=0).to(device)
            calib_labels_dict[name] = torch.cat(labels_list, dim=0).to(device)

    task_vectors = []
    for name in task_names:
        expert_state = expert_models[name].state_dict()
        vector = {}
        for key in pretrained_state:
            if pretrained_state[key].dtype in [torch.int64, torch.uint8]:
                continue
            if get_layer_group_idx(key) is None:
                continue
            vector[key] = expert_state[key] - pretrained_state[key]
        task_vectors.append(vector)

    calib_mean_feats = torch.stack([calib_features_dict[n].mean(dim=0) for n in task_names]) # K x D

    # 1. Untrained GatedChaosMergeModel with Gate initialized to high value (equivalent to standard chaotic lattice)
    print("\nEvaluating Untrained (Chaotic) G-CML...")
    chaos_model_chaotic = GatedChaosMergeModel(init_val=4.0).to(device) # High gate value (approx 1.0) means fully open chaos, no skip connection
    with torch.no_grad():
        lyapunovs_chaotic = chaos_model_chaotic.compute_lyapunov_exponents(calib_mean_feats)
    
    # 2. Train the model (exactly like in standard ChaosMerge run)
    print("\nTraining Gated ChaosMerge (G-CML)...")
    chaos_model_gated = GatedChaosMergeModel(init_val=-2.0).to(device) # Stable initialization gate_raw = -2.0 (lambda approx 0.12)
    optimizer = torch.optim.Adam(chaos_model_gated.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    for step in range(50):
        optimizer.zero_grad()
        total_loss = 0.
        alpha_spec = chaos_model_gated.forward_coefficients(calib_mean_feats, average_batch=False)
        for idx, name in enumerate(task_names):
            x_cal, y_cal = next(iter(calib_loaders[name]))
            x_cal, y_cal = x_cal.to(device), y_cal.to(device)
            task_alpha = alpha_spec[:, idx, :]
            merged_state = merge_weights(pretrained_state, task_vectors, task_alpha, device=device)
            task_merged_state = {k: v for k, v in merged_state.items()}
            task_merged_state['head.weight'] = task_heads[idx]['weight'].to(device)
            task_merged_state['head.bias'] = task_heads[idx]['bias'].to(device)
            logits = torch.func.functional_call(base_model, task_merged_state, (x_cal,))
            total_loss += criterion(logits, y_cal)
        total_loss.backward()
        optimizer.step()

    print("Evaluating Trained G-CML...")
    with torch.no_grad():
        lyapunovs_gated = chaos_model_gated.compute_lyapunov_exponents(calib_mean_feats)

    print("\n--- Lyapunov Exponents by Layer ---")
    print(f"{'Layer':<6} | {'Untrained Chaotic (gate=1.0)':<28} | {'Trained Gated (gate=0.12)':<25}")
    print("-" * 65)
    for l in range(14):
        print(f"{l+1:<6} | {lyapunovs_chaotic[l]:<28.4f} | {lyapunovs_gated[l]:<25.4f}")
    
    print("-" * 65)
    print(f"{'Average':<6} | {np.mean(lyapunovs_chaotic):<28.4f} | {np.mean(lyapunovs_gated):<25.4f}")

    # Generate Plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 15), lyapunovs_chaotic, 'o-', color='red', label='Untrained CML (No Gating, $\lambda_l \\approx 1.0$)', linewidth=2)
    plt.plot(range(1, 15), lyapunovs_gated, 's--', color='green', label='Trained G-CML (Learned Gating, $\lambda_l \\approx 0.12$)', linewidth=2)
    plt.axhline(0.0, color='black', linestyle=':', label='Chaos Boundary ($\lambda_{Lyapunov} = 0$)')
    plt.xlabel('Layer Group Index ($l$)', fontsize=12)
    plt.ylabel('Lyapunov Exponent ($\lambda_{Lyapunov}$)', fontsize=12)
    plt.title('Lyapunov Exponents across Layers of G-CML', fontsize=14, fontweight='bold')
    plt.xticks(range(1, 15))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('results/lyapunov.png', dpi=300)
    plt.close()
    print("\nLyapunov exponents plot saved to results/lyapunov.png!")

if __name__ == '__main__':
    main()
