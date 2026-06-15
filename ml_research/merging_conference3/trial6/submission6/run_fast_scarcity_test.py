import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms

DEVICE = torch.device("cpu")
NUM_LAYERS = 14
NUM_TASKS = 4
D_FEAT = 192
NUM_CLASSES = 10
NUM_TRAIN = 300
NUM_TEST = 500
D_HIDDEN = 64
THETA_UNIFORM = np.log(1.0 / (NUM_TASKS - 1))

# Load data
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = dset.MNIST(root='./data', train=True, download=False, transform=transform)
mnist_test = dset.MNIST(root='./data', train=False, download=False, transform=transform)
fmnist_train = dset.FashionMNIST(root='./data', train=True, download=False, transform=transform)
fmnist_test = dset.FashionMNIST(root='./data', train=False, download=False, transform=transform)
cifar_train = dset.CIFAR10(root='./data', train=True, download=False, transform=transform)
cifar_test = dset.CIFAR10(root='./data', train=False, download=False, transform=transform)
svhn_train = dset.SVHN(root='./data', split='train', download=False, transform=transform)
svhn_test = dset.SVHN(root='./data', split='test', download=False, transform=transform)

class DatasetPool:
    def __init__(self, ds_list):
        self.ds_list = ds_list
        self.total_len = sum(len(d) for d in ds_list)
    def __len__(self): return self.total_len
    def __getitem__(self, idx):
        offset = 0
        for d in self.ds_list:
            if idx - offset < len(d):
                img, lbl = d[idx - offset]
                return img, lbl
            offset += len(d)
        raise IndexError()

GLOBAL_POOLS = {
    0: DatasetPool([mnist_train, mnist_test]),
    1: DatasetPool([fmnist_train, fmnist_test]),
    2: DatasetPool([cifar_train, cifar_test]),
    3: DatasetPool([svhn_train, svhn_test])
}

class RealWorldSandbox:
    def __init__(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.proj_matrices = {
            0: torch.randn(784, D_FEAT) / np.sqrt(D_FEAT),
            1: torch.randn(784, D_FEAT) / np.sqrt(D_FEAT),
            2: torch.randn(3072, D_FEAT) / np.sqrt(D_FEAT),
            3: torch.randn(3072, D_FEAT) / np.sqrt(D_FEAT)
        }
        self.permuted_indices = {k: torch.randperm(len(GLOBAL_POOLS[k])) for k in range(NUM_TASKS)}
        self.drawn_indices = {k: 0 for k in range(NUM_TASKS)}
        
    def generate_data(self, num_samples_per_task):
        features_list, labels_list = [], []
        for k in range(NUM_TASKS):
            imgs, lbls = [], []
            start_ptr = self.drawn_indices[k]
            for idx in range(start_ptr, start_ptr + num_samples_per_task):
                real_idx = self.permuted_indices[k][idx].item()
                img, lbl = GLOBAL_POOLS[k][real_idx]
                imgs.append(img.flatten())
                lbls.append(lbl)
            self.drawn_indices[k] += num_samples_per_task
            imgs = torch.stack(imgs)
            lbls = torch.tensor(lbls, dtype=torch.long)
            projected = imgs @ self.proj_matrices[k]
            projected = (projected - projected.mean(dim=-1, keepdim=True)) / (projected.std(dim=-1, keepdim=True) + 1e-6)
            features_list.append(projected)
            labels_list.append(lbls)
        return features_list, labels_list

class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_in = nn.Linear(D_FEAT, D_HIDDEN)
        self.mid_layers = nn.ModuleList([nn.Linear(D_HIDDEN, D_HIDDEN) for _ in range(12)])
        self.fc_out = nn.Linear(D_HIDDEN, NUM_CLASSES)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.fc_in(x))
        for layer in self.mid_layers:
            h = h + 0.1 * self.relu(layer(h))
        return self.fc_out(h)

def get_weights(model):
    W, B = [], []
    W.append(model.fc_in.weight.data.clone())
    B.append(model.fc_in.bias.data.clone())
    for layer in model.mid_layers:
        W.append(layer.weight.data.clone())
        B.append(layer.bias.data.clone())
    W.append(model.fc_out.weight.data.clone())
    B.append(model.fc_out.bias.data.clone())
    return W, B

def set_weights(model, W, B):
    model.fc_in.weight.data = W[0].clone()
    model.fc_in.bias.data = B[0].clone()
    for idx, layer in enumerate(model.mid_layers):
        layer.weight.data = W[idx+1].clone()
        layer.bias.data = B[idx+1].clone()
    model.fc_out.weight.data = W[13].clone()
    model.fc_out.bias.data = B[13].clone()

def forward_functional(x, W, B):
    h = F.relu(F.linear(x, W[0], B[0]))
    for l in range(1, 13):
        h = h + 0.1 * F.relu(F.linear(h, W[l], B[l]))
    return F.linear(h, W[13], B[13])

def train_classifier(model, features, labels, num_epochs=20, lr=0.01):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = criterion(model(features), labels)
        loss.backward()
        optimizer.step()

def compute_coefficients(theta, device):
    t = torch.linspace(0, 1, NUM_LAYERS, device=device)
    alpha = torch.zeros(NUM_LAYERS, NUM_TASKS, device=device)
    for k in range(NUM_TASKS):
        poly = theta[k, 0] + theta[k, 1] * t + theta[k, 2] * (t ** 2) + theta[k, 3] * (t ** 4)
        alpha[:, k] = torch.sigmoid(poly)
    return alpha

def evaluate_merged_model_functional(W, B, test_features_list, test_labels_list):
    accs = []
    for k in range(NUM_TASKS):
        out = forward_functional(test_features_list[k], W, B)
        preds = out.argmax(dim=-1)
        accs.append((preds == test_labels_list[k]).float().mean().item())
    return accs

SEEDS = [10, 11, 12, 13, 14]
CACHED_EXPERTS = {}

for seed in SEEDS:
    sandbox = RealWorldSandbox(seed)
    train_feats, train_labels = sandbox.generate_data(NUM_TRAIN)
    cal_feats, cal_labels = sandbox.generate_data(2)  # M = 2
    test_feats, test_labels = sandbox.generate_data(NUM_TEST)
    
    base_model = MLPClassifier().to(DEVICE)
    W_base, B_base = get_weights(base_model)
    W_experts, B_experts = [], []
    for k in range(NUM_TASKS):
        expert_model = MLPClassifier()
        set_weights(expert_model, W_base, B_base)
        train_classifier(expert_model, train_feats[k], train_labels[k], num_epochs=20, lr=0.01)
        W_exp, B_exp = get_weights(expert_model)
        W_experts.append(W_exp)
        B_experts.append(B_exp)
        
    V_W, V_B = [], []
    for l in range(NUM_LAYERS):
        V_W.append(torch.stack([W_experts[k][l] - W_base[l] for k in range(NUM_TASKS)]).to(DEVICE))
        V_B.append(torch.stack([B_experts[k][l] - B_base[l] for k in range(NUM_TASKS)]).to(DEVICE))
        
    CACHED_EXPERTS[seed] = {
        'W_base': W_base, 'B_base': B_base, 'V_W': V_W, 'V_B': V_B,
        'cal_feats': cal_feats, 'cal_labels': cal_labels,
        'test_feats': test_feats, 'test_labels': test_labels
    }

# Compute baselines for M = 2
print("\n--- Computing M = 2 Baselines ---")
uniform_accs = []
unconstrained_accs = []

for seed in SEEDS:
    exp = CACHED_EXPERTS[seed]
    W_base, B_base, V_W, V_B = exp['W_base'], exp['B_base'], exp['V_W'], exp['V_B']
    cal_feats, cal_labels = exp['cal_feats'], exp['cal_labels']
    test_feats, test_labels = exp['test_feats'], exp['test_labels']
    
    alpha_uniform = torch.full((NUM_LAYERS, NUM_TASKS), 0.25, device=DEVICE)
    W_uniform, B_uniform = [], []
    for l in range(NUM_LAYERS):
        w_l = W_base[l] + torch.sum(alpha_uniform[l][:, None, None] * V_W[l], dim=0)
        b_l = B_base[l] + torch.sum(alpha_uniform[l][:, None] * V_B[l], dim=0)
        W_uniform.append(w_l)
        B_uniform.append(b_l)
    uniform_accs.append(np.mean(evaluate_merged_model_functional(W_uniform, B_uniform, test_feats, test_labels)))
    
    gamma = torch.full((NUM_TASKS, NUM_LAYERS), THETA_UNIFORM, device=DEVICE, requires_grad=True)
    optimizer = optim.AdamW([gamma], lr=0.01)
    criterion = nn.CrossEntropyLoss()
    for step in range(50):
        optimizer.zero_grad()
        alpha = torch.sigmoid(gamma)
        W_m, B_m = [], []
        for l in range(NUM_LAYERS):
            a_l = alpha[:, l]
            W_m.append(W_base[l] + torch.sum(a_l[:, None, None] * V_W[l], dim=0))
            B_m.append(B_base[l] + torch.sum(a_l[:, None] * V_B[l], dim=0))
        total_loss = 0.0
        for k in range(NUM_TASKS):
            logits = forward_functional(cal_feats[k], W_m, B_m)
            total_loss += torch.clamp(criterion(logits, cal_labels[k]), max=5.0)
        total_loss /= NUM_TASKS
        total_loss.backward()
        optimizer.step()
    with torch.no_grad():
        alpha_opt = torch.sigmoid(gamma)
        W_opt, B_opt = [], []
        for l in range(NUM_LAYERS):
            a_l = alpha_opt[:, l]
            W_opt.append(W_base[l] + torch.sum(a_l[:, None, None] * V_W[l], dim=0))
            B_opt.append(B_base[l] + torch.sum(a_l[:, None] * V_B[l], dim=0))
        unconstrained_accs.append(np.mean(evaluate_merged_model_functional(W_opt, B_opt, test_feats, test_labels)))

print(f"Static Uniform Mean: {np.mean(uniform_accs)*100:.3f}%")
print(f"Offline Unconstrained Mean: {np.mean(unconstrained_accs)*100:.3f}%")

print("\nRunning M = 2 Sweep...")
for l_pac in [0.01, 0.04, 0.08, 0.12, 0.16, 0.20, 0.30, 0.40]:
    for damp in [0.1, 0.3, 0.5, 0.7, 1.0]:
        det_accs = []
        for seed in SEEDS:
            exp = CACHED_EXPERTS[seed]
            W_base, B_base, V_W, V_B = exp['W_base'], exp['B_base'], exp['V_W'], exp['V_B']
            cal_feats, cal_labels = exp['cal_feats'], exp['cal_labels']
            test_feats, test_labels = exp['test_feats'], exp['test_labels']
            
            # Fisher
            theta_init = torch.zeros(NUM_TASKS, 4, device=DEVICE)
            theta_init[:, 0] = THETA_UNIFORM
            theta_init.requires_grad = True
            criterion = nn.CrossEntropyLoss()
            fisher_diagonal = torch.zeros_like(theta_init)
            
            for k in range(NUM_TASKS):
                for i in range(cal_feats[k].size(0)):
                    x_i = cal_feats[k][i:i+1]
                    y_i = cal_labels[k][i:i+1]
                    alpha = compute_coefficients(theta_init, DEVICE)
                    W_m, B_m = [], []
                    for l in range(NUM_LAYERS):
                        a_l = alpha[l, :]
                        W_m.append(W_base[l] + torch.sum(a_l[:, None, None] * V_W[l], dim=0))
                        B_m.append(B_base[l] + torch.sum(a_l[:, None] * V_B[l], dim=0))
                    logits = forward_functional(x_i, W_m, B_m)
                    loss = criterion(logits, y_i)
                    grads = torch.autograd.grad(loss, theta_init, retain_graph=False)[0]
                    fisher_diagonal += grads ** 2
            fisher_diagonal = fisher_diagonal / (NUM_TASKS * cal_feats[0].size(0))
            fisher_diagonal = torch.clamp(fisher_diagonal, min=1e-5)
            fisher_diagonal = fisher_diagonal / fisher_diagonal.mean()
            
            # Apply damping
            fisher_diagonal = damp + (1.0 - damp) * fisher_diagonal
            
            theta_pac_fim = torch.zeros(NUM_TASKS, 4, device=DEVICE)
            theta_pac_fim[:, 0] = THETA_UNIFORM
            theta_pac_fim.requires_grad = True
            optimizer = optim.AdamW([theta_pac_fim], lr=0.01)
            
            for step in range(50):
                optimizer.zero_grad()
                ce_loss = 0.0
                for _ in range(3):
                    noise = torch.randn_like(theta_pac_fim) * 0.05
                    theta_sampled = theta_pac_fim + noise
                    alpha = compute_coefficients(theta_sampled, DEVICE)
                    W_m, B_m = [], []
                    for l in range(NUM_LAYERS):
                        a_l = alpha[l, :]
                        W_m.append(W_base[l] + torch.sum(a_l[:, None, None] * V_W[l], dim=0))
                        B_m.append(B_base[l] + torch.sum(a_l[:, None] * V_B[l], dim=0))
                    sample_ce = 0.0
                    for k in range(NUM_TASKS):
                        logits = forward_functional(cal_feats[k], W_m, B_m)
                        sample_ce += torch.clamp(criterion(logits, cal_labels[k]), max=5.0)
                    ce_loss += sample_ce / NUM_TASKS
                ce_loss /= 3
                reg_loss = 0.0
                for k in range(NUM_TASKS):
                    reg_loss += fisher_diagonal[k, 0] * (theta_pac_fim[k, 0] - THETA_UNIFORM)**2
                    reg_loss += fisher_diagonal[k, 1] * theta_pac_fim[k, 1]**2
                    reg_loss += fisher_diagonal[k, 2] * theta_pac_fim[k, 2]**2
                    reg_loss += fisher_diagonal[k, 3] * theta_pac_fim[k, 3]**2
                total_loss = ce_loss + l_pac * reg_loss
                total_loss.backward()
                optimizer.step()
                
            with torch.no_grad():
                alpha_opt = compute_coefficients(theta_pac_fim, DEVICE)
                W_opt, B_opt = [], []
                for l in range(NUM_LAYERS):
                    a_l = alpha_opt[l, :]
                    W_opt.append(W_base[l] + torch.sum(a_l[:, None, None] * V_W[l], dim=0))
                    B_opt.append(B_base[l] + torch.sum(a_l[:, None] * V_B[l], dim=0))
                det_accs.append(np.mean(evaluate_merged_model_functional(W_opt, B_opt, test_feats, test_labels)))
                
        print(f"l_pac={l_pac}, damp={damp} -> Mean Det: {np.mean(det_accs)*100:.3f}%")
