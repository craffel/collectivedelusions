import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from scipy.stats import pearsonr

# Set seed function
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# Sigmoid for retention mapping
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# 1. Dataset Loading Utility
def load_physical_data(num_train_per_task=1500):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load MNIST
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Load Fashion-MNIST
    fmnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    # Slice MNIST train
    mnist_train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(mnist_train, list(range(num_train_per_task))), batch_size=64, shuffle=True
    )
    
    # Slice Fashion-MNIST train
    fmnist_train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(fmnist_train, list(range(num_train_per_task))), batch_size=64, shuffle=True
    )
    
    return mnist_train_loader, mnist_test, fmnist_train_loader, fmnist_test

# 2. PyTorch Model with Shared Trunk, Outputs, and LoRA Adapters
class SharedTrunk(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.fc1(x))

class LoRA_Adapter(nn.Module):
    def __init__(self, hidden_dim=128, r=4):
        super().__init__()
        self.A = nn.Linear(hidden_dim, r, bias=False)
        self.B = nn.Linear(r, hidden_dim, bias=False)
        # Initialize
        nn.init.kaiming_uniform_(self.A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.B.weight)
    def forward(self, x):
        return self.B(self.A(x))

class PhysicalEnsembleNet(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, num_classes=10, r=4):
        super().__init__()
        self.trunk = SharedTrunk(input_dim, hidden_dim)
        
        self.fc2_base = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        
        self.lora1 = LoRA_Adapter(hidden_dim, r)
        self.lora2 = LoRA_Adapter(hidden_dim, r)
        
        self.head1 = nn.Linear(hidden_dim, num_classes)
        self.head2 = nn.Linear(hidden_dim, num_classes)
        
    def forward_trunk(self, x):
        return self.trunk(x)
        
    def forward_subsequent(self, h1, alphas):
        # h1 shape: (B, hidden_dim)
        # alphas shape: (B, 2)
        h2_base = self.fc2_base(h1)
        h2_lora1 = self.lora1(h1)
        h2_lora2 = self.lora2(h1)
        
        # Element-wise blending across batch
        # h2_lora1 is shape (B, 128), alphas[:, 0] is shape (B, 1)
        blended_lora = alphas[:, 0].unsqueeze(1) * h2_lora1 + alphas[:, 1].unsqueeze(1) * h2_lora2
        h2 = h2_base + blended_lora
        h2 = self.relu2(h2)
        
        # Task-specific logits blended by alphas
        logits1 = self.head1(h2)
        logits2 = self.head2(h2)
        logits = alphas[:, 0].unsqueeze(1) * logits1 + alphas[:, 1].unsqueeze(1) * logits2
        return h2, logits

# Router module matching PAC-Kinetics formulation
class PAC_Kinetics_Router(nn.Module):
    def __init__(self, num_tasks=2, sigma0_sq=5.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.sigma0_sq = sigma0_sq
        
        self.register_buffer("u0", torch.zeros(num_tasks))
        self.register_buffer("W0", torch.eye(num_tasks))
        self.register_buffer("w0", torch.ones(num_tasks) * np.log(0.05))
        
        self.u = nn.Parameter(torch.zeros(num_tasks))
        self.W = nn.Parameter(torch.eye(num_tasks))
        self.w = nn.Parameter(torch.ones(num_tasks) * np.log(0.05))

    def forward_stream(self, coords_stream):
        T = coords_stream.size(0)
        s = torch.zeros(self.num_tasks, device=coords_stream.device)
        alphas_list = []
        
        a = torch.sigmoid(self.u)
        tau = torch.exp(self.w) + 0.01  # enforce tau_min = 0.01
        
        for t in range(T):
            e_t = coords_stream[t]
            if t > 0:
                e_prev = coords_stream[t-1]
                num = torch.dot(e_t, e_prev)
                den = torch.norm(e_t) * torch.norm(e_prev) + 1e-8
                cos_sim = num / den
                homogeneity = torch.clamp(cos_sim, min=0.0)
                a_t = a * homogeneity
            else:
                a_t = a
                
            s = a_t * s + torch.mv(self.W, e_t)
            alpha_t = torch.softmax(s / tau, dim=0)
            alphas_list.append(alpha_t)
            
        return torch.stack(alphas_list)

    def compute_kl(self):
        kl_u = torch.sum((self.u - self.u0)**2) / (2.0 * self.sigma0_sq)
        kl_W = torch.sum((self.W - self.W0)**2) / (2.0 * self.sigma0_sq)
        kl_w = torch.sum((self.w - self.w0)**2) / (2.0 * self.sigma0_sq)
        return kl_u + kl_W + kl_w

def run_physical_experiments_for_seed(seed):
    set_seed(seed)
    
    # Load data
    mnist_train_loader, mnist_test, fmnist_train_loader, fmnist_test = load_physical_data()
    
    # Create Net
    net = PhysicalEnsembleNet(input_dim=784, hidden_dim=128, num_classes=10, r=4)
    criterion = nn.CrossEntropyLoss()
    
    # 2.1 Pretrain Shared Model (Trunk + fc2_base + head) with Oracle Routing
    print(f"[Seed {seed}] Phase 1: Pretraining shared parameters on combined MNIST/Fashion-MNIST subsets with Oracle Routing...")
    optimizer_pre = optim.Adam(net.parameters(), lr=0.01)
    
    # Combined dataset iterator
    mnist_iter = iter(mnist_train_loader)
    fmnist_iter = iter(fmnist_train_loader)
    
    for epoch in range(3):
        net.train()
        for i in range(15): # run more steps
            try:
                images_m, labels_m = next(mnist_iter)
            except StopIteration:
                mnist_iter = iter(mnist_train_loader)
                images_m, labels_m = next(mnist_iter)
                
            try:
                images_f, labels_f = next(fmnist_iter)
            except StopIteration:
                fmnist_iter = iter(fmnist_train_loader)
                images_f, labels_f = next(fmnist_iter)
                
            # Mix MNIST and Fashion-MNIST
            images = torch.cat([images_m, images_f], dim=0).view(-1, 784)
            labels = torch.cat([labels_m, labels_f], dim=0)
            
            optimizer_pre.zero_grad()
            h1 = net.forward_trunk(images)
            # Oracle routing during pretraining!
            alphas_pre = torch.zeros(images.size(0), 2)
            alphas_pre[:images_m.size(0), 0] = 1.0  # MNIST gets lora1
            alphas_pre[images_m.size(0):, 1] = 1.0  # Fashion-MNIST gets lora2
            
            _, logits = net.forward_subsequent(h1, alphas_pre)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer_pre.step()
            
    # Freeze Shared Parameters
    for param in net.trunk.parameters():
        param.requires_grad = False
    for param in net.fc2_base.parameters():
        param.requires_grad = False
    for param in net.head1.parameters():
        param.requires_grad = False
    for param in net.head2.parameters():
        param.requires_grad = False
        
    # 2.2 Fine-tune Expert 1 (lora1) on MNIST
    print(f"[Seed {seed}] Phase 2: Fine-tuning MNIST expert (LoRA 1)...")
    optimizer_lora1 = optim.Adam(net.lora1.parameters(), lr=0.01)
    net.lora1.train()
    for epoch in range(5):
        for images_m, labels_m in mnist_train_loader:
            images = images_m.view(-1, 784)
            optimizer_lora1.zero_grad()
            h1 = net.forward_trunk(images)
            # MNIST-only alpha: [1.0, 0.0]
            alphas_mnist = torch.zeros(images.size(0), 2)
            alphas_mnist[:, 0] = 1.0
            _, logits = net.forward_subsequent(h1, alphas_mnist)
            loss = criterion(logits, labels_m)
            loss.backward()
            optimizer_lora1.step()
            
    # 2.3 Fine-tune Expert 2 (lora2) on Fashion-MNIST
    print(f"[Seed {seed}] Phase 3: Fine-tuning Fashion-MNIST expert (LoRA 2)...")
    optimizer_lora2 = optim.Adam(net.lora2.parameters(), lr=0.01)
    net.lora2.train()
    for epoch in range(5):
        for images_f, labels_f in fmnist_train_loader:
            images = images_f.view(-1, 784)
            optimizer_lora2.zero_grad()
            h1 = net.forward_trunk(images)
            # Fashion-MNIST-only alpha: [0.0, 1.0]
            alphas_fmnist = torch.zeros(images.size(0), 2)
            alphas_fmnist[:, 1] = 1.0
            _, logits = net.forward_subsequent(h1, alphas_fmnist)
            loss = criterion(logits, labels_f)
            loss.backward()
            optimizer_lora2.step()
            
    # Put model in eval mode
    net.eval()
    
    # 2.4 Subspace Extraction Split C_sub (8 samples per task)
    print(f"[Seed {seed}] Phase 4: Constructing Subspace Energy Projections...")
    mnist_sub = [mnist_test[i] for i in range(8)]
    fmnist_sub = [fmnist_test[i] for i in range(8)]
    
    subspace_reps = {0: [], 1: []}
    with torch.no_grad():
        for img, _ in mnist_sub:
            h1 = net.forward_trunk(img.view(1, 784))
            subspace_reps[0].append(h1.squeeze(0).numpy())
        for img, _ in fmnist_sub:
            h1 = net.forward_trunk(img.view(1, 784))
            subspace_reps[1].append(h1.squeeze(0).numpy())
            
    # Compute PCA projection matrices V_k
    projection_matrices = []
    centroids_sub = []
    for k in [0, 1]:
        Z_k = np.array(subspace_reps[k])
        centroids_sub.append(np.mean(Z_k, axis=0))
        # Unit-norm normalization
        Z_k_norm = Z_k / (np.linalg.norm(Z_k, axis=1, keepdims=True) + 1e-8)
        # SVD
        U, S, Vh = np.linalg.svd(Z_k_norm, full_matrices=False)
        # Top 4 principal components
        V_k = Vh.T[:, :4]
        projection_matrices.append(V_k)
        
    # 2.5 Calibration Split C_opt (8 samples per task)
    print(f"[Seed {seed}] Phase 5: Calibration & Router Optimization...")
    mnist_opt = [mnist_test[i] for i in range(8, 16)]
    fmnist_opt = [fmnist_test[i] for i in range(8, 16)]
    
    opt_images = []
    opt_labels = []
    for idx in range(4):
        opt_images.append(mnist_opt[2 * idx][0])
        opt_labels.append(0)  # task 0
        opt_images.append(mnist_opt[2 * idx + 1][0])
        opt_labels.append(0)  # task 0
        opt_images.append(fmnist_opt[2 * idx][0])
        opt_labels.append(1)  # task 1
        opt_images.append(fmnist_opt[2 * idx + 1][0])
        opt_labels.append(1)  # task 1
        
    # Extract coordinates e_t
    opt_coords = []
    with torch.no_grad():
        for img in opt_images:
            h1 = net.forward_trunk(img.view(1, 784)).squeeze(0).numpy()
            tilde_z = h1 / (np.linalg.norm(h1) + 1e-8)
            e = [np.linalg.norm(np.matmul(projection_matrices[k].T, tilde_z)) for k in [0, 1]]
            opt_coords.append(e)
            
    opt_coords = torch.tensor(opt_coords, dtype=torch.float32)
    opt_labels_torch = torch.tensor(opt_labels, dtype=torch.long)
    
    # Train PAC-Kinetics Router
    pac_router = PAC_Kinetics_Router(num_tasks=2, sigma0_sq=5.0)
    optimizer_pac = optim.Adam(pac_router.parameters(), lr=0.01)
    
    for epoch in range(150):
        optimizer_pac.zero_grad()
        alphas = pac_router.forward_stream(opt_coords)
        individual_losses = -torch.log(alphas[range(len(opt_labels)), opt_labels] + 1e-8)
        losses_clamped = torch.clamp(individual_losses, max=5.0)
        loss_ce = torch.mean(losses_clamped)
        
        L_max = 5.0
        lam = 0.5
        kl = pac_router.compute_kl()
        delta = 0.05
        a = len(opt_labels) / 4.0
        bound = (L_max / (1.0 - np.exp(-lam))) * (1.0 - torch.exp(-lam * loss_ce / L_max - 2.0 * (kl + np.log(2.0 / delta)) / a))
        bound.backward()
        optimizer_pac.step()
        
    u_opt = pac_router.u.detach().numpy()
    W_opt = pac_router.W.detach().numpy()
    w_opt = pac_router.w.detach().numpy()
    a_opt = sigmoid(u_opt)
    tau_opt = np.exp(w_opt) + 0.01
    
    # 2.6 Test-time Evaluation Stream (100 MNIST and 100 Fashion-MNIST)
    print(f"[Seed {seed}] Phase 6: Serving Evaluations...")
    # Gather distinct test items
    test_mnist_items = [mnist_test[i] for i in range(16, 116)]
    test_fmnist_items = [fmnist_test[i] for i in range(16, 116)]
    
    test_data = []
    # Each item: (image, task_label, raw_class_label)
    for img, lbl in test_mnist_items:
        test_data.append((img, 0, lbl))
    for img, lbl in test_fmnist_items:
        test_data.append((img, 1, lbl))
        
    # Extract coordinates and trunk representations for the whole test set
    test_coords = []
    test_h1_reps = []
    test_raw_labels = []
    test_task_labels = []
    test_images = []
    with torch.no_grad():
        for img, task_lbl, raw_lbl in test_data:
            h1 = net.forward_trunk(img.view(1, 784)).squeeze(0).numpy()
            tilde_z = h1 / (np.linalg.norm(h1) + 1e-8)
            e = [np.linalg.norm(np.matmul(projection_matrices[k].T, tilde_z)) for k in [0, 1]]
            test_coords.append(e)
            test_h1_reps.append(h1)
            test_raw_labels.append(raw_lbl)
            test_task_labels.append(task_lbl)
            test_images.append(img)
            
    # 1. Homogeneous Stream (Sorted by task label)
    homo_indices = np.argsort(test_task_labels)
    homo_coords = [test_coords[i] for i in homo_indices]
    homo_h1 = [test_h1_reps[i] for i in homo_indices]
    homo_raw_lbls = [test_raw_labels[i] for i in homo_indices]
    homo_task_lbls = [test_task_labels[i] for i in homo_indices]
    homo_imgs = [test_images[i] for i in homo_indices]
    
    # 2. Heterogeneous Stream (Completely mixed/interleaved)
    hetero_indices = list(range(len(test_task_labels)))
    rng = np.random.default_rng(seed + 999)
    rng.shuffle(hetero_indices)
    hetero_coords = [test_coords[i] for i in hetero_indices]
    hetero_h1 = [test_h1_reps[i] for i in hetero_indices]
    hetero_raw_lbls = [test_raw_labels[i] for i in hetero_indices]
    hetero_task_lbls = [test_task_labels[i] for i in hetero_indices]
    hetero_imgs = [test_images[i] for i in hetero_indices]
    
    # Soft accuracy scaling factor
    lambda_scale = 0.05
    
    methods_list = ["oracle", "uniform", "sable_raw", "pac_zca", "chemmerge", "pac_kinetics", "pac_kinetics_rand"]
    
    seed_results = {}
    
    for stream_name, stream_coords, stream_h1, stream_raw_lbls, stream_task_lbls, stream_imgs in [
        ("homo", homo_coords, homo_h1, homo_raw_lbls, homo_task_lbls, homo_imgs),
        ("hetero", hetero_coords, hetero_h1, hetero_raw_lbls, hetero_task_lbls, hetero_imgs)
    ]:
        T_eval = len(stream_task_lbls)
        
        # Initialize trackers
        rep_acc_sum = {m: 0.0 for m in methods_list}
        cls_acc_sum = {m: 0.0 for m in methods_list}
        alphas_history = {m: [] for m in methods_list}
        
        # Draw 10 randomized parameter sets for pac_kinetics_rand
        rand_params_stream = []
        std = np.sqrt(pac_router.sigma0_sq)
        rng_param = np.random.default_rng(seed + 1234)
        for _ in range(10):
            u_r = u_opt + rng_param.normal(0.0, std, size=u_opt.shape)
            W_r = W_opt + rng_param.normal(0.0, std, size=W_opt.shape)
            w_r = w_opt + rng_param.normal(0.0, std, size=w_opt.shape)
            a_r = sigmoid(u_r)
            tau_r = np.exp(w_r) + 0.01
            rand_params_stream.append((a_r, W_r, tau_r))
            
        s_kin_rand = [np.zeros(2) for _ in range(10)]
        alphas_history_rand = [[] for _ in range(10)]
        rep_acc_sum_rand = 0.0
        cls_acc_sum_rand = 0.0
        
        # We need to trace step-by-step for dynamic routers
        # Routers initial states
        s_chem = np.zeros(2)
        s_kin = np.zeros(2)
        
        # Precompute centroids of tasks in trunk representation space for soft representation accuracy
        c0 = centroids_sub[0]
        c1 = centroids_sub[1]
        
        # For correlation calculation, we collect paired proxy representation alignment accuracy and task classification success
        all_rep_accs = []
        all_cls_successes = []
        
        for t in range(T_eval):
            e_t = stream_coords[t]
            h1_t = stream_h1[t]
            true_task = stream_task_lbls[t]
            true_raw_class = stream_raw_lbls[t]
            img_t = stream_imgs[t]
            
            # Compute alphas for each method
            alphas = {}
            # Oracle: perfect task selection
            alphas["oracle"] = np.zeros(2)
            alphas["oracle"][true_task] = 1.0
            
            # Uniform: 0.5 each
            alphas["uniform"] = np.ones(2) * 0.5
            
            # SABLE Raw
            sum_e = np.sum(e_t) + 1e-8
            alphas["sable_raw"] = np.array(e_t) / sum_e
            
            # PAC-ZCA (stateless, optimized tau)
            tau_zca = 0.05 # standard zca temperature
            exp_zca = np.exp(np.array(e_t) / tau_zca)
            alphas["pac_zca"] = exp_zca / np.sum(exp_zca)
            
            # Heuristic ChemMerge (stateful, a=0.5, W=I)
            s_chem = 0.5 * s_chem + np.array(e_t)
            alphas["chemmerge"] = s_chem / (np.sum(s_chem) + 1e-8)
            
            # PAC-Kinetics (stateful, optimized with adaptive online kinetics)
            if t > 0:
                e_prev = np.array(stream_coords[t-1])
                num = np.dot(np.array(e_t), e_prev)
                den = np.linalg.norm(e_t) * np.linalg.norm(e_prev) + 1e-8
                cos_sim = num / den
                homogeneity = np.maximum(0.0, cos_sim)
                a_t = a_opt * homogeneity
            else:
                a_t = a_opt
                
            s_kin = a_t * s_kin + np.matmul(W_opt, np.array(e_t))
            exp_kin = np.exp(s_kin / tau_opt)
            alphas["pac_kinetics"] = exp_kin / np.sum(exp_kin)
            
            # Evaluate each method (except randomized which is handled separately)
            for m in methods_list:
                if m == "pac_kinetics_rand":
                    continue
                alpha_m = alphas[m]
                alphas_history[m].append(alpha_m)
                
                # Compute physical blended prediction and h2 activation
                alpha_tensor = torch.tensor(alpha_m, dtype=torch.float32).view(1, 2)
                h1_tensor = torch.tensor(h1_t, dtype=torch.float32).view(1, 128)
                with torch.no_grad():
                    h2, logits = net.forward_subsequent(h1_tensor, alpha_tensor)
                    pred_class = torch.argmax(logits, dim=1).item()
                    h2_np = h2.squeeze(0).numpy()
                    
                # Downstream classification accuracy
                is_correct = 1.0 if pred_class == true_raw_class else 0.0
                cls_acc_sum[m] += is_correct
                
                # Representation Alignment Accuracy
                # Target representation: centroid of the true task
                v_target = c0 if true_task == 0 else c1
                # Compute blending error in Layer 2 space
                # Ideal h2 under oracle
                with torch.no_grad():
                    oracle_alpha = torch.tensor([1.0, 0.0] if true_task == 0 else [0.0, 1.0]).view(1, 2)
                    h2_ideal, _ = net.forward_subsequent(h1_tensor, oracle_alpha)
                    h2_ideal_np = h2_ideal.squeeze(0).numpy()
                    
                rep_err_sq = np.sum((h2_np - h2_ideal_np) ** 2)
                rep_acc = np.exp(-lambda_scale * rep_err_sq)
                rep_acc_sum[m] += rep_acc
                
                if m == "pac_kinetics":
                    all_rep_accs.append(rep_acc)
                    all_cls_successes.append(is_correct)
                    
            # --- 7. Randomized PAC-Kinetics ---
            for i in range(10):
                a_r, W_r, tau_r = rand_params_stream[i]
                if t > 0:
                    e_prev_np = np.array(stream_coords[t-1])
                    num = np.dot(np.array(e_t), e_prev_np)
                    den = np.linalg.norm(e_t) * np.linalg.norm(e_prev_np) + 1e-8
                    cos_sim = num / den
                    homogeneity = np.maximum(0.0, cos_sim)
                    a_t_r = a_r * homogeneity
                else:
                    a_t_r = a_r
                    
                s_kin_rand[i] = a_t_r * s_kin_rand[i] + np.matmul(W_r, np.array(e_t))
                logits_r = s_kin_rand[i] / tau_r
                logits_r_stable = logits_r - np.max(logits_r)
                alphas_r = np.exp(logits_r_stable) / np.sum(np.exp(logits_r_stable))
                alphas_history_rand[i].append(alphas_r)
                
                # Evaluate
                alpha_tensor = torch.tensor(alphas_r, dtype=torch.float32).view(1, 2)
                h1_tensor = torch.tensor(h1_t, dtype=torch.float32).view(1, 128)
                with torch.no_grad():
                    h2, logits = net.forward_subsequent(h1_tensor, alpha_tensor)
                    pred_class = torch.argmax(logits, dim=1).item()
                    h2_np = h2.squeeze(0).numpy()
                    
                is_correct = 1.0 if pred_class == true_raw_class else 0.0
                cls_acc_sum_rand += is_correct
                
                with torch.no_grad():
                    oracle_alpha = torch.tensor([1.0, 0.0] if true_task == 0 else [0.0, 1.0]).view(1, 2)
                    h2_ideal, _ = net.forward_subsequent(h1_tensor, oracle_alpha)
                    h2_ideal_np = h2_ideal.squeeze(0).numpy()
                    
                rep_err_sq = np.sum((h2_np - h2_ideal_np) ** 2)
                rep_acc = np.exp(-lambda_scale * rep_err_sq)
                rep_acc_sum_rand += rep_acc
                
            mean_alpha_r = np.mean([alphas_history_rand[i][-1] for i in range(10)], axis=0)
            alphas_history["pac_kinetics_rand"].append(mean_alpha_r)
                    
        # Compute metrics
        stream_results = {}
        # Compute routing jitter of 10 randomized runs
        jitters_r = []
        for i in range(10):
            history_r = np.array(alphas_history_rand[i])
            diffs_r = np.abs(history_r[1:] - history_r[:-1])
            jit_r = np.mean(np.sum(diffs_r, axis=1))
            jitters_r.append(jit_r)
            
        for m in methods_list:
            if m == "pac_kinetics_rand":
                acc_rep = (rep_acc_sum_rand / (T_eval * 10.0)) * 100.0
                acc_cls = (cls_acc_sum_rand / (T_eval * 10.0)) * 100.0
                jitter = np.mean(jitters_r)
            else:
                acc_rep = (rep_acc_sum[m] / T_eval) * 100.0
                acc_cls = (cls_acc_sum[m] / T_eval) * 100.0
                # Jitter
                history = np.array(alphas_history[m])
                diffs = np.abs(history[1:] - history[:-1])
                jitter = np.mean(np.sum(diffs, axis=1))
            
            stream_results[m] = {
                "rep_acc": acc_rep,
                "cls_acc": acc_cls,
                "jitter": jitter
            }
            
        # Pearson correlation
        p_corr, _ = pearsonr(all_rep_accs, all_cls_successes)
        stream_results["pearson_correlation"] = p_corr
        
        seed_results[stream_name] = stream_results
        
    return seed_results

# 3. Main runner
if __name__ == "__main__":
    print("=========================================================================")
    print("STARTING PHYSICAL MODEL VALIDATION (MNIST & FASHION-MNIST ON REAL MLP)")
    print("=========================================================================")
    
    seeds = [101, 102, 103, 104, 105]
    all_seed_results = []
    
    for s in seeds:
        res = run_physical_experiments_for_seed(s)
        all_seed_results.append(res)
        
    # Aggregate results over seeds
    methods_list = ["oracle", "uniform", "sable_raw", "pac_zca", "chemmerge", "pac_kinetics", "pac_kinetics_rand"]
    
    aggregated = {"homo": {}, "hetero": {}}
    for s_name in ["homo", "hetero"]:
        for m in methods_list:
            rep_accs = [r[s_name][m]["rep_acc"] for r in all_seed_results]
            cls_accs = [r[s_name][m]["cls_acc"] for r in all_seed_results]
            jitters = [r[s_name][m]["jitter"] for r in all_seed_results]
            
            aggregated[s_name][m] = {
                "rep_acc_mean": np.mean(rep_accs), "rep_acc_std": np.std(rep_accs),
                "cls_acc_mean": np.mean(cls_accs), "cls_acc_std": np.std(cls_accs),
                "jitter_mean": np.mean(jitters), "jitter_std": np.std(jitters)
            }
            
        corrs = [r[s_name]["pearson_correlation"] for r in all_seed_results]
        aggregated[s_name]["pearson_correlation_mean"] = np.mean(corrs)
        aggregated[s_name]["pearson_correlation_std"] = np.std(corrs)
        
    # Write physical results md file
    with open("physical_results.md", "w") as f:
        f.write("# Physical Evaluation Results: MNIST & Fashion-MNIST on PyTorch MLP\n\n")
        f.write("This file documents the physical evaluation of PAC-Kinetics on real-world datasets using deep LoRA-blended neural networks. ")
        f.write("We pre-train a shared trunk MLP, freeze it, and fine-tune two active LoRA-style adapters on MNIST and Fashion-MNIST subsets. ")
        f.write("Then, we evaluate our stateful PAC-Kinetics router against 5 standard baselines over sequential query streams.\n\n")
        
        for s_name in ["homo", "hetero"]:
            f.write(f"## {s_name.capitalize()} Stream serving (5 seeds)\n\n")
            f.write("| Method | Representation Alignment Acc. (%) | Actual Classification Acc. (%) | Routing Jitter |\n")
            f.write("| :--- | :---: | :---: | :---: |\n")
            for m in methods_list:
                rep_mean = aggregated[s_name][m]['rep_acc_mean']
                rep_std = aggregated[s_name][m]['rep_acc_std']
                cls_mean = aggregated[s_name][m]['cls_acc_mean']
                cls_std = aggregated[s_name][m]['cls_acc_std']
                jit_mean = aggregated[s_name][m]['jitter_mean']
                jit_std = aggregated[s_name][m]['jitter_std']
                f.write(f"| {m} | {rep_mean:.2f}% &plusmn; {rep_std:.2f}% | {cls_mean:.2f}% &plusmn; {cls_std:.2f}% | {jit_mean:.4f} &plusmn; {jit_std:.4f} |\n")
            
            corr_mean = aggregated[s_name]['pearson_correlation_mean']
            corr_std = aggregated[s_name]['pearson_correlation_std']
            f.write(f"\n**Pearson correlation coefficient** between intermediate representation blending error and final downstream classification accuracy success: **{corr_mean:.4f} &plusmn; {corr_std:.4f}**\n\n")
            
    print("=========================================================================")
    print("PHYSICAL MODEL VALIDATION COMPLETE. RESULTS SAVED TO physical_results.md")
    print("=========================================================================")
