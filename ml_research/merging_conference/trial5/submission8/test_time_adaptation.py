import os
import random
import types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.func import functional_call
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Disable cuDNN to avoid initialization errors
torch.backends.cudnn.enabled = False

# Simplex Projection Helper
def project_simplex(v):
    sorted_v, _ = torch.sort(v, descending=True)
    cumsum_v = torch.cumsum(sorted_v, dim=0)
    indices = torch.arange(1, len(v) + 1, device=v.device, dtype=torch.long)
    res = sorted_v - (cumsum_v - 1.0) / indices.float()
    rho = torch.max(torch.where(res > 0, indices, torch.zeros_like(indices)))
    theta = (cumsum_v[rho - 1] - 1.0) / rho.float()
    return torch.clamp(v - theta, min=0)

# Helper to merge parameters differentiably for functional_call
def compute_merged_params(experts_list, lambdas, K):
    merged_params = {}
    for key in experts_list[0].keys():
        is_buffer = "running_mean" in key or "running_var" in key or "num_batches_tracked" in key
        if is_buffer:
            with torch.no_grad():
                merged_params[key] = sum(lambdas[k].detach() * experts_list[k][key] for k in range(K))
        else:
            merged_params[key] = sum(lambdas[k] * experts_list[k][key] for k in range(K))
    return merged_params

# Custom forward pass bound to standard ResNet18 to return logits and features
def custom_forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    feat = torch.flatten(x, 1)
    logits = self.fc(feat)
    return logits, feat

def get_bound_resnet(state_dict_path=None, device="cpu"):
    model = models.resnet18()
    model.fc = nn.Linear(512, 10)
    if state_dict_path and os.path.exists(state_dict_path):
        model.load_state_dict(torch.load(state_dict_path, map_location=device))
    model.forward = types.MethodType(custom_forward, model)
    return model

# Compute class prototypes using an expert model
def compute_prototypes(model, state_dict, loader, device, num_samples=300):
    model.eval()
    features_list = []
    labels_list = []
    count = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            logits, feats = functional_call(model, state_dict, inputs)
            features_list.append(feats.cpu())
            labels_list.append(labels)
            count += inputs.size(0)
            if count >= num_samples:
                break
                
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    prototypes = {}
    for c in range(10):
        mask = (labels == c)
        if mask.sum() > 0:
            c_feats = features[mask]
            c_feats_norm = F.normalize(c_feats, p=2, dim=1)
            mean_feat = c_feats_norm.mean(dim=0)
            prototypes[c] = F.normalize(mean_feat, p=2, dim=0)
        else:
            prototypes[c] = torch.zeros(512)
            
    return prototypes

def evaluate_test_stream():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")
    
    # Initialize standard ResNet-18 with custom bound forward
    model = get_bound_resnet(device=device).to(device)
    
    expert_cifar_path = "expert_cifar10.pth"
    expert_svhn_path = "expert_svhn.pth"
    expert_fmnist_path = "expert_fmnist.pth"
    
    if not (os.path.exists(expert_cifar_path) and os.path.exists(expert_svhn_path) and os.path.exists(expert_fmnist_path)):
        print("Error: Expert checkpoints are missing.")
        return
        
    expert_cifar = torch.load(expert_cifar_path, map_location=device)
    expert_svhn = torch.load(expert_svhn_path, map_location=device)
    expert_fmnist = torch.load(expert_fmnist_path, map_location=device)
    
    experts_list = [expert_cifar, expert_svhn, expert_fmnist]
    K = len(experts_list)
    
    # Datasets
    transform_color = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_fmnist = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    print("Loading test datasets...")
    test_cifar = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_color)
    test_svhn = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_color)
    test_fmnist = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_fmnist)
    
    batch_size = 64
    num_batches_per_block = 30
    subset_size = batch_size * num_batches_per_block
    
    loader_cifar = DataLoader(Subset(test_cifar, range(subset_size)), batch_size=batch_size, shuffle=False)
    loader_svhn = DataLoader(Subset(test_svhn, range(subset_size)), batch_size=batch_size, shuffle=False)
    loader_fmnist = DataLoader(Subset(test_fmnist, range(subset_size)), batch_size=batch_size, shuffle=False)
    
    print("Pre-computing class prototypes for CIFAR-10 and SVHN under the uniform model...")
    val_loader_cifar = DataLoader(Subset(test_cifar, range(subset_size, subset_size + 300)), batch_size=64, shuffle=False)
    val_loader_svhn = DataLoader(Subset(test_svhn, range(subset_size, subset_size + 300)), batch_size=64, shuffle=False)
    
    uniform_lambdas = torch.tensor([1.0/K] * K, device=device)
    uniform_params = compute_merged_params(experts_list, uniform_lambdas, K)
    
    prototypes_cifar = compute_prototypes(model, uniform_params, val_loader_cifar, device)
    prototypes_svhn = compute_prototypes(model, uniform_params, val_loader_svhn, device)
    
    # Compute global mean of pre-computed prototypes to center and normalize feature space (mitigate anisotropy)
    all_raw_protos = list(prototypes_cifar.values()) + list(prototypes_svhn.values())
    global_mean = torch.stack([p.to(device) for p in all_raw_protos]).mean(dim=0)
    print("Feature centering initialized. Global mean norm:", torch.norm(global_mean).item())
    
    # Center-and-normalize prototypes
    centered_prototypes_cifar = {c: F.normalize(p.to(device) - global_mean, p=2, dim=0) for c, p in prototypes_cifar.items()}
    centered_prototypes_svhn = {c: F.normalize(p.to(device) - global_mean, p=2, dim=0) for c, p in prototypes_svhn.items()}
    
    test_stream = []
    for inputs, labels in loader_cifar:
        test_stream.append((inputs, labels, 0))
    for inputs, labels in loader_svhn:
        test_stream.append((inputs, labels, 1))
    for inputs, labels in loader_fmnist:
        test_stream.append((inputs, labels, 2))
        
    print(f"Total batches in the evaluation stream: {len(test_stream)}")
    
    algorithms = ["Static", "TENT", "CPA-Merge", "PC-Merge", "PROTO-TTMM"]
    results = {alg: [] for alg in algorithms}
    
    for alg in algorithms:
        print(f"\nEvaluating Algorithm: {alg}")
        
        # Initialize lambdas
        lambdas = torch.tensor([1.0/K] * K, device=device, requires_grad=True)
        optimizer = optim.Adam([lambdas], lr=0.01)
        
        rolling_accuracies = []
        task_block_accuracies = {0: [], 1: [], 2: []}
        rolling_loss_history = []
        
        cpa_prototypes = {0: centered_prototypes_cifar, 1: centered_prototypes_svhn}
        
        proto_memory = {
            0: {c: p.clone() for c, p in centered_prototypes_cifar.items()},
            1: {c: p.clone() for c, p in centered_prototypes_svhn.items()}
        }
        
        for batch_idx, (inputs, labels, task_id) in enumerate(test_stream):
            inputs, labels = inputs.to(device), labels.to(device)
            
            if alg != "Static":
                # UNBIASED ROUTING: Temporarily compute features using uniform coefficients to avoid routing loops
                with torch.no_grad():
                    uniform_lambdas = torch.tensor([1.0/K] * K, device=device)
                    uniform_params = compute_merged_params(experts_list, uniform_lambdas, K)
                    _, routing_feats = functional_call(model, uniform_params, inputs)
                    centered_routing_feats = F.normalize(routing_feats - global_mean, p=2, dim=1)
                    mean_feat = F.normalize(centered_routing_feats.mean(dim=0), p=2, dim=0)
                
                # First-batch dynamic routing initialization to starting specialist to avoid uniform model noise
                if alg == "PROTO-TTMM" and batch_idx == 0:
                    with torch.no_grad():
                        mean_proto_0 = F.normalize(torch.stack([p for p in proto_memory[0].values()]).mean(dim=0), p=2, dim=0)
                        mean_proto_1 = F.normalize(torch.stack([p for p in proto_memory[1].values()]).mean(dim=0), p=2, dim=0)
                        sim_0 = torch.dot(mean_feat, mean_proto_0).item()
                        sim_1 = torch.dot(mean_feat, mean_proto_1).item()
                        initial_routed_task = 0 if sim_0 > sim_1 else 1
                        print(f"[PROTO-TTMM Init] Batch 0 - Centered Unbiased Sim to CIFAR: {sim_0:.4f}, Sim to SVHN: {sim_1:.4f} -> Routed Task: {initial_routed_task}")
                        if initial_routed_task == 0:
                            lambdas.data = torch.tensor([0.90, 0.05, 0.05], device=device)
                        else:
                            lambdas.data = torch.tensor([0.05, 0.90, 0.05], device=device)
                
                # Compute current adapted model's forward pass
                merged_params = compute_merged_params(experts_list, lambdas, K)
                logits, feats = functional_call(model, merged_params, inputs)
                probs = F.softmax(logits, dim=1)
                
                if alg == "TENT":
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
                    optimizer.zero_grad()
                    entropy.backward()
                    optimizer.step()
                    with torch.no_grad():
                        lambdas.data = project_simplex(lambdas.data)
                        
                elif alg == "CPA-Merge":
                    # 1. Routing using centered, UNBIASED task-level prototypes
                    mean_proto_0 = F.normalize(torch.stack([p for p in cpa_prototypes[0].values()]).mean(dim=0), p=2, dim=0)
                    mean_proto_1 = F.normalize(torch.stack([p for p in cpa_prototypes[1].values()]).mean(dim=0), p=2, dim=0)
                    
                    sim_0 = torch.dot(mean_feat, mean_proto_0).item()
                    sim_1 = torch.dot(mean_feat, mean_proto_1).item()
                    
                    routed_task_id = 0 if sim_0 > sim_1 else 1
                    
                    if batch_idx in [0, num_batches_per_block, 2*num_batches_per_block]:
                        print(f"[CPA-Merge Block Start] Batch {batch_idx} - Centered Unbiased Sim to CIFAR: {sim_0:.4f}, Sim to SVHN: {sim_1:.4f} -> Routed Task: {routed_task_id}")
                    
                    # 2. Hard Reset
                    with torch.no_grad():
                        if routed_task_id == 0:
                            lambdas.data = torch.tensor([0.90, 0.05, 0.05], device=device)
                        else:
                            lambdas.data = torch.tensor([0.05, 0.90, 0.05], device=device)
                    
                    # 3. Contrastive Alignment with centered prototypes
                    merged_params = compute_merged_params(experts_list, lambdas, K)
                    logits, feats = functional_call(model, merged_params, inputs)
                    probs = F.softmax(logits, dim=1)
                    centered_feats = F.normalize(feats - global_mean, p=2, dim=1)
                    
                    max_probs, preds = probs.max(dim=1)
                    conf_mask = max_probs > 0.7
                    
                    if conf_mask.sum() > 0:
                        conf_feats = centered_feats[conf_mask]
                        conf_preds = preds[conf_mask]
                        
                        loss = 0.0
                        for i_sample in range(conf_feats.size(0)):
                            pred_class = conf_preds[i_sample].item()
                            feat_i = conf_feats[i_sample]
                            
                            target_proto = cpa_prototypes[routed_task_id][pred_class]
                            pos_sim = torch.dot(feat_i, target_proto) / 0.1
                            all_proto_sims = torch.stack([torch.dot(feat_i, cpa_prototypes[routed_task_id][c]) for c in range(10)]) / 0.1
                            
                            loss += -torch.log(torch.exp(pos_sim) / torch.sum(torch.exp(all_proto_sims)) + 1e-8)
                            
                        loss = loss / conf_feats.size(0)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        with torch.no_grad():
                            lambdas.data = project_simplex(lambdas.data)
                            
                elif alg == "PC-Merge":
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
                    rolling_loss_history.append(entropy.item())
                    
                    if len(rolling_loss_history) > 5:
                        mean_prev = np.mean(rolling_loss_history[-6:-1])
                        current_loss = rolling_loss_history[-1]
                        if current_loss > 1.4 * mean_prev:
                            with torch.no_grad():
                                lambdas.data = torch.tensor([1.0/K]*K, device=device)
                            optimizer = optim.Adam([lambdas], lr=0.01)
                            merged_params = compute_merged_params(experts_list, lambdas, K)
                            logits, feats = functional_call(model, merged_params, inputs)
                            probs = F.softmax(logits, dim=1)
                            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
                    
                    optimizer.zero_grad()
                    entropy.backward()
                    optimizer.step()
                    with torch.no_grad():
                        lambdas.data = project_simplex(lambdas.data)
                        
                elif alg == "PROTO-TTMM":
                    if batch_idx == 0:
                        novelty_score = 0.0
                        cohesion_score = 0.0
                    else:
                        # 1. Compute Novelty Score in centered space using unbiased features
                        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean().item()
                        
                        # Sample-level maximum similarity to any active class prototype
                        sample_max_sims = []
                        for i_sample in range(centered_routing_feats.size(0)):
                            feat_i = centered_routing_feats[i_sample]
                            best_sim_i = -1.0
                            for task_k, task_protos in proto_memory.items():
                                for c, p in task_protos.items():
                                    sim_ic = torch.dot(feat_i, p.to(device)).item()
                                    best_sim_i = max(best_sim_i, sim_ic)
                            sample_max_sims.append(best_sim_i)
                        
                        mean_sample_max_sim = np.mean(sample_max_sims)
                        novelty_score = entropy * (1.0 - mean_sample_max_sim)
                        
                        # 2. Compute Cohesion Score in centered space using unbiased features
                        sim_matrix = torch.matmul(centered_routing_feats, centered_routing_feats.T)
                        cohesion_score = (sim_matrix.sum() - feats.size(0)) / (feats.size(0) * (feats.size(0) - 1))
                        cohesion_score = cohesion_score.item()
                    
                    # Centered thresholds
                    tau_novel = 1.10
                    tau_cohesion = 0.15
                    
                    # 3. Novel Task Discovery
                    if novelty_score > tau_novel and cohesion_score > tau_cohesion and len(proto_memory) < 3:
                        print(f"--- [Batch {batch_idx}] PROTO-TTMM: Novel Task Detected! (Novelty: {novelty_score:.4f}, Cohesion: {cohesion_score:.4f}) ---")
                        new_task_id = len(proto_memory)
                        proto_memory[new_task_id] = {}
                        
                        max_probs, preds = probs.max(dim=1)
                        for c in range(10):
                            c_mask = (preds == c) & (max_probs > 0.5)
                            if c_mask.sum() > 0:
                                c_mean = F.normalize(centered_routing_feats[c_mask].mean(dim=0), p=2, dim=0)
                                proto_memory[new_task_id][c] = c_mean.detach().clone()
                            else:
                                proto_memory[new_task_id][c] = mean_feat.detach().clone()
                                
                    # 4. Route to the best matched task
                    best_task_id = 0
                    best_sim = -1.0
                    for task_k, task_protos in proto_memory.items():
                        task_mean_proto = F.normalize(torch.stack([p for p in task_protos.values()]).mean(dim=0), p=2, dim=0)
                        sim_k = torch.dot(mean_feat, task_mean_proto).item()
                        if sim_k > best_sim:
                            best_sim = sim_k
                            best_task_id = task_k
                            
                    if batch_idx in [0, num_batches_per_block, 2*num_batches_per_block]:
                        sims_str = ", ".join([f"Sim {k}: {torch.dot(mean_feat, F.normalize(torch.stack([p.to(device) for p in task_protos.values()]).mean(dim=0), p=2, dim=0)).item():.4f}" for k, task_protos in proto_memory.items()])
                        print(f"[PROTO-TTMM Block Start] Batch {batch_idx} - {sims_str} - Novelty Score: {novelty_score:.4f}, Cohesion: {cohesion_score:.4f} -> Routed Task: {best_task_id}")
                    
                    # 5. Reset coefficients
                    with torch.no_grad():
                        if best_task_id == 0:
                            lambdas.data = torch.tensor([0.90, 0.05, 0.05], device=device)
                        elif best_task_id == 1:
                            lambdas.data = torch.tensor([0.05, 0.90, 0.05], device=device)
                        else:
                            lambdas.data = torch.tensor([0.05, 0.05, 0.90], device=device)
                            
                    # 6. Confidence-Masked Contrastive Alignment
                    merged_params = compute_merged_params(experts_list, lambdas, K)
                    logits, feats = functional_call(model, merged_params, inputs)
                    probs = F.softmax(logits, dim=1)
                    centered_feats = F.normalize(feats - global_mean, p=2, dim=1)
                    
                    max_probs, preds = probs.max(dim=1)
                    conf_mask = max_probs > 0.7
                    
                    if conf_mask.sum() > 0:
                        conf_feats = centered_feats[conf_mask]
                        conf_preds = preds[conf_mask]
                        
                        loss = 0.0
                        for i_sample in range(conf_feats.size(0)):
                            pred_class = conf_preds[i_sample].item()
                            feat_i = conf_feats[i_sample]
                            
                            target_proto = proto_memory[best_task_id][pred_class]
                            pos_sim = torch.dot(feat_i, target_proto) / 0.1
                            all_proto_sims = torch.stack([torch.dot(feat_i, proto_memory[best_task_id][c]) for c in range(10)]) / 0.1
                            
                            loss += -torch.log(torch.exp(pos_sim) / torch.sum(torch.exp(all_proto_sims)) + 1e-8)
                            
                        loss = loss / conf_feats.size(0)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        with torch.no_grad():
                            lambdas.data = project_simplex(lambdas.data)
                            
                        # 7. Prototype Refinement (EMA) only for newly discovered novel tasks to prevent representational drift of known tasks
                        if best_task_id >= 2:
                            with torch.no_grad():
                                for c in range(10):
                                    c_mask = (preds == c) & (max_probs > 0.8)
                                    if c_mask.sum() > 0:
                                        batch_c_mean = F.normalize(centered_routing_feats[c_mask].mean(dim=0), p=2, dim=0)
                                        proto_memory[best_task_id][c] = 0.9 * proto_memory[best_task_id][c] + 0.1 * batch_c_mean

            # --- Inference Evaluation ---
            with torch.no_grad():
                merged_params = compute_merged_params(experts_list, lambdas, K)
                logits, _ = functional_call(model, merged_params, inputs)
                _, preds = logits.max(dim=1)
                acc = (preds == labels).sum().item() / labels.size(0) * 100.0
                
            rolling_accuracies.append(acc)
            task_block_accuracies[task_id].append(acc)
            
            if (batch_idx+1) % 10 == 0:
                print(f"Batch [{batch_idx+1}/{len(test_stream)}] - Active Task: {task_id} - Accuracy: {acc:.2f}% - Lambda: {lambdas.detach().cpu().numpy()}")
                
        results[alg] = {
            "rolling": rolling_accuracies,
            "block_0": np.mean(task_block_accuracies[0]),
            "block_1": np.mean(task_block_accuracies[1]),
            "block_2": np.mean(task_block_accuracies[2]),
            "overall": np.mean(rolling_accuracies)
        }
        
    # Print markdown table
    print("\n" + "="*50)
    print("CONTINUAL TEST-TIME ADAPTATION RESULTS SUMMARY")
    print("="*50)
    print("| Method | Task A (CIFAR-10) | Task B (SVHN) | Task C (FMNIST - Novel) | Overall Accuracy |")
    print("|---|---|---|---|---|")
    for alg in algorithms:
        res = results[alg]
        print(f"| {alg} | {res['block_0']:.2f}% | {res['block_1']:.2f}% | {res['block_2']:.2f}% | {res['overall']:.2f}% |")
    print("="*50)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for alg in algorithms:
        plt.plot(results[alg]["rolling"], label=alg, linewidth=2)
    plt.axvline(x=num_batches_per_block, color='k', linestyle='--', alpha=0.5)
    plt.axvline(x=2*num_batches_per_block, color='k', linestyle='--', alpha=0.5)
    plt.text(num_batches_per_block/2, 95, "Task A\n(CIFAR-10)", fontsize=10, ha='center')
    plt.text(num_batches_per_block + num_batches_per_block/2, 95, "Task B\n(SVHN)", fontsize=10, ha='center')
    plt.text(2*num_batches_per_block + num_batches_per_block/2, 95, "Task C (Novel)\n(FashionMNIST)", fontsize=10, ha='center')
    plt.xlabel("Batch Index (Time step t)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Continual Test-Time Model Merging Adaptation Dynamics", fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc="lower left", fontsize=11)
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig("ctta_results.png", dpi=150)
    print("Saved plot to ctta_results.png")
    
    with open("progress.md", "a", encoding="utf-8") as f:
        f.write("\n## Phase 2: Experimentation & Results\n")
        f.write("\n### Experimental Evaluation of PROTO-TTMM on CTTA Benchmark (Robust Centered Run)\n")
        f.write("We evaluated 5 different Test-Time Model Merging algorithms on our newly proposed Continual Test-Time Adaptation (CTTA) benchmark.\n")
        f.write("The stream consists of 3 sequential blocks: CIFAR-10 (Task A), SVHN (Task B), and FashionMNIST (Task C - Novel task, for which no pre-computed prototypes are available to closed-world methods).\n\n")
        f.write("| Method | Task A (CIFAR-10) | Task B (SVHN) | Task C (FMNIST - Novel) | Overall Accuracy |\n")
        f.write("|---|---|---|---|---|\n")
        for alg in algorithms:
            res = results[alg]
            f.write(f"| {alg} | {res['block_0']:.2f}% | {res['block_1']:.2f}% | {res['block_2']:.2f}% | {res['overall']:.2f}% |\n")
        f.write("\n\n=== Phase 2 Complete ===\n")

if __name__ == "__main__":
    evaluate_test_stream()
