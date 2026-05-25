import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import numpy as np
import copy
import random
from torch.func import functional_call

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED errors on this cluster environment
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False
    print("Disabled cuDNN for stability.")

# ----------------------------------------------------
# 1. Dataset Preparation
# ----------------------------------------------------
def get_datasets():
    # Convert grayscale images to 3 channels and resize to 224x224 (ResNet-18 default)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading MNIST...")
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    print("Loading FashionMNIST...")
    fmnist_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fmnist_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    print("Loading KMNIST...")
    kmnist_train = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    kmnist_test = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)

    return {
        'mnist': (mnist_train, mnist_test),
        'fmnist': (fmnist_train, fmnist_test),
        'kmnist': (kmnist_train, kmnist_test)
    }

# ----------------------------------------------------
# 2. ResNet-18 Model Definition & Merging Helpers
# ----------------------------------------------------
class ResNet18Expert(nn.Module):
    def __init__(self):
        super(ResNet18Expert, self).__init__()
        # Load pre-trained ResNet-18 backbone
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_dim = self.resnet.fc.in_features
        # Extract encoder (everything except fc)
        self.encoder = nn.Sequential(*list(self.resnet.children())[:-1])
        # Task head is separate
        self.fc = nn.Linear(self.feature_dim, 10)

    def forward(self, x):
        features = self.encoder(x)
        features = torch.flatten(features, 1)
        out = self.fc(features)
        return out

def get_pretrained_base_encoder():
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    encoder = nn.Sequential(*list(resnet.children())[:-1])
    return encoder.to(device)

def get_task_vector(expert_encoder, base_encoder):
    task_vector = {}
    with torch.no_grad():
        for name, param in expert_encoder.named_parameters():
            base_param = base_encoder.state_dict()[name]
            task_vector[name] = param.clone() - base_param.clone()
    return task_vector

def reconstruct_merged_encoder(base_encoder, task_vectors, lambdas):
    # lambdas is a dict/list of weights for each expert
    merged_encoder = copy.deepcopy(base_encoder)
    merged_state_dict = merged_encoder.state_dict()
    
    with torch.no_grad():
        for name in merged_state_dict.keys():
            if name in task_vectors[0]:
                update = torch.zeros_like(merged_state_dict[name])
                for i, task_vec in enumerate(task_vectors):
                    update += lambdas[i] * task_vec[name].to(device)
                merged_state_dict[name] += update
            
    merged_encoder.load_state_dict(merged_state_dict)
    return merged_encoder

# ----------------------------------------------------
# 3. Training Experts (reused from existing checkpoints if available)
# ----------------------------------------------------
def train_expert(task_name, train_dataset, test_dataset, epochs=3, batch_size=128):
    print(f"\n--- Training Expert for {task_name.upper()} ---")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = ResNet18Expert().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    # Evaluate
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            test_total += y.size(0)
            test_correct += predicted.eq(y).sum().item()
            
    test_acc = (test_correct / test_total) * 100
    print(f"Standalone Test Accuracy for {task_name.upper()}: {test_acc:.2f}%")
    return model.encoder.to(device), model.fc.to(device), test_acc

# ----------------------------------------------------
# 4. Fisher Information Matrix Pre-computation (EWC)
# ----------------------------------------------------
def compute_diagonal_fisher(encoder, head, dataset, num_samples=200):
    print("Computing diagonal Fisher Information...")
    encoder.eval()
    head.eval()
    
    # We compute Fisher for the task classification head parameters (fc)
    # as they are the primary source of decision boundary drift.
    fisher = {}
    for name, param in head.named_parameters():
        fisher[name] = torch.zeros_like(param.data)
        
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    
    count = 0
    for x, y in loader:
        if count >= num_samples:
            break
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        with torch.enable_grad():
            features = encoder(x)
            features = torch.flatten(features, 1)
            outputs = head(features)
            loss = criterion(outputs, y)
            
            # Zero gradients
            head.zero_grad()
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in head.named_parameters():
                if param.grad is not None:
                    fisher[name] += (param.grad.data ** 2)
                    
        count += 1
        
    # Average across samples
    for name in fisher.keys():
        fisher[name] = fisher[name] / count
        # Add a tiny epsilon to avoid division by zero and stabilize
        fisher[name] += 1e-8
        
    print(f"Fisher computed over {count} samples.")
    return fisher

# ----------------------------------------------------
# 5. Reproducible Test Stream Generation
# ----------------------------------------------------
def generate_test_stream(test_datasets, stream_type, num_batches_per_task=50, batch_size=32, seed=42):
    # Set seed for reproducibility of stream generation
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # We use shuffle=True so that the test stream is a realistic, randomly ordered subset,
    # but since we set the seed right before, it will be 100% identical across all methods.
    loaders = {
        0: DataLoader(test_datasets[0], batch_size=batch_size, shuffle=True), # MNIST
        1: DataLoader(test_datasets[1], batch_size=batch_size, shuffle=True), # FMNIST
        2: DataLoader(test_datasets[2], batch_size=batch_size, shuffle=True)  # KMNIST
    }
    
    batches = []
    if stream_type == 'alternating':
        iters = [iter(loaders[i]) for i in range(3)]
        for step in range(num_batches_per_task): # 50 cycles of 3 tasks = 150 batches
            for task_id in range(3):
                try:
                    x, y = next(iters[task_id])
                    batches.append((task_id, x, y))
                except StopIteration:
                    iters[task_id] = iter(loaders[task_id])
                    x, y = next(iters[task_id])
                    batches.append((task_id, x, y))
    elif stream_type == 'sequential':
        iters = [iter(loaders[i]) for i in range(3)]
        for task_id in range(3):
            for step in range(num_batches_per_task): # 50 batches per task sequential
                try:
                    x, y = next(iters[task_id])
                    batches.append((task_id, x, y))
                except StopIteration:
                    iters[task_id] = iter(loaders[task_id])
                    x, y = next(iters[task_id])
                    batches.append((task_id, x, y))
                    
    return batches

# ----------------------------------------------------
# 6. Evaluation Protocols (Static and TTA Methods)
# ----------------------------------------------------
def evaluate_static_merged(base_encoder, task_vectors, expert_heads, batches):
    static_lambdas = torch.tensor([1/3, 1/3, 1/3], device=device)
    static_encoder = reconstruct_merged_encoder(base_encoder, task_vectors, static_lambdas)
    static_encoder.eval()
    
    correct = 0
    total = 0
    for task_id, x, y in batches:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            features = static_encoder(x)
            features = torch.flatten(features, 1)
            outputs = expert_heads[task_id](features)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    return (correct / total) * 100

def run_tta_evaluation(base_encoder, task_vectors, expert_encoders, expert_heads, batches, fisher_diags=None, reg_type='ewc', gamma=10.0, tta_lr_heads=1e-4, tta_lr_lambdas=0.05):
    # Initialize merging coefficients lambda to [1/3, 1/3, 1/3]
    lambdas = torch.tensor([1/3, 1/3, 1/3], requires_grad=True, device=device)
    
    # Create active/adapted classification heads (initialized from expert heads)
    adapted_heads = [copy.deepcopy(h).to(device) for h in expert_heads]
    
    # Persistent optimizers across batches so we don't reset optimizer states
    opt_heads = [optim.Adam(adapted_heads[i].parameters(), lr=tta_lr_heads) for i in range(3)]
    opt_lambdas = optim.Adam([lambdas], lr=tta_lr_lambdas)
    
    correct = 0
    total = 0
    
    for task_id, x, y in batches:
        x, y = x.to(device), y.to(device)
        
        # 1. Adapt on the current batch
        opt_heads[task_id].zero_grad()
        opt_lambdas.zero_grad()
        
        # Forward pass on expert to get target soft labels (extremely fast optimization using static expert_encoder)
        with torch.no_grad():
            expert_encoder = expert_encoders[task_id]
            expert_encoder.eval()
            expert_features = expert_encoder(x)
            expert_features = torch.flatten(expert_features, 1)
            expert_outputs = expert_heads[task_id](expert_features)
            p_expert = torch.softmax(expert_outputs, dim=1)
            
        # Reconstruct active merged model and get prediction differentiably
        merged_params = {}
        for name, param in base_encoder.named_parameters():
            if name in task_vectors[0]:
                update = torch.zeros_like(param)
                for i, task_vec in enumerate(task_vectors):
                    update = update + lambdas[i] * task_vec[name].to(device)
                merged_params[name] = base_encoder.state_dict()[name].to(device) + update
            else:
                merged_params[name] = param
                
        features = functional_call(base_encoder, merged_params, x)
        features = torch.flatten(features, 1)
        outputs = adapted_heads[task_id](features)
        p_merged = torch.softmax(outputs, dim=1)
        
        # KL loss (expert soft-labels to merged predictions)
        kl_loss = nn.KLDivLoss(reduction='batchmean')(torch.log(p_merged + 1e-8), p_expert)
        
        # Regularization penalty
        reg_loss = 0.0
        if reg_type == 'ewc' and fisher_diags is not None:
            for name, param in adapted_heads[task_id].named_parameters():
                init_param = expert_heads[task_id].state_dict()[name]
                fim = fisher_diags[task_id][name]
                reg_loss += torch.sum(fim * (param - init_param) ** 2)
        elif reg_type == 'l2':
            for name, param in adapted_heads[task_id].named_parameters():
                init_param = expert_heads[task_id].state_dict()[name]
                reg_loss += torch.sum((param - init_param) ** 2)
                
        total_loss = kl_loss + (gamma / 2.0) * reg_loss
        total_loss.backward()
        
        opt_heads[task_id].step()
        opt_lambdas.step()
        
        # Project lambdas to sum to 1 and be non-negative (standard in model merging)
        with torch.no_grad():
            lambdas.clamp_(0.0, 1.0)
            sum_l = lambdas.sum()
            if sum_l > 0:
                lambdas.div_(sum_l)
                
        # Evaluate on the batch AFTER adaptation
        with torch.no_grad():
            merged_params_eval = {}
            for name, param in base_encoder.named_parameters():
                if name in task_vectors[0]:
                    update = torch.zeros_like(param)
                    for i, task_vec in enumerate(task_vectors):
                        update = update + lambdas[i] * task_vec[name].to(device)
                    merged_params_eval[name] = base_encoder.state_dict()[name].to(device) + update
                else:
                    merged_params_eval[name] = param
            
            adapted_heads[task_id].eval()
            eval_features = functional_call(base_encoder, merged_params_eval, x)
            eval_features = torch.flatten(eval_features, 1)
            eval_outputs = adapted_heads[task_id](eval_features)
            _, predicted = eval_outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
    overall_acc = (correct / total) * 100
    return overall_acc

# ----------------------------------------------------
# 7. Main Runner Flow
# ----------------------------------------------------
if __name__ == "__main__":
    # Create directories
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./checkpoints', exist_ok=True)
    
    # Load Datasets
    data = get_datasets()
    
    tasks = ['mnist', 'fmnist', 'kmnist']
    expert_encoders = []
    expert_heads = []
    standalones = []
    
    # Load or train experts task-by-task to be resilient to interruptions
    for task in tasks:
        enc_path = f"./checkpoints/{task}_encoder.pt"
        head_path = f"./checkpoints/{task}_head.pt"
        if os.path.exists(enc_path) and os.path.exists(head_path):
            print(f"\nFound saved expert checkpoint for {task.upper()}! Loading...")
            resnet = models.resnet18()
            encoder = nn.Sequential(*list(resnet.children())[:-1])
            encoder.load_state_dict(torch.load(enc_path, map_location=device))
            encoder = encoder.to(device)
            
            head = nn.Linear(512, 10)
            head.load_state_dict(torch.load(head_path, map_location=device))
            head = head.to(device)
            
            expert_encoders.append(encoder)
            expert_heads.append(head)
            standalones.append(0.0)
        else:
            print(f"\nNo checkpoint found for {task.upper()}. Training from scratch...")
            train_set, test_set = data[task]
            enc, head, acc = train_expert(task, train_set, test_set, epochs=3, batch_size=128)
            expert_encoders.append(enc)
            expert_heads.append(head)
            standalones.append(acc)
            
            # Save checkpoints
            torch.save(enc.state_dict(), enc_path)
            torch.save(head.state_dict(), head_path)
            
    # Base Encoder
    base_encoder = get_pretrained_base_encoder()
    base_encoder.eval()
    
    # Freeze parameters to save memory and avoid in-place corruption during autograd backward pass
    base_encoder.requires_grad_(False)
    for enc in expert_encoders:
        enc.requires_grad_(False)
        enc.eval()
        
    # Task vectors
    task_vectors = []
    for i in range(3):
        t_vec = get_task_vector(expert_encoders[i], base_encoder)
        task_vectors.append(t_vec)
        
    # Pre-compute diagonal Fisher for expert heads
    fisher_diags = []
    for i, task in enumerate(tasks):
        train_set, _ = data[task]
        # Use a small subset (e.g. 200 samples) to compute FIM
        subset_indices = list(range(200))
        subset_train_set = Subset(train_set, subset_indices)
        fim = compute_diagonal_fisher(expert_encoders[i], expert_heads[i], subset_train_set, num_samples=200)
        fisher_diags.append(fim)
        
    # Test sets for the three tasks
    test_datasets = [data[tasks[i]][1] for i in range(3)]
    
    # Generate test streams once and reuse them for 100% fair and paired comparison
    all_results = {}
    gammas = [1.0, 10.0, 100.0]
    
    for stream in ['alternating', 'sequential']:
        print(f"\n=== Generating reproducible '{stream}' stream ===")
        batches = generate_test_stream(test_datasets, stream_type=stream, num_batches_per_task=50, batch_size=32, seed=42)
        
        # Evaluate Static Merged Baseline once
        static_acc = evaluate_static_merged(base_encoder, task_vectors, expert_heads, batches)
        print(f"[{stream.upper()}] Static Merged Accuracy: {static_acc:.2f}%")
        
        # Evaluate Standard unconstrained TTA (equivalent to gamma=0)
        std_acc = run_tta_evaluation(
            base_encoder=base_encoder,
            task_vectors=task_vectors,
            expert_encoders=expert_encoders,
            expert_heads=expert_heads,
            batches=batches,
            reg_type='none',
            gamma=0.0
        )
        print(f"[{stream.upper()}] Standard TTA (no regularization) Accuracy: {std_acc:.2f}%")
        
        stream_results = {
            'static': static_acc,
            'standard_tta': std_acc
        }
        
        # Sweep L2-TTA Baselines
        for gamma in gammas:
            l2_acc = run_tta_evaluation(
                base_encoder=base_encoder,
                task_vectors=task_vectors,
                expert_encoders=expert_encoders,
                expert_heads=expert_heads,
                batches=batches,
                reg_type='l2',
                gamma=gamma
            )
            print(f"[{stream.upper()}] L2-TTA (Gamma={gamma}) Accuracy: {l2_acc:.2f}%")
            stream_results[f'l2_tta_gamma_{gamma}'] = l2_acc
            
        # Sweep EWC-TTA (Ours)
        for gamma in gammas:
            ewc_acc = run_tta_evaluation(
                base_encoder=base_encoder,
                task_vectors=task_vectors,
                expert_encoders=expert_encoders,
                expert_heads=expert_heads,
                batches=batches,
                fisher_diags=fisher_diags,
                reg_type='ewc',
                gamma=gamma
            )
            print(f"[{stream.upper()}] EWC-TTA (Gamma={gamma}) Accuracy: {ewc_acc:.2f}%")
            stream_results[f'ewc_tta_gamma_{gamma}'] = ewc_acc
            
        all_results[stream] = stream_results
        
    # Print a summary table of results
    print("\n==========================================================================================================")
    print("EXPERIMENTAL SUMMARY RESULTS (SEED=42 REPRODUCIBLE STREAMS)")
    print("==========================================================================================================")
    print("Stream Type | Static | Std TTA | L2-TTA (G=1) | L2-TTA (G=10) | L2-TTA (G=100) | EWC-TTA (G=1) | EWC-TTA (G=10) | EWC-TTA (G=100)")
    print("----------------------------------------------------------------------------------------------------------")
    for stream in ['alternating', 'sequential']:
        res = all_results[stream]
        print(f"{stream:11} | {res['static']:5.2f}% | {res['standard_tta']:6.2f}% | {res['l2_tta_gamma_1.0']:11.2f}% | {res['l2_tta_gamma_10.0']:12.2f}% | {res['l2_tta_gamma_100.0']:13.2f}% | {res['ewc_tta_gamma_1.0']:12.2f}% | {res['ewc_tta_gamma_10.0']:13.2f}% | {res['ewc_tta_gamma_100.0']:14.2f}%")
    print("==========================================================================================================")
    
    # Save a comparison summary text to a file for easy reading
    with open("results_summary.txt", "w") as f:
        f.write("Stream Type | Static | Std TTA | L2-TTA (G=1) | L2-TTA (G=10) | L2-TTA (G=100) | EWC-TTA (G=1) | EWC-TTA (G=10) | EWC-TTA (G=100)\n")
        f.write("----------------------------------------------------------------------------------------------------------\n")
        for stream in ['alternating', 'sequential']:
            res = all_results[stream]
            f.write(f"{stream:11} | {res['static']:5.2f}% | {res['standard_tta']:6.2f}% | {res['l2_tta_gamma_1.0']:11.2f}% | {res['l2_tta_gamma_10.0']:12.2f}% | {res['l2_tta_gamma_100.0']:13.2f}% | {res['ewc_tta_gamma_1.0']:12.2f}% | {res['ewc_tta_gamma_10.0']:13.2f}% | {res['ewc_tta_gamma_100.0']:14.2f}%\n")
    print("Saved results to results_summary.txt")
