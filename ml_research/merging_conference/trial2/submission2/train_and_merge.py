import os
import argparse
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import timm

# Disable cuDNN to prevent cluster compatibility issues
torch.backends.cudnn.enabled = False

# --- SAM Optimizer ---
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (p.grad * scale).to(p)
                p.add_(e_w)  # perturb weights
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # restore weights
        self.base_optimizer.step()  # actual step
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def step(self, closure=None):
        raise NotImplementedError("SAM requires first_step and second_step")

# --- Deterministic Orthogonal Matrix Generator ---
def get_orthogonal_matrix(d_out, R, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    W = torch.randn(d_out, R, generator=g)
    Q, _ = torch.linalg.qr(W)
    return Q

# --- LoraWrapper ---
class LoraWrapper(nn.Module):
    def __init__(self, original_module, r=8, lora_alpha=16, num_tasks=2, so_lora=False, orthogonal_matrix=None):
        super().__init__()
        self.original_module = original_module
        # Freeze base module
        for p in self.original_module.parameters():
            p.requires_grad = False
            
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.num_tasks = num_tasks
        self.so_lora = so_lora
        
        # Determine shapes
        if isinstance(original_module, nn.Linear):
            self.layer_type = "linear"
            self.d_out, self.d_in = original_module.weight.shape
        elif isinstance(original_module, nn.Conv2d):
            self.layer_type = "conv2d"
            self.d_out = original_module.out_channels
            self.d_in = original_module.in_channels
            self.kernel_size = original_module.kernel_size
        else:
            raise ValueError("Unsupported module type")
            
        # Define parameters for each task
        self.lora_A = nn.ParameterList()
        if so_lora:
            self.register_buffer("P", orthogonal_matrix)
            self.lora_B_tilde = nn.ParameterList()
            for _ in range(num_tasks):
                self.lora_A.append(nn.Parameter(torch.zeros(r, self.d_in) if self.layer_type == "linear" else nn.Parameter(torch.zeros(r, self.d_in // original_module.groups, *self.kernel_size))))
                self.lora_B_tilde.append(nn.Parameter(torch.zeros(r, r)))
        else:
            self.lora_B = nn.ParameterList()
            for _ in range(num_tasks):
                self.lora_A.append(nn.Parameter(torch.zeros(r, self.d_in) if self.layer_type == "linear" else nn.Parameter(torch.zeros(r, self.d_in // original_module.groups, *self.kernel_size))))
                self.lora_B.append(nn.Parameter(torch.zeros(self.d_out, r)))
                
        self.reset_parameters()
        
        # Modes: 'train_task' or 'merged'
        self.mode = 'merged'
        self.active_task_id = 0
        self.merging_coefficients = [1.0] * num_tasks

    def reset_parameters(self):
        for i in range(self.num_tasks):
            nn.init.kaiming_uniform_(self.lora_A[i], a=math.sqrt(5))
            if self.so_lora:
                nn.init.zeros_(self.lora_B_tilde[i])
            else:
                nn.init.zeros_(self.lora_B[i])

    def forward(self, x):
        out = self.original_module(x)
        
        if self.mode == 'train_task':
            tid = self.active_task_id
            if self.layer_type == "linear":
                a_out = torch.matmul(x, self.lora_A[tid].t())
                if self.so_lora:
                    P_slice = self.P[:, tid * self.r : (tid + 1) * self.r]
                    B = torch.matmul(P_slice, self.lora_B_tilde[tid])
                else:
                    B = self.lora_B[tid]
                lora_out = torch.matmul(a_out, B.t()) * self.scaling
            else: # conv2d
                a_out = F.conv2d(
                    x, self.lora_A[tid], bias=None,
                    stride=self.original_module.stride,
                    padding=self.original_module.padding,
                    dilation=self.original_module.dilation,
                    groups=self.original_module.groups
                )
                if self.so_lora:
                    P_slice = self.P[:, tid * self.r : (tid + 1) * self.r]
                    B = torch.matmul(P_slice, self.lora_B_tilde[tid])
                else:
                    B = self.lora_B[tid]
                B = B.unsqueeze(-1).unsqueeze(-1)
                lora_out = F.conv2d(a_out, B, bias=None) * self.scaling
            return out + lora_out
            
        elif self.mode == 'merged':
            total_lora_out = 0.0
            for tid in range(self.num_tasks):
                coeff = self.merging_coefficients[tid]
                if coeff == 0.0:
                    continue
                if self.layer_type == "linear":
                    a_out = torch.matmul(x, self.lora_A[tid].t())
                    if self.so_lora:
                        P_slice = self.P[:, tid * self.r : (tid + 1) * self.r]
                        B = torch.matmul(P_slice, self.lora_B_tilde[tid])
                    else:
                        B = self.lora_B[tid]
                    lora_out = torch.matmul(a_out, B.t()) * (self.scaling * coeff)
                else: # conv2d
                    a_out = F.conv2d(
                        x, self.lora_A[tid], bias=None,
                        stride=self.original_module.stride,
                        padding=self.original_module.padding,
                        dilation=self.original_module.dilation,
                        groups=self.original_module.groups
                    )
                    if self.so_lora:
                        P_slice = self.P[:, tid * self.r : (tid + 1) * self.r]
                        B = torch.matmul(P_slice, self.lora_B_tilde[tid])
                    else:
                        B = self.lora_B[tid]
                    B = B.unsqueeze(-1).unsqueeze(-1)
                    lora_out = F.conv2d(a_out, B, bias=None) * (self.scaling * coeff)
                total_lora_out += lora_out
            return out + total_lora_out
        else:
            return out

# --- Helper functions to traverse and wrap model ---
def get_submodule(model, target_name):
    if target_name == '':
        return model
    parts = target_name.split('.')
    curr = model
    for part in parts:
        curr = getattr(curr, part)
    return curr

def set_submodule(model, target_name, new_module):
    parts = target_name.split('.')
    curr = model
    for part in parts[:-1]:
        curr = getattr(curr, part)
    setattr(curr, parts[-1], new_module)

def wrap_model_with_lora(model, r=8, lora_alpha=16, num_tasks=2, so_lora=False, seed=42):
    for name, module in list(model.named_modules()):
        if name == 'conv1' or name == 'fc' or name == '':
            continue
        if len(list(module.children())) > 0:
            continue
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            orthogonal_matrix = None
            if so_lora:
                if isinstance(module, nn.Linear):
                    d_out = module.weight.shape[0]
                else:
                    d_out = module.out_channels
                orthogonal_matrix = get_orthogonal_matrix(d_out, num_tasks * r, seed=seed)
            
            wrapper = LoraWrapper(module, r=r, lora_alpha=lora_alpha, num_tasks=num_tasks, so_lora=so_lora, orthogonal_matrix=orthogonal_matrix)
            set_submodule(model, name, wrapper)

# --- Set Active Task Mode ---
def set_active_task_mode(model, task_id):
    for module in model.modules():
        if isinstance(module, LoraWrapper):
            module.mode = 'train_task'
            module.active_task_id = task_id

def set_merged_mode(model, coefficients):
    for module in model.modules():
        if isinstance(module, LoraWrapper):
            module.mode = 'merged'
            module.merging_coefficients = coefficients

# --- Main Training & Evaluation ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["standard", "sam", "so_lora", "so_lora_sam"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--rho", type=float, default=0.05)
    args = parser.parse_args()

    print(f"==================================================")
    print(f"Running Experiment in Mode: {args.mode.upper()}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Rank: {args.r}")
    print(f"==================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # CIFAR-10 Transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Datasets
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=test_transform)

    # Split Indices
    task1_train_idx = [i for i, label in enumerate(train_dataset.targets) if label in [0, 1, 2, 3, 4]]
    task2_train_idx = [i for i, label in enumerate(train_dataset.targets) if label in [5, 6, 7, 8, 9]]
    task1_test_idx = [i for i, label in enumerate(test_dataset.targets) if label in [0, 1, 2, 3, 4]]
    task2_test_idx = [i for i, label in enumerate(test_dataset.targets) if label in [5, 6, 7, 8, 9]]

    # Loaders
    task1_train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dataset, task1_train_idx), batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    task2_train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dataset, task2_train_idx), batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    task1_test_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_dataset, task1_test_idx), batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    task2_test_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_dataset, task2_test_idx), batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # Task Configuration
    so_lora = "so_lora" in args.mode
    use_sam = "sam" in args.mode

    suffix = f"_r{args.r}" if args.r != 8 else ""

    # Train Task 1
    task1_checkpoint = f"task1_{args.mode}{suffix}.pt"
    if os.path.exists(task1_checkpoint):
        print(f"\nFound checkpoint for Task 1: {task1_checkpoint}. Skipping training.")
    else:
        print("\n--- Training Task 1 (Classes 0-4) ---")
        model_t1 = timm.create_model('resnet18', pretrained=True)
        model_t1.fc = nn.Linear(512, 10)
        wrap_model_with_lora(model_t1, r=args.r, lora_alpha=args.lora_alpha, num_tasks=2, so_lora=so_lora)
        model_t1 = model_t1.to(device)

        # Freeze base model, ensure only active task parameters and classification head are updated
        set_active_task_mode(model_t1, task_id=0)
        
        # Verify trainable params
        trainable_params_t1 = []
        for name, param in model_t1.named_parameters():
            # Check if it belongs to task 0 LoRA or the head fc
            if "lora_A.0" in name or "lora_B.0" in name or "lora_B_tilde.0" in name or "fc." in name:
                param.requires_grad = True
                trainable_params_t1.append(param)
            else:
                param.requires_grad = False

        # Optimizer
        if use_sam:
            optimizer_t1 = SAM(trainable_params_t1, torch.optim.SGD, rho=args.rho, lr=args.lr, momentum=0.9, weight_decay=1e-4)
        else:
            optimizer_t1 = torch.optim.SGD(trainable_params_t1, lr=args.lr, momentum=0.9, weight_decay=1e-4)

        criterion = nn.CrossEntropyLoss()

        # Training Loop Task 1
        for epoch in range(args.epochs):
            model_t1.train()
            total_loss = 0.0
            correct = 0
            total = 0
            for inputs, targets in task1_train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                if use_sam:
                    # First Step
                    outputs = model_t1(inputs)
                    loss = criterion(outputs[:, 0:5], targets)
                    loss.backward()
                    optimizer_t1.first_step(zero_grad=True)
                    
                    # Second Step
                    outputs = model_t1(inputs)
                    loss = criterion(outputs[:, 0:5], targets)
                    loss.backward()
                    optimizer_t1.second_step(zero_grad=True)
                else:
                    optimizer_t1.zero_grad()
                    outputs = model_t1(inputs)
                    loss = criterion(outputs[:, 0:5], targets)
                    loss.backward()
                    optimizer_t1.step()

                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs[:, 0:5].max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/total:.4f} | Train Acc: {100.*correct/total:.2f}%")

        # Save Task 1 state
        torch.save(model_t1.state_dict(), task1_checkpoint)


    # Train Task 2
    task2_checkpoint = f"task2_{args.mode}{suffix}.pt"
    if os.path.exists(task2_checkpoint):
        print(f"\nFound checkpoint for Task 2: {task2_checkpoint}. Skipping training.")
    else:
        print("\n--- Training Task 2 (Classes 5-9) ---")
        model_t2 = timm.create_model('resnet18', pretrained=True)
        model_t2.fc = nn.Linear(512, 10)
        wrap_model_with_lora(model_t2, r=args.r, lora_alpha=args.lora_alpha, num_tasks=2, so_lora=so_lora)
        model_t2 = model_t2.to(device)

        # Freeze base model, ensure only active task parameters and classification head are updated
        set_active_task_mode(model_t2, task_id=1)
        
        # Verify trainable params
        trainable_params_t2 = []
        for name, param in model_t2.named_parameters():
            if "lora_A.1" in name or "lora_B.1" in name or "lora_B_tilde.1" in name or "fc." in name:
                param.requires_grad = True
                trainable_params_t2.append(param)
            else:
                param.requires_grad = False

        # Optimizer
        if use_sam:
            optimizer_t2 = SAM(trainable_params_t2, torch.optim.SGD, rho=args.rho, lr=args.lr, momentum=0.9, weight_decay=1e-4)
        else:
            optimizer_t2 = torch.optim.SGD(trainable_params_t2, lr=args.lr, momentum=0.9, weight_decay=1e-4)

        criterion = nn.CrossEntropyLoss()

        # Training Loop Task 2
        for epoch in range(args.epochs):
            model_t2.train()
            total_loss = 0.0
            correct = 0
            total = 0
            for inputs, targets in task2_train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                # Map targets to 0-4
                targets_mapped = targets - 5
                
                if use_sam:
                    # First Step
                    outputs = model_t2(inputs)
                    loss = criterion(outputs[:, 5:10], targets_mapped)
                    loss.backward()
                    optimizer_t2.first_step(zero_grad=True)
                    
                    # Second Step
                    outputs = model_t2(inputs)
                    loss = criterion(outputs[:, 5:10], targets_mapped)
                    loss.backward()
                    optimizer_t2.second_step(zero_grad=True)
                else:
                    optimizer_t2.zero_grad()
                    outputs = model_t2(inputs)
                    loss = criterion(outputs[:, 5:10], targets_mapped)
                    loss.backward()
                    optimizer_t2.step()

                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs[:, 5:10].max(1)
                total += targets.size(0)
                correct += predicted.eq(targets_mapped).sum().item()

            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/total:.4f} | Train Acc: {100.*correct/total:.2f}%")

        # Save Task 2 state
        torch.save(model_t2.state_dict(), task2_checkpoint)


    # --- MERGING & EVALUATION ---
    print("\n--- Merging and Evaluating Model ---")
    merged_model = timm.create_model('resnet18', pretrained=True)
    merged_model.fc = nn.Linear(512, 10)
    wrap_model_with_lora(merged_model, r=args.r, lora_alpha=args.lora_alpha, num_tasks=2, so_lora=so_lora)
    merged_model = merged_model.to(device)

    # Load State Dicts
    state_t1 = torch.load(f"task1_{args.mode}{suffix}.pt", map_location=device)
    state_t2 = torch.load(f"task2_{args.mode}{suffix}.pt", map_location=device)

    # Custom merge of state dicts
    # We want merged_model to contain:
    # 1. Base weights (which are identical in both)
    # 2. Task 1 LoRA weights from state_t1
    # 3. Task 2 LoRA weights from state_t2
    # 4. Classification head: copy classes 0-4 from state_t1.fc and classes 5-9 from state_t2.fc
    merged_state = merged_model.state_dict()

    for k in merged_state.keys():
        if "lora_A.0" in k or "lora_B.0" in k or "lora_B_tilde.0" in k:
            merged_state[k].copy_(state_t1[k])
        elif "lora_A.1" in k or "lora_B.1" in k or "lora_B_tilde.1" in k:
            merged_state[k].copy_(state_t2[k])
        elif "fc.weight" in k:
            # First 5 rows from t1, last 5 rows from t2
            merged_state[k][0:5, :].copy_(state_t1[k][0:5, :])
            merged_state[k][5:10, :].copy_(state_t2[k][5:10, :])
        elif "fc.bias" in k:
            # First 5 from t1, last 5 from t2
            merged_state[k][0:5].copy_(state_t1[k][0:5])
            merged_state[k][5:10].copy_(state_t2[k][5:10])
        else:
            # Base model weights (from state_t1 or state_t2, they are frozen and identical)
            merged_state[k].copy_(state_t1[k])

    merged_model.load_state_dict(merged_state)

    # We evaluate standard merging coefficients: [lambda1, lambda2]
    # For standard/sam, the typical merging averages them (coeff = 0.5)
    # For so_lora, the updates are orthogonal, so they can be summed without scaling down (coeff = 1.0)
    # Let's evaluate a range of coefficients to find the optimal merging coefficient!
    coefficients_to_test = [
        [0.5, 0.5],
        [0.7, 0.7],
        [1.0, 1.0]
    ]

    results = {}

    for coeffs in coefficients_to_test:
        print(f"\nEvaluating Merging Coefficients: {coeffs}")
        set_merged_mode(merged_model, coeffs)
        merged_model.eval()

        # Task 1 Evaluation
        correct_t1 = 0
        total_t1 = 0
        with torch.no_grad():
            for inputs, targets in task1_test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = merged_model(inputs)
                # Predict over 10 classes
                _, predicted = outputs.max(1)
                total_t1 += targets.size(0)
                correct_t1 += predicted.eq(targets).sum().item()
        acc_t1 = 100. * correct_t1 / total_t1

        # Task 2 Evaluation
        correct_t2 = 0
        total_t2 = 0
        with torch.no_grad():
            for inputs, targets in task2_test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = merged_model(inputs)
                _, predicted = outputs.max(1)
                total_t2 += targets.size(0)
                correct_t2 += predicted.eq(targets).sum().item()
        acc_t2 = 100. * correct_t2 / total_t2

        avg_acc = (acc_t1 + acc_t2) / 2
        print(f"Task 1 (Classes 0-4) Acc: {acc_t1:.2f}%")
        print(f"Task 2 (Classes 5-9) Acc: {acc_t2:.2f}%")
        print(f"Multi-Task Average Acc: {avg_acc:.2f}%")

        results[f"coeffs_{coeffs[0]}_{coeffs[1]}"] = {
            "task1_acc": acc_t1,
            "task2_acc": acc_t2,
            "avg_acc": avg_acc
        }

    # Save results to JSON
    with open(f"results_{args.mode}{suffix}.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved results to results_{args.mode}{suffix}.json")

if __name__ == "__main__":
    main()
