import os
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# Set seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.enabled = False

# Define SAM Optimizer
class SAM:
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        self.params = list(params)
        self.optimizer = base_optimizer(self.params, **kwargs)
        self.rho = rho
        self.e_ws = {}

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for p in self.params:
            if p.grad is None: continue
            scale = self.rho / (grad_norm + 1e-12)
            e_w = p.grad * scale
            p.add_(e_w) # perturb
            self.e_ws[p] = e_w
        if zero_grad:
            self.optimizer.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for p in self.params:
            if p in self.e_ws:
                p.sub_(self.e_ws[p]) # restore original
        self.e_ws.clear()
        self.optimizer.step()
        if zero_grad:
            self.optimizer.zero_grad()

    def _grad_norm(self):
        shared_device = self.params[0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for p in self.params if p.grad is not None
            ]),
            p=2
        )
        return norm

    def zero_grad(self):
        self.optimizer.zero_grad()

# Block-diagonal Orthogonal Layer (OFT)
class OFTLayer(nn.Module):
    def __init__(self, original_layer, block_size=8):
        super().__init__()
        self.original_layer = original_layer
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False
            
        self.block_size = block_size
        self.out_features = original_layer.weight.shape[0]
        
        # Adjust block_size to be a divisor of out_features
        for b in range(self.block_size, 0, -1):
            if self.out_features % b == 0:
                self.block_size = b
                break
        
        self.num_blocks = self.out_features // self.block_size
        self.tri_size = self.block_size * (self.block_size - 1) // 2
        
        if self.tri_size > 0:
            self.q_params = nn.Parameter(torch.zeros(self.num_blocks, self.tri_size))
            # Initialize with small random values to break symmetry if needed, or zeros
            nn.init.normal_(self.q_params, std=1e-4)
        else:
            self.q_params = None

    def get_R(self):
        device = self.original_layer.weight.device
        if self.q_params is None:
            return None
            
        Q = torch.zeros(self.num_blocks, self.block_size, self.block_size, device=device)
        tri_indices = torch.triu_indices(self.block_size, self.block_size, offset=1, device=device)
        
        Q[:, tri_indices[0], tri_indices[1]] = self.q_params
        Q = Q - Q.transpose(-1, -2) # skew-symmetry
        
        I = torch.eye(self.block_size, device=device).unsqueeze(0).expand(self.num_blocks, -1, -1)
        # Cayley transform
        R_blocks = torch.matmul(I + Q, torch.inverse(I - Q))
        return R_blocks

    def forward(self, x):
        weight_0 = self.original_layer.weight
        bias = self.original_layer.bias
        C_out = weight_0.shape[0]
        weight_flat = weight_0.view(C_out, -1)
        
        R_blocks = self.get_R()
        if R_blocks is not None:
            weight_reshaped = weight_flat.view(self.num_blocks, self.block_size, -1)
            weight_oft = torch.matmul(R_blocks, weight_reshaped)
            weight_oft = weight_oft.view(C_out, -1)
        else:
            weight_oft = weight_flat
            
        weight = weight_oft.view_as(weight_0)
        
        if isinstance(self.original_layer, nn.Conv2d):
            return F.conv2d(x, weight, bias, 
                           stride=self.original_layer.stride,
                           padding=self.original_layer.padding,
                           dilation=self.original_layer.dilation,
                           groups=self.original_layer.groups)
        elif isinstance(self.original_layer, nn.Linear):
            return F.linear(x, weight, bias)
        else:
            return self.original_layer(x)

def apply_oft(model, block_size=8):
    for name, module in list(model.named_children()):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if name == 'fc':
                continue
            setattr(model, name, OFTLayer(module, block_size))
        else:
            apply_oft(module, block_size)

# Load data
def get_data():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Split into Task 1 (classes 0-4) and Task 2 (classes 5-9)
    train_targets = torch.tensor(trainset.targets)
    test_targets = torch.tensor(testset.targets)

    task1_train_idx = torch.where(train_targets < 5)[0]
    task2_train_idx = torch.where(train_targets >= 5)[0]

    task1_test_idx = torch.where(test_targets < 5)[0]
    task2_test_idx = torch.where(test_targets >= 5)[0]

    train_loader1 = DataLoader(Subset(trainset, task1_train_idx), batch_size=128, shuffle=True, num_workers=2)
    train_loader2 = DataLoader(Subset(trainset, task2_train_idx), batch_size=128, shuffle=True, num_workers=2)

    test_loader1 = DataLoader(Subset(testset, task1_test_idx), batch_size=128, shuffle=False, num_workers=2)
    test_loader2 = DataLoader(Subset(testset, task2_test_idx), batch_size=128, shuffle=False, num_workers=2)

    return train_loader1, train_loader2, test_loader1, test_loader2

# Train function
def train(model, loader, optimizer, epochs, device, sam_opt=None, is_task2=False):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # Adjust targets for task 2 to be 0-4
            if is_task2:
                targets = targets - 5
                
            if sam_opt is not None:
                # First step of SAM
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                sam_opt.first_step(zero_grad=True)
                
                # Second step of SAM
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                sam_opt.second_step(zero_grad=True)
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

# Evaluate function
def evaluate(model, loader, device, is_task2=False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if is_task2:
                targets = targets - 5
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

# Merging helpers
def extract_state_dicts(model):
    return copy.deepcopy(model.state_dict())

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running experiments on: {device}")
    
    # Get split datasets
    train_loader1, train_loader2, test_loader1, test_loader2 = get_data()
    
    epochs = 10
    print(f"Fine-tuning models for {epochs} epochs each.")

    results = {}

    # ==========================================
    # EXPERIMENT 1: Task Arithmetic (Euclidean Standard)
    # ==========================================
    print("\n--- Running Experiment 1: Task Arithmetic ---")
    # Task 1
    model1 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model1.fc = nn.Linear(model1.fc.in_features, 5)
    model1 = model1.to(device)
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-4)
    print("Training Task 1 model...")
    train(model1, train_loader1, optimizer1, epochs, device)
    
    # Task 2
    model2 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model2.fc = nn.Linear(model2.fc.in_features, 5)
    model2 = model2.to(device)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-4)
    print("Training Task 2 model...")
    train(model2, train_loader2, optimizer2, epochs, device, is_task2=True)
    
    # Save heads
    fc1_state = copy.deepcopy(model1.fc.state_dict())
    fc2_state = copy.deepcopy(model2.fc.state_dict())
    
    # Merge backbones (TA)
    # W_merged = 0.5 * W_1 + 0.5 * W_2
    ta_state_dict = {}
    m1_sd = model1.state_dict()
    m2_sd = model2.state_dict()
    for key in m1_sd.keys():
        if 'fc' not in key:
            ta_state_dict[key] = 0.5 * m1_sd[key] + 0.5 * m2_sd[key]
            
    # Evaluate TA
    merged_model = torchvision.models.resnet18()
    merged_model.fc = nn.Linear(merged_model.fc.in_features, 5)
    merged_model = merged_model.to(device)
    
    # Eval on Task 1
    merged_model.load_state_dict(ta_state_dict, strict=False)
    merged_model.fc.load_state_dict(fc1_state)
    acc1 = evaluate(merged_model, test_loader1, device)
    
    # Eval on Task 2
    merged_model.fc.load_state_dict(fc2_state)
    acc2 = evaluate(merged_model, test_loader2, device, is_task2=True)
    
    avg_acc_ta = (acc1 + acc2) / 2.0
    print(f"Task Arithmetic -> Task 1 Acc: {acc1:.2f}%, Task 2 Acc: {acc2:.2f}%, Avg Acc: {avg_acc_ta:.2f}%")
    results['Task_Arithmetic'] = {'task1': acc1, 'task2': acc2, 'avg': avg_acc_ta}

    # ==========================================
    # EXPERIMENT 2: SAIM (Euclidean Flatness + Isotropic Merging)
    # ==========================================
    print("\n--- Running Experiment 2: SAIM ---")
    # We fine-tune ResNet-18 with SAM, then perform isotropic merging on the backbone
    model1_sam = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model1_sam.fc = nn.Linear(model1_sam.fc.in_features, 5)
    model1_sam = model1_sam.to(device)
    sam_opt1 = SAM(model1_sam.parameters(), torch.optim.Adam, rho=0.05, lr=1e-4)
    print("Training Task 1 model with SAM...")
    train(model1_sam, train_loader1, None, epochs, device, sam_opt=sam_opt1)
    
    model2_sam = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model2_sam.fc = nn.Linear(model2_sam.fc.in_features, 5)
    model2_sam = model2_sam.to(device)
    sam_opt2 = SAM(model2_sam.parameters(), torch.optim.Adam, rho=0.05, lr=1e-4)
    print("Training Task 2 model with SAM...")
    train(model2_sam, train_loader2, None, epochs, device, sam_opt=sam_opt2, is_task2=True)
    
    # Save heads
    fc1_sam_state = copy.deepcopy(model1_sam.fc.state_dict())
    fc2_sam_state = copy.deepcopy(model2_sam.fc.state_dict())
    
    # Get base pre-trained model for task vectors
    base_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    base_sd = base_model.state_dict()
    
    # Isotropic Merging
    saim_state_dict = {}
    m1_sam_sd = model1_sam.state_dict()
    m2_sam_sd = model2_sam.state_dict()
    
    for key in base_sd.keys():
        if 'fc' in key or 'num_batches_tracked' in key:
            continue

        w_pre = base_sd[key].to(device)
        w_1 = m1_sam_sd[key]
        w_2 = m2_sam_sd[key]        
        # Check if parameter has dimensions for SVD
        if len(w_pre.shape) >= 2:
            # Task vectors
            delta1 = w_1 - w_pre
            delta2 = w_2 - w_pre
            # Combined delta
            delta_com = 0.5 * delta1 + 0.5 * delta2
            
            # Flatten to 2D
            orig_shape = delta_com.shape
            M = delta_com.view(orig_shape[0], -1)
            
            try:
                # SVD on M
                U, S, V = torch.linalg.svd(M, full_matrices=False)
                # Mean singular value
                S_mean = S.mean()
                # Interpolate singular values
                S_hat = S_mean + (S - S_mean) * (1.0 / (2.0 ** 0.5))
                # Reconstruct
                M_merged = torch.matmul(U * S_hat.unsqueeze(0), V)
                delta_merged = M_merged.view(orig_shape)
                saim_state_dict[key] = w_pre + delta_merged
            except Exception as e:
                # Fallback to standard arithmetic if SVD fails
                saim_state_dict[key] = 0.5 * w_1 + 0.5 * w_2
        else:
            # 1D or 0D parameters (e.g. bias, batchnorm weights)
            saim_state_dict[key] = 0.5 * w_1 + 0.5 * w_2
            
    # Evaluate SAIM
    merged_model.load_state_dict(saim_state_dict, strict=False)
    merged_model.fc.load_state_dict(fc1_sam_state)
    acc1_saim = evaluate(merged_model, test_loader1, device)
    
    merged_model.fc.load_state_dict(fc2_sam_state)
    acc2_saim = evaluate(merged_model, test_loader2, device, is_task2=True)
    
    avg_acc_saim = (acc1_saim + acc2_saim) / 2.0
    print(f"SAIM -> Task 1 Acc: {acc1_saim:.2f}%, Task 2 Acc: {acc2_saim:.2f}%, Avg Acc: {avg_acc_saim:.2f}%")
    results['SAIM'] = {'task1': acc1_saim, 'task2': acc2_saim, 'avg': avg_acc_saim}

    # ==========================================
    # EXPERIMENT 3: OrthoMerge (Manifold Standard)
    # ==========================================
    print("\n--- Running Experiment 3: OrthoMerge ---")
    # Task 1 with OFT
    model1_oft = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    apply_oft(model1_oft, block_size=16)
    model1_oft.fc = nn.Linear(model1_oft.fc.in_features, 5)
    model1_oft = model1_oft.to(device)
    # Train only Q parameters and final fc layer
    oft_params1 = [p for n, p in model1_oft.named_parameters() if 'q_params' in n or 'fc' in n]
    optimizer1_oft = torch.optim.Adam(oft_params1, lr=1e-3)
    print("Training Task 1 model with OFT (Adam)...")
    train(model1_oft, train_loader1, optimizer1_oft, epochs, device)
    
    # Task 2 with OFT
    model2_oft = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    apply_oft(model2_oft, block_size=16)
    model2_oft.fc = nn.Linear(model2_oft.fc.in_features, 5)
    model2_oft = model2_oft.to(device)
    oft_params2 = [p for n, p in model2_oft.named_parameters() if 'q_params' in n or 'fc' in n]
    optimizer2_oft = torch.optim.Adam(oft_params2, lr=1e-3)
    print("Training Task 2 model with OFT (Adam)...")
    train(model2_oft, train_loader2, optimizer2_oft, epochs, device, is_task2=True)
    
    # Save heads
    fc1_oft_state = copy.deepcopy(model1_oft.fc.state_dict())
    fc2_oft_state = copy.deepcopy(model2_oft.fc.state_dict())
    
    # Merge Q parameters via magnitude-corrected average
    m1_oft_sd = model1_oft.state_dict()
    m2_oft_sd = model2_oft.state_dict()
    orthomerge_state_dict = {}
    
    for key in m1_oft_sd.keys():
        if 'fc' in key:
            continue
        if 'q_params' in key:
            q1 = m1_oft_sd[key]
            q2 = m2_oft_sd[key]
            # Magnitude-corrected average
            norm_sum = torch.norm(q1) + torch.norm(q2)
            q_avg = 0.5 * (q1 + q2)
            norm_avg = torch.norm(q_avg)
            c = norm_sum / (2.0 * norm_avg + 1e-12)
            orthomerge_state_dict[key] = c * q_avg
        else:
            # Backbone parameters (frozen and identical across task models)
            orthomerge_state_dict[key] = m1_oft_sd[key]
            
    # Evaluate OrthoMerge
    merged_oft_model = torchvision.models.resnet18()
    apply_oft(merged_oft_model, block_size=16)
    merged_oft_model.fc = nn.Linear(merged_oft_model.fc.in_features, 5)
    merged_oft_model = merged_oft_model.to(device)
    
    merged_oft_model.load_state_dict(orthomerge_state_dict, strict=False)
    merged_oft_model.fc.load_state_dict(fc1_oft_state)
    acc1_ortho = evaluate(merged_oft_model, test_loader1, device)
    
    merged_oft_model.fc.load_state_dict(fc2_oft_state)
    acc2_ortho = evaluate(merged_oft_model, test_loader2, device, is_task2=True)
    
    avg_acc_ortho = (acc1_ortho + acc2_ortho) / 2.0
    print(f"OrthoMerge -> Task 1 Acc: {acc1_ortho:.2f}%, Task 2 Acc: {acc2_ortho:.2f}%, Avg Acc: {avg_acc_ortho:.2f}%")
    results['OrthoMerge'] = {'task1': acc1_ortho, 'task2': acc2_ortho, 'avg': avg_acc_ortho}

    # ==========================================
    # EXPERIMENT 4: Sharpness-Aware Manifold Merging (SMM) (Ours)
    # ==========================================
    print("\n--- Running Experiment 4: SMM (Ours) ---")
    # Task 1 with OFT + SAM
    model1_smm = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    apply_oft(model1_smm, block_size=16)
    model1_smm.fc = nn.Linear(model1_smm.fc.in_features, 5)
    model1_smm = model1_smm.to(device)
    smm_params1 = [p for n, p in model1_smm.named_parameters() if 'q_params' in n or 'fc' in n]
    sam_opt1_smm = SAM(smm_params1, torch.optim.Adam, rho=0.05, lr=1e-3)
    print("Training Task 1 model with SMM (OFT + SAM)...")
    train(model1_smm, train_loader1, None, epochs, device, sam_opt=sam_opt1_smm)
    
    # Task 2 with OFT + SAM
    model2_smm = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    apply_oft(model2_smm, block_size=16)
    model2_smm.fc = nn.Linear(model2_smm.fc.in_features, 5)
    model2_smm = model2_smm.to(device)
    smm_params2 = [p for n, p in model2_smm.named_parameters() if 'q_params' in n or 'fc' in n]
    sam_opt2_smm = SAM(smm_params2, torch.optim.Adam, rho=0.05, lr=1e-3)
    print("Training Task 2 model with SMM (OFT + SAM)...")
    train(model2_smm, train_loader2, None, epochs, device, sam_opt=sam_opt2_smm, is_task2=True)
    
    # Save heads
    fc1_smm_state = copy.deepcopy(model1_smm.fc.state_dict())
    fc2_smm_state = copy.deepcopy(model2_smm.fc.state_dict())
    
    # Merge Q parameters via magnitude-corrected average
    m1_smm_sd = model1_smm.state_dict()
    m2_smm_sd = model2_smm.state_dict()
    smm_state_dict = {}
    
    for key in m1_smm_sd.keys():
        if 'fc' in key:
            continue
        if 'q_params' in key:
            q1 = m1_smm_sd[key]
            q2 = m2_smm_sd[key]
            # Magnitude-corrected average
            norm_sum = torch.norm(q1) + torch.norm(q2)
            q_avg = 0.5 * (q1 + q2)
            norm_avg = torch.norm(q_avg)
            c = norm_sum / (2.0 * norm_avg + 1e-12)
            smm_state_dict[key] = c * q_avg
        else:
            # Backbone parameters
            smm_state_dict[key] = m1_smm_sd[key]
            
    # Evaluate SMM
    merged_oft_model.load_state_dict(smm_state_dict, strict=False)
    merged_oft_model.fc.load_state_dict(fc1_smm_state)
    acc1_smm = evaluate(merged_oft_model, test_loader1, device)
    
    merged_oft_model.fc.load_state_dict(fc2_smm_state)
    acc2_smm = evaluate(merged_oft_model, test_loader2, device, is_task2=True)
    
    avg_acc_smm = (acc1_smm + acc2_smm) / 2.0
    print(f"SMM (Ours) -> Task 1 Acc: {acc1_smm:.2f}%, Task 2 Acc: {acc2_smm:.2f}%, Avg Acc: {avg_acc_smm:.2f}%")
    results['SMM'] = {'task1': acc1_smm, 'task2': acc2_smm, 'avg': avg_acc_smm}

    # ==========================================
    # SAVE ALL RESULTS
    # ==========================================
    print("\n=================== FINAL SUMMARY ===================")
    print(f"Task Arithmetic: {avg_acc_ta:.2f}% (T1: {acc1:.2f}%, T2: {acc2:.2f}%)")
    print(f"SAIM (Eucl+SAM): {avg_acc_saim:.2f}% (T1: {acc1_saim:.2f}%, T2: {acc2_saim:.2f}%)")
    print(f"OrthoMerge (Manifold Standard): {avg_acc_ortho:.2f}% (T1: {acc1_ortho:.2f}%, T2: {acc2_ortho:.2f}%)")
    print(f"SMM (Manifold SAM - Ours): {avg_acc_smm:.2f}% (T1: {acc1_smm:.2f}%, T2: {acc2_smm:.2f}%)")
    
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Results saved to experiment_results.json")

if __name__ == '__main__':
    main()
