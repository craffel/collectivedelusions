import sys
sys.modules['flash_attn'] = None
sys.modules['flash_attn_2_cuda'] = None
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
from train_eval import LoRALinear, apply_lora, evaluate

# Cayley transform functions
def cayley(X):
    I = torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
    return torch.linalg.solve(I - X, I + X)

def inv_cayley(Q):
    I = torch.eye(Q.shape[0], device=Q.device, dtype=Q.dtype)
    return torch.linalg.solve(Q + I, Q - I)

# Perform low-rank orthogonal merging on B and A
def low_rank_ortho_merge(B1, A1, B2, A2, lambda_val, method="lrom"):
    # B1, B2 shapes: [out_features, r]
    # A1, A2 shapes: [r, in_features]
    r = B1.shape[1]
    
    # 1. Align B2 to B1 using Orthogonal Procrustes
    # M = B2^T B1 [r, r]
    M = torch.matmul(B2.t(), B1)
    try:
        U_R, S_R, Vh_R = torch.linalg.svd(M)
        R = torch.matmul(U_R, Vh_R)
    except RuntimeError:
        # Fallback if SVD fails to converge (very rare on 8x8)
        R = torch.eye(r, device=B1.device, dtype=B1.dtype)
        
    B2_prime = torch.matmul(B2, R)
    A2_prime = torch.matmul(R.t(), A2)
    
    # 2. Align A2_prime to A1 from the left using Orthogonal Procrustes
    # N = A2_prime A1^T [r, r]
    N = torch.matmul(A2_prime, A1.t())
    try:
        U_S, S_S, Vh_S = torch.linalg.svd(N)
        # S = V_S U_S^T
        S = torch.matmul(Vh_S.t(), U_S.t())
    except RuntimeError:
        S = torch.eye(r, device=B1.device, dtype=B1.dtype)
        
    B2_double_prime = torch.matmul(B2_prime, S.t())
    A2_double_prime = torch.matmul(S, A2_prime)
    
    if method == "palm":
        # Option A: Procrustes-Aligned Linear Merging (PALM)
        B_merged = lambda_val * B1 + (1.0 - lambda_val) * B2_double_prime
        A_merged = lambda_val * A1 + (1.0 - lambda_val) * A2_double_prime
    else:
        # Option B: Low-Rank Orthogonal Manifold Merging (LROM-SR) with Lie Algebra Interpolation
        try:
            # Map rotations to Lie algebra via inverse Cayley transform
            X_R = inv_cayley(R)
            X_S = inv_cayley(S)
            
            # Interpolate the Lie algebra components
            # At lambda_val=1 (Task 1), we apply full alignment (R, S) to Task 2.
            # At lambda_val=0 (Task 2), we apply no alignment (I) to Task 2.
            # So the interpolation scale for Task 2's alignment is lambda_val.
            X_R_interp = lambda_val * X_R
            X_S_interp = lambda_val * X_S
            
            # Map back to orthogonal group
            R_interp = cayley(X_R_interp)
            S_interp = cayley(X_S_interp)
        except RuntimeError:
            # Fallback to linear rotations if Cayley fails
            R_interp = torch.eye(r, device=B1.device, dtype=B1.dtype)
            S_interp = torch.eye(r, device=B1.device, dtype=B1.dtype)
            
        B2_rotated = torch.matmul(torch.matmul(B2, R_interp), S_interp.t())
        A2_rotated = torch.matmul(torch.matmul(S_interp, R_interp.t()), A2)
        
        B_merged = lambda_val * B1 + (1.0 - lambda_val) * B2_rotated
        A_merged = lambda_val * A1 + (1.0 - lambda_val) * A2_rotated
        
    return B_merged, A_merged

# Perform SVD-based Low-Rank Merging (SVDM)
def svd_merge(B1, A1, B2, A2, lambda_val, r=8, scaling=2.0):
    # W_update = B * A * scaling
    U1 = torch.matmul(B1, A1) * scaling
    U2 = torch.matmul(B2, A2) * scaling
    
    # Combined update matrix
    U_comb = lambda_val * U1 + (1.0 - lambda_val) * U2
    
    # Run SVD
    try:
        U, S, Vh = torch.linalg.svd(U_comb / scaling)
        B_merged = U[:, :r] * torch.sqrt(S[:r])
        A_merged = torch.sqrt(S[:r]).unsqueeze(1) * Vh[:r, :]
    except RuntimeError:
        # Fallback to standard average if SVD fails
        B_merged = lambda_val * B1 + (1.0 - lambda_val) * B2
        A_merged = lambda_val * A1 + (1.0 - lambda_val) * A2
        
    return B_merged, A_merged

# Load weights from a state dict into a module
def merge_and_load(model, checkpoint1, checkpoint2, lambda_val, merge_method, r=8, scaling=2.0):
    merged_state_dict = {}
    
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # Get A and B parameter names
            lora_A_name = f"{name}.lora_A"
            lora_B_name = f"{name}.lora_B"
            
            B1 = checkpoint1[lora_B_name].to(model.device)
            A1 = checkpoint1[lora_A_name].to(model.device)
            B2 = checkpoint2[lora_B_name].to(model.device)
            A2 = checkpoint2[lora_A_name].to(model.device)
            
            if merge_method == "dpm":
                # Direct Parameter Merging
                B_merged = lambda_val * B1 + (1.0 - lambda_val) * B2
                A_merged = lambda_val * A1 + (1.0 - lambda_val) * A2
            elif merge_method == "svdm":
                # SVD Low-Rank Merging
                B_merged, A_merged = svd_merge(B1, A1, B2, A2, lambda_val, r=r, scaling=scaling)
            elif merge_method == "palm":
                # Procrustes-Aligned Linear Merging (PALM)
                B_merged, A_merged = low_rank_ortho_merge(B1, A1, B2, A2, lambda_val, method="palm")
            elif merge_method == "lrom":
                # Low-Rank Orthogonal Manifold Merging (LROM-SR)
                B_merged, A_merged = low_rank_ortho_merge(B1, A1, B2, A2, lambda_val, method="lrom")
            else:
                raise ValueError(f"Unknown merge method: {merge_method}")
                
            merged_state_dict[lora_B_name] = B_merged
            merged_state_dict[lora_A_name] = A_merged
            
    # Load into model
    model.load_state_dict(merged_state_dict, strict=False)

def main():
    # Disable cuDNN to bypass cuDNN initialization errors on certain GPU cluster configurations
    torch.backends.cudnn.enabled = False
    
    parser = argparse.ArgumentParser(description="Evaluate Merged LoRA Models")
    parser.add_argument("--ckpt1", type=str, required=True, help="Path to CIFAR-10 checkpoint")
    parser.add_argument("--ckpt2", type=str, required=True, help="Path to SVHN checkpoint")
    parser.add_argument("--merge_method", type=str, required=True, choices=["dpm", "svdm", "palm", "lrom"])
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Merging Method: {args.merge_method}")
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load test sets
    cifar_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    svhn_test = datasets.SVHN(root="./data", split="test", download=True, transform=transform)
    
    cifar_loader = DataLoader(cifar_test, batch_size=args.batch_size, shuffle=False, num_workers=4)
    svhn_loader = DataLoader(svhn_test, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Load base model
    print("Loading base pre-trained model...")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    model.classifier = nn.Linear(model.classifier.in_features, 10)
    apply_lora(model)
    model.to(device)
    
    # Load checkpoints
    checkpoint1 = torch.load(args.ckpt1, map_location=device)
    checkpoint2 = torch.load(args.ckpt2, map_location=device)
    
    # Extract task classification heads
    head1 = {k: v for k, v in checkpoint1.items() if "classifier" in k}
    head2 = {k: v for k, v in checkpoint2.items() if "classifier" in k}
    
    lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = []
    
    print("\nStarting evaluation over lambda from 0.0 (SVHN) to 1.0 (CIFAR-10)...")
    print(f"{'Lambda':<10}{'CIFAR-10 Acc (%)':<20}{'SVHN Acc (%)':<20}{'Average Acc (%)':<20}")
    print("-" * 70)
    
    for l_val in lambdas:
        # Merge LoRA adapters and load them
        merge_and_load(model, checkpoint1, checkpoint2, l_val, args.merge_method)
        
        # 1. Evaluate CIFAR-10 using CIFAR-10 Head
        model.load_state_dict(head1, strict=False)
        cifar_acc = evaluate(model, cifar_loader, device)
        
        # 2. Evaluate SVHN using SVHN Head
        model.load_state_dict(head2, strict=False)
        svhn_acc = evaluate(model, svhn_loader, device)
        
        avg_acc = (cifar_acc + svhn_acc) / 2.0
        results.append((l_val, cifar_acc, svhn_acc, avg_acc))
        print(f"{l_val:<10.1f}{cifar_acc:<20.2f}{svhn_acc:<20.2f}{avg_acc:<20.2f}")
        
    # Save results to text file
    os.makedirs("./results", exist_ok=True)
    # Extract method name from checkpoint paths (e.g. "./models/cifar10_standard.pt" -> "standard")
    ckpt_name = os.path.basename(args.ckpt1).replace("cifar10_", "").replace(".pt", "")
    results_filename = f"results_{ckpt_name}_{args.merge_method}.txt"
    results_path = os.path.join("./results", results_filename)
    with open(results_path, "w") as f:
        f.write("lambda,cifar10,svhn,average\n")
        for r in results:
            f.write(f"{r[0]:.1f},{r[1]:.2f},{r[2]:.2f},{r[3]:.2f}\n")
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()
