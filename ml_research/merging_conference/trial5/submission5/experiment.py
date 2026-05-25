import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from models import MultiTaskCNN, merge_backbone

# Simplex projection
def project_simplex(v):
    """
    Projects a vector v onto the probability simplex (sum to 1, all >= 0).
    """
    sorted_v, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(sorted_v, dim=0)
    indices = torch.arange(1, len(v) + 1, device=v.device)
    cond = sorted_v - (cssv - 1.0) / indices > 0
    rho = indices[cond][-1]
    theta = (cssv[rho - 1] - 1.0) / rho
    w = torch.clamp(v - theta, min=0.0)
    return w

# Corruption helper in raw image space [0, 1]
def apply_corruption(images, corruption_type):
    """
    Applies corruptions on raw [0, 1] image tensors.
    """
    if corruption_type == "clean":
        return images
    elif corruption_type == "noise":
        # Gaussian Noise with sigma = 0.4
        noisy = images + torch.randn_like(images) * 0.4
        return torch.clamp(noisy, 0.0, 1.0)
    elif corruption_type == "blur":
        # Gaussian Blur with kernel_size = 5, sigma = 2.0
        return TF.gaussian_blur(images, [5, 5], [2.0, 2.0])
    elif corruption_type == "contrast":
        # Contrast with factor = 0.15
        contrast_img = 0.5 + 0.15 * (images - 0.5)
        return torch.clamp(contrast_img, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")

# Evaluation function
def run_evaluation():
    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")
    
    # 1. Load expert checkpoints and prototypes
    experts = []
    for k in range(3):
        model = MultiTaskCNN(num_tasks=3, num_classes=10)
        ckpt_path = f"./checkpoints/expert_{k}.pt"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Expert checkpoint {ckpt_path} not found. Please train experts first.")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model = model.to(device)
        model.eval()
        experts.append(model)
        
    proto_path = "./checkpoints/prototypes.pt"
    if not os.path.exists(proto_path):
        raise FileNotFoundError(f"Prototypes checkpoint {proto_path} not found. Please train experts first.")
    prototypes = torch.load(proto_path, map_location=device) # [3, 10, 128]
    print("Successfully loaded experts and prototypes.")
    
    # 2. Load raw test datasets (without normalization, since we corrupt in raw space first)
    raw_transform = transforms.ToTensor()
    mnist_test_raw = torchvision.datasets.MNIST("./data", train=False, download=True, transform=raw_transform)
    fmnist_test_raw = torchvision.datasets.FashionMNIST("./data", train=False, download=True, transform=raw_transform)
    kmnist_test_raw = torchvision.datasets.KMNIST("./data", train=False, download=True, transform=raw_transform)
    
    # Subsets of 3200 images each (50 batches of size 64)
    mnist_sub = Subset(mnist_test_raw, list(range(3200)))
    fmnist_sub = Subset(fmnist_test_raw, list(range(3200)))
    kmnist_sub = Subset(kmnist_test_raw, list(range(3200)))
    
    # Create DataLoader batches
    mnist_loader = DataLoader(mnist_sub, batch_size=64, shuffle=False)
    fmnist_loader = DataLoader(fmnist_sub, batch_size=64, shuffle=False)
    kmnist_loader = DataLoader(kmnist_sub, batch_size=64, shuffle=False)
    
    # Collect all batches as lists of (images, labels, task_idx)
    mnist_batches = [(imgs, lbls, 0) for imgs, lbls in mnist_loader]
    fmnist_batches = [(imgs, lbls, 1) for imgs, lbls in fmnist_loader]
    kmnist_batches = [(imgs, lbls, 2) for imgs, lbls in kmnist_loader]
    
    # Define Streams
    # 1. Sequential Stream: MNIST -> FashionMNIST -> KMNIST
    seq_batches = mnist_batches + fmnist_batches + kmnist_batches
    
    # 2. Alternating Stream: task-alternating on every batch
    alt_batches = []
    for b_idx in range(50):
        alt_batches.append(mnist_batches[b_idx])
        alt_batches.append(fmnist_batches[b_idx])
        alt_batches.append(kmnist_batches[b_idx])
        
    streams = {
        "sequential": seq_batches,
        "alternating": alt_batches
    }
    
    corruptions = ["clean", "noise", "blur", "contrast"]
    methods = ["static", "adamerging", "pc_merge_opr", "cpa_merge", "cp_cadr"]
    
    # Results dictionary
    results = {m: {s: {} for s in streams} for m in methods}
    
    # Helper to normalize corrupted image tensors
    def normalize_batch(imgs):
        # standard MNIST normalization: mean=0.1307, std=0.3081
        return (imgs - 0.1307) / 0.3081

    # Main evaluation loop
    for method in methods:
        for stream_name, batches in streams.items():
            for corr in corruptions:
                print(f"Evaluating Method: {method.upper()} | Stream: {stream_name.upper()} | Corruption: {corr.upper()}")
                
                # Create merged model instance
                merged_model = MultiTaskCNN(num_tasks=3, num_classes=10).to(device)
                
                # Method-specific initializations
                lambdas = torch.tensor([1/3, 1/3, 1/3], device=device, dtype=torch.float32, requires_grad=True)
                # We optimize lambdas via a simple optimizer
                optimizer = optim.SGD([lambdas], lr=0.01)
                
                # For PC-Merge OPR:
                loss_history = []
                moving_avg_loss = 1.0 # seed moving average
                
                # For CP-CADR (Ours):
                tracked_task = None
                
                correct_predictions = 0
                total_samples = 0
                
                for batch_idx, (raw_imgs, labels, true_task_idx) in enumerate(batches):
                    raw_imgs, labels = raw_imgs.to(device), labels.to(device)
                    
                    # Apply corruption in raw space and normalize
                    corrupted_imgs = apply_corruption(raw_imgs, corr)
                    norm_imgs = normalize_batch(corrupted_imgs)
                    
                    # 1. Unsupervised Prototype Task Detection (Anchor Pass)
                    with torch.no_grad():
                        # Anchor features using uniform static merge
                        static_lambdas = torch.tensor([1/3, 1/3, 1/3], device=device)
                        merge_backbone(merged_model, experts, static_lambdas)
                        anchor_features = merged_model.backbone(norm_imgs) # [B, 128]
                        # L2 normalize anchor features
                        anchor_norm = anchor_features / torch.norm(anchor_features, p=2, dim=1, keepdim=True)
                        
                        # Cosine similarity with L2-normalized prototypes
                        # prototypes is [3, 10, 128]
                        # Compute task affinities
                        task_affinities = torch.zeros(3, device=device)
                        for k in range(3):
                            # anchor_norm is [B, 128], prototypes[k] is [10, 128]
                            # sim_matrix is [B, 10]
                            sim_matrix = torch.matmul(anchor_norm, prototypes[k].t())
                            max_sims, _ = sim_matrix.max(dim=1) # [B]
                            task_affinities[k] = max_sims.mean()
                            
                        # Softmax with low temperature τ = 0.02
                        lambdas_prior = torch.softmax(task_affinities / 0.02, dim=0)
                        predicted_task_idx = torch.argmax(task_affinities).item()
                        
                    # 2. Execute Method-specific Adaptation/Resets
                    if method == "static":
                        # Fixed uniform weights
                        lambdas_val = torch.tensor([1/3, 1/3, 1/3], device=device)
                        merge_backbone(merged_model, experts, lambdas_val)
                        active_task_idx = predicted_task_idx # route using predicted task
                        
                    elif method == "adamerging":
                        # Continuous adaptation via prediction entropy minimization
                        optimizer.zero_grad()
                        # Merge backbone
                        merge_backbone(merged_model, experts, lambdas)
                        active_task_idx = predicted_task_idx
                        
                        # Forward pass
                        logits, _ = merged_model(norm_imgs, active_task_idx)
                        probs = torch.softmax(logits, dim=1)
                        entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
                        
                        entropy.backward()
                        optimizer.step()
                        with torch.no_grad():
                            lambdas.copy_(project_simplex(lambdas))
                        
                        # Final pass with updated weights
                        with torch.no_grad():
                            merge_backbone(merged_model, experts, lambdas)
                            
                    elif method == "pc_merge_opr":
                        # PC-Merge with unsupervised loss spike resets (OPR)
                        optimizer.zero_grad()
                        merge_backbone(merged_model, experts, lambdas)
                        active_task_idx = predicted_task_idx
                        
                        logits, _ = merged_model(norm_imgs, active_task_idx)
                        probs = torch.softmax(logits, dim=1)
                        entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
                        
                        # OPR Trigger: if entropy spikes above moving average
                        if entropy.item() > 2.0 * moving_avg_loss and len(loss_history) > 5:
                            # Reset parameters and clear optimizer
                            with torch.no_grad():
                                lambdas.copy_(torch.tensor([1/3, 1/3, 1/3], device=device))
                            optimizer = optim.SGD([lambdas], lr=0.01)
                            # Re-merge and compute again
                            merge_backbone(merged_model, experts, lambdas)
                            logits, _ = merged_model(norm_imgs, active_task_idx)
                            probs = torch.softmax(logits, dim=1)
                            entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
                        
                        # Update moving average
                        loss_history.append(entropy.item())
                        moving_avg_loss = np.mean(loss_history[-10:])
                        
                        entropy.backward()
                        optimizer.step()
                        with torch.no_grad():
                            lambdas.copy_(project_simplex(lambdas))
                        with torch.no_grad():
                            merge_backbone(merged_model, experts, lambdas)
                            
                    elif method == "cpa_merge":
                        # CPA-Merge: reset to prior at EVERY step, then 1-step dual objective TTA
                        with torch.no_grad():
                            lambdas.copy_(lambdas_prior)
                        
                        optimizer = optim.SGD([lambdas], lr=0.01)
                        optimizer.zero_grad()
                        
                        merge_backbone(merged_model, experts, lambdas)
                        active_task_idx = predicted_task_idx
                        
                        # Forward and extract embeddings
                        logits, embeddings = merged_model(norm_imgs, active_task_idx)
                        
                        # 1. Prediction Entropy
                        probs = torch.softmax(logits, dim=1)
                        entropy_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
                        
                        # 2. Confidence-Masked Contrastive Alignment
                        embeddings_norm = embeddings / torch.norm(embeddings, p=2, dim=1, keepdim=True)
                        sim_matrix = torch.matmul(embeddings_norm, prototypes[active_task_idx].t()) # [B, 10]
                        
                        max_probs, pred_classes = probs.max(dim=1)
                        confidence_mask = max_probs > 0.85
                        
                        if confidence_mask.sum() > 0:
                            # InfoNCE contrastive loss over classes on high-confidence samples
                            # κ = 0.1
                            exp_sim = torch.exp(sim_matrix / 0.1)
                            # numerator for predicted classes
                            num = exp_sim[range(exp_sim.size(0)), pred_classes]
                            den = exp_sim.sum(dim=1)
                            infonce = -torch.log(num / den + 1e-8)
                            contrastive_loss = infonce[confidence_mask].mean()
                        else:
                            contrastive_loss = torch.tensor(0.0, device=device)
                            
                        # Dual loss: Lent + β * Lcontra (β = 0.1)
                        total_loss = entropy_loss + 0.1 * contrastive_loss
                        
                        total_loss.backward()
                        optimizer.step()
                        with torch.no_grad():
                            lambdas.copy_(project_simplex(lambdas))
                        with torch.no_grad():
                            merge_backbone(merged_model, experts, lambdas)
                            
                    elif method == "cp_cadr":
                        # CP-CADR (Ours): Continuous adaptation with PROTOTYPE task shift resets
                        active_task_idx = predicted_task_idx
                        
                        # Check for Task Shift
                        if tracked_task is None:
                            tracked_task = active_task_idx
                        elif active_task_idx != tracked_task:
                            # Dynamic Reset Trigger!
                            # Reset coefficients to the prototype prior and flush optimizer
                            with torch.no_grad():
                                lambdas.copy_(lambdas_prior)
                            optimizer = optim.SGD([lambdas], lr=0.01)
                            tracked_task = active_task_idx
                            
                        # Continuous adaptation Step via Dual Objective
                        optimizer.zero_grad()
                        merge_backbone(merged_model, experts, lambdas)
                        
                        logits, embeddings = merged_model(norm_imgs, active_task_idx)
                        probs = torch.softmax(logits, dim=1)
                        entropy_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
                        
                        embeddings_norm = embeddings / torch.norm(embeddings, p=2, dim=1, keepdim=True)
                        sim_matrix = torch.matmul(embeddings_norm, prototypes[active_task_idx].t())
                        
                        max_probs, pred_classes = probs.max(dim=1)
                        confidence_mask = max_probs > 0.85
                        
                        if confidence_mask.sum() > 0:
                            exp_sim = torch.exp(sim_matrix / 0.1)
                            num = exp_sim[range(exp_sim.size(0)), pred_classes]
                            den = exp_sim.sum(dim=1)
                            infonce = -torch.log(num / den + 1e-8)
                            contrastive_loss = infonce[confidence_mask].mean()
                        else:
                            contrastive_loss = torch.tensor(0.0, device=device)
                            
                        total_loss = entropy_loss + 0.1 * contrastive_loss
                        
                        total_loss.backward()
                        optimizer.step()
                        with torch.no_grad():
                            lambdas.copy_(project_simplex(lambdas))
                        with torch.no_grad():
                            merge_backbone(merged_model, experts, lambdas)
                            
                    # 3. Final Prediction & Accuracy Tracking
                    with torch.no_grad():
                        # We evaluate the performance on the TRUE task head
                        logits, _ = merged_model(norm_imgs, true_task_idx)
                        _, predicted = logits.max(1)
                        correct_predictions += predicted.eq(labels).sum().item()
                        total_samples += labels.size(0)
                        
                # Batch loop completed for this corruption
                final_accuracy = (correct_predictions / total_samples) * 100
                results[method][stream_name][corr] = final_accuracy
                print(f"--> Accuracy: {final_accuracy:.2f}%\n")
                
    # Print a neat markdown table
    print("\n" + "="*80)
    print("EXPERIMENTAL RESULTS TABLE")
    print("="*80)
    
    for stream_name in streams:
        print(f"\n### {stream_name.upper()} STREAM ACCURACIES")
        print("| Method | Clean | Noise | Blur | Contrast | Average |")
        print("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for method in methods:
            c = results[method][stream_name]["clean"]
            n = results[method][stream_name]["noise"]
            b = results[method][stream_name]["blur"]
            con = results[method][stream_name]["contrast"]
            avg = (c + n + b + con) / 4.0
            print(f"| {method.upper()} | {c:.2f}% | {n:.2f}% | {b:.2f}% | {con:.2f}% | {avg:.2f}% |")
            
    # Save the results dictionary for future reference
    torch.save(results, "./checkpoints/eval_results.pt")
    print("\nResults dictionary saved to ./checkpoints/eval_results.pt")

if __name__ == "__main__":
    run_evaluation()
