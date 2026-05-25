import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class TTMM_Simulation:
    def __init__(self, D=512, K=3, C=10):
        self.D = D  # Representation dimension
        self.K = K  # Number of tasks
        self.C = C  # Number of classes per task
        
        # 1. Generate clean prototypes for each task
        self.true_prototypes = {}
        for k in range(K):
            protos = torch.randn(C, D)
            protos = F.normalize(protos, p=2, dim=1)
            self.true_prototypes[k] = protos
            
        # 2. Model 3-layer expert weights and base model
        self.W_base = {}
        self.W_expert = {}
        self.tau = {}
        
        for l in range(3):  # 3 layers
            # Base weight is a scaled identity matrix to preserve features
            self.W_base[l] = 0.5 * torch.eye(D)
            self.W_expert[l] = {}
            self.tau[l] = {}
            for k in range(K):
                # Expert weights: base + random task vector
                # Correct expert has a well-behaved matrix, others introduce interference
                expert_weight = 0.5 * torch.eye(D) + 0.1 * torch.randn(D, D)
                # Ensure correct expert product is close to identity for clean transfer
                if k == l:  # Just some structure
                    expert_weight = torch.eye(D) + 0.05 * torch.randn(D, D)
                self.W_expert[l][k] = expert_weight
                self.tau[l][k] = expert_weight - self.W_base[l]
                
        # 3. Define the true combined transformations and classification heads
        self.heads = {}
        for k in range(K):
            # True expert transformations product: W2_k * W1_k * W0_k
            T_k = self.W_expert[2][k] @ self.W_expert[1][k] @ self.W_expert[0][k]
            # Classification head aligns with clean prototypes after correct transformations
            self.heads[k] = self.true_prototypes[k] @ T_k
            
        # 4. Define layer-wise sensitivities (Fisher Information)
        # Layer 0 (early) and Layer 2 (late/head adjacent) are highly sensitive.
        # Layer 1 (mid) is robust.
        self.Fisher = {0: 12.0, 1: 0.1, 2: 6.0}

    def generate_batch(self, task, corruption="clean", batch_size=32):
        """
        Generate a batch of features and labels for a specific task and corruption.
        """
        labels = torch.randint(0, self.C, (batch_size,))
        protos = self.true_prototypes[task][labels]
        
        if corruption == "clean":
            noise = 0.02 * torch.randn(batch_size, self.D)
            x = protos + noise
        elif corruption == "noise":
            noise = 0.20 * torch.randn(batch_size, self.D)
            x = protos + noise
        elif corruption == "blur":
            # Blur mixes prototypes with task-level global mean, reducing inter-class separation
            mean_proto = self.true_prototypes[task].mean(dim=0, keepdim=True)
            blurred_protos = 0.6 * protos + 0.4 * mean_proto
            noise = 0.02 * torch.randn(batch_size, self.D)
            x = blurred_protos + noise
        elif corruption == "contrast":
            # Contrast shifts and scales prototypes
            x = 0.3 * protos + 0.1 * torch.ones(batch_size, self.D)
            noise = 0.01 * torch.randn(batch_size, self.D)
            x = x + noise
        else:
            x = protos
            
        return x, labels

    def get_merged_transformation(self, Lambdas):
        """
        Compute the layer-wise merged transformation.
        Lambdas is a list of 3 tensors of shape (K,), representing coefficients for each layer.
        """
        W_merged = {}
        for l in range(3):
            W_merged[l] = self.W_base[l] + sum(Lambdas[l][k] * self.tau[l][k] for k in range(self.K))
        
        # Product of merged layers: W2 * W1 * W0
        T_merged = W_merged[2] @ W_merged[1] @ W_merged[0]
        return T_merged

    def evaluate_batch(self, x, labels, task, Lambdas):
        """
        Evaluate accuracy of merged coefficients on a batch.
        """
        T_merged = self.get_merged_transformation(Lambdas)
        z = x @ T_merged.t()
        logits = z @ self.heads[task].t()
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean().item()
        return acc, logits, z

    def project_to_simplex(self, v):
        """
        Project vector v onto the probability simplex.
        """
        n_features = v.shape[0]
        u, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(u, dim=0)
        ind = torch.arange(n_features, dtype=torch.float32) + 1
        cond = u - (cssv - 1.0) / ind > 0
        # Find the last index where condition is met
        # To avoid index issues, we do a safe argmax-like extraction
        indices = torch.nonzero(cond)
        if len(indices) == 0:
            return torch.ones_like(v) / n_features
        rho = indices[-1].item()
        theta = (cssv[rho] - 1.0) / (rho + 1)
        return torch.clamp(v - theta, min=0.0)

    def run_stream(self, stream_type="sequential", corruption="clean", method="STATIC", 
                   lr=0.05, alpha=0.5, beta=0.1, tau_temp=0.02, gamma=0.1, num_batches=150,
                   h_thresh=0.90, h_contra=0.85, seed=42, block_length=None):
        """
        Simulate a stream of batches and perform test-time model merging.
        """
        # Reset random seeds at the start of each stream for 100% fair and reproducible evaluation
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Initialize layer-wise merging coefficients to uniform [1/3, 1/3, 1/3]
        Lambdas = [torch.tensor([1/3, 1/3, 1/3], requires_grad=True) for _ in range(3)]
        
        # Build stream task indices
        if block_length is not None:
            task_stream = [(i // block_length) % self.K for i in range(num_batches)]
        elif stream_type == "sequential":
            # 50 batches of Task 0, 50 of Task 1, 50 of Task 2
            task_stream = [0]*50 + [1]*50 + [2]*50
        else:
            # Alternating task stream
            task_stream = [i % self.K for i in range(num_batches)]
            
        accuracies = []
        lambda_history = []
        
        # PC-Merge variables for resetting
        ema_loss = None
        beta_ema = 0.9
        opr_threshold = 2.5 if corruption != "clean" else 4.0
        
        # SR-CPA variables
        self.sr_prototypes = {}
        # We start with empty self-refined prototypes.
        # They will be initialized from high-confidence predictions during stream.
        for k in range(self.K):
            self.sr_prototypes[k] = torch.zeros(self.C, self.D)
            
        for step, active_task in enumerate(task_stream[:num_batches]):
            # 1. Generate incoming batch
            x, labels = self.generate_batch(active_task, corruption=corruption)
            
            # Detach and wrap in tensors for optimization
            x_var = x.clone().detach()
            labels_var = labels.clone().detach()
            
            # 2. Record initial coefficients
            current_lambdas = [L.clone().detach() for L in Lambdas]
            lambda_history.append([L.numpy() for L in current_lambdas])
            
            # 3. Evaluate initial accuracy on the incoming batch
            acc, logits, z = self.evaluate_batch(x_var, labels_var, active_task, current_lambdas)
            accuracies.append(acc)
            
            # 4. Perform adaptation step if not STATIC
            if method != "STATIC":
                # Compute predictions with current coefficients
                T_merged = self.get_merged_transformation(Lambdas)
                z_adapt = x_var @ T_merged.t()
                logits_adapt = z_adapt @ self.heads[active_task].t()
                probs = F.softmax(logits_adapt, dim=1)
                
                # --- Unconstrained Prediction Entropy Loss ---
                ent_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
                
                # --- Optimizer and Parameter Resets (OPR) for PC-Merge ---
                if "OPR" in method or method == "PC-Merge":
                    loss_val = ent_loss.item()
                    if ema_loss is None:
                        ema_loss = loss_val
                    else:
                        # Detect task shift spike
                        if loss_val > opr_threshold * ema_loss:
                            # Reset coefficients to uniform and clear gradients
                            with torch.no_grad():
                                for L in Lambdas:
                                    L.fill_(1/3)
                                # Reset self-refined prototypes
                                if "SR-CPA" in method:
                                    for k in range(self.K):
                                        self.sr_prototypes[k].fill_(0.0)
                            # Recompute forward pass with uniform weights
                            T_merged = self.get_merged_transformation(Lambdas)
                            z_adapt = x_var @ T_merged.t()
                            logits_adapt = z_adapt @ self.heads[active_task].t()
                            probs = F.softmax(logits_adapt, dim=1)
                            ent_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
                            loss_val = ent_loss.item()
                        
                        # Update loss EMA
                        ema_loss = beta_ema * ema_loss + (1 - beta_ema) * loss_val
                
                # --- Prototype-driven Dynamic Routing (PD-Routing) for CPA-Merge and SR-CPA ---
                if "CPA" in method:
                    # Choose which prototypes to use: Clean (CPA-Merge) or Self-Refined (SR-CPA)
                    if "SR-CPA" in method:
                        protos_to_use = self.sr_prototypes
                    else:
                        protos_to_use = self.true_prototypes
                        
                    # Forward anchor pass through uniform merged model
                    Lambda_static = [torch.tensor([1/3, 1/3, 1/3]) for _ in range(3)]
                    T_static = self.get_merged_transformation(Lambda_static)
                    z_anchor = x_var @ T_static.t()
                    z_anchor_norm = F.normalize(z_anchor, p=2, dim=1)
                    
                    # Compute task affinity scores using prototypes
                    scores = []
                    for k in range(self.K):
                        if "SR-CPA" in method and torch.sum(protos_to_use[k]) == 0:
                            # If self-refined prototypes are not initialized yet, use neutral similarity
                            scores.append(0.0)
                        else:
                            # Normalize prototypes
                            protos_norm = F.normalize(protos_to_use[k], p=2, dim=1)
                            # Cosine similarities
                            sims = z_anchor_norm @ protos_norm.t() # (B, C)
                            max_sims, _ = torch.max(sims, dim=1)
                            scores.append(torch.mean(max_sims).item())
                            
                    # Set merging coefficients using a sharp softmax routing prior
                    scores_t = torch.tensor(scores)
                    if torch.sum(scores_t) == 0:
                        prior = torch.tensor([1/3, 1/3, 1/3])
                    else:
                        if "AT" in method:
                            # Adaptive routing temperature based on task affinity uncertainty (entropy)
                            # First compute a baseline softmax task probability distribution
                            q = F.softmax(scores_t / 0.05, dim=0)
                            ent = -torch.sum(q * torch.log(q + 1e-8))
                            max_ent = np.log(self.K)
                            uncertainty = ent / max_ent
                            tau_min = 0.01
                            tau_max = 0.50
                            tau_temp_dynamic = tau_min + (tau_max - tau_min) * uncertainty.item()
                            prior = F.softmax(scores_t / tau_temp_dynamic, dim=0)
                        else:
                            prior = F.softmax(scores_t / tau_temp, dim=0)
                        
                    # Reset current active coefficients to prior (PD-Routing)
                    with torch.no_grad():
                        for L in Lambdas:
                            L.copy_(prior)
                            
                    # Recompute representation under newly routed coefficients
                    T_merged = self.get_merged_transformation(Lambdas)
                    z_adapt = x_var @ T_merged.t()
                    logits_adapt = z_adapt @ self.heads[active_task].t()
                    probs = F.softmax(logits_adapt, dim=1)
                    ent_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
                
                # --- Dynamic Self-Refined Prototype Extraction for SR-CPA ---
                if "SR-CPA" in method:
                    # Check for high-confidence predictions to initialize/update self-refined prototypes
                    max_probs, pred_classes = torch.max(probs, dim=1)
                    high_conf_mask = max_probs > h_thresh
                    
                    if torch.any(high_conf_mask):
                        z_norm = F.normalize(z_adapt.detach(), p=2, dim=1)
                        for i in range(len(x_var)):
                            if high_conf_mask[i]:
                                c = pred_classes[i].item()
                                feat = z_norm[i]
                                
                                # Update self-refined prototypes via EMA
                                current_proto = self.sr_prototypes[active_task][c]
                                if torch.sum(current_proto) == 0:
                                    # Initialize first prototype
                                    self.sr_prototypes[active_task][c] = feat
                                else:
                                    # Refine prototype
                                    # If CW in method name, scale gamma by confidence of prediction
                                    g_scaled = gamma * max_probs[i].item() if "CW" in method else gamma
                                    self.sr_prototypes[active_task][c] = (1 - g_scaled) * current_proto + g_scaled * feat
                                # Normalize
                                self.sr_prototypes[active_task][c] = F.normalize(self.sr_prototypes[active_task][c].unsqueeze(0), p=2, dim=1).squeeze(0)

                # --- Compute Loss for Backpropagation ---
                loss = ent_loss
                
                # Add Contrastive Prototype Alignment Loss for CPA-Merge and SR-CPA
                if "CPA" in method:
                    if "SR-CPA" in method:
                        protos_to_use = self.sr_prototypes
                    else:
                        protos_to_use = self.true_prototypes
                        
                    if torch.sum(protos_to_use[active_task]) != 0:
                        max_probs, pred_classes = torch.max(probs, dim=1)
                        high_conf_mask = max_probs > h_contra
                        
                        if torch.any(high_conf_mask):
                            z_norm = F.normalize(z_adapt, p=2, dim=1)
                            protos_norm = F.normalize(protos_to_use[active_task], p=2, dim=1)
                            
                            # Logits as cosine similarities scaled by contrastive temperature
                            M = (z_norm @ protos_norm.t()) / 0.1  # (B, C), κ=0.1
                            
                            # Filter high confidence samples
                            M_masked = M[high_conf_mask]
                            classes_masked = pred_classes[high_conf_mask]
                            
                            # InfoNCE cross-entropy loss over high-confidence samples
                            contra_loss = F.cross_entropy(M_masked, classes_masked)
                            loss = loss + beta * contra_loss
                
                # --- Backward Pass and Parameter Updates ---
                # Clear gradients
                for L in Lambdas:
                    if L.grad is not None:
                        L.grad.zero_()
                        
                # Zero out specific layer gradients if PC-Merge gradient surgery is applied
                if method == "PC-Merge":
                    # Class-specific gradient projection
                    max_probs, pred_classes = torch.max(probs, dim=1)
                    class_losses = []
                    class_grads = []
                    
                    # Compute gradients per active class
                    unique_classes = torch.unique(pred_classes)
                    for c in unique_classes:
                        c_mask = pred_classes == c
                        if torch.any(c_mask):
                            c_probs = probs[c_mask]
                            c_ent = -torch.mean(torch.sum(c_probs * torch.log(c_probs + 1e-8), dim=1))
                            
                            # Compute gradient of class-specific entropy with respect to Lambdas
                            grads = torch.autograd.grad(c_ent, Lambdas, retain_graph=True, allow_unused=True)
                            class_grads.append([g.clone() if g is not None else torch.zeros_like(Lambdas[l]) for l, g in enumerate(grads)])
                    
                    # Perform Gradient Surgery (project conflicting class gradients)
                    num_classes = len(class_grads)
                    final_grads = [torch.zeros_like(Lambdas[l]) for l in range(3)]
                    
                    if num_classes > 0:
                        # Initialize projected gradients
                        proj_grads = [[g[l].clone() for l in range(3)] for g in class_grads]
                        
                        for i in range(num_classes):
                            for j in range(num_classes):
                                if i != j:
                                    for l in range(3):
                                        dot_prod = torch.dot(proj_grads[i][l], class_grads[j][l])
                                        if dot_prod < 0:
                                            # Conflict: project i's gradient onto the normal plane of j
                                            proj_grads[i][l] = proj_grads[i][l] - (dot_prod / (torch.norm(class_grads[j][l])**2 + 1e-8)) * class_grads[j][l]
                                            
                        # Sum up conflict-free class gradients
                        for l in range(3):
                            final_grads[l] = sum(proj_grads[i][l] for i in range(num_classes))
                            
                    # Update coefficients manually
                    with torch.no_grad():
                        for l in range(3):
                            Lambdas[l] -= lr * final_grads[l]
                            Lambdas[l].copy_(self.project_to_simplex(Lambdas[l]))
                            
                else:
                    # Standard backpropagation
                    loss.backward()
                    
                    # Update coefficients
                    with torch.no_grad():
                        for l in range(3):
                            # --- LFWA: Layer-wise Fisher-Weighted Adaptation ---
                            if method == "LFWA" or ("SR-CPA" in method and alpha > 0.0):
                                lr_scaled = lr / (self.Fisher[l] + 1e-8)**alpha
                            else:
                                lr_scaled = lr
                                
                            if Lambdas[l].grad is not None:
                                Lambdas[l] -= lr_scaled * Lambdas[l].grad
                            Lambdas[l].copy_(self.project_to_simplex(Lambdas[l]))
                            
                # Re-enable grad tracking
                for L in Lambdas:
                    L.requires_grad_(True)
                    
        return np.mean(accuracies), accuracies, np.array(lambda_history)

def run_experiment_suite():
    print("Initializing TTMM Simulation and Experiment Suite...")
    sim = TTMM_Simulation()
    
    corruptions = ["clean", "noise", "blur", "contrast"]
    streams = ["sequential", "alternating"]
    methods = ["STATIC", "TTA", "LFWA", "PC-Merge", "CPA-Merge", "SR-CPA", "FW-SR-CPA", "CW-SR-CPA", "FW-CW-SR-CPA", "AT-SR-CPA", "FW-CW-AT-SR-CPA", "OPR-SR-CPA", "FW-CW-OPR-AT-SR-CPA"]
    
    results = {}
    for stream in streams:
        results[stream] = {}
        for corr in corruptions:
            results[stream][corr] = {}
            for m in methods:
                # Set hyperparameters based on papers
                # LFWA uses alpha=0.5, TTA/PC-Merge uses alpha=0, etc.
                lr_val = 0.10 if m == "PC-Merge" else (0.01 if "SR-CPA" in m or m == "CPA-Merge" else 0.05)
                avg_acc, accs, lambdas = sim.run_stream(
                    stream_type=stream,
                    corruption=corr,
                    method=m,
                    lr=lr_val,
                    alpha=0.5 if m == "LFWA" else (0.3 if "FW" in m else 0.0),
                    beta=0.1,
                    tau_temp=0.02,
                    gamma=0.1
                )
                results[stream][corr][m] = (avg_acc, accs, lambdas)
                print(f"[{stream.upper()} - {corr.upper()}] {m}: {avg_acc*100:.2f}%")
                
    # --- Generate Latex / Text Tables ---
    print("\n--- RESULTS SUMMARY TABLE ---")
    for stream in streams:
        print(f"\n{stream.capitalize()} Stream:")
        print(f"{'Method':<12} | {'Clean':<8} | {'Noise':<8} | {'Blur':<8} | {'Contrast':<8} | {'Average':<8}")
        print("-" * 65)
        for m in methods:
            row_accs = [results[stream][corr][m][0] for corr in corruptions]
            avg_acc = np.mean(row_accs)
            print(f"{m:<12} | " + " | ".join(f"{acc*100:.2f}%" for acc in row_accs) + f" | {avg_acc*100:.2f}%")
            
    # --- Save Coefficient Trajectory Plots (like Figure 1) ---
    print("\nGenerating coefficient trajectory plots...")
    # Get sequential clean histories for TTA vs SR-CPA
    tta_lambdas = results["sequential"]["clean"]["TTA"][2]      # shape (150, 3, 3) -> steps, layers, experts
    srcpa_lambdas = results["sequential"]["clean"]["SR-CPA"][2]  # shape (150, 3, 3)
    
    # Plot Layer 2 (head-adjacent) coefficients for MNIST (Expert 0), FashionMNIST (Expert 1), KMNIST (Expert 2)
    steps = np.arange(150)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    # Standard TTA (Left)
    axes[0].plot(steps, tta_lambdas[:, 2, 0], label="Task 0 Expert", color="tab:blue", linewidth=2)
    axes[0].plot(steps, tta_lambdas[:, 2, 1], label="Task 1 Expert", color="tab:orange", linewidth=2)
    axes[0].plot(steps, tta_lambdas[:, 2, 2], label="Task 2 Expert", color="tab:green", linewidth=2)
    axes[0].axvline(50, color="red", linestyle="--", alpha=0.5)
    axes[0].axvline(100, color="red", linestyle="--", alpha=0.5)
    axes[0].set_title("Standard TTA (Softmax Saturation Collapse)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Test Stream Batch Index", fontsize=10)
    axes[0].set_ylabel("Merging Coefficient (Layer 2)", fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # SR-CPA (Right)
    axes[1].plot(steps, srcpa_lambdas[:, 2, 0], label="Task 0 Expert", color="tab:blue", linewidth=2)
    axes[1].plot(steps, srcpa_lambdas[:, 2, 1], label="Task 1 Expert", color="tab:orange", linewidth=2)
    axes[1].plot(steps, srcpa_lambdas[:, 2, 2], label="Task 2 Expert", color="tab:green", linewidth=2)
    axes[1].axvline(50, color="red", linestyle="--", alpha=0.5)
    axes[1].axvline(100, color="red", linestyle="--", alpha=0.5)
    axes[1].set_title("SR-CPA (Ours, Calibration-Free Active Tracking)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Test Stream Batch Index", fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig("coefficient_trajectory.png", dpi=300)
    plt.close()
    print("Saved coefficient_trajectory.png.")
    
    # --- Save Average Accuracy Bar Chart ---
    print("Generating accuracy comparison bar chart...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, stream in enumerate(streams):
        x_indices = np.arange(len(corruptions))
        bar_width = 0.12
        
        for j, m in enumerate(methods):
            accs = [results[stream][corr][m][0] * 100 for corr in corruptions]
            axes[i].bar(x_indices + j * bar_width, accs, bar_width, label=m)
            
        axes[i].set_title(f"{stream.capitalize()} Stream Accuracy (%)", fontsize=12, fontweight="bold")
        axes[i].set_xticks(x_indices + 3.0 * bar_width)
        axes[i].set_xticklabels([c.capitalize() for c in corruptions])
        axes[i].set_ylabel("Average Accuracy (%)", fontsize=10)
        axes[i].set_ylim(0, 105)
        axes[i].grid(True, alpha=0.3, axis="y")
        axes[i].legend()
        
    plt.tight_layout()
    plt.savefig("accuracy_comparison.png", dpi=300)
    plt.close()
    print("Saved accuracy_comparison.png.")
    
    # --- Sensitivity Sweeps for our paper (Table 2 replica) ---
    print("\nRunning hyperparameter sensitivity sweeps for SR-CPA...")
    # Softmax Routing Temperature Sweep (Sequential Stream, Clean)
    temp_sweep = [0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]
    temp_results = []
    for temp in temp_sweep:
        avg_acc, _, _ = sim.run_stream(
            stream_type="sequential",
            corruption="clean",
            method="SR-CPA",
            lr=0.01,
            tau_temp=temp
        )
        temp_results.append(avg_acc)
        print(f"Temp Sweep: tau={temp} -> Acc={avg_acc*100:.2f}%")
        
    # Prototype EMA Update Rate Sweep (Sequential Stream, Clean)
    gamma_sweep = [0.01, 0.05, 0.10, 0.20, 0.50]
    gamma_results = []
    for g in gamma_sweep:
        avg_acc, _, _ = sim.run_stream(
            stream_type="sequential",
            corruption="clean",
            method="SR-CPA",
            lr=0.01,
            gamma=g
        )
        gamma_results.append(avg_acc)
        print(f"Gamma Sweep: gamma={g} -> Acc={avg_acc*100:.2f}%")
        
    # Fisher Scaling Exponent Sweep (Alternating Stream, Contrast)
    alpha_sweep = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    alpha_results = []
    for a in alpha_sweep:
        avg_acc, _, _ = sim.run_stream(
            stream_type="alternating",
            corruption="contrast",
            method="SR-CPA",
            lr=0.01,
            alpha=a
        )
        alpha_results.append(avg_acc)
        print(f"Alpha Sweep: alpha={a} -> Acc={avg_acc*100:.2f}%")
        
    # Contrastive Weight Beta Sweep (Alternating Stream, Contrast)
    beta_sweep = [0.0, 0.01, 0.05, 0.10, 0.20, 0.50]
    beta_results = []
    for b in beta_sweep:
        avg_acc, _, _ = sim.run_stream(
            stream_type="alternating",
            corruption="contrast",
            method="SR-CPA",
            lr=0.01,
            alpha=0.0,
            beta=b
        )
        beta_results.append(avg_acc)
        print(f"Beta Sweep: beta={b} -> Acc={avg_acc*100:.2f}%")

    # Seeding Threshold H_thresh Sweep (Alternating Stream, Clean)
    h_thresh_sweep = [0.70, 0.80, 0.85, 0.90, 0.95, 0.98]
    h_thresh_results = []
    for ht in h_thresh_sweep:
        avg_acc, _, _ = sim.run_stream(
            stream_type="alternating",
            corruption="clean",
            method="SR-CPA",
            lr=0.01,
            h_thresh=ht
        )
        h_thresh_results.append(avg_acc)
        print(f"H_thresh Sweep: h_thresh={ht} -> Acc={avg_acc*100:.2f}%")

    # Contrastive Threshold H_contra Sweep (Alternating Stream, Clean)
    h_contra_sweep = [0.70, 0.80, 0.85, 0.90, 0.95]
    h_contra_results = []
    for hc in h_contra_sweep:
        avg_acc, _, _ = sim.run_stream(
            stream_type="alternating",
            corruption="clean",
            method="SR-CPA",
            lr=0.01,
            h_contra=hc
        )
        h_contra_results.append(avg_acc)
        print(f"H_contra Sweep: h_contra={hc} -> Acc={avg_acc*100:.2f}%")
        
    # Task Block Length Sweep (Clean corruption)
    block_lengths = [1, 2, 5, 10, 25, 50]
    block_methods = ["STATIC", "TTA", "CPA-Merge", "SR-CPA", "FW-CW-AT-SR-CPA"]
    block_results = {m: [] for m in block_methods}
    
    print("\nRunning task block length sweep...")
    for bl in block_lengths:
        for m in block_methods:
            # Match method-specific hyperparameters
            lr_val = 0.01 if "SR-CPA" in m or m == "CPA-Merge" else 0.05
            alpha_val = 0.3 if "FW" in m else 0.0
            
            avg_acc, _, _ = sim.run_stream(
                stream_type="alternating", # dummy, overridden by block_length
                corruption="clean",
                method=m,
                lr=lr_val,
                alpha=alpha_val,
                beta=0.1,
                tau_temp=0.02,
                gamma=0.1,
                block_length=bl
            )
            block_results[m].append(avg_acc)
            print(f"Block Length {bl} | {m}: {avg_acc*100:.2f}%")
        
    # Write sweep results to a txt file for latex extraction
    with open("sweep_results.txt", "w") as f:
        f.write("=== SOFTMAX TEMPERATURE SWEEP ===\n")
        for temp, acc in zip(temp_sweep, temp_results):
            f.write(f"tau={temp}: {acc*100:.2f}%\n")
        f.write("\n=== PROTOTYPE EMA UPDATE RATE SWEEP ===\n")
        for g, acc in zip(gamma_sweep, gamma_results):
            f.write(f"gamma={g}: {acc*100:.2f}%\n")
        f.write("\n=== FISHER SCALING ALPHA SWEEP ===\n")
        for a, acc in zip(alpha_sweep, alpha_results):
            f.write(f"alpha={a}: {acc*100:.2f}%\n")
        f.write("\n=== CONTRASTIVE WEIGHT BETA SWEEP ===\n")
        for b, acc in zip(beta_sweep, beta_results):
            f.write(f"beta={b}: {acc*100:.2f}%\n")
        f.write("\n=== SEEDING THRESHOLD H_THRESH SWEEP ===\n")
        for ht, acc in zip(h_thresh_sweep, h_thresh_results):
            f.write(f"h_thresh={ht}: {acc*100:.2f}%\n")
        f.write("\n=== CONTRASTIVE THRESHOLD H_CONTRA SWEEP ===\n")
        for hc, acc in zip(h_contra_sweep, h_contra_results):
            f.write(f"h_contra={hc}: {acc*100:.2f}%\n")
        f.write("\n=== TASK BLOCK LENGTH SWEEP (CLEAN CORRUPTION) ===\n")
        f.write(f"{'Method':<16} | " + " | ".join(f"B={bl}" for bl in block_lengths) + "\n")
        f.write("-" * 65 + "\n")
        for m in block_methods:
            row = [f"{acc*100:.2f}%" for acc in block_results[m]]
            f.write(f"{m:<16} | " + " | ".join(row) + "\n")
            
    print("Saved sweep_results.txt.")

if __name__ == "__main__":
    run_experiment_suite()
