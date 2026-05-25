import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import os
import copy
from model import CNNEncoder, ClassifierHead, MergedModel

# ==========================================
# 1. Corruptions
# ==========================================
def apply_gaussian_noise(x, sigma=0.4):
    noise = torch.randn_like(x) * sigma
    return torch.clamp(x + noise, 0.0, 1.0)

def apply_gaussian_blur(x, sigma=2.0):
    return TF.gaussian_blur(x, kernel_size=[5, 5], sigma=[sigma, sigma])

def apply_contrast(x, alpha=0.15):
    return torch.clamp(0.5 + alpha * (x - 0.5), 0.0, 1.0)

def preprocess_and_normalize(x, corruption_type="clean"):
    if corruption_type == "noise":
        x = apply_gaussian_noise(x, sigma=0.4)
    elif corruption_type == "blur":
        x = apply_gaussian_blur(x, sigma=2.0)
    elif corruption_type == "contrast":
        x = apply_contrast(x, alpha=0.15)
    # Normalize from [0, 1] to [-1, 1]
    return (x - 0.5) / 0.5

# ==========================================
# 2. Data Loaders and Stream Builders
# ==========================================
def load_test_data():
    transform_raw = transforms.Compose([transforms.ToTensor()])
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform_raw)
    fmnist_test = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform_raw)
    kmnist_test = datasets.KMNIST(root="./data", train=False, download=True, transform=transform_raw)
    
    # Extract 3200 samples per task (50 batches of size 64)
    mnist_imgs = torch.stack([mnist_test[i][0] for i in range(3200)])
    mnist_lbls = torch.tensor([mnist_test[i][1] for i in range(3200)])
    
    fmnist_imgs = torch.stack([fmnist_test[i][0] for i in range(3200)])
    fmnist_lbls = torch.tensor([fmnist_test[i][1] for i in range(3200)])
    
    kmnist_imgs = torch.stack([kmnist_test[i][0] for i in range(3200)])
    kmnist_lbls = torch.tensor([kmnist_test[i][1] for i in range(3200)])
    
    return {
        "mnist": (mnist_imgs, mnist_lbls),
        "fmnist": (fmnist_imgs, fmnist_lbls),
        "kmnist": (kmnist_imgs, kmnist_lbls)
    }

def build_stream(test_data, stream_type="sequential", corruption_type="clean"):
    mnist_imgs, mnist_lbls = test_data["mnist"]
    fmnist_imgs, fmnist_lbls = test_data["fmnist"]
    kmnist_imgs, kmnist_lbls = test_data["kmnist"]
    
    num_batches = 50
    batch_size = 64
    
    batches = []
    if stream_type == "sequential":
        for b in range(num_batches):
            idx = b * batch_size
            imgs = preprocess_and_normalize(mnist_imgs[idx:idx+batch_size], corruption_type)
            batches.append((imgs, mnist_lbls[idx:idx+batch_size], 0))
        for b in range(num_batches):
            idx = b * batch_size
            imgs = preprocess_and_normalize(fmnist_imgs[idx:idx+batch_size], corruption_type)
            batches.append((imgs, fmnist_lbls[idx:idx+batch_size], 1))
        for b in range(num_batches):
            idx = b * batch_size
            imgs = preprocess_and_normalize(kmnist_imgs[idx:idx+batch_size], corruption_type)
            batches.append((imgs, kmnist_lbls[idx:idx+batch_size], 2))
    elif stream_type == "alternating":
        for b in range(num_batches):
            # MNIST batch b
            idx = b * batch_size
            imgs0 = preprocess_and_normalize(mnist_imgs[idx:idx+batch_size], corruption_type)
            batches.append((imgs0, mnist_lbls[idx:idx+batch_size], 0))
            # FMNIST batch b
            imgs1 = preprocess_and_normalize(fmnist_imgs[idx:idx+batch_size], corruption_type)
            batches.append((imgs1, fmnist_lbls[idx:idx+batch_size], 1))
            # KMNIST batch b
            imgs2 = preprocess_and_normalize(kmnist_imgs[idx:idx+batch_size], corruption_type)
            batches.append((imgs2, kmnist_lbls[idx:idx+batch_size], 2))
            
    return batches

# ==========================================
# 3. Fisher Sensitivity Computation
# ==========================================
def compute_joint_fisher(encoder_paths, head_paths, device):
    print("Computing joint layer-wise Fisher sensitivity prior...")
    tasks = ["mnist", "fmnist", "kmnist"]
    fisher_dict = {}
    
    # Names of trainable tensors in encoder
    # conv1.weight, conv1.bias, conv2.weight, conv2.bias, conv3.weight, conv3.bias, fc.weight, fc.bias
    dummy_enc = CNNEncoder()
    tensor_names = [name for name, _ in dummy_enc.named_parameters()]
    
    for name in tensor_names:
        fisher_dict[name] = []
        
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    for idx, task in enumerate(tasks):
        encoder = CNNEncoder().to(device)
        encoder.load_state_dict(torch.load(encoder_paths[idx], map_location=device))
        head = ClassifierHead().to(device)
        head.load_state_dict(torch.load(head_paths[idx], map_location=device))
        
        encoder.eval()
        head.eval()
        
        # Load calibration data (small subset of training data, say 500 samples)
        if task == "mnist":
            cal_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        elif task == "fmnist":
            cal_ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        elif task == "kmnist":
            cal_ds = datasets.KMNIST(root="./data", train=True, download=True, transform=transform)
            
        cal_loader = torch.utils.data.DataLoader(cal_ds, batch_size=64, shuffle=True)
        
        accum_grads = {}
        for name, param in encoder.named_parameters():
            accum_grads[name] = torch.zeros_like(param)
            
        num_samples = 0
        for images, labels in cal_loader:
            images = images.to(device)
            features = encoder(images)
            outputs = head(features)
            
            # Empirical Fisher using predicted classes
            log_probs = F.log_softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            loss = F.nll_loss(log_probs, preds, reduction='sum')
            
            encoder.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                for name, param in encoder.named_parameters():
                    if param.grad is not None:
                        accum_grads[name] += param.grad.data.pow(2)
            num_samples += images.size(0)
            if num_samples >= 500:
                break
                
        # Compute scalar average Fisher for each tensor
        for name, param in encoder.named_parameters():
            avg_fisher_tensor = accum_grads[name] / num_samples
            scalar_avg = avg_fisher_tensor.mean().item()
            fisher_dict[name].append(scalar_avg)
            
    # Compute joint average Fisher sensitivity across all tasks
    joint_fisher = {}
    for name in tensor_names:
        joint_fisher[name] = sum(fisher_dict[name]) / len(tasks)
        print(f"  {name}: {joint_fisher[name]:.6f}")
        
    return joint_fisher

# ==========================================
# 3.5. Class Prototype Extraction (CPA-Merge)
# ==========================================
def get_or_compute_prototypes(encoder_paths, device):
    proto_path = "checkpoints/prototypes.pth"
    if os.path.exists(proto_path):
        print("Loading pre-computed class prototypes...")
        return torch.load(proto_path, map_location=device)
        
    print("Computing class prototypes from training data...")
    tasks = ["mnist", "fmnist", "kmnist"]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 3 tasks, 10 classes, 128 feature dimension
    prototypes = torch.zeros(3, 10, 128, device=device)
    
    for idx, task in enumerate(tasks):
        encoder = CNNEncoder().to(device)
        encoder.load_state_dict(torch.load(encoder_paths[idx], map_location=device))
        encoder.eval()
        
        if task == "mnist":
            ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        elif task == "fmnist":
            ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        elif task == "kmnist":
            ds = datasets.KMNIST(root="./data", train=True, download=True, transform=transform)
            
        loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
        
        class_features = {c: [] for c in range(10)}
        counts = {c: 0 for c in range(10)}
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                features = encoder(images)
                for f, lbl in zip(features, labels):
                    c = lbl.item()
                    if len(class_features[c]) < 100:  # use 100 samples per class for stability
                        class_features[c].append(f)
                        counts[c] += 1
                if all(counts[c] >= 100 for c in range(10)):
                    break
                    
        for c in range(10):
            mean_feat = torch.stack(class_features[c]).mean(dim=0)
            normalized_feat = mean_feat / (mean_feat.norm(p=2) + 1e-8)
            prototypes[idx, c] = normalized_feat
            
    torch.save(prototypes, proto_path)
    print(f"Saved computed prototypes to {proto_path}")
    return prototypes

# ==========================================
# 4. Evaluation and Adaptation Loop
# ==========================================
def evaluate_method(method_name, test_batches, experts_sds, base_sd, expert_encoders, expert_heads, joint_fisher, device, lr=0.10, alpha_lfwa=0.5, ad_window=10, ad_threshold=0.40, prototypes=None):
    """
    Evaluates a specific model merging adaptation strategy on a test stream.
    """
    # Initialize merged model
    merged_model = MergedModel(experts_sds, base_sd, num_experts=3).to(device)
    
    # Active expert models (for self-labeling targets)
    experts = []
    heads = []
    for k in range(3):
        enc = CNNEncoder().to(device)
        enc.load_state_dict(expert_encoders[k])
        enc.eval()
        experts.append(enc)
        
        hd = ClassifierHead().to(device)
        hd.load_state_dict(expert_heads[k])
        hd.eval()
        heads.append(hd)
        
    # Setup TTA optimizer for raw_lambdas
    # Standard SGD with momentum as described in paper appendix
    optimizer = optim.SGD([merged_model.raw_lambdas], lr=lr, momentum=0.9)
    
    # Fisher preconditioning factor for each tensor idx
    # Names are in the exact order as merged_model.tensor_names
    fisher_scalars = []
    if joint_fisher is not None:
        for name in merged_model.tensor_names:
            # Fisher scalar
            fisher_scalars.append(joint_fisher[name])
    fisher_scalars = torch.tensor(fisher_scalars, device=device)
    # Fisher preconditioning multiplier (F_w + eps)^(-alpha)
    fisher_multipliers = (fisher_scalars + 1e-8).pow(-alpha_lfwa)
    
    # OPR parameters
    running_ema_loss = 0.0
    beta_ema = 0.90
    opr_threshold = 4.0 if "clean" in method_name else 2.5 # adjust based on domain
    
    # Tracking accuracy
    total_samples = 0
    correct_predictions = 0
    
    # Log loss trajectory
    loss_history = []
    
    # We step through the stream
    for t, (imgs, lbls, active_task_idx) in enumerate(test_batches):
        imgs, lbls = imgs.to(device), lbls.to(device)
        
        # --- CPA-Merge PD-Routing Step ---
        if "CPA-Merge" in method_name and prototypes is not None:
            with torch.no_grad():
                original_lambdas = merged_model.raw_lambdas.clone()
                merged_model.raw_lambdas.zero_()
                merged_sd = merged_model.get_merged_state_dict()
                z_anchor = torch.func.functional_call(merged_model.encoder, merged_sd, imgs)
                merged_model.raw_lambdas.copy_(original_lambdas)
                
                # L2 normalize
                z_anchor_norm = z_anchor / (z_anchor.norm(p=2, dim=1, keepdim=True) + 1e-8)
                
                S = torch.zeros(3, device=device)
                for k in range(3):
                    sims = torch.matmul(z_anchor_norm, prototypes[k].t())
                    max_sims, _ = sims.max(dim=1)
                    S[k] = max_sims.mean()
                    
                tau = 0.02
                lambda_prior = F.softmax(S / tau, dim=0)
                
                log_prior = torch.log(lambda_prior + 1e-8)
                merged_model.raw_lambdas.copy_(log_prior.unsqueeze(0).expand(merged_model.raw_lambdas.size(0), -1))
                
            # Reset optimizer momentum
            optimizer = optim.SGD([merged_model.raw_lambdas], lr=lr, momentum=0.9)
            
        # 1. Unmerged, frozen expert k targets
        with torch.no_grad():
            expert_features = experts[active_task_idx](imgs)
            expert_logits = heads[active_task_idx](expert_features)
            expert_probs = F.softmax(expert_logits, dim=1)
            
        # 2. Compute current self-labeling loss
        # Before any parameter resets, to evaluate on this batch
        merged_logits = merged_model(imgs, heads[active_task_idx])
        merged_probs = F.softmax(merged_logits, dim=1)
        
        # Self labeling loss: KL(expert || merged)
        # To avoid log(0)
        kl_loss = F.kl_div(F.log_softmax(merged_logits, dim=1), expert_probs, reduction='batchmean')
        loss_val = kl_loss.item()
        loss_history.append(loss_val)
        
        # Evaluate performance on this batch
        with torch.no_grad():
            _, preds = merged_logits.max(1)
            correct_predictions += preds.eq(lbls).sum().item()
            total_samples += imgs.size(0)
            
        # --- TEST TIME ADAPTATION STEP ---
        if method_name == "static":
            continue # no adaptation
            
        if "CPA-Merge" in method_name and prototypes is not None:
            # Fine-grained adaptation: entropy + contrastive alignment
            merged_logits = merged_model(imgs, heads[active_task_idx])
            probs = F.softmax(merged_logits, dim=1)
            ent_loss = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
            
            # Extract features of the current merged model
            merged_sd = merged_model.get_merged_state_dict()
            features = torch.func.functional_call(merged_model.encoder, merged_sd, imgs)
            features_norm = features / (features.norm(p=2, dim=1, keepdim=True) + 1e-8)
            
            M = torch.matmul(features_norm, prototypes[active_task_idx].t())
            max_probs, preds_merged = probs.max(dim=1)
            mask = (max_probs > 0.85).float()
            
            kappa = 0.1
            log_sims = F.log_softmax(M / kappa, dim=1)
            pred_log_sims = log_sims.gather(1, preds_merged.unsqueeze(1)).squeeze(1)
            masked_log_sims = pred_log_sims * mask
            num_masked = mask.sum()
            
            if num_masked > 0:
                contra_loss = -masked_log_sims.sum() / num_masked
            else:
                contra_loss = torch.tensor(0.0, device=device)
                
            beta_cpa = 0.1
            loss_total = ent_loss + beta_cpa * contra_loss
            
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
            with torch.no_grad():
                merged_model.raw_lambdas.clamp_(-10.0, 10.0)
                
            continue
            
        # OPR Reset Check (for PC-Merge and Fisher-PC-Merge)
        has_reset = False
        if "PC-Merge" in method_name:
            if t > 0:
                threshold = 4.0 if "clean" in method_name else 2.5
                if loss_val > threshold * running_ema_loss:
                    # Unsupervised Task Shift Detected!
                    # Reset Lambda to 0 (uniform weights)
                    with torch.no_grad():
                        merged_model.raw_lambdas.zero_()
                    # Reset optimizer momentum
                    optimizer = optim.SGD([merged_model.raw_lambdas], lr=lr, momentum=0.9)
                    # Recompute logits & loss
                    merged_logits = merged_model(imgs, heads[active_task_idx])
                    kl_loss = F.kl_div(F.log_softmax(merged_logits, dim=1), expert_probs, reduction='batchmean')
                    loss_val = kl_loss.item()
                    has_reset = True
                    
            # Update EMA loss
            if t == 0 or has_reset:
                running_ema_loss = loss_val
            else:
                running_ema_loss = beta_ema * running_ema_loss + (1.0 - beta_ema) * loss_val
                
        # --- AD-Merge Adaptive Dampening ---
        current_lr = lr
        if "AD-Merge" in method_name and len(loss_history) >= ad_window:
            avg_loss_win = sum(loss_history[-ad_window:]) / float(ad_window)
            if avg_loss_win > ad_threshold:
                current_lr = 0.0
                with torch.no_grad():
                    merged_model.raw_lambdas.zero_()
                    
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
            
        if current_lr == 0.0:
            continue
                
        # Zero gradients
        optimizer.zero_grad()
        
        # Adaptation updates
        if "PC-Merge" in method_name:
            # Class-Specific Gradient Projection
            # Group samples by predicted classes from expert model
            expert_preds = expert_logits.argmax(dim=1)
            unique_classes = expert_preds.unique()
            
            class_grads = {}
            for cls in unique_classes:
                mask = expert_preds.eq(cls)
                if mask.sum() == 0:
                    continue
                # Compute class loss
                cls_merged_logits = merged_model(imgs[mask], heads[active_task_idx])
                cls_expert_probs = expert_probs[mask]
                cls_kl_loss = F.kl_div(F.log_softmax(cls_merged_logits, dim=1), cls_expert_probs, reduction='batchmean')
                
                # Backward to get gradients on raw_lambdas
                merged_model.raw_lambdas.grad = None
                cls_kl_loss.backward(retain_graph=True)
                
                if merged_model.raw_lambdas.grad is not None:
                    class_grads[cls.item()] = merged_model.raw_lambdas.grad.clone()
                    
            # Perform pairwise projection on class gradients
            projected_grads = {}
            active_clses = list(class_grads.keys())
            for idx_a, cls_a in enumerate(active_clses):
                g_a = class_grads[cls_a].clone()
                for idx_b, cls_b in enumerate(active_clses):
                    if cls_a == cls_b:
                        continue
                    g_b = class_grads[cls_b]
                    dot_prod = torch.sum(g_a * g_b)
                    if dot_prod < 0:
                        norm_b_sq = torch.sum(g_b * g_b) + 1e-8
                        g_a = g_a - (dot_prod / norm_b_sq) * g_b
                projected_grads[cls_a] = g_a
                
            # Sum conflict-free gradients
            if len(projected_grads) > 0:
                g_final = torch.stack(list(projected_grads.values())).sum(dim=0)
            else:
                g_final = torch.zeros_like(merged_model.raw_lambdas)
                
            # If our proposed Fisher-PC-Merge, apply Fisher weight scaling directly to the final projected gradient!
            if "Fisher" in method_name:
                # Multiply final gradient of tensor w by (F_w + eps)^(-alpha)
                with torch.no_grad():
                    g_final = g_final * fisher_multipliers.unsqueeze(1)
                    
            # Set gradient and step
            merged_model.raw_lambdas.grad = g_final
            optimizer.step()
            
        else:
            # Standard TTA (AdaMerging) or LFWA (standard gradient step without OPR or projection)
            kl_loss.backward()
            
            if "LFWA" in method_name:
                # Multiply standard gradient of tensor w by (F_w + eps)^(-alpha)
                with torch.no_grad():
                    merged_model.raw_lambdas.grad = merged_model.raw_lambdas.grad * fisher_multipliers.unsqueeze(1)
                    
            optimizer.step()
            
        # Clamp raw lambdas to avoid extreme exploding coefficients (e.g. [-10.0, 10.0])
        # This is a standard optimization safety constraint in TTA
        with torch.no_grad():
            merged_model.raw_lambdas.clamp_(-10.0, 10.0)
            
    accuracy = 100. * correct_predictions / total_samples
    return accuracy
