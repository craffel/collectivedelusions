import os
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Global configuration of layers (stages) matching the 12 layers of ViT-B/16
STAGES = [f"layer{i}" for i in range(1, 13)]

# -------------------------------------------------------------
# 1. SYNTHETIC IMAGE GENERATOR (CIRCLE, SQUARE, TRIANGLE, CROSS)
# -------------------------------------------------------------

def generate_shape_image(shape_type, size=224, seed=None):
    """
    Generates a 3-channel PIL image containing a randomized geometric shape
    on a noisy background to simulate diverse real-world task samples.
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Create noisy background
    bg_color = np.random.randint(20, 100, size=3)
    img = Image.new("RGB", (size, size), tuple(bg_color))
    draw = ImageDraw.Draw(img)
    
    # Add background pixel noise
    pixels = np.array(img)
    noise = np.random.normal(0, 15, pixels.shape).astype(np.int16)
    pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels)
    draw = ImageDraw.Draw(img)
    
    # Randomize shape properties
    shape_color = tuple(np.random.randint(150, 255, size=3))
    center_x = np.random.randint(int(size*0.3), int(size*0.7))
    center_y = np.random.randint(int(size*0.3), int(size*0.7))
    r = np.random.randint(int(size*0.15), int(size*0.3))
    
    if shape_type == "circle":
        draw.ellipse([center_x - r, center_y - r, center_x + r, center_y + r], fill=shape_color)
    elif shape_type == "square":
        draw.rectangle([center_x - r, center_y - r, center_x + r, center_y + r], fill=shape_color)
    elif shape_type == "triangle":
        p1 = (center_x, center_y - r)
        p2 = (center_x - r, center_y + r)
        p3 = (center_x + r, center_y + r)
        draw.polygon([p1, p2, p3], fill=shape_color)
    elif shape_type == "cross":
        w = np.random.randint(8, 20)
        draw.rectangle([center_x - w, center_y - r, center_x + w, center_y + r], fill=shape_color)
        draw.rectangle([center_x - r, center_y - w, center_x + r, center_y + w], fill=shape_color)
        
    return img

# -------------------------------------------------------------
# 2. FEATURE EXTRACTION PIPELINE USING PRE-TRAINED VIT-B/16
# -------------------------------------------------------------

class FeatureExtractor:
    """
    Hooks into a pre-trained ViT-B/16 to extract activation representations
    from the outputs of its 12 encoder layers.
    """
    def __init__(self):
        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        self.model.eval()
        
        self.activations = {}
        self.hooks = []
        
        # Register hooks for all 12 encoder layers
        for i in range(12):
            layer_name = f"layer{i+1}"
            self.hooks.append(
                self.model.encoder.layers[i].register_forward_hook(self._get_hook(layer_name))
            )
            
        # Image normalization transforms matching ImageNet-1k pre-training for ViT
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _get_hook(self, name):
        def hook(model, input, output):
            # Spatially/sequence-wise average pool the token features [B, S, D] to form [B, D]
            # output has shape [B, 197, 768] (1 CLS token + 196 patch tokens)
            self.activations[name] = torch.mean(output, dim=1).detach()
        return hook
        
    def extract(self, img):
        tensor = self.transform(img).unsqueeze(0) # add batch dimension
        with torch.no_grad():
            self.model(tensor)
            
        # Process and normalize activations at each layer
        features = {}
        for name in STAGES:
            feat = self.activations[name].squeeze(0).numpy() # shape: [D] (768)
            norm = np.linalg.norm(feat)
            features[name] = feat / norm if norm > 0 else feat
            
        return features
        
    def close(self):
        for h in self.hooks:
            h.remove()

# -------------------------------------------------------------
# 3. ROUTING ENGINE IMPLEMENTATIONS (SABLE, SPS-ZCA, CHEMMERGE)
# -------------------------------------------------------------

def run_sable_routing(features, centroids, temp=0.05):
    """
    SABLE: Stateless cosine similarity Softmax stage-by-stage.
    `features` is a list of dicts. `centroids` is a dict of arrays per stage.
    """
    num_samples = len(features)
    L = len(STAGES)
    K = len(centroids[STAGES[0]])
    
    weights = []
    for i in range(num_samples):
        sample_weights = []
        for l, stage in enumerate(STAGES):
            h = features[i][stage]
            # Compare to layer-specific centroids
            sims = np.array([np.dot(h, centroids[stage][k]) for k in range(K)])
            sims_stable = sims - np.max(sims)
            exp_sims = np.exp(sims_stable / temp)
            alpha = exp_sims / np.sum(exp_sims)
            sample_weights.append(alpha)
        weights.append(sample_weights)
    return np.array(weights) # Shape: [N, L, K]

def run_sps_zca_routing(features, centroids, expected_sims, temp=0.001):
    """
    SPS-ZCA: Stateless nearest-centroid similarity with dispersion calibration.
    """
    num_samples = len(features)
    L = len(STAGES)
    K = len(centroids[STAGES[0]])
    
    weights = []
    for i in range(num_samples):
        sample_weights = []
        for l, stage in enumerate(STAGES):
            h = features[i][stage]
            sims = np.array([np.dot(h, centroids[stage][k]) for k in range(K)])
            # Apply IDC dispersion calibration per stage
            cal_sims = sims / expected_sims[stage]
            cal_sims_stable = cal_sims - np.max(cal_sims)
            exp_sims = np.exp(cal_sims_stable / temp)
            alpha = exp_sims / np.sum(exp_sims)
            sample_weights.append(alpha)
        weights.append(sample_weights)
    return np.array(weights) # Shape: [N, L, K]

def run_chemmerge_routing(features, centroids, temp=0.01, delta_t=1.5, k_decay=0.3):
    """
    ChemMerge: Non-Equilibrium Chemical Reaction Kinetics for Dynamic Model Merging.
    Tracks state concentrations layer-by-layer across residual stages.
    """
    num_samples = len(features)
    L = len(STAGES)
    K = len(centroids[STAGES[0]])
    
    weights = []
    for i in range(num_samples):
        # Initialize concentrations at stage 1: C = 1/K
        C_layer = np.ones(K) / K
        sample_weights = []
        
        for l, stage in enumerate(STAGES):
            h = features[i][stage]
            # 1. Cosine similarity
            sims = np.array([np.dot(h, centroids[stage][k]) for k in range(K)])
            # 2. Subtract max
            sims_stable = sims - np.max(sims)
            # 3. Arrhenius rate equation
            exp_u = np.exp(sims_stable / temp)
            k_rate = exp_u / np.sum(exp_u)
            
            # 4. Discretized Euler step update with clipping to [0, 1]
            C_next = C_layer + delta_t * (k_rate * (1.0 - C_layer) - k_decay * C_layer)
            C_layer = np.clip(C_next, 0.0, 1.0)
            
            # 5. Law of Mass Action normalization to form ensembling weights
            alpha = C_layer / np.sum(C_layer)
            sample_weights.append(alpha)
            
        weights.append(sample_weights)
    return np.array(weights) # Shape: [N, L, K]

# -------------------------------------------------------------
# 4. MAIN EXPERIMENTAL PIPELINE
# -------------------------------------------------------------

def evaluate_routing_accuracy(weights, y_true):
    """
    Computes routing accuracy (how often highest weight is equal to true task).
    Weights shape: [N, L, K]. We measure final-stage routing accuracy (l = L - 1).
    """
    num_samples = len(y_true)
    L = weights.shape[1]
    
    correct = 0
    for i in range(num_samples):
        final_weights = weights[i, L-1]
        pred_task = np.argmax(final_weights)
        if pred_task == y_true[i]:
            correct += 1
    return correct / num_samples

def compute_routing_jitter(weights):
    """
    Computes mean routing weight jitter (layer-to-layer ensembling weight variance)
    across all samples. Lower is better (smoother trajectories).
    """
    # Shape: [N, L, K]
    N, L, K = weights.shape
    jitters = []
    for i in range(N):
        jitter_sample = 0
        for l in range(1, L):
            jitter_sample += np.sum((weights[i, l] - weights[i, l-1])**2)
        jitters.append(jitter_sample / (L - 1))
    return np.mean(jitters)

def run_real_world_validation():
    print("====================================================")
    print("RUNNING REAL-WORLD PRE-TRAINED VISION TRANSFORMER (ViT) ACCELERATION")
    print("====================================================")
    
    extractor = FeatureExtractor()
    shape_types = ["circle", "square", "triangle", "cross"]
    K = len(shape_types)
    
    seeds = list(range(42, 47)) # 5 independent evaluation seeds
    num_cal = 32 # samples per task for calibration
    num_eval = 25 # samples per task for evaluation
    
    results = {
        "SABLE": {"acc": [], "jitter": []},
        "SPS-ZCA": {"acc": [], "jitter": []},
        "ChemMerge (Ours)": {"acc": [], "jitter": []}
    }
    
    sample_trajectory_data = {} # For plotting example trajectories
    
    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        np.random.seed(seed)
        
        # 1. Generate Calibration Dataset
        print(f"Generating {num_cal * K} calibration images...")
        cal_features = []
        cal_y = []
        for k, shape in enumerate(shape_types):
            for _ in range(num_cal):
                img = generate_shape_image(shape, seed=seed + _ + k*100)
                feats = extractor.extract(img)
                cal_features.append(feats)
                cal_y.append(k)
                
        # 2. Extract Layer-Specific Centroids (Layer-Specific Centroid Routing)
        print("Extracting layer-specific task centroids...")
        centroids = {stage: [] for stage in STAGES}
        for stage in STAGES:
            for k in range(K):
                k_feats = [cal_features[i][stage] for i in range(len(cal_y)) if cal_y[i] == k]
                centroid = np.mean(k_feats, axis=0)
                centroid = centroid / np.linalg.norm(centroid) # normalize
                centroids[stage].append(centroid)
            centroids[stage] = np.array(centroids[stage])
            
        # Calculate layer-specific expected similarities for SPS-ZCA's calibration
        expected_sims = {stage: [] for stage in STAGES}
        for stage in STAGES:
            for k in range(K):
                k_feats = [cal_features[i][stage] for i in range(len(cal_y)) if cal_y[i] == k]
                sims = [np.dot(f, centroids[stage][k]) for f in k_feats]
                expected_sims[stage].append(np.mean(sims))
            expected_sims[stage] = np.array(expected_sims[stage])
            
        # 3. Generate Evaluation Dataset (Mixed heterogeneous stream)
        print(f"Generating {num_eval * K} evaluation images...")
        eval_features = []
        eval_y = []
        for k, shape in enumerate(shape_types):
            for _ in range(num_eval):
                img = generate_shape_image(shape, seed=seed + _ + k*200 + 1000)
                feats = extractor.extract(img)
                eval_features.append(feats)
                eval_y.append(k)
                
        eval_y = np.array(eval_y)
        
        # 4. Evaluate Routing Engines
        print("Evaluating routing performance across all methods...")
        
        # SABLE (stateless) - Using temperature 0.05
        weights_sable = run_sable_routing(eval_features, centroids, temp=0.05)
        sable_acc = evaluate_routing_accuracy(weights_sable, eval_y)
        sable_jitter = compute_routing_jitter(weights_sable)
        results["SABLE"]["acc"].append(sable_acc)
        results["SABLE"]["jitter"].append(sable_jitter)
        
        # SPS-ZCA (stateless + calibrated) - Using temperature 0.001
        weights_zca = run_sps_zca_routing(eval_features, centroids, expected_sims, temp=0.001)
        zca_acc = evaluate_routing_accuracy(weights_zca, eval_y)
        zca_jitter = compute_routing_jitter(weights_zca)
        results["SPS-ZCA"]["acc"].append(zca_acc)
        results["SPS-ZCA"]["jitter"].append(zca_jitter)
        
        # ChemMerge (Ours) - Using temperature 0.01
        weights_chem = run_chemmerge_routing(eval_features, centroids, temp=0.01, delta_t=1.5, k_decay=0.3)
        chem_acc = evaluate_routing_accuracy(weights_chem, eval_y)
        chem_jitter = compute_routing_jitter(weights_chem)
        results["ChemMerge (Ours)"]["acc"].append(chem_acc)
        results["ChemMerge (Ours)"]["jitter"].append(chem_jitter)
        
        # Keep example trajectory for visual validation
        if seed == 42:
            # We save the first sample of each task type (Task 0 = Circle)
            task_idx = 0 # first circle image
            sample_trajectory_data["SABLE"] = weights_sable[task_idx]
            sample_trajectory_data["SPS-ZCA"] = weights_zca[task_idx]
            sample_trajectory_data["ChemMerge"] = weights_chem[task_idx]
            
        print(f"  SABLE           : Acc = {sable_acc*100:5.2f}%, Jitter = {sable_jitter:6.4f}")
        print(f"  SPS-ZCA         : Acc = {zca_acc*100:5.2f}%, Jitter = {zca_jitter:6.4f}")
        print(f"  ChemMerge (Ours): Acc = {chem_acc*100:5.2f}%, Jitter = {chem_jitter:6.4f}")
        
    extractor.close()
    
    # --- Print Aggregated Results ---
    print("\n====================================================")
    print("AGGREGATED PERFORMANCE ON PRE-TRAINED ViT-B/16")
    print("====================================================")
    for m in results:
        m_acc = np.mean(results[m]["acc"]) * 100
        std_acc = np.std(results[m]["acc"]) * 100
        m_jit = np.mean(results[m]["jitter"])
        std_jit = np.std(results[m]["jitter"])
        print(f"{m:<16} | Routing Accuracy: {m_acc:5.2f}% +/- {std_acc:4.2f}% | Routing Jitter: {m_jit:6.4f} +/- {std_jit:6.4f}")
    print("====================================================")
    
    # --- Generate Visualization of Real-World Trajectories ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    x_ticks = [f"L{i}" for i in range(1, 13)]
    
    # Plot example trajectories for a Circle Image (Task 0)
    # We plot the ensembling weight of the correct task (Task 0) across network depth (layers)
    # sample_trajectory_data[method] has shape [L, K]
    
    # SABLE Trajectory
    for k in range(K):
        lbl = f"Expert {k} ({shape_types[k]})"
        axes[0].plot(x_ticks, sample_trajectory_data["SABLE"][:, k], "o-", color=colors[k], label=lbl, linewidth=2)
    axes[0].set_title("SABLE (Stateless Cos-Sim Softmax)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Encoder Layer", fontsize=10)
    axes[0].set_ylabel("Ensembling Weight", fontsize=10)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=9, loc="lower left")
    
    # SPS-ZCA Trajectory
    for k in range(K):
        axes[1].plot(x_ticks, sample_trajectory_data["SPS-ZCA"][:, k], "s-", color=colors[k], linewidth=2)
    axes[1].set_title("SPS-ZCA (Stateless Calibrated)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Encoder Layer", fontsize=10)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    # ChemMerge Trajectory
    for k in range(K):
        axes[2].plot(x_ticks, sample_trajectory_data["ChemMerge"][:, k], "D-", color=colors[k], linewidth=2)
    axes[2].set_title("ChemMerge (Ours: Chemical Kinetics ODE)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Encoder Layer", fontsize=10)
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(True, linestyle="--", alpha=0.5)
    
    plt.suptitle("Ensembling Weight Trajectories across Pre-Trained ViT-B/16 Encoder Layers", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("submission/results", exist_ok=True)
    plt.savefig("results/real_world_vit_test.png", dpi=150)
    plt.savefig("submission/results/real_world_vit_test.png", dpi=150)
    plt.close()
    
    print("\nReal-world trajectory visualization saved to results/real_world_vit_test.png and submission/results/real_world_vit_test.png")

if __name__ == "__main__":
    run_real_world_validation()
