import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from run_experiments import (
    AIR, get_task_signatures, generate_stream, set_seed, propagate_sandbox, evaluate_output
)
from run_advanced_evals import evaluate_output_custom

class TaskCAE(nn.Module):
    def __init__(self, D, d):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(D, d),
            nn.Sigmoid()
        )
        self.decoder = nn.Linear(d, D)
        
    def forward(self, x):
        h = self.encoder(x)
        x_recon = self.decoder(h)
        return x_recon, h

def train_cae_models(v_signatures, sigmas, d=4, num_samples=64, epochs=300, lr=0.01, lambda_contract=1e-3, config="orthogonal"):
    K, D = v_signatures.shape
    cae_models = []
    
    for k in range(K):
        # Generate calibration samples for task k
        if config == "nonlinear":
            # Heavy-tailed Student's t-noise with df=3
            z = torch.randn(num_samples, D)
            v_chi = torch.sum(torch.randn(3, num_samples, D) ** 2, dim=0) / 3.0
            noise = (z / torch.sqrt(v_chi + 1e-8)) * sigmas[k]
            samples_linear = v_signatures[k].unsqueeze(0) + noise
            # Apply non-linear sinusoidal-quadratic warping (non-invertible)
            samples = torch.sin(samples_linear) + 0.1 * torch.sign(samples_linear) * (samples_linear ** 2)
        else:
            noise = torch.randn(num_samples, D) * sigmas[k]
            samples = v_signatures[k].unsqueeze(0) + noise
            
        model = TaskCAE(D, d)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Train CAE
        for epoch in range(epochs):
            optimizer.zero_grad()
            x_recon, h = model(samples)
            recon_loss = F.mse_loss(x_recon, samples)
            
            # Analytical contractive penalty (Frobenius norm of Jacobian)
            W = model.encoder[0].weight # d x D
            W_norm = torch.sum(W ** 2, dim=1) # d
            h_deriv = h * (1.0 - h) # B x d
            jacobian_norm = torch.mean(torch.sum((h_deriv ** 2) * W_norm.unsqueeze(0), dim=1))
            
            loss = recon_loss + lambda_contract * jacobian_norm
            loss.backward()
            optimizer.step()
            
        cae_models.append(model)
        
    return cae_models

def compute_cae_projections(z_t, cae_models, kappa=2.0):
    # z_t shape: B x D
    B, D = z_t.shape
    K = len(cae_models)
    e = torch.zeros(B, K, device=z_t.device)
    
    # Normalize z_t
    z_norm = z_t / (torch.norm(z_t, p=2, dim=1, keepdim=True) + 1e-8)
    
    for k in range(K):
        model = cae_models[k]
        model.eval()
        with torch.no_grad():
            x_recon, _ = model(z_norm)
            # Reconstruct error: L2 distance
            recon_err = torch.norm(z_norm - x_recon, p=2, dim=1)
            # Map reconstruction error to [0, 1] using Gaussian kernel
            e[:, k] = torch.exp(-kappa * (recon_err ** 2))
            
    return e

def run_cae_experiment():
    print("=== Running Non-Linear Contractive Autoencoder (CAE) Projection Experiment ===")
    set_seed(42)
    
    # Experiment settings
    K = 4
    D = 192
    B = 16
    T_cal = 32
    T_test = 200
    sigmas = [0.15, 0.15, 0.15, 0.15]
    
    # Evaluate across Orthogonal and Nonlinear manifolds
    configs = ["orthogonal", "nonlinear"]
    
    for cfg in configs:
        print(f"\n--- Manifold Config: {cfg.upper()} ---")
        v_signatures = get_task_signatures(cfg)
        
        # Train Contractive Autoencoders as alternative projection spaces
        cae_models = train_cae_models(v_signatures, sigmas, d=4, config=cfg)
        
        # Generate Calibration streams
        cal_hom_h3, cal_hom_target_y = generate_stream(v_signatures, sigmas, stream_type="homogeneous", T=T_cal, B=B, config=cfg)
        
        # Generate Test streams
        hom_test_h3, hom_test_target_y = generate_stream(v_signatures, sigmas, stream_type="homogeneous", T=T_test, B=B, config=cfg)
        het_test_h3, het_test_target_y = generate_stream(v_signatures, sigmas, stream_type="heterogeneous", T=T_test, B=B, config=cfg)
        
        # Compute CAE projection coordinates for calibration and test
        def get_cae_coords(h3):
            T, B, D = h3.shape
            coords = torch.zeros(T, B, K)
            for t in range(T):
                coords[t] = compute_cae_projections(h3[t], cae_models, kappa=2.0)
            return coords
            
        e_cal = get_cae_coords(cal_hom_h3)
        e_hom = get_cae_coords(hom_test_h3)
        e_het = get_cae_coords(het_test_h3)
        
        # Train AIR model using CAE projection coordinates
        model = AIR(K, N_steps=5, eta_test=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Training loop
        for epoch in range(200):
            optimizer.zero_grad()
            model.reset(e_cal[0])
            ce_loss = 0.0
            smoothness_loss = 0.0
            prev_alpha = None
            
            for t in range(1, T_cal):
                logits_t = model(e_cal[t], return_logits=True)
                ce_loss += F.cross_entropy(logits_t, cal_hom_target_y[t])
                
                alpha_t = F.softmax(logits_t, dim=1)
                if prev_alpha is not None:
                    smoothness_loss += torch.sum((alpha_t - prev_alpha) ** 2, dim=1).mean()
                prev_alpha = alpha_t
                
            ce_loss = ce_loss / (T_cal - 1)
            smoothness_loss = smoothness_loss / (T_cal - 2)
            loss = ce_loss + 0.05 * smoothness_loss
            loss.backward()
            optimizer.step()
            
        # Evaluate AIR on homogeneous and heterogeneous streams using CAE
        def eval_air(coords, h3, target_y):
            T, B, _ = coords.shape
            model.reset(coords[0])
            all_alphas = []
            all_align_accs = []
            all_cat_accs = []
            
            for t in range(T):
                if t == 0:
                    tau = torch.exp(model.w) + model.tau_min
                    logits = model.mu_prev / tau.unsqueeze(0)
                    alpha_t = F.softmax(logits, dim=1)
                else:
                    alpha_t = model(coords[t])
                    
                all_alphas.append(alpha_t.unsqueeze(0))
                h14 = propagate_sandbox(h3[t], alpha_t, v_signatures)
                cat_acc, align_acc, _ = evaluate_output_custom(h14, v_signatures, target_y[t])
                all_align_accs.append(align_acc.item())
                all_cat_accs.append(cat_acc.item())
                
            all_alphas = torch.cat(all_alphas, dim=0)
            
            # Compute Jitter
            jitters = []
            for b in range(B):
                stream_alphas = all_alphas[:, b, :]
                diff = torch.abs(stream_alphas[1:] - stream_alphas[:-1])
                l1_diff = torch.sum(diff, dim=1)
                jitters.append(l1_diff.mean().item())
                
            return np.mean(all_cat_accs), np.mean(all_align_accs), np.mean(jitters)
            
        hom_cat, hom_align, hom_jit = eval_air(e_hom, hom_test_h3, hom_test_target_y)
        het_cat, het_align, het_jit = eval_air(e_het, het_test_h3, het_test_target_y)
        
        print(f"Homogeneous Test - Cat Acc: {hom_cat*100.0:.2f}%, Align Acc: {hom_align*100.0:.2f}%, Jitter: {hom_jit:.4f}")
        print(f"Heterogeneous Test - Cat Acc: {het_cat*100.0:.2f}%, Align Acc: {het_align*100.0:.2f}%, Jitter: {het_jit:.4f}")

if __name__ == "__main__":
    run_cae_experiment()
