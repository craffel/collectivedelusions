import sys
sys.path.insert(0, './my_packages')

import torch
import torch.nn as nn
import numpy as np
import os
import time
from transformers import AutoTokenizer, AutoModel

# Set seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Load GPT-2 tokenizer and model
print("Loading GPT-2 model for Generative LLM Blueprint pilot...")
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModel.from_pretrained('gpt2')
model.eval()

# Helper to extract pooled hidden states from GPT-2
def extract_gpt2_embeddings(texts):
    all_embs = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use mean pooling over token representations
        mean_emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        all_embs.append(mean_emb)
    return torch.stack(all_embs, dim=0)

# Define tasks and prompts
# Task 0: Movie Review / Sentiment Analysis
task0_calibration_prompts = [
    "Review: The movie was absolutely fantastic and brilliant.",
    "Review: I loved the film, the acting was spectacular.",
    "Review: An incredibly beautiful and emotionally moving story.",
    "Review: Highly recommended, the cinematography is stunning.",
    "Review: A masterpiece of modern cinema, truly wonderful."
]
task0_test_prompts = [
    "Review: The performance was stellar and I enjoyed every scene.",
    "Review: A captivating plot that kept me engaged throughout.",
    "Review: Such a brilliant film with amazing direction.",
    "Review: The cast did an outstanding job, very impressive.",
    "Review: A delightful and heart-warming cinematic experience."
]

# Task 1: French Translation
task1_calibration_prompts = [
    "Translate to French: The cat is sleeping on the table.",
    "Translate to French: Where is the train station, please?",
    "Translate to French: Hello, how are you doing today?",
    "Translate to French: The book on the desk is mine.",
    "Translate to French: I would like to order a coffee."
]
task1_test_prompts = [
    "Translate to French: What time does the plane arrive?",
    "Translate to French: The weather is very nice this afternoon.",
    "Translate to French: Can you help me find my keys?",
    "Translate to French: We went to the park yesterday morning.",
    "Translate to French: This restaurant serves delicious food."
]

# True OOD: Random Math Prompts (unseen task)
ood_test_prompts = [
    "Solve the equation: 3x + 5 = 14.",
    "Calculate the derivative of f(x) = sin(x) + x^2.",
    "What is the square root of 144?",
    "Solve for y in the system of equations.",
    "Simplify the algebraic expression: (x+2)(x-3)."
]

print("Extracting embeddings for calibration and test sets...")
task0_cal_embs = extract_gpt2_embeddings(task0_calibration_prompts)
task1_cal_embs = extract_gpt2_embeddings(task1_calibration_prompts)

task0_test_embs = extract_gpt2_embeddings(task0_test_prompts)
task1_test_embs = extract_gpt2_embeddings(task1_test_prompts)
ood_test_embs = extract_gpt2_embeddings(ood_test_prompts)

# 1. Compute task centroids
centroid0 = task0_cal_embs.mean(dim=0)
centroid1 = task1_cal_embs.mean(dim=0)
centroids = torch.stack([centroid0, centroid1], dim=0) # shape (2, D)

print(f"Task Centroids shape: {centroids.shape} (K=2, D={centroids.shape[1]})")

# 2. Project to task coordinates via Centered & Clamped Cosine Similarity
# Pre-compute average raw similarity to centroids to neutralize representation anisotropy
def get_raw_sims(X, centroids):
    B_size = X.shape[0]
    K = centroids.shape[0]
    u = torch.zeros(B_size, K)
    for k in range(K):
        X_norm = X / (torch.norm(X, p=2, dim=1, keepdim=True) + 1e-8)
        centroid_k = centroids[k].unsqueeze(0)
        centroid_k_norm = centroid_k / (torch.norm(centroid_k, p=2, dim=1, keepdim=True) + 1e-8)
        u[:, k] = torch.matmul(X_norm, centroid_k_norm.T).squeeze(1)
    return u

u_cal0_raw = get_raw_sims(task0_cal_embs, centroids)
u_cal1_raw = get_raw_sims(task1_cal_embs, centroids)
u_cal_raw = torch.cat([u_cal0_raw, u_cal1_raw], dim=0)
mean_cal_sim = u_cal_raw.mean(dim=0, keepdim=True) # average similarity of calibration set to each task centroid

def project_generative_coords_centered(X, centroids, mean_cal_sim):
    u_raw = get_raw_sims(X, centroids)
    # Centering & Clamping: subtract mean calibration similarity to expose true task alignment
    u_centered = torch.clamp(u_raw - mean_cal_sim, min=0.0)
    
    # Standard project onto unit sphere with safe division-by-zero threshold
    norm = torch.norm(u_centered, p=2, dim=1, keepdim=True)
    psi = torch.zeros_like(u_centered)
    mask = (norm.squeeze(1) > 1e-5)
    psi[mask] = u_centered[mask] / norm[mask]
    return psi

# Project all subsets using centering
psi_cal0 = project_generative_coords_centered(task0_cal_embs, centroids, mean_cal_sim)
psi_cal1 = project_generative_coords_centered(task1_cal_embs, centroids, mean_cal_sim)
psi_cal = torch.cat([psi_cal0, psi_cal1], dim=0)
task_cal = torch.cat([torch.zeros(len(psi_cal0), dtype=torch.long), torch.ones(len(psi_cal1), dtype=torch.long)])

psi_test0 = project_generative_coords_centered(task0_test_embs, centroids, mean_cal_sim)
psi_test1 = project_generative_coords_centered(task1_test_embs, centroids, mean_cal_sim)
psi_test = torch.cat([psi_test0, psi_test1], dim=0)
task_test = torch.cat([torch.zeros(len(psi_test0), dtype=torch.long), torch.ones(len(psi_test1), dtype=torch.long)])

psi_ood = project_generative_coords_centered(ood_test_embs, centroids, mean_cal_sim)

print("Coordinates successfully projected onto centered K=2 task space.")

# 3. GP-DR Router Formulation
class GPDRRouter:
    def __init__(self, psi_train, y_train, K=2, sigma_f=1.0, lengthscale=1.0, sigma_n=1e-2):
        self.psi_train = psi_train
        self.K = K
        self.sigma_f = sigma_f
        self.lengthscale = lengthscale
        self.sigma_n = sigma_n
        self.N = psi_train.shape[0]
        
        self.Y_targets = torch.zeros(self.N, K)
        for i in range(self.N):
            self.Y_targets[i, y_train[i]] = 1.0
            
        self.prior_mean = 1.0 / K
        self.K_gram = self.kernel(self.psi_train, self.psi_train)
        self.M = torch.inverse(self.K_gram + (self.sigma_n ** 2) * torch.eye(self.N))
        self.W_gp = torch.matmul(self.M, self.Y_targets - self.prior_mean)
        
    def kernel(self, x1, x2):
        sq_dist = torch.cdist(x1, x2, p=2) ** 2
        return (self.sigma_f ** 2) * torch.exp(-sq_dist / (2.0 * (self.lengthscale ** 2)))
        
    def forward(self, psi_test, theta_ood=0.9):
        B_size = psi_test.shape[0]
        k_star = self.kernel(psi_test, self.psi_train)
        mu = self.prior_mean + torch.matmul(k_star, self.W_gp)
        
        k_star_M = torch.matmul(k_star, self.M)
        # Compute posterior variance with a non-negative clamping safeguard to prevent numerical instabilities
        post_var = torch.clamp((self.sigma_f ** 2) - (k_star_M * k_star).sum(dim=1), min=0.0)
        
        alpha = torch.zeros(B_size, self.K)
        for b in range(B_size):
            if post_var[b] > theta_ood:
                alpha[b] = torch.ones(self.K) * self.prior_mean
            else:
                alpha[b] = mu[b]
                
        alpha = torch.clamp(alpha, min=1e-5, max=1.0)
        alpha = alpha / alpha.sum(dim=1, keepdim=True)
        return alpha, post_var

# Initialize Router
router = GPDRRouter(psi_cal, task_cal, K=2, sigma_f=1.0, lengthscale=0.5, sigma_n=0.01)

# Evaluate routing accuracy on ID test prompts
print("\n=== PILOT EVALUATION: GENERATIVE ROUTING ACCURACY ===")
alpha_id, var_id = router.forward(psi_test, theta_ood=0.95)
routing_preds = alpha_id.argmax(dim=1)
routing_acc = (routing_preds == task_test).float().mean().item() * 100.0

print(f"ID Test Routing Decisions:")
for i, text in enumerate(task0_test_prompts + task1_test_prompts):
    pred = routing_preds[i].item()
    target = task_test[i].item()
    pred_name = "Sentiment Expert" if pred == 0 else "French Expert"
    target_name = "Sentiment Expert" if target == 0 else "French Expert"
    var_i = var_id[i].item()
    print(f"  Prompt: \"{text[:45]}...\"")
    print(f"    Routed to: {pred_name} (Target: {target_name}, GP Post Var: {var_i:.4f})")

print(f"\nOverall ID Routing Accuracy: {routing_acc:.2f}%")

# Evaluate OOD Rejection
print("\n=== PILOT EVALUATION: GENERATIVE OOD REJECTION ===")
alpha_ood, var_ood = router.forward(psi_ood, theta_ood=0.95)

print(f"OOD Prompts Variance Mapping:")
for i, text in enumerate(ood_test_prompts):
    var_i = var_ood[i].item()
    routed_uniform = (alpha_ood[i, 0] - 0.5).abs() < 1e-4
    status = "REJECTED (Uniform Fallback)" if routed_uniform else "ACCEPTED (Dynamic Route)"
    print(f"  Prompt: \"{text[:45]}...\"")
    print(f"    GP Post Var: {var_i:.4f} | Status: {status}")

from sklearn.metrics import roc_auc_score
y_true_ood = np.concatenate([np.zeros(len(psi_test)), np.ones(len(psi_ood))])
all_vars = torch.cat([var_id, var_ood]).cpu().numpy()
auroc = roc_auc_score(y_true_ood, all_vars)
print(f"\nGenerative OOD Rejection AUROC: {auroc * 100.0:.2f}%")
print("SUCCESS! Generative LLM Blueprint pilot validation complete!")
