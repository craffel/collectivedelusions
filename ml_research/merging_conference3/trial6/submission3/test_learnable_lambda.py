import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json

from model_routing import (
    generate_synthetic_dataset,
    train_experts,
    PCAPreprojector,
    set_seed,
    train_router,
    evaluate_router
)

# Custom BWS Router with learnable lambda_max
class LearnableLambdaBWS_Router(nn.Module):
    def __init__(self, L=12, G=4, d=4, K=4, activation='Sigmoid', init_lambda_max=0.3, init_bias=-2.0):
        super().__init__()
        self.L = L
        self.G = G
        self.d = d
        self.K = K
        self.activation_name = activation
        
        # Learnable lambda_max parameter
        self.lambda_max = nn.Parameter(torch.tensor(init_lambda_max))
        
        # Trainable routing weights and biases
        self.W = nn.Parameter(torch.zeros(G, K, d))
        self.B = nn.Parameter(torch.zeros(G, K))
        nn.init.normal_(self.W, std=0.01)
        nn.init.constant_(self.B, init_bias)
        
    def forward(self, psi):
        # psi shape: [B, d]
        logits = torch.einsum("gkd,bd->bgk", self.W, psi) + self.B.unsqueeze(0) # [B, G, K]
        
        if self.activation_name == 'Sigmoid':
            # Ensure lambda_max stays positive via softplus or absolute value
            lambda_val = torch.abs(self.lambda_max)
            alpha = lambda_val * torch.sigmoid(logits) # [B, G, K]
        elif self.activation_name == 'Softmax':
            alpha = torch.softmax(logits, dim=-1)
        else:
            alpha = logits
            
        # Repeat block-wise groups to match L layers
        M = self.L // self.G
        alpha_list = []
        for g in range(self.G):
            for _ in range(M):
                alpha_list.append(alpha[:, g])
        return alpha_list

# Training function customized for learnable parameters
def train_learnable_router(router, pca_proj, experts, calib_data, epochs=120, lr=0.05, lambda_wd=1e-4):
    router.train()
    X_cal, Y_cal, T_cal = calib_data
    X_cal_t = torch.tensor(X_cal)
    Y_cal_t = torch.tensor(Y_cal)
    
    # Project calibration features using PCA
    psi = pca_proj.project(X_cal_t)
    
    optimizer = optim.AdamW(router.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        alpha_list = router(psi)
        
        # Forward pass on expert linear layer activations
        logits_experts = torch.stack([experts[k](X_cal_t) for k in range(4)], dim=0) # [4, B, C]
        
        # Layer ensembling: average logits across all L layers
        logits_layers = []
        for l in range(router.L):
            alpha_l = alpha_list[l] # [B, K]
            # Blend expert logits for layer l
            blended_logits = torch.einsum("bk,kb...->b...", alpha_l, logits_experts) # [B, C]
            logits_layers.append(blended_logits)
            
        # Mean logits across all L layers
        logits = torch.stack(logits_layers, dim=0).mean(dim=0)
        loss = criterion(logits, Y_cal_t)
        
        # L2 weight regularization excluding biases and the lambda_max parameter
        l2_reg = torch.tensor(0.0, requires_grad=True)
        for name, param in router.named_parameters():
            if 'W' in name:
                l2_reg = l2_reg + torch.norm(param, p=2)**2
                
        total_loss = loss + lambda_wd * l2_reg
        total_loss.backward()
        optimizer.step()

def evaluate_learnable_router(router, pca_proj, experts, test_data, mode='Homogeneous_B256'):
    router.eval()
    K = 4
    
    with torch.no_grad():
        if mode == 'Homogeneous_B256':
            accuracies = []
            for k in range(K):
                X, Y = test_data[k]
                X_t = torch.tensor(X)
                Y_t = torch.tensor(Y)
                
                psi = pca_proj.project(X_t)
                alpha_list = router(psi)
                
                logits_experts = torch.stack([experts[i](X_t) for i in range(4)], dim=0)
                
                logits_layers = []
                for l in range(router.L):
                    alpha_l = alpha_list[l]
                    blended_logits = torch.einsum("bk,kb...->b...", alpha_l, logits_experts)
                    logits_layers.append(blended_logits)
                    
                logits = torch.stack(logits_layers, dim=0).mean(dim=0)
                preds = logits.argmax(dim=-1)
                acc = (preds == Y_t).float().mean().item() * 100
                accuracies.append(acc)
            return accuracies, np.mean(accuracies)

def main():
    seeds = [42, 43, 44, 45, 46]
    
    print("=" * 80)
    print("EVALUATING LEARNABLE TASK CEILING (lambda_max) ACROSS 5 SEEDS")
    print("=" * 80)
    
    static_accs = []
    learnable_accs = []
    final_lambdas = []
    
    for seed in seeds:
        set_seed(seed)
        train_data, test_data, calib_data = generate_synthetic_dataset(seed)
        experts = train_experts(train_data, test_data)
        
        pca_proj = PCAPreprojector(n_components=4)
        pca_proj.fit(calib_data[0])
        
        # 1. Static ceiling (lambda_max = 0.3)
        static_router = LearnableLambdaBWS_Router(
            L=12, G=4, d=4, K=4, activation='Sigmoid', init_lambda_max=0.3, init_bias=-2.0
        )
        # Freeze lambda_max to act as static
        static_router.lambda_max.requires_grad = False
        
        train_learnable_router(static_router, pca_proj, experts, calib_data, lr=0.05, lambda_wd=1e-4)
        _, static_mean = evaluate_learnable_router(static_router, pca_proj, experts, test_data)
        static_accs.append(static_mean)
        
        # 2. Learnable ceiling (initialized at 0.3)
        learnable_router = LearnableLambdaBWS_Router(
            L=12, G=4, d=4, K=4, activation='Sigmoid', init_lambda_max=0.3, init_bias=-2.0
        )
        learnable_router.lambda_max.requires_grad = True
        
        train_learnable_router(learnable_router, pca_proj, experts, calib_data, lr=0.05, lambda_wd=1e-4)
        _, learnable_mean = evaluate_learnable_router(learnable_router, pca_proj, experts, test_data)
        learnable_accs.append(learnable_mean)
        
        final_lambda = torch.abs(learnable_router.lambda_max).item()
        final_lambdas.append(final_lambda)
        
        print(f"Seed {seed} | Static (0.3) Acc: {static_mean:.2f}% | Learnable Acc: {learnable_mean:.2f}% | Final lambda_max: {final_lambda:.4f}")
        
    print("\n" + "=" * 80)
    print("FINAL SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Static lambda_max (0.3)      : {np.mean(static_accs):.2f}% \u00b1 {np.std(static_accs):.2f}%")
    print(f"Learnable lambda_max         : {np.mean(learnable_accs):.2f}% \u00b1 {np.std(learnable_accs):.2f}%")
    print(f"Convergenced lambda_max value: {np.mean(final_lambdas):.4f} \u00b1 {np.std(final_lambdas):.4f}")
    print("=" * 80)
    
if __name__ == '__main__':
    main()
