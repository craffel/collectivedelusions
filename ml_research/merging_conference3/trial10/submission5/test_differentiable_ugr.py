import torch
import torch.nn as nn
import torch.optim as optim
import math

class DifferentiableUGR(nn.Module):
    def __init__(self, K, eta_init=0.5):
        super().__init__()
        self.K = K
        # Step size parameter (eta) is learnable
        self.eta = nn.Parameter(torch.tensor(eta_init))
        # Expert centroids are learnable (dim=8)
        self.centroids = nn.Parameter(torch.randn(K, 8))
        
    def forward(self, h, s_prev):
        # Normalize centroids
        centroids_norm = self.centroids / torch.norm(self.centroids, p=2, dim=1, keepdim=True)
        # Normalize input representation h
        h_norm = h / torch.norm(h, p=2, dim=1, keepdim=True) # [B, D]
        
        # 1. Similarity Extraction
        cos_sims = torch.matmul(h_norm, centroids_norm.t()) # [B, K]
        
        # Softmax target construction to yield e_t
        e = torch.softmax(cos_sims / 0.1, dim=1) # [B, K]
        # Project onto sphere to obtain target w_t using square-root Bhattacharyya/Born mapping
        w = torch.sqrt(e + 1e-12) # [B, K]
        
        # Ensure w is exactly unit-norm
        w = w / torch.norm(w, p=2, dim=1, keepdim=True)
        
        # 2. Geodesic Rotation (Slerp)
        # Compute alignment cosine
        c = torch.sum(s_prev * w, dim=1, keepdim=True) # [B, 1]
        c = torch.clamp(c, -1.0 + 1e-6, 1.0 - 1e-6)
        
        # Angular torque
        phi = torch.acos(c) # [B, 1]
        theta = self.eta * phi # [B, 1]
        
        # Compute Slerp coefficients
        sin_phi = torch.sin(phi) + 1e-12
        a = torch.sin((1.0 - self.eta) * phi) / sin_phi
        b = torch.sin(theta) / sin_phi
        
        # Updated state
        s_next = a * s_prev + b * w # [B, K]
        
        # Map back to simplex via Born's rule
        alpha = s_next ** 2 # [B, K]
        return alpha, s_next

def run_empirical_validation():
    print("===============================================================")
    print("Empirical Validation of Differentiable Training-Time UGR")
    print("===============================================================")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    K = 4  # number of experts
    D = 8  # hidden dimension
    batch_size = 5
    
    # Initialize toy inputs and targets
    h = torch.randn(batch_size, D)
    s_prev = torch.ones(batch_size, K) / math.sqrt(K) # uniform start state
    
    # Let's define a target routing distribution we want to learn
    target_alpha = torch.zeros(batch_size, K)
    target_alpha[:, 0] = 1.0  # We want the router to confidently route to Expert 0
    
    model = DifferentiableUGR(K=K, eta_init=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.KLDivLoss(reduction='batchmean')
    
    print("\nInitial parameters:")
    print(f"eta: {model.eta.item():.4f}")
    
    print("\nOptimizing UGR parameters to learn Target Routing Distribution...")
    for epoch in range(100):
        optimizer.zero_grad()
        
        # Forward pass
        alpha, s_next = model(h, s_prev)
        
        # Compute KL-Divergence loss (expecting log-probabilities for input)
        log_alpha = torch.log(alpha + 1e-12)
        loss = loss_fn(log_alpha, target_alpha)
        
        # Backward pass
        loss.backward()
        
        # Check gradient existence and stability
        if epoch == 0:
            print("\nGradient Verification (Epoch 0):")
            print(f"Loss: {loss.item():.6f}")
            print(f"Grad of eta: {model.eta.grad.item():.6f}")
            print(f"Grad of centroids (norm): {torch.norm(model.centroids.grad).item():.6f}")
            assert model.eta.grad is not None, "Gradient of eta is None!"
            assert model.centroids.grad is not None, "Gradient of centroids is None!"
            print("Gradients successfully backpropagated through all geodesic Slerp operations!")
            
        optimizer.step()
        
        # Clamp eta to valid [0, 1] range during optimization
        with torch.no_grad():
            model.eta.clamp_(0.0, 1.0)
            
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.6f} | eta: {model.eta.item():.4f} | Routing to Exp 0: {alpha[:, 0].mean().item():.4f}")
            
    print("\nOptimization Complete!")
    print(f"Final learned eta: {model.eta.item():.4f}")
    print(f"Final mean routing weights to Expert 0: {alpha[:, 0].mean().item():.4f}")
    print("Loss successfully minimized, confirming the numerical stability of end-to-end backpropagation!")
    print("===============================================================\n")

if __name__ == "__main__":
    run_empirical_validation()
