import torch
import torch.nn.functional as F

def compute_euclidean_cpal(features, prototypes):
    # features: (B, D), prototypes: (K*C, D)
    feats_sq = torch.sum(features**2, dim=-1, keepdim=True)
    protos_sq = torch.sum(prototypes**2, dim=-1, keepdim=True).t()
    eucl_dists = feats_sq - 2.0 * torch.matmul(features, prototypes.t()) + protos_sq
    eucl_dists = torch.clamp(eucl_dists, min=0.0)
    min_dists, _ = eucl_dists.min(dim=-1)
    sigma = eucl_dists.mean().detach() + 1e-6
    loss = torch.mean(min_dists / sigma + torch.logsumexp(-eucl_dists / sigma, dim=-1))
    return loss

def compute_cosine_cpal(features, prototypes):
    # features: (B, D), prototypes: (K*C, D)
    features_norm = F.normalize(features, p=2, dim=-1)
    prototypes_norm = F.normalize(prototypes, p=2, dim=-1)
    cos_sims = torch.matmul(features_norm, prototypes_norm.t())
    cos_dists = 1.0 - cos_sims
    min_dists, _ = cos_dists.min(dim=-1)
    sigma = cos_dists.mean().detach() + 1e-6
    loss = torch.mean(min_dists / sigma + torch.logsumexp(-cos_dists / sigma, dim=-1))
    return loss

def test_scale_invariance():
    print("Testing scale invariance...")
    B, D, N = 10, 128, 20
    torch.manual_seed(42)
    
    # Generate random features and prototypes
    features = torch.randn(B, D).abs() + 0.1 # strictly positive to mimic ReLU
    prototypes = torch.randn(N, D).abs() + 0.1
    
    # Scale features and prototypes
    scale_feat = torch.rand(B, 1) * 10.0 + 1.0
    scale_proto = torch.rand(N, 1) * 5.0 + 1.0
    
    scaled_features = features * scale_feat
    scaled_prototypes = prototypes * scale_proto
    
    # 1. Test Euclidean CPAL - Not Scale Invariant
    loss_eucl = compute_euclidean_cpal(features, prototypes)
    loss_eucl_scaled = compute_euclidean_cpal(scaled_features, scaled_prototypes)
    print(f"Euclidean CPAL: Clean Loss = {loss_eucl.item():.6f}, Scaled Loss = {loss_eucl_scaled.item():.6f}")
    assert not torch.allclose(loss_eucl, loss_eucl_scaled, rtol=1e-3, atol=1e-3), "Euclidean should not be scale-invariant!"
    
    # 2. Test Cosine CPAL - Perfectly Scale Invariant
    loss_cos = compute_cosine_cpal(features, prototypes)
    loss_cos_scaled = compute_cosine_cpal(scaled_features, scaled_prototypes)
    print(f"Cosine CPAL (C-CPAL): Clean Loss = {loss_cos.item():.6f}, Scaled Loss = {loss_cos_scaled.item():.6f}")
    assert torch.allclose(loss_cos, loss_cos_scaled, rtol=1e-5, atol=1e-5), "C-CPAL is not scale-invariant!"
    print("Verification Successful: Cosine-Normalized CPAL is perfectly scale-invariant under arbitrary positive scaling.")

if __name__ == '__main__':
    test_scale_invariance()
