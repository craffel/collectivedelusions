import torch

def test_hns_tv():
    # Simulate a layer weight of shape (128, 64, 3, 3)
    shape = (128, 64, 3, 3)
    torch.manual_seed(42)

    # Initial model
    W_init = torch.randn(shape)
    
    # Task updates (task vectors) are orthogonal
    tv1 = torch.randn(shape) * 0.2
    tv2 = torch.randn(shape) * 0.2
    tv3 = torch.randn(shape) * 0.2

    # Expert models
    W1 = W_init + tv1
    W2 = W_init + tv2
    W3 = W_init + tv3

    # Merged task vector (WA is equivalent to W_init + averaged task vectors)
    tv_merged = (tv1 + tv2 + tv3) / 3.0
    W_merged = W_init + tv_merged

    # Check task vector norms
    norm_tv1 = torch.norm(tv1, p=2, dim=(1, 2, 3))
    norm_tv_merged = torch.norm(tv_merged, p=2, dim=(1, 2, 3))

    print("Average Task Vector Channel Norms:")
    print(f"Expert 1: {norm_tv1.mean().item():.4f}")
    print(f"Merged: {norm_tv_merged.mean().item():.4f} (Ratio: {norm_tv_merged.mean().item()/norm_tv1.mean().item()*100:.2f}%)")

    # Compute Holographic Norm Scaling (HNS) factors for task vectors!
    gamma1 = norm_tv1 / (norm_tv_merged + 1e-8)
    print(f"Average scale factor (gamma): {gamma1.mean().item():.4f}")

    # Scale the merged task vector
    tv1_recon = tv_merged * gamma1.view(-1, 1, 1, 1)
    W1_recon = W_init + tv1_recon

    # Compare cosine similarities of the task-specific part
    cos_sim_orig = torch.nn.functional.cosine_similarity(tv1.flatten().unsqueeze(0), tv_merged.flatten().unsqueeze(0)).item()
    cos_sim_recon = torch.nn.functional.cosine_similarity(tv1.flatten().unsqueeze(0), tv1_recon.flatten().unsqueeze(0)).item()

    print("\nComparison of task updates:")
    print(f"Original Cosine Similarity: {cos_sim_orig:.4f}")
    print(f"Reconstructed Cosine Similarity: {cos_sim_recon:.4f}")

test_hns_tv()
