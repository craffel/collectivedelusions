import torch

def test_hns():
    # Simulate a conv layer weight tensor of shape (out_channels, in_channels, k, k)
    shape = (128, 64, 3, 3)
    torch.manual_seed(42)

    # Expert weights fine-tuned from W_init
    W_init = torch.randn(shape)
    # Drift is orthogonal
    tv1 = torch.randn(shape) * 0.2
    tv2 = torch.randn(shape) * 0.2
    tv3 = torch.randn(shape) * 0.2

    W1 = W_init + tv1
    W2 = W_init + tv2
    W3 = W_init + tv3

    # Weight Averaging (WA)
    W_merged = (W1 + W2 + W3) / 3.0

    # Let's check channel-wise L2 norms for W1 vs W_merged
    # Norm along dimensions (1, 2, 3)
    norm_W1 = torch.norm(W1, p=2, dim=(1, 2, 3))
    norm_merged = torch.norm(W_merged, p=2, dim=(1, 2, 3))

    print("Average Channel Norms:")
    print(f"Expert 1: {norm_W1.mean().item():.4f}")
    print(f"Merged: {norm_merged.mean().item():.4f}")

    # Compute Holographic Norm Scaling (HNS) factors
    gamma1 = norm_W1 / (norm_merged + 1e-8)

    # Scale the merged weights
    # Reshape gamma1 to (out_channels, 1, 1, 1) for broadcasting
    W1_recon = W_merged * gamma1.view(-1, 1, 1, 1)

    # Check cosine similarity and MSE of W1 vs W1_recon vs W_merged
    cos_sim_orig = torch.nn.functional.cosine_similarity(W1.flatten().unsqueeze(0), W_merged.flatten().unsqueeze(0)).item()
    cos_sim_recon = torch.nn.functional.cosine_similarity(W1.flatten().unsqueeze(0), W1_recon.flatten().unsqueeze(0)).item()

    print("\nComparison:")
    print(f"Original Cosine Similarity (W1 vs W_merged): {cos_sim_orig:.4f}")
    print(f"Reconstructed Cosine Similarity (W1 vs W1_recon): {cos_sim_recon:.4f}")

test_hns()
