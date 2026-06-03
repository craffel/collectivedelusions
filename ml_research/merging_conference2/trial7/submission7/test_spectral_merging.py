import torch

def test_holographic_spectral_merging():
    # Simulate a weight matrix of shape (64, 64) - typical conv kernel or linear weight
    shape = (64, 64)
    torch.manual_seed(42)

    # Expert weights (fine-tuned from a shared progenitor)
    W_init = torch.randn(shape)
    W1 = W_init + torch.randn(shape) * 0.1
    W2 = W_init + torch.randn(shape) * 0.1
    W3 = W_init + torch.randn(shape) * 0.1

    # Take the 2D FFT of the task vectors (weight updates)
    tv1 = W1 - W_init
    tv2 = W2 - W_init
    tv3 = W3 - W_init

    F1 = torch.fft.fft2(tv1)
    F2 = torch.fft.fft2(tv2)
    F3 = torch.fft.fft2(tv3)

    # Let's sum the FFT representations to create a merged hologram
    F_merged = F1 + F2 + F3

    # If we use the exact filter: H_i = F_i / F_merged
    # To avoid division by zero, we add a small epsilon
    eps = 1e-8
    H1 = F1 / (F_merged + eps)
    H2 = F2 / (F_merged + eps)
    H3 = F3 / (F_merged + eps)

    # Now let's compress the filters!
    # A visionary idea: we can downsample the complex filters H_i to a much smaller size (e.g., 8x8)
    # and then bilinearly interpolate them back to 64x64 during reconstruction!
    # Let's see how much we can compress them.
    for low_res_size in [8, 16, 32]:
        # Downsample the filter H1 in the frequency domain
        # Since H1 is complex, we downsample real and imaginary parts separately
        H1_real = H1.real.unsqueeze(0).unsqueeze(0) # (1, 1, 64, 64)
        H1_imag = H1.imag.unsqueeze(0).unsqueeze(0)

        # Use interpolate to downsample
        H1_real_low = torch.nn.functional.interpolate(H1_real, size=(low_res_size, low_res_size), mode='bilinear', align_corners=False)
        H1_imag_low = torch.nn.functional.interpolate(H1_imag, size=(low_res_size, low_res_size), mode='bilinear', align_corners=False)

        # Upsample back to original size
        H1_real_up = torch.nn.functional.interpolate(H1_real_low, size=shape, mode='bilinear', align_corners=False).squeeze()
        H1_imag_up = torch.nn.functional.interpolate(H1_imag_low, size=shape, mode='bilinear', align_corners=False).squeeze()

        # Reconstruct the complex filter
        H1_recon = torch.complex(H1_real_up, H1_imag_up)

        # Retrieve the task vector
        tv1_ret = torch.fft.ifft2(F_merged * H1_recon).real

        # Compute cosine similarity and MSE
        cos_sim = torch.nn.functional.cosine_similarity(tv1.flatten().unsqueeze(0), tv1_ret.flatten().unsqueeze(0)).item()
        mse = torch.mean((tv1 - tv1_ret)**2).item()

        compression_ratio = (low_res_size * low_res_size) / (shape[0] * shape[1]) * 100
        print(f"Low Res: {low_res_size}x{low_res_size} ({compression_ratio:.2f}% size) -> Cosine Similarity: {cos_sim:.4f}, MSE: {mse:.6e}")

test_holographic_spectral_merging()
