import torch

X = torch.randn(2, 3, 32, 32)
# Compute 2D FFT along spatial dimensions
X_fft = torch.fft.fft2(X, dim=(-2, -1))
print("X_fft shape:", X_fft.shape, "dtype:", X_fft.dtype)

# Extract magnitude and phase
mag = torch.abs(X_fft)
phase = torch.angle(X_fft)
print("mag shape:", mag.shape, "phase shape:", phase.shape)

# Reconstruct complex tensor from magnitude and phase
# We can do this via polar coordinates: mag * exp(i * phase)
# In PyTorch, we can do torch.polar(mag, phase) or mag * torch.exp(1j * phase)
X_fft_reconstructed = torch.polar(mag, phase)
X_reconstructed = torch.fft.ifft2(X_fft_reconstructed, dim=(-2, -1)).real
print("X_reconstructed shape:", X_reconstructed.shape)
print("Difference:", torch.max(torch.abs(X - X_reconstructed)).item())
