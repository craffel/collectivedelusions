import torch

def circular_convolution(a, b):
    # a and b are PyTorch tensors of shape (d,)
    # We can use FFT for fast circular convolution
    A = torch.fft.fft(a)
    B = torch.fft.fft(b)
    return torch.fft.ifft(A * B).real

def circular_correlation(a, b):
    # Circular correlation: a correlated with b
    # In FFT domain, it is conj(A) * B
    A = torch.fft.fft(a)
    B = torch.fft.fft(b)
    return torch.fft.ifft(torch.conj(A) * B).real

# Or using the standard HRR definition:
# To retrieve vector x from holographic memory M using key k:
# M = sum(x_i * k_i), where * is circular convolution
# x_retrieved = M # k, where # is circular correlation
# Let's test this!
d = 36864  # typical ResNet conv layer size
torch.manual_seed(42)

# Generate three random task vectors (representing expert weight updates)
tv1 = torch.randn(d) * 0.1
tv2 = torch.randn(d) * 0.1
tv3 = torch.randn(d) * 0.1

# Generate random keys (reference beams)
# Standard HRR keys are generated as random normal vectors with mean 0 and var 1/d,
# or we can just use standard normal and normalize them to unit norm.
k1 = torch.randn(d)
k2 = torch.randn(d)
k3 = torch.randn(d)

k1 = k1 / torch.norm(k1)
k2 = k2 / torch.norm(k2)
k3 = k3 / torch.norm(k3)

# Bind each task vector with its key
bind1 = circular_convolution(tv1, k1)
bind2 = circular_convolution(tv2, k2)
bind3 = circular_convolution(tv3, k3)

# Holographic merge (superposition)
M = bind1 + bind2 + bind3

# Retrieve each task vector using circular correlation with its key
# In HRR, retrieval is: tv_retrieved = M # k_inv
# Wait! Let's check the relation:
# (tv * k) # k. Let's see if we correlate M with k_inv or k.
# Let's write a function that computes the inverse key or correlation.
# Circular correlation of M with k:
ret1 = circular_correlation(k1, M)
ret2 = circular_correlation(k2, M)
ret3 = circular_correlation(k3, M)

# Let's check the cosine similarity and MSE between original and retrieved task vectors!
def metrics(orig, ret):
    cos_sim = torch.nn.functional.cosine_similarity(orig.unsqueeze(0), ret.unsqueeze(0)).item()
    mse = torch.mean((orig - ret)**2).item()
    print(f"Cosine Similarity: {cos_sim:.4f}, MSE: {mse:.6e}")

print("Retrieval Metrics:")
print("Task 1:")
metrics(tv1, ret1)
print("Task 2:")
metrics(tv2, ret2)
print("Task 3:")
metrics(tv3, ret3)
