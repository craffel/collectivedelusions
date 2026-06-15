import torch
import numpy as np
import matplotlib.pyplot as plt

def circular_convolution(v, k):
    V = torch.fft.fft(v)
    K = torch.fft.fft(k)
    return torch.fft.ifft(V * K).real

def circular_correlation(x, k):
    X = torch.fft.fft(x)
    K = torch.fft.fft(k)
    return torch.fft.ifft(X * torch.conj(K)).real

# Sweep dimensions
dimensions = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
num_tasks = 4
correct_similarities = []
wrong_similarities_max = []

for D in dimensions:
    # Generate random task vectors
    V_tasks = [torch.randn(D) for _ in range(num_tasks)]
    
    # --- Circular Convolution Binding ---
    keys_circular = [torch.randn(D) / np.sqrt(D) for _ in range(num_tasks)]
    W_holo_circular = sum(circular_convolution(V_tasks[k], keys_circular[k]) for k in range(num_tasks))
    demod_circular = circular_correlation(W_holo_circular, keys_circular[0])
    
    # Cosine Similarity with Correct Item (v_0)
    sim_correct = torch.dot(demod_circular, V_tasks[0]) / (torch.norm(demod_circular) * torch.norm(V_tasks[0]) + 1e-8)
    correct_similarities.append(sim_correct.item() * 100.0)
    
    # Cosine Similarity with Wrong Items (v_j, j != 0)
    sim_wrongs = []
    for j in range(1, num_tasks):
        sim_w = torch.dot(demod_circular, V_tasks[j]) / (torch.norm(demod_circular) * torch.norm(V_tasks[j]) + 1e-8)
        sim_wrongs.append(sim_w.item() * 100.0)
    wrong_similarities_max.append(max(np.abs(sim_wrongs)))

# Print results
print("Dimension Sweep - Associative Retrieval results:")
for i, D in enumerate(dimensions):
    print(f"  D={D:<5} | Correct CosSim: {correct_similarities[i]:.2f}% | Max Wrong CosSim: {wrong_similarities_max[i]:.2f}% | Gap: {correct_similarities[i] - wrong_similarities_max[i]:.2f}%")

# Generate comparison plot
plt.figure(figsize=(8, 5))
plt.plot(dimensions, correct_similarities, marker='s', linestyle='-', color='blue', label='Correct Item Cosine Similarity (~1/sqrt(K))')
plt.plot(dimensions, wrong_similarities_max, marker='x', linestyle='--', color='red', label='Max Wrong Item Cosine Similarity (O(1/sqrt(D)) noise)')
plt.xlabel("Representation Dimension (D)")
plt.ylabel("Demodulated Cosine Similarity (%)")
plt.title("VSA Clean Associative Retrieval Gap vs. Dimension")
plt.grid(True, linestyle="--", alpha=0.6)
plt.xscale("log")
plt.xticks(dimensions, [str(d) for d in dimensions])
plt.ylim(-10, 110)
plt.legend()
plt.tight_layout()
plt.savefig("results/fig4_circular_convolution_decay.png", dpi=150)
plt.close()
print("Saved comparison plot to 'results/fig4_circular_convolution_decay.png'.")
