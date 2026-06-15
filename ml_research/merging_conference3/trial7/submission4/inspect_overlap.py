import torch
from run_experiments import RepresentationSandbox, get_oracle_experts

sandbox = RepresentationSandbox(42, rho=0.33)
experts = get_oracle_experts(sandbox)

# Extract centroids
K = len(experts)
D = experts[0].weight.shape[1]
v = torch.zeros(K, D)
for k in range(K):
    W_k = experts[k].weight.data
    _, _, Vh = torch.linalg.svd(W_k, full_matrices=False)
    v[k] = Vh[0]
v_bar = v / (torch.norm(v, dim=1, keepdim=True) + 1.0e-8)

S = torch.matmul(v_bar, v_bar.t())
print("Centroid Overlap Matrix S under rho=0.33:")
print(S)
