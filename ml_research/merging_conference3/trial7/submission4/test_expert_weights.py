import torch
from run_experiments import RepresentationSandbox, get_oracle_experts

seed = 42
sandbox = RepresentationSandbox(seed, rho=0.0)
experts = get_oracle_experts(sandbox)

# Check expert 0 weights vs task 0 prototypes
for c in range(10):
    proto_c = sandbox.prototypes[0, c]
    weight_c = experts[0].weight[c]
    dot = torch.dot(proto_c, weight_c).item()
    print(f"Class {c}: Prototype norm = {torch.norm(proto_c).item():.3f}, Weight norm = {torch.norm(weight_c).item():.3f}, dot product = {dot:.3f}")
