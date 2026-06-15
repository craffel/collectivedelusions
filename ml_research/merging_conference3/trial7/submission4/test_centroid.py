import torch
from run_experiments import RepresentationSandbox, get_oracle_experts, PFSRRouter

seed = 42
sandbox = RepresentationSandbox(seed, rho=0.0)
experts = get_oracle_experts(sandbox)
pfsr = PFSRRouter(experts)

X_test, Y_test, task_test = sandbox.generate_split(250)
mask_0 = (task_test == 0)
z = X_test[mask_0][0]
tilde_z = z / torch.norm(z)

print("tilde_z norm:", torch.norm(tilde_z).item())
print("v_bar[0] norm:", torch.norm(pfsr.v_bar[0]).item())
print("Dot product:", torch.dot(tilde_z, pfsr.v_bar[0]).item())

# Let's inspect the active dimensions of tilde_z and v_bar[0]
print("tilde_z active dimensions (0 to 48 sum):", torch.sum(tilde_z[:48]**2).item())
print("v_bar[0] active dimensions (0 to 48 sum):", torch.sum(pfsr.v_bar[0][:48]**2).item())

# Let's inspect the weights of expert 0
print("expert 0 weights shape:", experts[0].weight.shape)
# Let's see the sum of rows of expert 0
expert_0_sum = torch.sum(experts[0].weight, dim=0)
print("expert_0_sum norm:", torch.norm(expert_0_sum).item())
