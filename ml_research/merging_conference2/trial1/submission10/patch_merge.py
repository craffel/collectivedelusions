with open("merge_models.py", "r") as f:
    code = f.read()

# Replace the start of the loops in each function to skip non-floating-point tensors

# 1. merge_arithmetic
old_arithmetic = """def merge_arithmetic(base_sd, expert_sds, alpha=0.5):
    merged_sd = {}
    N = len(expert_sds)
    for k in base_sd.keys():
        # Compute average task vector"""

new_arithmetic = """def merge_arithmetic(base_sd, expert_sds, alpha=0.5):
    merged_sd = {}
    N = len(expert_sds)
    for k in base_sd.keys():
        if not torch.is_floating_point(base_sd[k]):
            merged_sd[k] = base_sd[k]
            continue
        # Compute average task vector"""

# 2. merge_ties
old_ties = """def merge_ties(base_sd, expert_sds, fraction=0.2, alpha=0.5):
    merged_sd = {}
    N = len(expert_sds)
    for k in base_sd.keys():
        # Task vectors"""

new_ties = """def merge_ties(base_sd, expert_sds, fraction=0.2, alpha=0.5):
    merged_sd = {}
    N = len(expert_sds)
    for k in base_sd.keys():
        if not torch.is_floating_point(base_sd[k]):
            merged_sd[k] = base_sd[k]
            continue
        # Task vectors"""

# 3. merge_dare
old_dare = """def merge_dare(base_sd, expert_sds, drop_rate=0.9, alpha=0.5):
    merged_sd = {}
    N = len(expert_sds)
    for k in base_sd.keys():
        tvs = [expert_sds[i][k] - base_sd[k] for i in range(N)]"""

new_dare = """def merge_dare(base_sd, expert_sds, drop_rate=0.9, alpha=0.5):
    merged_sd = {}
    N = len(expert_sds)
    for k in base_sd.keys():
        if not torch.is_floating_point(base_sd[k]):
            merged_sd[k] = base_sd[k]
            continue
        tvs = [expert_sds[i][k] - base_sd[k] for i in range(N)]"""

# 4. merge_orthomerge
old_orthomerge = """def merge_orthomerge(base_sd, expert_sds, alpha=0.5, device='cpu'):
    merged_sd = {}
    N = len(expert_sds)
    
    for k in base_sd.keys():
        base_w = base_sd[k].to(device)"""

new_orthomerge = """def merge_orthomerge(base_sd, expert_sds, alpha=0.5, device='cpu'):
    merged_sd = {}
    N = len(expert_sds)
    
    for k in base_sd.keys():
        if not torch.is_floating_point(base_sd[k]):
            merged_sd[k] = base_sd[k]
            continue
        base_w = base_sd[k].to(device)"""

# 5. merge_saim
old_saim = """def merge_saim(base_sd, expert_sds, alpha=0.5, gamma=0.5, device='cpu'):
    merged_sd = {}
    N = len(expert_sds)
    for k in base_sd.keys():
        base_w = base_sd[k].to(device)"""

new_saim = """def merge_saim(base_sd, expert_sds, alpha=0.5, gamma=0.5, device='cpu'):
    merged_sd = {}
    N = len(expert_sds)
    for k in base_sd.keys():
        if not torch.is_floating_point(base_sd[k]):
            merged_sd[k] = base_sd[k]
            continue
        base_w = base_sd[k].to(device)"""

# 6. merge_dor_saim
old_dor_saim = """def merge_dor_saim(base_sd, expert_sds, alpha=0.5, gamma=0.5, device='cpu'):
    # Decoupled Orthogonal-Residual Sharpness-Aware Isotropic Merging
    merged_sd = {}
    N = len(expert_sds)
    
    decoupled_info = {} # Keep track of R, base, rho for post-hoc scaling
    
    for k in base_sd.keys():
        base_w = base_sd[k].to(device)"""

new_dor_saim = """def merge_dor_saim(base_sd, expert_sds, alpha=0.5, gamma=0.5, device='cpu'):
    # Decoupled Orthogonal-Residual Sharpness-Aware Isotropic Merging
    merged_sd = {}
    N = len(expert_sds)
    
    decoupled_info = {} # Keep track of R, base, rho for post-hoc scaling
    
    for k in base_sd.keys():
        if not torch.is_floating_point(base_sd[k]):
            merged_sd[k] = base_sd[k]
            continue
        base_w = base_sd[k].to(device)"""

code = code.replace(old_arithmetic, new_arithmetic)
code = code.replace(old_ties, new_ties)
code = code.replace(old_dare, new_dare)
code = code.replace(old_orthomerge, new_orthomerge)
code = code.replace(old_saim, new_saim)
code = code.replace(old_dor_saim, new_dor_saim)

with open("merge_models.py", "w") as f:
    f.write(code)

print("Patching completed successfully!")
