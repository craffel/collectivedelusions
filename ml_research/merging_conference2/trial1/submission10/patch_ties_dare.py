with open("merge_models.py", "r") as f:
    code = f.read()

# 1. Modify merge_ties
old_ties_body = """def merge_ties(base_sd, expert_sds, fraction=0.2, alpha=0.5):
    merged_sd = {}
    N = len(expert_sds)
    for k in base_sd.keys():
        if not torch.is_floating_point(base_sd[k]):
            merged_sd[k] = base_sd[k]
            continue
        # Task vectors
        tvs = [expert_sds[i][k] - base_sd[k] for i in range(N)]"""

new_ties_body = """def merge_ties(base_sd, expert_sds, fraction=0.2, alpha=0.5):
    merged_sd = {}
    N = len(expert_sds)
    for k in base_sd.keys():
        if not torch.is_floating_point(base_sd[k]):
            merged_sd[k] = base_sd[k]
            continue
        
        # If 1D parameter, use simple average
        if base_sd[k].dim() < 2:
            task_vectors = [expert_sds[i][k] - base_sd[k] for i in range(N)]
            avg_task_vector = torch.stack(task_vectors).mean(dim=0)
            merged_sd[k] = base_sd[k] + alpha * avg_task_vector
            continue
            
        # Task vectors
        tvs = [expert_sds[i][k] - base_sd[k] for i in range(N)]"""

# 2. Modify merge_dare
old_dare_body = """def merge_dare(base_sd, expert_sds, drop_rate=0.9, alpha=0.5):
    merged_sd = {}
    N = len(expert_sds)
    for k in base_sd.keys():
        if not torch.is_floating_point(base_sd[k]):
            merged_sd[k] = base_sd[k]
            continue
        tvs = [expert_sds[i][k] - base_sd[k] for i in range(N)]"""

new_dare_body = """def merge_dare(base_sd, expert_sds, drop_rate=0.9, alpha=0.5):
    merged_sd = {}
    N = len(expert_sds)
    for k in base_sd.keys():
        if not torch.is_floating_point(base_sd[k]):
            merged_sd[k] = base_sd[k]
            continue
            
        # If 1D parameter, use simple average
        if base_sd[k].dim() < 2:
            task_vectors = [expert_sds[i][k] - base_sd[k] for i in range(N)]
            avg_task_vector = torch.stack(task_vectors).mean(dim=0)
            merged_sd[k] = base_sd[k] + alpha * avg_task_vector
            continue
            
        tvs = [expert_sds[i][k] - base_sd[k] for i in range(N)]"""

code = code.replace(old_ties_body, new_ties_body)
code = code.replace(old_dare_body, new_dare_body)

with open("merge_models.py", "w") as f:
    f.write(code)

print("TIES and DARE dimension guards patched successfully!")
