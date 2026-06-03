import torch

def merge_ties_test(base_sd, task_vectors, p=0.2, lam=0.5):
    merged_tv = {}
    for k in base_sd.keys():
        w_pre = base_sd[k]
        if w_pre.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            merged_tv[k] = w_pre.clone()
            continue
            
        updates = torch.stack([tv[k] for tv in task_vectors], dim=0) # [K, ...]
        flat_updates = updates.view(updates.shape[0], -1) # [K, N]
        
        # 1. Trim
        k_val = int(p * flat_updates.shape[1])
        print(f"Key: {k}, Shape: {w_pre.shape}, k_val: {k_val}, flat_size: {flat_updates.shape[1]}")
        if k_val > 0:
            thresholds = torch.topk(torch.abs(flat_updates), k_val, dim=1).values[:, -1] # [K]
            mask = torch.abs(updates) >= thresholds.view(-1, *([1] * (updates.ndim - 1)))
            trimmed_updates = updates * mask
        else:
            trimmed_updates = updates
            
        # 2. Elect sign
        signs = torch.sign(trimmed_updates)
        sum_signs = torch.sum(signs, dim=0)
        majority_sign = torch.sign(sum_signs)
        
        # 3. Disjoint merge
        same_sign_mask = (torch.sign(trimmed_updates) == majority_sign.unsqueeze(0)) & (majority_sign.unsqueeze(0) != 0)
        sum_matching = torch.sum(trimmed_updates * same_sign_mask, dim=0)
        count_matching = torch.sum(same_sign_mask.float(), dim=0)
        
        merged_val = torch.where(count_matching > 0, sum_matching / count_matching, torch.zeros_like(sum_matching))
        merged_tv[k] = w_pre + lam * merged_val
        
        # Print some stats
        print(f"  Trimmed updates non-zero count: {torch.count_nonzero(trimmed_updates).item()} / {trimmed_updates.numel()}")
        print(f"  Same sign mask non-zero count: {torch.count_nonzero(same_sign_mask).item()}")
        print(f"  Merged val non-zero count: {torch.count_nonzero(merged_val).item()} / {merged_val.numel()}")
        print(f"  Merged val norm: {torch.norm(merged_val).item():.4f}")
        
    return merged_tv

if __name__ == "__main__":
    base_sd = {"weight": torch.randn(10, 10)}
    task_vectors = [
        {"weight": torch.randn(10, 10)},
        {"weight": torch.randn(10, 10)},
        {"weight": torch.randn(10, 10)}
    ]
    merge_ties_test(base_sd, task_vectors, p=0.2, lam=0.5)
