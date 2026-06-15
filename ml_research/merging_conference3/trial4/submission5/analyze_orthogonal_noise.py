import os
import torch
import timm

def get_task_vector(expert_state, base_state):
    task_vector = {}
    for k in base_state.keys():
        if not k.startswith("head."):  # ignore task-specific head
            task_vector[k] = expert_state[k] - base_state[k]
    return task_vector

def flatten_state_dict(sd):
    all_vals = []
    for k in sorted(sd.keys()):
        all_vals.append(sd[k].flatten())
    return torch.cat(all_vals)

def cosine_similarity(v1, v2):
    return torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-8)

def main():
    print("Loading pre-trained base model...")
    base_model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    base_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}
    
    task_names = ["MNIST", "FashionMNIST", "CIFAR10", "SVHN"]
    expert_states = {}
    task_vectors = {}
    
    print("Loading expert checkpoints...")
    for t in task_names:
        path = f"./checkpoints/{t}_expert.pt"
        if not os.path.exists(path):
            print(f"Error: {path} does not exist.")
            return
        state = torch.load(path, map_location="cpu")
        expert_states[t] = state
        task_vectors[t] = get_task_vector(state, base_state)
    
    print("\nFlattening full task vectors...")
    flat_tvs = {t: flatten_state_dict(task_vectors[t]) for t in task_names}
    
    print("\n--- 1. Pairwise Cosine Similarity of Full Task Vectors ---")
    for i in range(len(task_names)):
        for j in range(i+1, len(task_names)):
            t1, t2 = task_names[i], task_names[j]
            cos_sim = cosine_similarity(flat_tvs[t1], flat_tvs[t2]).item()
            print(f"{t1} <-> {t2}: {cos_sim:.4f}")
            
    # GQ Masking keep-ratio k = 0.3
    k = 0.3
    print(f"\nApplying Global Quantile (GQ) Masking (k = {k})...")
    
    kept_tvs = {}
    pruned_tvs = {}
    masks = {}
    
    for t in task_names:
        tv = task_vectors[t]
        all_vals = []
        for key in sorted(tv.keys()):
            all_vals.append(tv[key].flatten())
        all_vals_tensor = torch.cat(all_vals)
        num_keep = int(k * len(all_vals_tensor))
        threshold = torch.topk(torch.abs(all_vals_tensor), num_keep).values[-1]
        
        kept_sd = {}
        pruned_sd = {}
        mask_sd = {}
        
        for key, val in tv.items():
            mask = torch.abs(val) >= threshold
            mask_sd[key] = mask.float()
            kept_sd[key] = torch.where(mask, val, torch.zeros_like(val))
            pruned_sd[key] = torch.where(~mask, val, torch.zeros_like(val))
            
        kept_tvs[t] = flatten_state_dict(kept_sd)
        pruned_tvs[t] = flatten_state_dict(pruned_sd)
        masks[t] = flatten_state_dict(mask_sd)
        
    print("\n--- 2. Pairwise Cosine Similarity of KEPT Updates (High-magnitude) ---")
    for i in range(len(task_names)):
        for j in range(i+1, len(task_names)):
            t1, t2 = task_names[i], task_names[j]
            cos_sim = cosine_similarity(kept_tvs[t1], kept_tvs[t2]).item()
            print(f"{t1} <-> {t2}: {cos_sim:.4f}")
            
    print("\n--- 3. Pairwise Cosine Similarity of PRUNED Updates (Low-magnitude, Background noise) ---")
    for i in range(len(task_names)):
        for j in range(i+1, len(task_names)):
            t1, t2 = task_names[i], task_names[j]
            cos_sim = cosine_similarity(pruned_tvs[t1], pruned_tvs[t2]).item()
            print(f"{t1} <-> {t2}: {cos_sim:.4f}")
            
    print("\n--- 4. Mask Overlap Statistics ---")
    print(f"Target Keep-Ratio: {k:.2f} (Expected random overlap: {k*k:.4f} or {k*k*100:.2f}%)")
    for i in range(len(task_names)):
        for j in range(i+1, len(task_names)):
            t1, t2 = task_names[i], task_names[j]
            m1, m2 = masks[t1], masks[t2]
            overlap_count = torch.sum(m1 * m2).item()
            total_params = len(m1)
            overlap_fraction = overlap_count / total_params
            print(f"{t1} <-> {t2} mask overlap: {overlap_fraction:.4f} ({overlap_fraction*100:.2f}%)")

if __name__ == "__main__":
    main()
