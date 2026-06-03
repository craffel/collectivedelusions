import torch
from transformers import CLIPModel, CLIPVisionModel

def debug():
    base_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    expert_model = CLIPVisionModel.from_pretrained("tanganke/clip-vit-base-patch32_mnist")
    
    base_sd = base_model.vision_model.state_dict()
    expert_sd = expert_model.vision_model.state_dict()
    
    # Let's create dummy task vectors
    tv = {k: expert_sd[k] - base_sd[k] for k in base_sd.keys()}
    task_vectors = [tv, tv, tv] # 3 identical task vectors
    
    # SCS Merge
    gamma = 1.0
    lam = 0.5
    
    k = "embeddings.class_embedding"
    w_pre = base_sd[k]
    updates = torch.stack([v[k] for v in task_vectors], dim=0)
    
    mean_tv = torch.mean(updates, dim=0)
    signs = torch.sign(updates)
    mean_signs = torch.mean(signs, dim=0)
    scs_factor = torch.pow(torch.abs(mean_signs), gamma)
    scaled_tv = mean_tv * scs_factor
    merged_w = w_pre + lam * scaled_tv
    
    print("w_pre norm:", torch.norm(w_pre).item())
    print("updates norm:", torch.norm(updates).item())
    print("mean_tv norm:", torch.norm(mean_tv).item())
    print("mean_signs norm:", torch.norm(mean_signs).item())
    print("scs_factor norm:", torch.norm(scs_factor).item())
    print("scaled_tv norm:", torch.norm(scaled_tv).item())
    print("merged_w norm:", torch.norm(merged_w).item())
    
    print("\nCheck first 5 elements of w_pre:")
    print(w_pre.flatten()[:5])
    print("\nCheck first 5 elements of mean_tv:")
    print(mean_tv.flatten()[:5])
    print("\nCheck first 5 elements of scs_factor:")
    print(scs_factor.flatten()[:5])
    print("\nCheck first 5 elements of merged_w:")
    print(merged_w.flatten()[:5])

if __name__ == "__main__":
    debug()
