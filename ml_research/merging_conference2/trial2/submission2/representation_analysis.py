import torch
import torch.nn as nn
import numpy as np
from run_experiments import load_expert, get_test_loader, build_merged_real_model, device
from cpos_merging import CPOSResNet

def compute_cosine_similarity(feat1, feat2):
    # Flatten features to shape (Batch, -1)
    f1 = feat1.view(feat1.size(0), -1)
    f2 = feat2.view(feat2.size(0), -1)
    # Compute cosine similarity per sample, then average
    cos = nn.functional.cosine_similarity(f1, f2, dim=1)
    return cos.mean().item()

def extract_activations(model, x, is_cpos=False):
    activations = []
    if is_cpos:
        # Stem
        y_A = model.model_A.relu(model.model_A.bn1(model.model_A.conv1(x)))
        y_B = model.model_B.relu(model.model_B.bn1(model.model_B.conv1(x)))
        x_next = torch.sqrt(model.alpha**2 * y_A**2 + model.beta**2 * y_B**2 + 1e-8)
        activations.append(x_next)
        x_next = model.model_A.maxpool(x_next)
        
        # Layers
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            layer_A = getattr(model.model_A, layer_name)
            layer_B = getattr(model.model_B, layer_name)
            for i in range(len(layer_A)):
                block_A = layer_A[i]
                block_B = layer_B[i]
                out_A = block_A(x_next)
                out_B = block_B(x_next)
                x_next = torch.sqrt(model.alpha**2 * out_A**2 + model.beta**2 * out_B**2 + 1e-8)
                activations.append(x_next)
    else:
        # Standard ResNet-18
        x_next = model.relu(model.bn1(model.conv1(x)))
        activations.append(x_next)
        x_next = model.maxpool(x_next)
        
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            layer = getattr(model, layer_name)
            for i in range(len(layer)):
                x_next = layer[i](x_next)
                activations.append(x_next)
                
    return activations

def main():
    print("==========================================================================")
    print("REPRESENTATION SIMILARITY ANALYSIS: CPOS VS WEIGHT AVERAGING")
    print("==========================================================================")
    
    # 1. Load Experts
    task_A = "cifar10"
    task_B = "fmnist"
    model_A = load_expert(task_A).to(device)
    model_B = load_expert(task_B).to(device)
    
    # 2. Setup Merged Models
    # Weight Averaging State Dict
    import copy
    state_A = model_A.state_dict()
    state_B = model_B.state_dict()
    merged_wa_state = {}
    for k in state_A.keys():
        if k.startswith("fc."):
            continue
        if k in state_B:
            merged_wa_state[k] = 0.5 * state_A[k] + 0.5 * state_B[k]
            
    # Build WA model (for Task A and Task B evaluation of backbone activations)
    fc_A_state = {k.replace("fc.", ""): v for k, v in model_A.state_dict().items() if k.startswith("fc.")}
    fc_B_state = {k.replace("fc.", ""): v for k, v in model_B.state_dict().items() if k.startswith("fc.")}
    wa_model_A = build_merged_real_model(merged_wa_state, fc_A_state, device)
    wa_model_B = build_merged_real_model(merged_wa_state, fc_B_state, device)
    
    # CPOS model
    alpha_val = 1.0 / np.sqrt(2)
    beta_val = 1.0 / np.sqrt(2)
    cpos_model = CPOSResNet(model_A, model_B, alpha=alpha_val, beta=beta_val).to(device)
    
    # 3. Load Dataloaders
    loader_A = get_test_loader(task_A, max_samples=256)
    loader_B = get_test_loader(task_B, max_samples=256)
    
    # Fetch a batch from both
    inputs_A, _ = next(iter(loader_A))
    inputs_B, _ = next(iter(loader_B))
    inputs_A = inputs_A.to(device)
    inputs_B = inputs_B.to(device)
    
    model_A.eval()
    model_B.eval()
    wa_model_A.eval()
    wa_model_B.eval()
    cpos_model.eval()
    
    with torch.no_grad():
        # Extracted activations
        act_A_expert = extract_activations(model_A, inputs_A, is_cpos=False)
        act_A_wa = extract_activations(wa_model_A, inputs_A, is_cpos=False)
        act_A_cpos = extract_activations(cpos_model, inputs_A, is_cpos=True)
        
        act_B_expert = extract_activations(model_B, inputs_B, is_cpos=False)
        act_B_wa = extract_activations(wa_model_B, inputs_B, is_cpos=False)
        act_B_cpos = extract_activations(cpos_model, inputs_B, is_cpos=True)
        
    print("\n--- Similarity to Expert A (CIFAR-10 Input) ---")
    print("| Block | WA CosSim | CPOS CosSim | Relative Gain |")
    print("|---|---|---|---|")
    block_names = ["Stem", "Layer 1.1", "Layer 1.2", "Layer 2.1", "Layer 2.2", "Layer 3.1", "Layer 3.2", "Layer 4.1", "Layer 4.2"]
    
    cossim_A_wa = []
    cossim_A_cpos = []
    for i, name in enumerate(block_names):
        sim_wa = compute_cosine_similarity(act_A_expert[i], act_A_wa[i])
        sim_cpos = compute_cosine_similarity(act_A_expert[i], act_A_cpos[i])
        cossim_A_wa.append(sim_wa)
        cossim_A_cpos.append(sim_cpos)
        gain = sim_cpos - sim_wa
        print(f"| {name} | {sim_wa:.4f} | {sim_cpos:.4f} | {gain:+.4f} |")
        
    print("\n--- Similarity to Expert B (FashionMNIST Input) ---")
    print("| Block | WA CosSim | CPOS CosSim | Relative Gain |")
    print("|---|---|---|---|")
    cossim_B_wa = []
    cossim_B_cpos = []
    for i, name in enumerate(block_names):
        sim_wa = compute_cosine_similarity(act_B_expert[i], act_B_wa[i])
        sim_cpos = compute_cosine_similarity(act_B_expert[i], act_B_cpos[i])
        cossim_B_wa.append(sim_wa)
        cossim_B_cpos.append(sim_cpos)
        gain = sim_cpos - sim_wa
        print(f"| {name} | {sim_wa:.4f} | {sim_cpos:.4f} | {gain:+.4f} |")

if __name__ == "__main__":
    main()
