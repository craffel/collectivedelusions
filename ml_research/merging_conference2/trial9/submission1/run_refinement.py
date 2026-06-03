import os
import copy
import json
import torch
import torch.nn as nn
from merge_eval import ExpertModel, get_dataloader, merge_models, apply_calibration, evaluate_model

def run_de_bn_sweep():
    print("="*50)
    print("RUNNING DE-BN SAMPLE EFFICIENCY SWEEP")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    progenitor = ExpertModel()
    progenitor.load_state_dict(torch.load("checkpoints/progenitor.pt", map_location="cpu"))
    progenitor_state = progenitor.state_dict()
    
    expert_states = []
    task_heads = {}
    tasks = ["mnist", "fmnist", "cifar10"]
    for task in tasks:
        expert = ExpertModel()
        expert.load_state_dict(torch.load(f"checkpoints/{task}_expert.pt", map_location="cpu"))
        expert_states.append(expert.state_dict())
        task_heads[task] = copy.deepcopy(expert.fc)
        
    wa_backbone_state = merge_models(progenitor_state, expert_states, alg="wa")
    wa_model = ExpertModel()
    wa_model.load_state_dict(wa_backbone_state)
    
    sweep_ns = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    sweep_results = {}
    
    for n in sweep_ns:
        print(f"Evaluating N={n}...")
        accs = evaluate_model(wa_model, task_heads, de_bn_samples=n)
        sweep_results[str(n)] = accs
        print(f"N={n} | Average Acc: {accs['average']:.2f}% | MNIST: {accs['mnist']:.2f}% | FMNIST: {accs['fmnist']:.2f}% | CIFAR-10: {accs['cifar10']:.2f}%")
        
    return sweep_results

def collect_activation_statistics():
    print("\n" + "="*50)
    print("COLLECTING LAYER-WISE ACTIVATION STATISTICS")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    progenitor = ExpertModel().to(device)
    progenitor.load_state_dict(torch.load("checkpoints/progenitor.pt", map_location=device))
    progenitor_state = progenitor.state_dict()
    
    expert_states = []
    tasks = ["mnist", "fmnist", "cifar10"]
    for task in tasks:
        expert = ExpertModel()
        expert.load_state_dict(torch.load(f"checkpoints/{task}_expert.pt", map_location="cpu"))
        expert_states.append(expert.state_dict())
        
    wa_backbone_state = merge_models(progenitor_state, expert_states, alg="wa")
    
    # WA + None
    wa_none = ExpertModel().to(device)
    wa_none.load_state_dict(wa_backbone_state)
    
    # WA + HNS
    wa_hns = apply_calibration(wa_none, progenitor_state, expert_states, cal_method="hns").to(device)
    
    # WA + CBVC (standard)
    wa_cbvc = apply_calibration(wa_none, progenitor_state, expert_states, cal_method="cbvc", dynamic_clip=False).to(device)
    
    models = {
        "Progenitor": progenitor,
        "WA + None": wa_none,
        "WA + HNS": wa_hns,
        "WA + CBVC": wa_cbvc
    }
    
    # Set to eval mode
    for name, model in models.items():
        model.eval()
        
    # Get a batch of 128 images from CIFAR-10 test set
    test_loader = get_dataloader("cifar10", batch_size=128, is_train=False)
    inputs, _ = next(iter(test_loader))
    inputs = inputs.to(device)
    
    # Hook targets (blocks/layers in ResNet-18 backbone)
    # We will hook:
    # 0: backbone.relu (after initial conv)
    # 1: backbone.layer1[0]
    # 2: backbone.layer1[1]
    # 3: backbone.layer2[0]
    # 4: backbone.layer2[1]
    # 5: backbone.layer3[0]
    # 6: backbone.layer3[1]
    # 7: backbone.layer4[0]
    # 8: backbone.layer4[1]
    
    hook_targets = [
        ("relu_init", progenitor.backbone.relu),
        ("layer1_0", progenitor.backbone.layer1[0]),
        ("layer1_1", progenitor.backbone.layer1[1]),
        ("layer2_0", progenitor.backbone.layer2[0]),
        ("layer2_1", progenitor.backbone.layer2[1]),
        ("layer3_0", progenitor.backbone.layer3[0]),
        ("layer3_1", progenitor.backbone.layer3[1]),
        ("layer4_0", progenitor.backbone.layer4[0]),
        ("layer4_1", progenitor.backbone.layer4[1])
    ]
    
    activation_stats = {m_name: [] for m_name in models.keys()}
    
    for m_name, model in models.items():
        # Register hooks for this model
        hooks = []
        layer_stats = {}
        
        def make_hook(layer_name):
            def hook_fn(module, inp, out):
                # Standard deviation of the activations
                std = out.std().item()
                layer_stats[layer_name] = std
            return hook_fn
            
        # Hook target reference on current model's modules
        # We need to find the modules in the current model
        curr_targets = [
            ("relu_init", model.backbone.relu),
            ("layer1_0", model.backbone.layer1[0]),
            ("layer1_1", model.backbone.layer1[1]),
            ("layer2_0", model.backbone.layer2[0]),
            ("layer2_1", model.backbone.layer2[1]),
            ("layer3_0", model.backbone.layer3[0]),
            ("layer3_1", model.backbone.layer3[1]),
            ("layer4_0", model.backbone.layer4[0]),
            ("layer4_1", model.backbone.layer4[1])
        ]
        
        for l_name, module in curr_targets:
            hooks.append(module.register_forward_hook(make_hook(l_name)))
            
        # Forward pass
        with torch.no_grad():
            _ = model(inputs)
            
        # Remove hooks
        for h in hooks:
            h.remove()
            
        # Collect stats in order of layers
        for l_name, _ in curr_targets:
            activation_stats[m_name].append(layer_stats[l_name])
            
    print("Activation statistics collected:")
    for name, stats in activation_stats.items():
        print(f"{name}: {[round(s, 4) for s in stats]}")
        
    return activation_stats

if __name__ == "__main__":
    # Disable cuDNN to avoid issues
    torch.backends.cudnn.enabled = False
    
    sweep_res = run_de_bn_sweep()
    act_stats = collect_activation_statistics()
    
    out_data = {
        "de_bn_sweep": sweep_res,
        "activation_stats": act_stats
    }
    
    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/refinement_results.json", "w") as f:
        json.dump(out_data, f, indent=4)
        
    print("\nRefinement experiments complete! Saved data to checkpoints/refinement_results.json")
