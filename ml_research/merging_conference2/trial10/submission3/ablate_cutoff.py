import torch
import numpy as np
import copy
from evaluate_merging_complete import ADSRModel, loaders, device, evaluate_model, nn, os, resnet18, quantize_tensor

# Load pretrained experts
experts = []
for task in ["mnist", "fmnist", "cifar10"]:
    chk_path = f"checkpoints/{task}_expert.pt"
    model = resnet18()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(chk_path, map_location=device))
    model = model.to(device)
    experts.append(model)

print("Cutoff Ratio | FP32 Clean (%) | 8-bit PTQ (%) | Noise (std=0.2) (%)")
print("-------------------------------------------------------------------")
for cutoff in [0.1, 0.2, 0.3, 0.4, 0.5]:
    # 1. FP32 Clean
    ss_model = ADSRModel(experts, temp=-5.0, temp_cw=-0.5, bits=None, channel_wise=False, spectral_mode=True, cutoff_ratio=cutoff).to(device)
    ss_accs = []
    for i, loader in enumerate(loaders):
        acc = evaluate_model(ss_model, loader, device, task_id=i)
        ss_accs.append(acc)
    avg_clean = np.mean(ss_accs)
    
    # 2. 8-bit PTQ
    q_ss_model = ADSRModel(experts, temp=-5.0, temp_cw=-0.5, bits=8, channel_wise=False, spectral_mode=True, cutoff_ratio=cutoff).to(device)
    for exp in q_ss_model.experts:
        for name, param in exp.named_parameters():
            param.data = quantize_tensor(param.data, 8)
    q_ss_accs = []
    for i, loader in enumerate(loaders):
        acc = evaluate_model(q_ss_model, loader, device, task_id=i)
        q_ss_accs.append(acc)
    avg_q8 = np.mean(q_ss_accs)
    
    # 3. Noise std=0.2
    n_ss_accs = []
    for i, loader in enumerate(loaders):
        acc = evaluate_model(ss_model, loader, device, task_id=i, noise_std=0.2)
        n_ss_accs.append(acc)
    avg_noise = np.mean(n_ss_accs)
    
    print(f"   {cutoff:.3f}     |     {avg_clean:.2f}     |     {avg_q8:.2f}     |      {avg_noise:.2f}")
