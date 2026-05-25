import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import functional_call
import matplotlib.pyplot as plt

from train_experts import CustomCNN, ExpertModel
from eval_tta import set_seed, apply_corruption, augment_batch, build_test_stream, reconstruct_merged_params

def get_trajectories(method_name, stream, base_encoder, expert_states, original_heads, fisher_priors, device,
                     lr_lambda=0.1, lr_head=1e-4, gamma_ewc=100.0, gamma_const=1.0, gamma_fwar=25.0):
    set_seed(42)
    encoder_keys = [name for name, param in base_encoder.named_parameters() if param.requires_grad]
    raw_lambdas = {k: torch.zeros(3, device=device, requires_grad=True) for k in encoder_keys}
    
    heads = {t_idx: copy.deepcopy(original_heads[t_idx]).to(device) for t_idx in original_heads}
    for t_idx in heads:
        heads[t_idx].train()
        
    sensitivities = {}
    for k in encoder_keys:
        f0 = torch.mean(fisher_priors[0][f"base_encoder.{k}"])
        f1 = torch.mean(fisher_priors[1][f"base_encoder.{k}"])
        f2 = torch.mean(fisher_priors[2][f"base_encoder.{k}"])
        sensitivities[k] = (f0 + f1 + f2).item() / 3.0
        
    mean_sens = np.mean(list(sensitivities.values()))
    for k in sensitivities:
        sensitivities[k] /= (mean_sens + 1e-12)
        
    conv1_traj = []
    fc_traj = []
    
    for step, (task_id, images, labels) in enumerate(stream):
        images = images.to(device)
        
        # --- Adaptation Step ---
        param_groups = []
        alpha = 0.5
        for k in encoder_keys:
            scaled_lr = lr_lambda / (sensitivities[k] + alpha)
            param_groups.append({"params": [raw_lambdas[k]], "lr": scaled_lr})
            
        optimizer = optim.Adam(param_groups)
        merged_params = reconstruct_merged_params(base_encoder, expert_states, raw_lambdas, use_softmax=True)
        
        # Unsupervised Teacher-Free Adaptation
        z = functional_call(base_encoder, merged_params, images)
        merged_outputs = heads[task_id](z)
        p_merged = torch.softmax(merged_outputs, dim=-1)
        loss_ent = -torch.mean(torch.sum(p_merged * torch.log(p_merged + 1e-12), dim=-1))
        
        images_aug = augment_batch(images)
        z_aug = functional_call(base_encoder, merged_params, images_aug)
        merged_outputs_aug = heads[task_id](z_aug)
        p_merged_aug = torch.softmax(merged_outputs_aug, dim=-1)
        loss_const = torch.mean(torch.sum(p_merged.detach() * (torch.log(p_merged.detach() + 1e-12) - torch.log(p_merged_aug + 1e-12)), dim=-1))
        
        loss = loss_ent + gamma_const * loss_const
        
        if "fwar" in method_name:
            loss_fwar = 0.0
            w_0 = torch.tensor([1/3, 1/3, 1/3], device=device)
            for k in encoder_keys:
                w = torch.softmax(raw_lambdas[k], dim=0)
                loss_fwar += sensitivities[k] * torch.sum((w - w_0) ** 2)
            loss += gamma_fwar * loss_fwar
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record weights
        with torch.no_grad():
            w_conv1 = torch.softmax(raw_lambdas["conv1.weight"], dim=0).cpu().numpy()
            w_fc = torch.softmax(raw_lambdas["fc.weight"], dim=0).cpu().numpy()
            conv1_traj.append(w_conv1)
            fc_traj.append(w_fc)
            
    return np.array(conv1_traj), np.array(fc_traj)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    base_encoder = CustomCNN().to(device)
    
    expert_states = []
    original_heads = {}
    expert_names = ["mnist", "fashion", "kmnist"]
    for idx, name in enumerate(expert_names):
        path = f"./experts/expert_{name}.pt"
        state = torch.load(path, map_location=device)
        encoder_state = {k.replace("base_encoder.", ""): v for k, v in state.items() if k.startswith("base_encoder.")}
        head_state = {k.replace("head.", ""): v for k, v in state.items() if k.startswith("head.")}
        expert_states.append(encoder_state)
        head_model = nn.Linear(128, 10).to(device)
        head_model.load_state_dict(head_state)
        original_heads[idx] = head_model
        
    fisher_priors = {}
    for idx, name in enumerate(expert_names):
        path = f"./fisher/fisher_{name}.pt"
        fisher_priors[idx] = torch.load(path, map_location=device)
        
    mnist_test = torch.load("./data/processed/mnist_test.pt")
    fashion_test = torch.load("./data/processed/fashion_test.pt")
    kmnist_test = torch.load("./data/processed/kmnist_test.pt")
    datasets = [mnist_test, fashion_test, kmnist_test]
    
    # Build clean sequential stream
    stream = build_test_stream(datasets, stream_type="sequential", batch_size=32, num_batches_per_task=50, seed=42)
    
    print("Running trajectories without FWAR...")
    conv1_no, fc_no = get_trajectories("fw_cms_tf", stream, base_encoder, expert_states, original_heads, fisher_priors, device)
    
    print("Running trajectories with FWAR...")
    conv1_fwar, fc_fwar = get_trajectories("fw_cms_tf_fwar", stream, base_encoder, expert_states, original_heads, fisher_priors, device)
    
    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    
    # Subplot [0, 0]: conv1.weight without FWAR
    axes[0, 0].plot(conv1_no[:, 0], label="MNIST expert", color="blue", alpha=0.8)
    axes[0, 0].plot(conv1_no[:, 1], label="Fashion expert", color="green", alpha=0.8)
    axes[0, 0].plot(conv1_no[:, 2], label="KMNIST expert", color="red", alpha=0.8)
    axes[0, 0].set_title("conv1.weight (Sensitive) WITHOUT FWAR", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Merging Weight")
    axes[0, 0].set_ylim(-0.05, 1.05)
    axes[0, 0].axvline(50, color="gray", linestyle="--")
    axes[0, 0].axvline(100, color="gray", linestyle="--")
    
    # Subplot [0, 1]: conv1.weight with FWAR
    axes[0, 1].plot(conv1_fwar[:, 0], label="MNIST expert", color="blue", alpha=0.8)
    axes[0, 1].plot(conv1_fwar[:, 1], label="Fashion expert", color="green", alpha=0.8)
    axes[0, 1].plot(conv1_fwar[:, 2], label="KMNIST expert", color="red", alpha=0.8)
    axes[0, 1].set_title("conv1.weight (Sensitive) WITH FWAR", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].axvline(50, color="gray", linestyle="--")
    axes[0, 1].axvline(100, color="gray", linestyle="--")
    
    # Subplot [1, 0]: fc.weight without FWAR
    axes[1, 0].plot(fc_no[:, 0], color="blue", alpha=0.8)
    axes[1, 0].plot(fc_no[:, 1], color="green", alpha=0.8)
    axes[1, 0].plot(fc_no[:, 2], color="red", alpha=0.8)
    axes[1, 0].set_title("fc.weight (Flexible) WITHOUT FWAR", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Merging Weight")
    axes[1, 0].set_xlabel("Stream Step")
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].axvline(50, color="gray", linestyle="--")
    axes[1, 0].axvline(100, color="gray", linestyle="--")
    
    # Subplot [1, 1]: fc.weight with FWAR
    axes[1, 1].plot(fc_fwar[:, 0], color="blue", alpha=0.8)
    axes[1, 1].plot(fc_fwar[:, 1], color="green", alpha=0.8)
    axes[1, 1].plot(fc_fwar[:, 2], color="red", alpha=0.8)
    axes[1, 1].set_title("fc.weight (Flexible) WITH FWAR", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Stream Step")
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].axvline(50, color="gray", linestyle="--")
    axes[1, 1].axvline(100, color="gray", linestyle="--")
    
    # Task Labels
    for ax in axes.flat:
        ax.text(25, 0.9, "MNIST", color="gray", fontsize=9, ha="center", fontweight="semibold")
        ax.text(75, 0.9, "Fashion", color="gray", fontsize=9, ha="center", fontweight="semibold")
        ax.text(125, 0.9, "KMNIST", color="gray", fontsize=9, ha="center", fontweight="semibold")
        
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.98), fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig("weight_trajectories.png", dpi=300, bbox_inches="tight")
    print("Trajectories plotted and saved to weight_trajectories.png.")

if __name__ == "__main__":
    main()
