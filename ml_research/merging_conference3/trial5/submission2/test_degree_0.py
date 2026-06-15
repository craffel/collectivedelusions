import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.func as tf
from run_experiments import set_seed, Deep12LayerCNN, get_datasets, get_merged_params_dict_for_task, evaluate_model

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running baseline evaluation on device: {device}", flush=True)
    
    datasets = get_datasets(device)
    base_model = Deep12LayerCNN().to(device)
    
    # Pre-train
    base_optimizer = optim.Adam(base_model.parameters(), lr=2e-3)
    mixed_imgs, mixed_labels = [], []
    for k in range(4):
        train_ds, _ = datasets[k]
        loader = DataLoader(train_ds, batch_size=1000, shuffle=False)
        imgs, labels = next(iter(loader))
        mixed_imgs.append(imgs)
        mixed_labels.append(labels)
    mixed_imgs = torch.cat(mixed_imgs)
    mixed_labels = torch.cat(mixed_labels)
    mixed_dataset = TensorDataset(mixed_imgs, mixed_labels)
    mixed_loader = DataLoader(mixed_dataset, batch_size=32, shuffle=True)
    
    base_model.train()
    for epoch in range(3):
        for imgs, labels in mixed_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            base_optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(base_model(imgs), labels)
            loss.backward()
            base_optimizer.step()
            
    # Fine-tune experts
    expert_models = []
    for k in range(4):
        expert = Deep12LayerCNN().to(device)
        expert.load_state_dict(base_model.state_dict())
        expert_optimizer = optim.Adam(expert.parameters(), lr=1e-3)
        train_ds, _ = datasets[k]
        loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        
        expert.train()
        for epoch in range(8):
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                expert_optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(expert(imgs), labels)
                loss.backward()
                expert_optimizer.step()
        expert_models.append(expert)
        
    merged_model = Deep12LayerCNN().to(device)
    
    # Calibration set M = 10
    calibration_data = []
    for k in range(4):
        train_ds, _ = datasets[k]
        loader_cal = DataLoader(train_ds, batch_size=500, shuffle=False)
        all_imgs, all_labels = next(iter(loader_cal))
        cal_imgs_list, cal_labels_list = [], []
        for class_idx in range(10):
            matches = (all_labels == class_idx).nonzero().flatten()
            cal_imgs_list.append(all_imgs[matches[0].item()])
            cal_labels_list.append(all_labels[matches[0].item()])
        calibration_data.append((torch.stack(cal_imgs_list).to(device), torch.tensor(cal_labels_list).to(device)))
        
    # --- Globally Scaled Task Arithmetic (d = 0) ---
    print("\n--- Running Globally Scaled Task Arithmetic (Single Scalar / d=0) ---", flush=True)
    merged_model.load_state_dict(base_model.state_dict())
    beta = torch.zeros(4, 1, requires_grad=True, device=device) # Single scalar per task
    with torch.no_grad():
        beta[:, 0] = -1.0986
        
    optimizer = optim.Adam([beta], lr=1e-1)
    criterion = nn.CrossEntropyLoss()
    
    for step in range(30):
        optimizer.zero_grad()
        alphas = beta.expand(4, 12) # Expand to 12 layers
        alphas = torch.sigmoid(alphas)
        
        loss = 0.0
        for k in range(4):
            cal_imgs, cal_labels = calibration_data[k]
            params = get_merged_params_dict_for_task(base_model, expert_models, alphas, k)
            outputs = tf.functional_call(merged_model, params, cal_imgs)
            loss += criterion(outputs, cal_labels)
            
        loss.backward()
        optimizer.step()
        
    alphas_d0 = torch.sigmoid(beta.expand(4, 12)).detach()
    accs_d0 = evaluate_model(merged_model, datasets, base_model, expert_models, alphas_d0, device)
    print(f"Globally Scaled Task Arithmetic (d=0) Accs: {accs_d0}, Average: {np.mean(accs_d0):.2f}%", flush=True)

if __name__ == '__main__':
    main()
