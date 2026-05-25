import torch
import copy
from experiment import get_datasets, generate_test_stream, run_tta_evaluation, get_pretrained_base_encoder, get_task_vector, ResNet18Expert
from torch.func import functional_call
import torch.nn as nn
import torch.optim as optim

tasks = ['mnist', 'fmnist', 'kmnist']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = get_datasets()
test_datasets = [data[tasks[i]][1] for i in range(3)]

expert_encoders = []
expert_heads_list = []
for task in tasks:
    resnet = ResNet18Expert()
    resnet.encoder.load_state_dict(torch.load(f"./checkpoints/{task}_encoder.pt", map_location=device))
    resnet.fc.load_state_dict(torch.load(f"./checkpoints/{task}_head.pt", map_location=device))
    expert_encoders.append(resnet.encoder.to(device))
    expert_heads_list.append(resnet.fc.to(device))

base_encoder = get_pretrained_base_encoder()
base_encoder.eval()
task_vectors = [get_task_vector(expert_encoders[i], base_encoder) for i in range(3)]

# Sweep different learning rates for lambdas
lrs = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
batches = generate_test_stream(test_datasets, 'sequential', seed=42)

for lr in lrs:
    # Standard TTA
    lambdas = torch.tensor([1/3, 1/3, 1/3], requires_grad=True, device=device)
    adapted_heads = [copy.deepcopy(h).to(device) for h in expert_heads_list]
    opt_heads = [optim.Adam(adapted_heads[i].parameters(), lr=1e-4) for i in range(3)]
    opt_lambdas = optim.Adam([lambdas], lr=lr)
    
    correct = 0
    total = 0
    
    for task_id, x, y in batches:
        x, y = x.to(device), y.to(device)
        
        opt_heads[task_id].zero_grad()
        opt_lambdas.zero_grad()
        
        with torch.no_grad():
            expert_features = expert_encoders[task_id](x)
            expert_features = torch.flatten(expert_features, 1)
            expert_outputs = expert_heads_list[task_id](expert_features)
            p_expert = torch.softmax(expert_outputs, dim=1)
            
        merged_params = {}
        for name, param in base_encoder.named_parameters():
            if name in task_vectors[0]:
                update = torch.zeros_like(param)
                for i, task_vec in enumerate(task_vectors):
                    update = update + lambdas[i] * task_vec[name].to(device)
                merged_params[name] = base_encoder.state_dict()[name].to(device) + update
            else:
                merged_params[name] = param
                
        features = functional_call(base_encoder, merged_params, x)
        features = torch.flatten(features, 1)
        outputs = adapted_heads[task_id](features)
        p_merged = torch.softmax(outputs, dim=1)
        
        kl_loss = nn.KLDivLoss(reduction='batchmean')(torch.log(p_merged + 1e-8), p_expert)
        kl_loss.backward()
        
        opt_heads[task_id].step()
        opt_lambdas.step()
        
        with torch.no_grad():
            lambdas.clamp_(0.0, 1.0)
            sum_l = lambdas.sum()
            if sum_l > 0:
                lambdas.div_(sum_l)
                
        # Eval AFTER
        with torch.no_grad():
            merged_params_eval = {}
            for name, param in base_encoder.named_parameters():
                if name in task_vectors[0]:
                    update = torch.zeros_like(param)
                    for i, task_vec in enumerate(task_vectors):
                        update = update + lambdas[i] * task_vec[name].to(device)
                    merged_params_eval[name] = base_encoder.state_dict()[name].to(device) + update
                else:
                    merged_params_eval[name] = param
                    
            eval_features = functional_call(base_encoder, merged_params_eval, x)
            eval_features = torch.flatten(eval_features, 1)
            eval_outputs = adapted_heads[task_id](eval_features)
            _, predicted = eval_outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
    print(f"Lambda LR: {lr:.2f} | Sequential Accuracy: {correct/total*100:.2f}%")
