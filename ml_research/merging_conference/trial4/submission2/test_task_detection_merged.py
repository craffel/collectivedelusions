import torch
import torch.nn as nn
from train_experts import BaseEncoder, ClassHead, ExpertModel, get_dataloader
from torch.func import functional_call
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_entropy(logits, eps=1e-12):
    probs = torch.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)
    return torch.mean(entropy)

def test_detection_merged():
    expert_names = ['mnist', 'fashion', 'kmnist']
    experts = {}
    
    # Load base encoder and experts
    base_encoder = BaseEncoder().to(device)
    base_encoder.load_state_dict(torch.load('./experts/base_encoder_init.pt', map_location=device, weights_only=True))
    base_params = {name: param.cpu() for name, param in base_encoder.named_parameters()}
    
    task_vectors = []
    for k, name in enumerate(expert_names):
        ckpt_path = f'./experts/{name}_expert.pt'
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        
        expert_encoder = BaseEncoder().to(device)
        expert_encoder.load_state_dict(ckpt['encoder_state_dict'])
        expert_head = ClassHead().to(device)
        expert_head.load_state_dict(ckpt['head_state_dict'])
        
        experts[name] = ExpertModel(expert_encoder, expert_head).to(device)
        experts[name].eval()
        
        tv = {}
        for p_name, p_val in expert_encoder.named_parameters():
            base_val = base_encoder.state_dict()[p_name]
            tv[p_name] = (p_val - base_val).cpu()
        task_vectors.append(tv)
        
    # Build static merged encoder
    merged_params = {}
    lambdas = torch.tensor([1.0/3.0, 1.0/3.0, 1.0/3.0])
    for p_name, base_p in base_params.items():
        w_merged = base_p.clone().to(device)
        for k in range(3):
            tv_val = task_vectors[k][p_name].to(device)
            w_merged = w_merged + lambdas[k] * tv_val
        merged_params[p_name] = w_merged
        
    # Get test dataloaders
    test_loaders = {}
    for name in expert_names:
        test_loaders[name] = get_dataloader(name, batch_size=64, train=False)
        
    print("Evaluating Task Detection via prediction entropy of static merged encoder...")
    
    for true_task in expert_names:
        loader = test_loaders[true_task]
        iterator = iter(loader)
        
        correct_detections = 0
        total = 0
        
        for i in range(20):
            try:
                images, _ = next(iterator)
            except StopIteration:
                break
                
            images = images.to(device)
            
            # Use static merged encoder to extract features
            with torch.no_grad():
                features = functional_call(base_encoder, merged_params, images)
                
                entropies = {}
                for task_name in expert_names:
                    logits = experts[task_name].head(features)
                    entropies[task_name] = compute_entropy(logits).item()
                    
            predicted_task = min(entropies, key=entropies.get)
            if predicted_task == true_task:
                correct_detections += 1
            total += 1
            
        print(f"True Task: {true_task.upper()} | Detection Accuracy: {100.0 * correct_detections / total:.2f}%")
        
if __name__ == '__main__':
    test_detection_merged()
