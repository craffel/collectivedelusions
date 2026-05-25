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

def test_detection():
    expert_names = ['mnist', 'fashion', 'kmnist']
    experts = {}
    
    # Load base encoder and experts
    base_encoder = BaseEncoder().to(device)
    base_encoder.load_state_dict(torch.load('./experts/base_encoder_init.pt', map_location=device, weights_only=True))
    
    for name in expert_names:
        ckpt_path = f'./experts/{name}_expert.pt'
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        
        expert_encoder = BaseEncoder().to(device)
        expert_encoder.load_state_dict(ckpt['encoder_state_dict'])
        expert_head = ClassHead().to(device)
        expert_head.load_state_dict(ckpt['head_state_dict'])
        
        experts[name] = ExpertModel(expert_encoder, expert_head).to(device)
        experts[name].eval()
        
    # Get test dataloaders
    test_loaders = {}
    for name in expert_names:
        test_loaders[name] = get_dataloader(name, batch_size=64, train=False)
        
    print("Evaluating Task Detection via prediction entropy...")
    
    # We will test on 20 batches of each task
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
            
            # For each head, we compute the entropy of predictions when using the merged/expert representations.
            # Here we can use the specific expert encoder or the shared/merged encoder.
            # Let's first check with the true expert models themselves.
            entropies = {}
            for task_name in expert_names:
                with torch.no_grad():
                    logits = experts[task_name](images)
                    entropies[task_name] = compute_entropy(logits).item()
                    
            predicted_task = min(entropies, key=entropies.get)
            if predicted_task == true_task:
                correct_detections += 1
            total += 1
            
        print(f"True Task: {true_task.upper()} | Detection Accuracy: {100.0 * correct_detections / total:.2f}%")
        
if __name__ == '__main__':
    test_detection()
