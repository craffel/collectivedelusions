import torch
import os
import copy
from models import get_model, get_dataloaders, train_expert, evaluate_model, set_seed

def main():
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs('./checkpoints', exist_ok=True)
    
    architectures = ['resnet18', 'mlp']
    datasets_list = ['mnist', 'fmnist', 'cifar10']
    
    # 1. Train or load progenitor and experts
    for arch in architectures:
        print(f"\n======================================")
        print(f"Processing Architecture: {arch}")
        print(f"======================================")
        
        progenitor_path = f'./checkpoints/{arch}_progenitor.pt'
        if os.path.exists(progenitor_path):
            print(f"Loading existing progenitor from {progenitor_path}")
            progenitor = get_model(arch)
            progenitor.load_state_dict(torch.load(progenitor_path, map_location='cpu'))
        else:
            print(f"Initializing new progenitor for {arch}")
            progenitor = get_model(arch)
            torch.save(progenitor.state_dict(), progenitor_path)
            
        for dataset_name in datasets_list:
            expert_path = f'./checkpoints/{arch}_{dataset_name}.pt'
            if os.path.exists(expert_path):
                print(f"Expert for {dataset_name} already exists. Skipping training.")
                continue
                
            print(f"\n--- Training Expert on {dataset_name} ---")
            train_loader, test_loader = get_dataloaders(dataset_name, batch_size=256)
            
            # Start from progenitor weights
            expert = get_model(arch)
            expert.load_state_dict(copy.deepcopy(progenitor.state_dict()))
            
            # Train the expert
            train_expert(expert, train_loader, epochs=5, lr=1e-3, device=device)
            
            # Evaluate the expert
            acc = evaluate_model(expert, test_loader, device=device)
            print(f"Finished training {arch} on {dataset_name}. Test Accuracy: {acc:.2f}%")
            
            # Save the expert
            torch.save(expert.state_dict(), expert_path)

if __name__ == '__main__':
    main()
