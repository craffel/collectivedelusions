import torch
import torch.nn as nn
from run_physical_validation import SimpleCNN, load_experts, build_data_splits, TASKS, blend_parameters_functional
from test_improved_router import ImprovedPhysicalRoutingHead, extract_improved_features, train_improved_router

def evaluate_sweep(experts, head, mean, std, test_data, test_labels, T=0.1):
    head.eval()
    base_model = SimpleCNN()
    
    print("\n" + "-"*80)
    print("--- SWEEP OF PARTITION DEPTH k (IMPROVED ROUTER) ---")
    print("-"*80)
    print(f"{'Depth (k)':9s} | {'MNIST':10s} | {'FMNIST':10s} | {'CIFAR10':10s} | {'SVHN':10s} | {'Joint Mean':12s}")
    print("-"*75)
    
    with torch.no_grad():
        for k in [0, 1, 2, 3, 4]:
            results = {}
            total_correct = 0
            total_samples = 0
            
            for task_id, task in enumerate(TASKS):
                imgs = test_data[task]
                lbls = test_labels[task]
                
                task_h_features = extract_improved_features(base_model, blend_parameters_functional(experts, torch.tensor([0.25, 0.25, 0.25, 0.25]), k=0), imgs)
                task_norm_features = (task_h_features - mean) / std
                
                logits = head(task_norm_features)
                alphas = torch.softmax(logits / T, dim=-1)
                mean_alphas = alphas.mean(dim=0)
                
                task_params = blend_parameters_functional(experts, mean_alphas, k=k)
                out = torch.func.functional_call(base_model, task_params, imgs)
                _, predicted = out.max(1)
                correct = predicted.eq(lbls).sum().item()
                acc = 100.0 * correct / len(lbls)
                
                results[task] = acc
                total_correct += correct
                total_samples += len(lbls)
                
            joint_mean = 100.0 * total_correct / total_samples
            print(f"   {k:2d}     | {results['MNIST']:.2f}%     | {results['FashionMNIST']:.2f}%     | {results['CIFAR10']:.2f}%     | {results['SVHN']:.2f}%     | {joint_mean:.2f}%")

if __name__ == '__main__':
    experts = load_experts()
    cal_data, cal_labels, cal_task_ids, test_data, test_labels = build_data_splits(seed=42)
    head, mean, std = train_improved_router(experts, cal_data, cal_labels, cal_task_ids, num_epochs=300)
    evaluate_sweep(experts, head, mean, std, test_data, test_labels)
