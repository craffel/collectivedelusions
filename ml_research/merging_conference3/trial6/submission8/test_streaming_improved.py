import torch
import numpy as np
from run_physical_validation import SimpleCNN, load_experts, build_data_splits, TASKS, blend_parameters_functional, evaluate_physical_heterogeneous_stream
from test_improved_router import ImprovedPhysicalRoutingHead, extract_improved_features, train_improved_router

def evaluate_improved_heterogeneous_stream(experts, head, mean, std, test_data, test_labels, batch_size=16, k=4, T=0.1, use_dbf=False, seed=42):
    head.eval()
    base_model = SimpleCNN()
    uniform_params = blend_parameters_functional(experts, torch.tensor([0.25, 0.25, 0.25, 0.25]), k=0)
    
    # Create the combined stream
    stream_imgs = []
    stream_lbls = []
    
    # Ensure deterministic shuffle per seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    for task in TASKS:
        imgs = test_data[task]
        lbls = test_labels[task]
        for idx in range(len(imgs)):
            stream_imgs.append(imgs[idx])
            stream_lbls.append(lbls[idx])
            
    stream_imgs = torch.stack(stream_imgs)
    stream_lbls = torch.tensor(stream_lbls, dtype=torch.long)
    
    # Shuffle stream
    shuffle_idx = torch.randperm(len(stream_imgs))
    stream_imgs = stream_imgs[shuffle_idx]
    stream_lbls = stream_lbls[shuffle_idx]
    
    num_batches = len(stream_imgs) // batch_size
    total_correct = 0
    total_samples = num_batches * batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            batch_imgs = stream_imgs[i*batch_size : (i+1)*batch_size]
            batch_lbls = stream_lbls[i*batch_size : (i+1)*batch_size]
            
            # 1. Extract features for the batch
            batch_h_features = extract_improved_features(base_model, uniform_params, batch_imgs)
            batch_norm_features = (batch_h_features - mean) / std
            
            if use_dbf and batch_size > 1:
                # Use DBF: cluster features into M=4 groups
                from run_physical_validation import kmeans_pytorch
                cluster_ids = kmeans_pytorch(batch_norm_features, num_clusters=4, num_iters=10, seed=seed)
                
                for c in range(4):
                    cluster_mask = (cluster_ids == c)
                    if cluster_mask.sum() == 0:
                        continue
                    sub_imgs = batch_imgs[cluster_mask]
                    sub_lbls = batch_lbls[cluster_mask]
                    sub_norm_feats = batch_norm_features[cluster_mask]
                    
                    logits = head(sub_norm_feats)
                    alphas = torch.softmax(logits / T, dim=-1)
                    mean_alphas = alphas.mean(dim=0)
                    
                    task_params = blend_parameters_functional(experts, mean_alphas, k=k)
                    out = torch.func.functional_call(base_model, task_params, sub_imgs)
                    _, predicted = out.max(1)
                    total_correct += predicted.eq(sub_lbls).sum().item()
            else:
                logits = head(batch_norm_features)
                alphas = torch.softmax(logits / T, dim=-1)
                mean_alphas = alphas.mean(dim=0)
                
                task_params = blend_parameters_functional(experts, mean_alphas, k=k)
                out = torch.func.functional_call(base_model, task_params, batch_imgs)
                _, predicted = out.max(1)
                total_correct += predicted.eq(batch_lbls).sum().item()
                
    accuracy = 100.0 * total_correct / total_samples
    return accuracy

def test_stream_seeds():
    experts = load_experts()
    SEEDS = [42, 43, 44]
    
    results = {
        'B16_Std': [],
        'B16_DBF': [],
        'B64_Std': [],
        'B64_DBF': []
    }
    
    for seed in SEEDS:
        print(f"\n[Seed {seed}] Training improved routing head for stream evaluation...")
        cal_data, cal_labels, cal_task_ids, test_data, test_labels = build_data_splits(seed=seed)
        head, mean, std = train_improved_router(experts, cal_data, cal_labels, cal_task_ids, num_epochs=300)
        
        acc = evaluate_improved_heterogeneous_stream(experts, head, mean, std, test_data, test_labels, batch_size=16, k=4, T=0.1, use_dbf=False, seed=seed)
        results['B16_Std'].append(acc)
        
        acc = evaluate_improved_heterogeneous_stream(experts, head, mean, std, test_data, test_labels, batch_size=16, k=4, T=0.1, use_dbf=True, seed=seed)
        results['B16_DBF'].append(acc)
        
        acc = evaluate_improved_heterogeneous_stream(experts, head, mean, std, test_data, test_labels, batch_size=64, k=4, T=0.1, use_dbf=False, seed=seed)
        results['B64_Std'].append(acc)
        
        acc = evaluate_improved_heterogeneous_stream(experts, head, mean, std, test_data, test_labels, batch_size=64, k=4, T=0.1, use_dbf=True, seed=seed)
        results['B64_DBF'].append(acc)
        
    print("\n" + "="*80)
    print("--- STREAMING RESULTS WITH IMPROVED ROUTER ---")
    print("="*80)
    for cfg in ['B16_Std', 'B16_DBF', 'B64_Std', 'B64_DBF']:
        accs = results[cfg]
        print(f"  {cfg:10s} -> Accuracy: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")

if __name__ == '__main__':
    test_stream_seeds()
