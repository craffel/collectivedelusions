import torch
import torch.nn as nn
import numpy as np
from run_experiments import generate_datasets, ExpertModel, set_seed, DEVICE, K, NUM_LAYERS

def eval_zcr_for_seed(seed):
    train_data, cal_data, test_data, prototypes = generate_datasets(seed)
    
    # Train Experts (same as main experiments)
    experts = {}
    for k in range(K):
        model = ExpertModel().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        x, y = train_data[k]
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        for epoch in range(60):
            model.train()
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        experts[k] = model
        
    # Prepare stream datasets
    homo_x = []
    homo_y = []
    homo_tasks = []
    for k in range(K):
        test_x, test_y = test_data[k]
        homo_x.append(test_x)
        homo_y.append(test_y)
        homo_tasks.extend([k]*len(test_x))
    homo_x = torch.cat(homo_x).to(DEVICE)
    homo_y = torch.cat(homo_y).to(DEVICE)
    
    set_seed(seed + 100)
    shuffled_idx = torch.randperm(len(homo_x))
    hete_x = homo_x[shuffled_idx]
    hete_y = homo_y[shuffled_idx]
    hete_tasks = np.array(homo_tasks)[shuffled_idx.numpy()]
    
    # Define Zero-Shot Cosine Routing (ZCR)
    # Weight-space task prototypes from expert heads
    weight_centroids = {}
    for k in range(K):
        # average the class projection vectors in the classification head weight matrix (shape C, D)
        weight_centroids[k] = experts[k].head.weight.mean(dim=0)
        weight_centroids[k] = weight_centroids[k] / (weight_centroids[k].norm(p=2) + 1e-8)
        
    correct = 0
    total = 0
    task_correct = [0]*K
    task_total = [0]*K
    
    with torch.no_grad():
        for b in range(len(hete_x)):
            x = hete_x[b:b+1]
            y = hete_y[b:b+1]
            t_true = hete_tasks[b]
            
            # Compute cosine similarity with each weight centroid
            u = torch.zeros(K)
            for k in range(K):
                u[k] = torch.dot(x[0], weight_centroids[k]) / (x[0].norm(p=2) * weight_centroids[k].norm(p=2) + 1e-8)
                
            # Route to the expert with highest cosine similarity
            k_star = u.argmax().item()
            
            logits_star, _ = experts[k_star](x)
            pred = logits_star.argmax(dim=1)
            
            is_correct = (pred == y).item()
            correct += is_correct
            total += 1
            task_correct[t_true] += is_correct
            task_total[t_true] += 1
            
    return correct/max(1, total), [task_correct[k]/max(1, task_total[k]) for k in range(K)]

if __name__ == "__main__":
    SEEDS = [10, 11, 12, 13, 14]
    all_accs = []
    all_task_accs = []
    for seed in SEEDS:
        print(f"Running seed {seed}...")
        acc, task_accs = eval_zcr_for_seed(seed)
        all_accs.append(acc)
        all_task_accs.append(task_accs)
        print(f"Seed {seed} ZCR Joint Accuracy: {acc*100:.2f}%")
        
    mean_acc = np.mean(all_accs)
    std_acc = np.std(all_accs)
    mean_task_accs = np.mean(all_task_accs, axis=0)
    print(f"\nAGGREGATED ZCR RESULTS over 5 seeds:")
    print(f"MNIST: {mean_task_accs[0]*100:.2f}%")
    print(f"FashionMNIST: {mean_task_accs[1]*100:.2f}%")
    print(f"CIFAR-10: {mean_task_accs[2]*100:.2f}%")
    print(f"SVHN: {mean_task_accs[3]*100:.2f}%")
    print(f"Joint Mean: {mean_acc*100:.2f} ± {std_acc*100:.2f}%")
