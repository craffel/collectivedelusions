import torch
import torch.nn as nn
import torch.optim as optim

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def generate_sandbox_data(seed=42):
    setup_seed(seed)
    D = 192
    K = 4
    C = 10
    
    # 48 dimensions per task
    subspace_dim = 48
    
    # Generate class prototypes for each task
    # Shape: (K, C, subspace_dim)
    prototypes = torch.randn(K, C, subspace_dim)
    prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)
    
    # Noise scales for MNIST, F-MNIST, CIFAR-10, SVHN
    noise_scales = [0.05, 0.15, 0.40, 1.20]
    
    def generate_split(num_samples_per_task):
        X = []
        Y_task = []
        Y_class = []
        for k in range(K):
            for _ in range(num_samples_per_task):
                c = torch.randint(0, C, (1,)).item()
                # Initialize representation as zeros
                z = torch.zeros(D)
                # Populate the task-specific subspace
                proto = prototypes[k, c]
                noise = noise_scales[k] * torch.randn(subspace_dim)
                z[k*subspace_dim : (k+1)*subspace_dim] = proto + noise
                
                # Overlapping bleed
                bleed = 0.08 * torch.randn(D)
                bleed[k*subspace_dim : (k+1)*subspace_dim] = 0.0
                z = z + bleed
                
                X.append(z)
                Y_task.append(k)
                Y_class.append(c)
        return torch.stack(X), torch.tensor(Y_task), torch.tensor(Y_class)
    
    # Generate splits
    X_train, Y_train_task, Y_train_class = generate_split(1000)
    X_cal, Y_cal_task, Y_cal_class = generate_split(16)
    X_test, Y_test_task, Y_test_class = generate_split(250)
    
    return prototypes, noise_scales, (X_train, Y_train_task, Y_train_class), (X_cal, Y_cal_task, Y_cal_class), (X_test, Y_test_task, Y_test_class)

def evaluate_ceilings():
    prototypes, noise_scales, train, cal, test = generate_sandbox_data()
    X_test, Y_test_task, Y_test_class = test
    
    # Build expert classification heads
    # Each expert has shape (10, 192)
    D = 192
    K = 4
    C = 10
    subspace_dim = 48
    
    heads = []
    for k in range(K):
        W = torch.randn(C, D) * 0.05
        # Place prototypes in the corresponding subspace
        W[:, k*subspace_dim : (k+1)*subspace_dim] += prototypes[k]
        heads.append(W)
        
    # Evaluate each expert on its own task
    for k in range(K):
        task_mask = (Y_test_task == k)
        X_task = X_test[task_mask]
        Y_class_task = Y_test_class[task_mask]
        
        # Expert head prediction
        logits = X_task @ heads[k].t()
        preds = logits.argmax(dim=-1)
        acc = (preds == Y_class_task).float().mean().item() * 100
        print(f"Expert {k} Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    evaluate_ceilings()
