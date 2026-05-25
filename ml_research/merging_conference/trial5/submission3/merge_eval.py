import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import copy
import os

from models import ResNetEncoder, ClassificationHead

# Preprocessing transforms (ImageNet standard)
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_test_streams(seed=42, batch_size=64):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    
    # Load full datasets
    cifar_full_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    svhn_full_test = datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    
    # Extract deterministic test subsets (16 batches of 64 images = 1,024 images each)
    cifar_indices = torch.randperm(len(cifar_full_test), generator=g)[:1024].tolist()
    cifar_subset = Subset(cifar_full_test, cifar_indices)
    
    svhn_indices = torch.randperm(len(svhn_full_test), generator=g)[:1024].tolist()
    svhn_subset = Subset(svhn_full_test, svhn_indices)
    
    # Dataloaders (no shuffle, batch size 64)
    cifar_loader = DataLoader(cifar_subset, batch_size=batch_size, shuffle=False)
    svhn_loader = DataLoader(svhn_subset, batch_size=batch_size, shuffle=False)
    
    # Collect all batches
    cifar_batches = list(cifar_loader)
    svhn_batches = list(svhn_loader)
    
    assert len(cifar_batches) == 16, f"Expected 16 batches of CIFAR-10, got {len(cifar_batches)}"
    assert len(svhn_batches) == 16, f"Expected 16 batches of SVHN, got {len(svhn_batches)}"
    
    # 1. Alternating Stream: Interleaves batches (CIFAR, SVHN, CIFAR, SVHN...)
    alternating_stream = []
    for i in range(16):
        alternating_stream.append((cifar_batches[i][0], cifar_batches[i][1], 0)) # 0 for CIFAR-10
        alternating_stream.append((svhn_batches[i][0], svhn_batches[i][1], 1)) # 1 for SVHN
        
    # 2. Block-Sequential Stream: 16 batches of CIFAR-10, followed by 16 batches of SVHN
    block_sequential_stream = []
    for i in range(16):
        block_sequential_stream.append((cifar_batches[i][0], cifar_batches[i][1], 0))
    for i in range(16):
        block_sequential_stream.append((svhn_batches[i][0], svhn_batches[i][1], 1))
        
    return alternating_stream, block_sequential_stream

class TestTimeModelMerger:
    def __init__(self, device):
        self.device = device
        
        # Load base model encoder (pretrained)
        self.base_encoder = ResNetEncoder().to(device)
        self.base_encoder.eval()
        
        # Store base model parameter names and tensors
        self.parameter_names = [name for name, _ in self.base_encoder.named_parameters()]
        self.base_params = {name: param.clone().detach() for name, param in self.base_encoder.named_parameters()}
        
        # Load Expert 1 (CIFAR-10)
        cifar_ckpt = torch.load("expert_cifar10.pth", map_location=device)
        self.cifar_head = ClassificationHead().to(device)
        self.cifar_head.load_state_dict(cifar_ckpt['head'])
        self.cifar_head.eval()
        
        # Load Expert 2 (SVHN)
        svhn_ckpt = torch.load("expert_svhn.pth", map_location=device)
        self.svhn_head = ClassificationHead().to(device)
        self.svhn_head.load_state_dict(svhn_ckpt['head'])
        self.svhn_head.eval()
        
        # Extract expert parameters to compute task vectors
        cifar_encoder_state = cifar_ckpt['encoder']
        svhn_encoder_state = svhn_ckpt['encoder']
        
        self.task_vectors = []
        # Expert 1 Task Vector
        tv1 = {}
        for name in self.parameter_names:
            tv1[name] = (cifar_encoder_state[name] - self.base_params[name]).clone().detach()
        self.task_vectors.append(tv1)
        
        # Expert 2 Task Vector
        tv2 = {}
        for name in self.parameter_names:
            tv2[name] = (svhn_encoder_state[name] - self.base_params[name]).clone().detach()
        self.task_vectors.append(tv2)
        
        self.heads = [self.cifar_head, self.svhn_head]
        
    def compute_fisher_information(self, num_samples=256):
        """
        Pre-compute the diagonal Fisher Information of the base model on each expert's training set
        as a static layer-wise sensitivity prior (for LFWA baseline).
        """
        print(f"\nComputing Fisher Information (num_samples={num_samples}) for both tasks...")
        fisher_sums = [ {name: torch.zeros_like(param) for name, param in self.base_params.items()} for _ in range(2) ]
        
        # CIFAR-10 Train subset
        cifar_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
        g = torch.Generator().manual_seed(42)
        cifar_indices = torch.randperm(len(cifar_train), generator=g)[:num_samples].tolist()
        cifar_loader = DataLoader(Subset(cifar_train, cifar_indices), batch_size=32, shuffle=False)
        
        # SVHN Train subset
        svhn_train = datasets.SVHN(root='./data', split='train', download=True, transform=transform_test)
        svhn_indices = torch.randperm(len(svhn_train), generator=g)[:num_samples].tolist()
        svhn_loader = DataLoader(Subset(svhn_train, svhn_indices), batch_size=32, shuffle=False)
        
        loaders = [cifar_loader, svhn_loader]
        
        for k in range(2):
            # Load expert weights into temporary encoder to compute task-specific gradients
            temp_encoder = ResNetEncoder().to(self.device)
            expert_state = torch.load("expert_cifar10.pth" if k == 0 else "expert_svhn.pth", map_location=self.device)['encoder']
            # Realign state dict keys
            realigned_state = {name: expert_state[name] for name in self.parameter_names}
            temp_encoder.load_state_dict(realigned_state, strict=False)
            temp_encoder.eval()
            
            head = self.heads[k]
            criterion = nn.CrossEntropyLoss()
            
            total_gradient_steps = 0
            for images, labels in loaders[k]:
                images, labels = images.to(self.device), labels.to(self.device)
                
                for i in range(len(images)):
                    # Compute sample-wise gradient
                    img = images[i:i+1]
                    lbl = labels[i:i+1]
                    
                    temp_encoder.zero_grad()
                    head.zero_grad()
                    
                    features = temp_encoder(img)
                    outputs = head(features)
                    loss = criterion(outputs, lbl)
                    loss.backward()
                    
                    for name, param in temp_encoder.named_parameters():
                        if param.grad is not None:
                            fisher_sums[k][name] += param.grad.data.pow(2)
                    
                    total_gradient_steps += 1
                    
            # Normalize Fisher
            for name in self.parameter_names:
                fisher_sums[k][name] /= total_gradient_steps
                
        # Compute layer-wise average Fisher Information
        self.layer_fisher = {}
        for name in self.parameter_names:
            # Average over elements in this parameter tensor
            avg_f1 = fisher_sums[0][name].mean().item()
            avg_f2 = fisher_sums[1][name].mean().item()
            # Joint Fisher (average across the two tasks)
            self.layer_fisher[name] = 0.5 * (avg_f1 + avg_f2)
            
        print("Fisher Information computation completed.")
        return self.layer_fisher
        
    def evaluate_stream(self, stream, method, lr_base=0.1, beta1=0.9, beta2=0.99, eps=1e-8):
        """
        Evaluate the merged model on a non-stationary test stream.
        method can be: 'Static', 'Uniform', 'LFWA', or 'AdaSNR'.
        """
        # Initialize merging coefficients: lambdas_raw of shape (60, 2), initialized to 0.5
        lambdas_raw = torch.ones(len(self.parameter_names), 2, device=self.device) * 0.5
        lambdas_raw.requires_grad = True
        
        # Moving averages for AdaSNR
        m = torch.zeros_like(lambdas_raw)
        v = torch.zeros_like(lambdas_raw)
        
        pre_accuracies = []
        task_accuracies = {0: [], 1: []} # 0: CIFAR-10, 1: SVHN
        
        for step, (images, labels, task_id) in enumerate(stream):
            images, labels = images.to(self.device), labels.to(self.device)
            head = self.heads[task_id]
            
            # 1. EVALUATION (PRE-ADAPTATION ACCURACY)
            with torch.no_grad():
                # Merge parameters with current lambdas
                merged_params = {}
                for i, name in enumerate(self.parameter_names):
                    l1 = torch.clamp(lambdas_raw[i, 0], 0.0, 1.0)
                    l2 = torch.clamp(lambdas_raw[i, 1], 0.0, 1.0)
                    merged_params[name] = self.base_params[name] + l1 * self.task_vectors[0][name] + l2 * self.task_vectors[1][name]
                
                # Forward pass
                features = torch.func.functional_call(self.base_encoder, merged_params, images)
                outputs = head(features)
                _, predicted = outputs.max(1)
                correct = predicted.eq(labels).sum().item()
                acc = 100.0 * correct / len(labels)
                pre_accuracies.append(acc)
                task_accuracies[task_id].append(acc)
                
            # 2. ADAPTATION (IF NOT STATIC)
            if method != 'Static':
                # Re-compute forward pass with gradients enabled on lambdas_raw
                merged_params = {}
                for i, name in enumerate(self.parameter_names):
                    l1 = torch.clamp(lambdas_raw[i, 0], 0.0, 1.0)
                    l2 = torch.clamp(lambdas_raw[i, 1], 0.0, 1.0)
                    merged_params[name] = self.base_params[name] + l1 * self.task_vectors[0][name] + l2 * self.task_vectors[1][name]
                
                features = torch.func.functional_call(self.base_encoder, merged_params, images)
                outputs = head(features)
                
                # Unsupervised entropy loss
                probs = F.softmax(outputs, dim=1)
                log_probs = F.log_softmax(outputs, dim=1)
                entropy_loss = -(probs * log_probs).sum(dim=1).mean()
                
                # Compute gradient with respect to lambdas_raw
                if lambdas_raw.grad is not None:
                    lambdas_raw.grad.zero_()
                entropy_loss.backward()
                
                grad = lambdas_raw.grad.data.clone().detach()
                
                # Apply optimizer update
                if method == 'Uniform':
                    # Simple SGD with fixed learning rate
                    lambdas_raw.data -= lr_base * grad
                    
                elif method == 'LFWA':
                    # Scale layer-wise learning rates by the inverse of the static Fisher sensitivity
                    scaled_grad = torch.zeros_like(grad)
                    for i, name in enumerate(self.parameter_names):
                        f_val = self.layer_fisher[name]
                        # Clamp Fisher to avoid division by extremely small values
                        scale_factor = 1.0 / (f_val + 1e-5)
                        scaled_grad[i] = scale_factor * grad[i]
                    lambdas_raw.data -= lr_base * scaled_grad
                    
                elif method == 'AdaSNR':
                    # Compute online gradient SNR for dynamic scaling
                    # 1-based step index for bias correction
                    t_step = step + 1
                    m = beta1 * m + (1.0 - beta1) * grad
                    v = beta2 * v + (1.0 - beta2) * grad.pow(2)
                    
                    m_hat = m / (1.0 - beta1 ** t_step)
                    v_hat = v / (1.0 - beta2 ** t_step)
                    
                    # Compute SNR for each merging coefficient
                    snr = m_hat.pow(2) / (v_hat + eps)
                    
                    # Smooth, bounded scaling: scale = alpha0 + (1 - alpha0) * (1 - exp(-snr))
                    # alpha0 = 0.1 prevents learning from completely freezing on noisy updates
                    alpha0 = 0.1
                    scale = alpha0 + (1.0 - alpha0) * (1.0 - torch.exp(-snr))
                    
                    # Dynamic gradient update
                    lambdas_raw.data -= lr_base * scale * grad
                    
                # Project/clamp lambdas_raw back to [0, 1] to keep parameters in a stable interpolative range
                lambdas_raw.data.clamp_(0.0, 1.0)
                
        avg_acc = np.mean(pre_accuracies)
        cifar_acc = np.mean(task_accuracies[0])
        svhn_acc = np.mean(task_accuracies[1])
        
        return avg_acc, cifar_acc, svhn_acc


def main():
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running evaluation on device: {device}")
    
    # Disable cuDNN to prevent initialization errors
    torch.backends.cudnn.enabled = False
    
    # Generate non-stationary streams
    alt_stream, seq_stream = get_test_streams(batch_size=64)
    
    # Initialize merger
    merger = TestTimeModelMerger(device)
    
    # Pre-compute Fisher Information for LFWA
    merger.compute_fisher_information(num_samples=256)
    
    # Hyperparameters
    # We will search for a good base learning rate for Uniform, LFWA and AdaSNR
    # Let's use lr=0.1 for Uniform TTA, lr=1e-5 for LFWA (as Fisher values are large, so lr needs to be scaled accordingly), and lr=0.5 for AdaSNR
    results_alt = {}
    results_seq = {}
    
    # Method configurations (Name: (method_key, lr_base))
    configs = {
        'Static (Task Arithmetic)': ('Static', 0.0),
        'Uniform TTA (SGD)': ('Uniform', 0.1),
        'LFWA TTA (SGD)': ('LFWA', 5e-7), # Scaled because average parameter-level Fisher can be large (around 10^5 to 10^6)
        'AdaSNR TTA (Ours)': ('AdaSNR', 0.5)
    }
    
    print("\n" + "="*80)
    print("EVALUATING ALTERNATING STREAM (High-frequency task switches)")
    print("="*80)
    for name, (method, lr) in configs.items():
        avg, cifar, svhn = merger.evaluate_stream(alt_stream, method, lr_base=lr)
        results_alt[name] = (avg, cifar, svhn)
        print(f"Method: {name:<26} | Average Acc: {avg:.2f}% | CIFAR-10 Acc: {cifar:.2f}% | SVHN Acc: {svhn:.2f}%")
        
    print("\n" + "="*80)
    print("EVALUATING BLOCK-SEQUENTIAL STREAM (Low-frequency task blocks)")
    print("="*80)
    for name, (method, lr) in configs.items():
        avg, cifar, svhn = merger.evaluate_stream(seq_stream, method, lr_base=lr)
        results_seq[name] = (avg, cifar, svhn)
        print(f"Method: {name:<26} | Average Acc: {avg:.2f}% | CIFAR-10 Acc: {cifar:.2f}% | SVHN Acc: {svhn:.2f}%")
        
    # Let's save the results to progress.md
    print("\nSaving results to progress.md...")
    with open("results_summary.txt", "w") as f:
        f.write("Evaluation Results Summary:\n\n")
        f.write("Alternating Stream:\n")
        for name, (avg, cifar, svhn) in results_alt.items():
            f.write(f"- {name}: Avg Acc={avg:.2f}%, CIFAR={cifar:.2f}%, SVHN={svhn:.2f}%\n")
        f.write("\nBlock-Sequential Stream:\n")
        for name, (avg, cifar, svhn) in results_seq.items():
            f.write(f"- {name}: Avg Acc={avg:.2f}%, CIFAR={cifar:.2f}%, SVHN={svhn:.2f}%\n")
            
    print("Done!")

if __name__ == "__main__":
    main()
