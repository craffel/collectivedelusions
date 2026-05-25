import os
import copy
import argparse
import random
import torch
import torch.nn as nn
from torch.func import functional_call
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

# Disable cuDNN to avoid initialization errors on cluster GPUs
torch.backends.cudnn.enabled = False

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluation device: {device}")

# Define directories
SAVE_DIR = "/fsx/craffel/collectivedelusions/ml_research/merging_conference/trial4/submission3"

# Standard transforms (normalized to match pre-trained ImageNet ResNet-18)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Task-aware translation augmentation for consistency regularization
def translate_augmentation(x, shift_range=2):
    """
    Randomly translates images by a few pixels using zero padding.
    """
    batch_size, channels, height, width = x.shape
    dx = random.randint(-shift_range, shift_range)
    dy = random.randint(-shift_range, shift_range)
    
    # Pad and crop to translate
    padded = nn.functional.pad(x, (shift_range, shift_range, shift_range, shift_range), mode='constant', value=0.0)
    start_y = shift_range + dy
    start_x = shift_range + dx
    return padded[:, :, start_y:start_y+height, start_x:start_x+width]

def get_test_datasets():
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    fashion_test = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    kmnist_test = datasets.KMNIST(root="./data", train=False, download=True, transform=transform)
    return mnist_test, fashion_test, kmnist_test

class ExpertModel(nn.Module):
    def __init__(self, dropout_prob=0.1):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.head = nn.Linear(512, 10)
        
    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        logits = self.head(features)
        return logits

def load_experts():
    experts = []
    task_names = ["mnist", "fashion", "kmnist"]
    for name in task_names:
        ckpt_path = os.path.join(SAVE_DIR, f"{name}_expert.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Expert checkpoint not found at {ckpt_path}. Please run train_experts.py first!")
        
        print(f"Loading {name.upper()} expert from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model = ExpertModel().to(device)
        model.backbone.load_state_dict(checkpoint['backbone_state_dict'])
        model.head.load_state_dict(checkpoint['head_state_dict'])
        model.eval()
        experts.append(model)
    return experts

def construct_test_streams(mnist_test, fashion_test, kmnist_test, batch_size=32, num_batches_per_task=50):
    # Total samples needed per task = 50 * 32 = 1600
    num_samples = num_batches_per_task * batch_size
    
    # Subsets to ensure alignment and reproducibility
    mnist_subset = Subset(mnist_test, list(range(num_samples)))
    fashion_subset = Subset(fashion_test, list(range(num_samples)))
    kmnist_subset = Subset(kmnist_test, list(range(num_samples)))
    
    # Create dataloaders
    mnist_loader = DataLoader(mnist_subset, batch_size=batch_size, shuffle=False)
    fashion_loader = DataLoader(fashion_subset, batch_size=batch_size, shuffle=False)
    kmnist_loader = DataLoader(kmnist_subset, batch_size=batch_size, shuffle=False)
    
    # Extract all batches
    mnist_batches = list(mnist_loader)
    fashion_batches = list(fashion_loader)
    kmnist_batches = list(kmnist_loader)
    
    # 1. Sequential Stream (50 MNIST, then 50 Fashion, then 50 KMNIST)
    sequential_stream = []
    for i, batch in enumerate(mnist_batches):
        sequential_stream.append((batch[0], batch[1], 0)) # images, labels, task_idx
    for i, batch in enumerate(fashion_batches):
        sequential_stream.append((batch[0], batch[1], 1))
    for i, batch in enumerate(kmnist_batches):
        sequential_stream.append((batch[0], batch[1], 2))
        
    # 2. Alternating Stream (alternate MNIST, Fashion, KMNIST)
    alternating_stream = []
    for step in range(num_batches_per_task):
        alternating_stream.append((mnist_batches[step][0], mnist_batches[step][1], 0))
        alternating_stream.append((fashion_batches[step][0], fashion_batches[step][1], 1))
        alternating_stream.append((kmnist_batches[step][0], kmnist_batches[step][1], 2))
        
    return sequential_stream, alternating_stream

def get_merged_params(lambda_coeff, base_state, task_vectors, parameter_names):
    merged_params = {}
    active_idx = torch.argmax(lambda_coeff).item() if lambda_coeff.requires_grad else 0
    for k in base_state.keys():
        if base_state[k].is_floating_point():
            tensor = base_state[k] + \
                     lambda_coeff[0] * task_vectors[0][k] + \
                     lambda_coeff[1] * task_vectors[1][k] + \
                     lambda_coeff[2] * task_vectors[2][k]
            if k not in parameter_names:
                tensor = tensor.detach()
            merged_params[k] = tensor
        else:
            merged_params[k] = task_vectors[active_idx][k]
    return merged_params

def run_evaluation(method, stream, experts, base_backbone, lr_lambda=0.5, lr_head=1e-4, gamma_reg=100.0, num_mc_passes=5):
    """
    Evaluates the model on a test stream using the specified adaptation method.
    """
    # Initialize merging coefficients lambda to uniform: [1/3, 1/3, 1/3]
    lambda_coeff = torch.tensor([1/3, 1/3, 1/3], device=device, requires_grad=True)
    
    # Copy expert heads so we don't modify the originals
    heads = [copy.deepcopy(expert.head).to(device) for expert in experts]
    initial_heads = [copy.deepcopy(expert.head).to(device) for expert in experts]
    
    # Extract parameter and buffer dictionaries for backbone via state_dict()
    base_state = {k: v for k, v in base_backbone.state_dict().items()}
    
    # Extract parameter names to identify buffers
    parameter_names = set(dict(base_backbone.named_parameters()).keys())
    
    # Extract expert backbone parameters and construct task-specific update vectors
    expert_backbones = [expert.backbone for expert in experts]
    task_vectors = []
    for exp_bb in expert_backbones:
        exp_state = {k: v for k, v in exp_bb.state_dict().items()}
        vec = {}
        for k, v in exp_state.items():
            if v.is_floating_point():
                vec[k] = v - base_state[k]
            else:
                vec[k] = v
        task_vectors.append(vec)
        
    # We will compute the Fisher Information Matrices for EWC-TTA if needed
    # (Using online/estimated Fisher since we don't have training splits)
    online_fims = [None] * len(experts)
    
    correct_predictions = 0
    total_samples = 0
    
    # Results per batch
    batch_accuracies = []
    lambda_history = []
    
    for step, (images, labels, task_idx) in enumerate(stream):
        images, labels = images.to(device), labels.to(device)
        active_head = heads[task_idx]
        initial_head = initial_heads[task_idx]
        
        # 1. Evaluate current state BEFORE adaptation
        # Construct merged parameters using our helper
        merged_params = get_merged_params(lambda_coeff, base_state, task_vectors, parameter_names)
                                  
        with torch.no_grad():
            features = functional_call(base_backbone, merged_params, images)
            # Evaluate using active head without dropout
            logits = active_head(features)
            _, predicted = logits.max(1)
            correct = predicted.eq(labels).sum().item()
            correct_predictions += correct
            total_samples += labels.size(0)
            batch_acc = 100.0 * correct / labels.size(0)
            batch_accuracies.append(batch_acc)
            lambda_history.append(lambda_coeff.detach().cpu().numpy().tolist())
            
        # 2. Adaptation Step
        if method == "static":
            # No adaptation
            continue
            
        # Setup optimizer for adaptation
        # We optimize both the active head and lambda
        params_to_opt = []
        if method != "s2c_merge": # S2C keeps classification heads frozen
            params_to_opt.append({'params': active_head.parameters(), 'lr': lr_head})
        params_to_opt.append({'params': [lambda_coeff], 'lr': lr_lambda})
        
        optimizer = torch.optim.SGD(params_to_opt)
        optimizer.zero_grad()
        
        # Construct merged params for forward pass in training/adaptation mode
        merged_params = get_merged_params(lambda_coeff, base_state, task_vectors, parameter_names)
                                  
        if method == "standard_tta":
            # Self-supervised prediction entropy minimization
            features = functional_call(base_backbone, merged_params, images)
            logits = active_head(features)
            probs = torch.softmax(logits, dim=-1)
            loss = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).mean()
            loss.backward()
            optimizer.step()
            
        elif method == "s2c_merge":
            # Frozen heads, prediction entropy + translation consistency
            # 1. Original features and logits
            features = functional_call(base_backbone, merged_params, images)
            logits = active_head(features)
            probs = torch.softmax(logits, dim=-1)
            entropy_loss = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).mean()
            
            # 2. Augmented features and logits
            images_aug = translate_augmentation(images)
            features_aug = functional_call(base_backbone, merged_params, images_aug)
            logits_aug = active_head(features_aug)
            probs_aug = torch.softmax(logits_aug, dim=-1)
            
            # KL divergence consistency loss (probs as stable target)
            kl_loss = torch.sum(probs_aug * (torch.log(probs_aug + 1e-12) - torch.log(probs.detach() + 1e-12)), dim=-1).mean()
            
            loss = entropy_loss + kl_loss
            loss.backward()
            optimizer.step()
            
        elif method == "ewc_tta":
            # Optimizes both heads and lambda, applies diagonal Fisher quadratic constraint
            # 1. Self-supervised prediction entropy
            features = functional_call(base_backbone, merged_params, images)
            logits = active_head(features)
            probs = torch.softmax(logits, dim=-1)
            entropy_loss = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).mean()
            
            # 2. EWC Penalty
            # Initialize or update online Fisher Information Matrix
            if online_fims[task_idx] is None:
                # Estimate a diagonal FIM on the first batch of the task
                fim = {}
                for p_name, p in active_head.named_parameters():
                    fim[p_name] = torch.zeros_like(p)
                # Compute gradients of log-likelihood for estimating Fisher
                log_probs = torch.log_softmax(logits, dim=-1)
                for class_idx in range(10):
                    grad_sum = torch.zeros_like(logits)
                    grad_sum[:, class_idx] = 1.0
                    active_head.zero_grad()
                    logits.backward(gradient=grad_sum, retain_graph=True)
                    for p_name, p in active_head.named_parameters():
                        if p.grad is not None:
                            fim[p_name] += (p.grad ** 2) / 10.0
                online_fims[task_idx] = {k: v + 1e-5 for k, v in fim.items()}
                
            # Compute quadratic EWC penalty
            ewc_penalty = 0.0
            for p_name, p in active_head.named_parameters():
                init_p = dict(initial_head.named_parameters())[p_name]
                fim_p = online_fims[task_idx][p_name]
                ewc_penalty += torch.sum(fim_p * (p - init_p) ** 2)
                
            loss = entropy_loss + gamma_reg * 0.5 * ewc_penalty
            loss.backward()
            optimizer.step()
            
        elif method == "mc_vti":
            # OUR METHOD: Monte Carlo Variational Task-Information Merging
            # 1. Perform multiple MC dropout passes
            logits_list = []
            for _ in range(num_mc_passes):
                # Construct merged params with dropout
                features_mc = functional_call(base_backbone, merged_params, images)
                # Keep test-time dropout active:
                features_mc = nn.functional.dropout(features_mc, p=0.1, training=True)
                logits_mc = active_head(features_mc)
                logits_list.append(logits_mc)
                
            logits_stack = torch.stack(logits_list, dim=0) # (M, B, C)
            probs_stack = torch.softmax(logits_stack, dim=-1)
            avg_probs = probs_stack.mean(dim=0) # (B, C)
            
            # MC expected entropy loss
            entropy_loss = -torch.sum(avg_probs * torch.log(avg_probs + 1e-12), dim=-1).mean()
            
            # MC expected consistency loss on augmented view
            images_aug = translate_augmentation(images)
            logits_list_aug = []
            for _ in range(num_mc_passes):
                features_mc_aug = functional_call(base_backbone, merged_params, images_aug)
                features_mc_aug = nn.functional.dropout(features_mc_aug, p=0.1, training=True)
                logits_mc_aug = active_head(features_mc_aug)
                logits_list_aug.append(logits_mc_aug)
            logits_stack_aug = torch.stack(logits_list_aug, dim=0)
            probs_stack_aug = torch.softmax(logits_stack_aug, dim=-1)
            avg_probs_aug = probs_stack_aug.mean(dim=0)
            
            # KL divergence between original and augmented average probabilities
            kl_loss = torch.sum(avg_probs_aug * (torch.log(avg_probs_aug + 1e-12) - torch.log(avg_probs.detach() + 1e-12)), dim=-1).mean()
            
            loss_ss = entropy_loss + kl_loss
            
            # 2. Bayesian-Guided logit variance regularization on heads
            # Compute logit variance across MC passes for each class
            logit_vars = logits_stack.var(dim=0).mean(dim=0) # Shape: (10,)
            
            # Penalty scales quadratic drift inversely with model confidence (directly with variance)
            reg_penalty = 0.0
            for c in range(10):
                weight_diff = active_head.weight[c] - initial_head.weight[c]
                bias_diff = active_head.bias[c] - initial_head.bias[c]
                # penalize adaptation of unstable classes proportionally to logit_vars[c]
                reg_penalty += logit_vars[c].detach() * (torch.sum(weight_diff ** 2) + bias_diff ** 2)
                
            loss = loss_ss + gamma_reg * reg_penalty
            loss.backward()
            optimizer.step()
            
        # Project lambda back to the simplex (lambda >= 0 and sum(lambda) = 1)
        with torch.no_grad():
            lambda_coeff.clamp_(min=0.0)
            sum_lambda = lambda_coeff.sum()
            if sum_lambda > 0:
                lambda_coeff.div_(sum_lambda)
                
    overall_acc = 100.0 * correct_predictions / total_samples
    return overall_acc, batch_accuracies, lambda_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr_lambda", type=float, default=0.5)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--gamma_reg", type=float, default=100.0)
    parser.add_argument("--num_mc_passes", type=int, default=5)
    parser.add_argument("--save_suffix", type=str, default="")
    args = parser.parse_args()
    
    # Load test datasets
    mnist_test, fashion_test, kmnist_test = get_test_datasets()
    
    # Construct streams
    seq_stream, alt_stream = construct_test_streams(mnist_test, fashion_test, kmnist_test)
    
    # Load experts and base backbone
    experts = load_experts()
    base_backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    base_backbone.fc = nn.Identity()
    base_backbone.to(device).eval()
    
    # Evaluate methods
    methods = ["static", "standard_tta", "s2c_merge", "ewc_tta", "mc_vti"]
    streams = {"sequential": seq_stream, "alternating": alt_stream}
    
    results = {}
    for stream_name, stream in streams.items():
        print(f"\n==========================================")
        print(f"Evaluating stream: {stream_name.upper()}")
        print(f"==========================================")
        results[stream_name] = {}
        for method in methods:
            acc, batch_accs, lambdas = run_evaluation(
                method, stream, experts, base_backbone,
                lr_lambda=args.lr_lambda, lr_head=args.lr_head,
                gamma_reg=args.gamma_reg, num_mc_passes=args.num_mc_passes
            )
            print(f"[{method.upper()}] Overall Accuracy: {acc:.2f}%")
            results[stream_name][method] = {
                "overall_accuracy": acc,
                "batch_accuracies": batch_accs,
                "lambda_history": lambdas
            }
            
    # Save evaluation results
    filename = "evaluation_results.pt" if not args.save_suffix else f"evaluation_results_{args.save_suffix}.pt"
    results_path = os.path.join(SAVE_DIR, filename)
    torch.save(results, results_path)
    print(f"\nSaved evaluation results to {results_path}")
