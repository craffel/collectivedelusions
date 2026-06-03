import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from src.dataset import get_dataset
from src.models import get_model
from src.merge import get_task_vectors, get_merged_task_vector, calibrate_model, average_batchnorm_stats, weight_averaging

# 1. Symmetric Uniform Quantization helpers
def quantize_tensor_per_tensor(tensor, num_bits=8):
    if num_bits is None:
        return tensor
    qmax = 2**(num_bits - 1) - 1
    max_val = torch.max(torch.abs(tensor))
    if max_val == 0:
        return tensor
    delta = max_val / qmax
    quantized = torch.clamp(torch.round(tensor / delta), -qmax, qmax) * delta
    return quantized

def quantize_tensor_per_channel(tensor, num_bits=8):
    if num_bits is None or tensor.ndim < 2:
        return quantize_tensor_per_tensor(tensor, num_bits)
    
    qmax = 2**(num_bits - 1) - 1
    C_out = tensor.shape[0]
    quantized = torch.zeros_like(tensor)
    for c in range(C_out):
        max_val_c = torch.max(torch.abs(tensor[c]))
        if max_val_c == 0:
            quantized[c] = tensor[c]
            continue
        delta_c = max_val_c / qmax
        quantized[c] = torch.clamp(torch.round(tensor[c] / delta_c), -qmax, qmax) * delta_c
    return quantized

def apply_quantization(model, num_bits=8, mode="per_tensor"):
    if num_bits is None:
        return
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if mode == "per_tensor":
                module.weight.data = quantize_tensor_per_tensor(module.weight.data, num_bits)
            elif mode == "per_channel":
                module.weight.data = quantize_tensor_per_channel(module.weight.data, num_bits)

# 2. Environmental corruptions
def add_gaussian_noise(x, std=0.1):
    noise = torch.randn_like(x) * std
    return x + noise

def apply_gaussian_blur(x):
    # torchvision gaussian_blur expects [..., C, H, W]
    # kernel size 3, standard deviation default or 1.0
    return TF.gaussian_blur(x, kernel_size=[3, 3], sigma=[1.0, 1.0])

def evaluate_merged_model(model, loader, device, corruption=None, num_bits=None, quant_mode="per_tensor"):
    # Apply quantization to a copy of the model to avoid corrupting original
    import copy
    eval_model = copy.deepcopy(model)
    eval_model.to(device)
    
    if num_bits is not None:
        apply_quantization(eval_model, num_bits=num_bits, mode=quant_mode)
        
    eval_model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Apply input corruption if specified
            if corruption == "noise":
                inputs = add_gaussian_noise(inputs, std=0.1)
            elif corruption == "blur":
                inputs = apply_gaussian_blur(inputs)
                
            outputs = eval_model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return correct / total

def main():
    parser = argparse.ArgumentParser(description="Evaluate Model Merging")
    parser.add_argument("--arch", type=str, required=True, choices=["resnet18", "mlp"])
    parser.add_argument("--merge_method", type=str, default="ta", choices=["ta", "ties", "dare", "wa"])
    parser.add_argument("--calib_method", type=str, default="none", choices=["none", "u_ipr", "hns", "qr_ipr", "wcpr", "qr_sc_wcpr"])
    parser.add_argument("--scale", type=float, default=1.0, help="Global scaling parameter lambda for Task Arithmetic")
    parser.add_argument("--reset_thresh", type=float, default=20, help="TIES reset threshold")
    parser.add_argument("--drop_rate", type=float, default=0.2, help="DARE drop rate")
    parser.add_argument("--gamma", type=float, default=2.0, help="Outlier rejection threshold for QR methods")
    parser.add_argument("--compensation", type=str, default="inverse", choices=["inverse", "sqrt", "none"], help="Sparsity compensation type")
    parser.add_argument("--num_bits", type=int, default=None, help="Quantization bits (e.g. 8 for INT8)")
    parser.add_argument("--quant_mode", type=str, default="per_tensor", choices=["per_tensor", "per_channel"])
    parser.add_argument("--corruption", type=str, default=None, choices=["noise", "blur"])
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Disable cuDNN due to driver compatibility issues on this cluster
    torch.backends.cudnn.enabled = False

    device = torch.device(args.device)
    tasks = ["mnist", "fmnist", "cifar10"]
    
    # 1. Load checkpoints
    progenitor_path = os.path.join(args.checkpoints_dir, f"{args.arch}_progenitor.pt")
    if not os.path.exists(progenitor_path):
        raise FileNotFoundError(f"Progenitor model not found at {progenitor_path}. Run train.py first.")
        
    progenitor_state = torch.load(progenitor_path, map_location="cpu")
    
    expert_states = []
    for task in tasks:
        expert_path = os.path.join(args.checkpoints_dir, f"{args.arch}_{task}_expert.pt")
        expert_states.append(torch.load(expert_path, map_location="cpu"))
        
    # 2. Compute task vectors and perform merging
    if args.merge_method == "wa":
        # Weight Averaging directly averages expert state dicts
        merged_state = weight_averaging(expert_states, progenitor_state)
    else:
        # Task vectors-based merging
        task_vectors = get_task_vectors(expert_states, progenitor_state)
        merged_tv = get_merged_task_vector(
            args.merge_method, task_vectors, progenitor_state,
            reset_thresh=args.reset_thresh, drop_rate=args.drop_rate
        )
        # Apply calibration
        merged_state = calibrate_model(
            merged_tv, task_vectors, progenitor_state, args.calib_method,
            scale=args.scale, gamma=args.gamma, compensation=args.compensation
        )
        
    # Standard BatchNorm averaging (if ResNet-18)
    if args.arch == "resnet18":
        merged_state = average_batchnorm_stats(expert_states, merged_state)
        
    # 3. Create merged model and load calibrated weights
    merged_model = get_model(args.arch, num_classes=10, pretrained=False)
    merged_model.load_state_dict(merged_state)
    
    # 4. Evaluate across all tasks
    accuracies = {}
    for task in tasks:
        # Load task-specific classification head from the corresponding expert
        # Find the correct expert index
        task_idx = tasks.index(task)
        expert_state = expert_states[task_idx]
        
        # Attach the expert's fc classifier head to our merged model
        fc_state = {k[3:]: v for k, v in expert_state.items() if k.startswith("fc.")}
        merged_model.fc.load_state_dict(fc_state)
        
        test_loader = get_dataset(task, train=False, batch_size=256)
        acc = evaluate_merged_model(
            merged_model, test_loader, device,
            corruption=args.corruption, num_bits=args.num_bits, quant_mode=args.quant_mode
        )
        accuracies[task] = acc
        print(f"Task: {task.upper()} | Accuracy: {acc*100:.2f}%")
        
    avg_acc = sum(accuracies.values()) / len(accuracies)
    print(f"\nAverage Multi-Task Accuracy: {avg_acc*100:.2f}%")
    return accuracies, avg_acc

if __name__ == "__main__":
    main()
