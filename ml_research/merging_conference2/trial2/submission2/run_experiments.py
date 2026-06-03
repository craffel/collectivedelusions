import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
import os
import copy
import numpy as np

from merging_baselines import (
    merge_weight_averaging,
    merge_task_arithmetic,
    merge_ties,
    merge_dare,
    merge_multi_weight_averaging,
    merge_multi_task_arithmetic,
    merge_multi_ties,
    merge_multi_dare
)
from cpos_merging import CPOSResNet, QCPOSResNet, HCPOSResNet, GeneralizedCPOSResNet, ChannelWiseCPOSResNet, DPRCPOSResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
MAX_SAMPLES = 1000 if device.type == "cpu" else None
if MAX_SAMPLES is not None:
    print(f"CPU detected: limiting evaluation to {MAX_SAMPLES} samples for fast execution.")
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False  # Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED on Slurm nodes

# Transforms for 3-channel 32x32 inputs
transform_cifar_svhn = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_fmnist = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_test_loader(dataset_name, batch_size=128, max_samples=None):
    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_cifar_svhn)
    elif dataset_name == "svhn":
        dataset = datasets.SVHN(root="./data", split="test", download=True, transform=transform_cifar_svhn)
    elif dataset_name == "fmnist":
        dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform_fmnist)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    if max_samples is not None:
        dataset = torch.utils.data.Subset(dataset, list(range(min(max_samples, len(dataset)))))
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

def evaluate_model(model, dataloader, noise_sigma=0.0):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            if noise_sigma > 0.0:
                inputs = inputs + torch.randn_like(inputs) * noise_sigma
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total

def load_expert(dataset_name):
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, 10)
    checkpoint_path = f"checkpoints/expert_{dataset_name}.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Expert checkpoint not found at {checkpoint_path}. Please train experts first.")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    return model

def build_merged_real_model(merged_state_dict, original_fc_state, device):
    """
    Builds a real-valued model using merged backbone weights and specific task's FC weights.
    """
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, 10)
    
    # Load merged weights
    model.load_state_dict(merged_state_dict, strict=False)
    
    # Overwrite FC layer with the task-specific FC weights
    fc_dict = {f"fc.{k}": v for k, v in original_fc_state.items()}
    model.load_state_dict(fc_dict, strict=False)
    
    model = model.to(device)
    return model

def measure_activation_variances(model_A, model_B, merged_state_dict, cpos_model, dataloader):
    """
    Measures and compares activation variances at the end of each residual block.
    """
    # Build Weight Averaging model
    fc_A_state = {k.replace("fc.", ""): v for k, v in model_A.state_dict().items() if k.startswith("fc.")}
    wa_model = build_merged_real_model(merged_state_dict, fc_A_state, device)
    
    wa_model.eval()
    cpos_model.eval()
    model_A.eval()
    
    # Fetch one batch of inputs
    inputs, _ = next(iter(dataloader))
    inputs = inputs.to(device)
    
    variances_wa = []
    variances_cpos = []
    variances_expert = []
    
    # Hook activations at layer outputs
    # For simplicity, we can do forward passes and extract outputs manually or use hooks.
    # Let's write a clean manual forward pass extraction for ResNet-18
    def get_layer_activations(model, x, is_cpos=False):
        activations = []
        if is_cpos:
            # Stem
            y_A = model.model_A.relu(model.model_A.bn1(model.model_A.conv1(x)))
            y_B = model.model_B.relu(model.model_B.bn1(model.model_B.conv1(x)))
            x = torch.sqrt(model.alpha**2 * y_A**2 + model.beta**2 * y_B**2 + 1e-8)
            activations.append(x.var().item())
            x = model.model_A.maxpool(x)
            
            # Layers
            for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
                layer_A = getattr(model.model_A, layer_name)
                layer_B = getattr(model.model_B, layer_name)
                for i in range(len(layer_A)):
                    out_A = layer_A[i](x)
                    out_B = layer_B[i](x)
                    x = torch.sqrt(model.alpha**2 * out_A**2 + model.beta**2 * out_B**2 + 1e-8)
                    activations.append(x.var().item())
        else:
            # Real model
            x = model.relu(model.bn1(model.conv1(x)))
            activations.append(x.var().item())
            x = model.maxpool(x)
            
            for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
                layer = getattr(model, layer_name)
                for i in range(len(layer)):
                    x = layer[i](x)
                    activations.append(x.var().item())
                    
        return activations

    with torch.no_grad():
        vars_expert = get_layer_activations(model_A, inputs, is_cpos=False)
        vars_wa = get_layer_activations(wa_model, inputs, is_cpos=False)
        vars_cpos = get_layer_activations(cpos_model, inputs, is_cpos=True)
        
    return vars_expert, vars_wa, vars_cpos

def run_experiment_pair(task_A, task_B):
    print(f"\n=======================================================")
    print(f"RUNNING EXPERIMENTS FOR PAIR: {task_A.upper()} + {task_B.upper()}")
    print(f"=======================================================")
    
    # 1. Load experts
    model_A = load_expert(task_A)
    model_B = load_expert(task_B)
    
    # Load base pre-trained model (ImageNet ResNet18)
    base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    base_model.fc = nn.Linear(512, 10) # Adjust fc dimension
    base_state = base_model.state_dict()
    
    # Extract FC states for task evaluation
    fc_A_state = {k.replace("fc.", ""): v for k, v in model_A.state_dict().items() if k.startswith("fc.")}
    fc_B_state = {k.replace("fc.", ""): v for k, v in model_B.state_dict().items() if k.startswith("fc.")}
    
    # 2. Load dataloaders
    loader_A = get_test_loader(task_A, max_samples=MAX_SAMPLES)
    loader_B = get_test_loader(task_B, max_samples=MAX_SAMPLES)
    
    # 3. Evaluate individual experts (Oracle)
    acc_A_expert = evaluate_model(model_A, loader_A)
    acc_B_expert = evaluate_model(model_B, loader_B)
    print(f"Oracle Expert {task_A.upper()}: {acc_A_expert:.2f}%")
    print(f"Oracle Expert {task_B.upper()}: {acc_B_expert:.2f}%")
    
    # 4. Evaluate baselines
    results = {}
    
    # --- Weight Averaging ---
    merged_wa = merge_weight_averaging(model_A.state_dict(), model_B.state_dict(), alpha=0.5)
    model_wa_A = build_merged_real_model(merged_wa, fc_A_state, device)
    model_wa_B = build_merged_real_model(merged_wa, fc_B_state, device)
    acc_wa_A = evaluate_model(model_wa_A, loader_A)
    acc_wa_B = evaluate_model(model_wa_B, loader_B)
    results["Weight Averaging"] = (acc_wa_A, acc_wa_B)
    
    # --- Task Arithmetic ---
    merged_ta = merge_task_arithmetic(base_state, model_A.state_dict(), model_B.state_dict(), lambda_val=0.5)
    model_ta_A = build_merged_real_model(merged_ta, fc_A_state, device)
    model_ta_B = build_merged_real_model(merged_ta, fc_B_state, device)
    acc_ta_A = evaluate_model(model_ta_A, loader_A)
    acc_ta_B = evaluate_model(model_ta_B, loader_B)
    results["Task Arithmetic"] = (acc_ta_A, acc_ta_B)
    
    # --- TIES-Merging ---
    merged_ties = merge_ties(base_state, model_A.state_dict(), model_B.state_dict(), trim_percent=0.20, lambda_val=0.5)
    model_ties_A = build_merged_real_model(merged_ties, fc_A_state, device)
    model_ties_B = build_merged_real_model(merged_ties, fc_B_state, device)
    acc_ties_A = evaluate_model(model_ties_A, loader_A)
    acc_ties_B = evaluate_model(model_ties_B, loader_B)
    results["TIES-Merging"] = (acc_ties_A, acc_ties_B)
    
    # --- DARE-Merging ---
    merged_dare = merge_dare(base_state, model_A.state_dict(), model_B.state_dict(), drop_prob=0.5, lambda_val=0.5)
    model_dare_A = build_merged_real_model(merged_dare, fc_A_state, device)
    model_dare_B = build_merged_real_model(merged_dare, fc_B_state, device)
    acc_dare_A = evaluate_model(model_dare_A, loader_A)
    acc_dare_B = evaluate_model(model_dare_B, loader_B)
    results["DARE-Merging"] = (acc_dare_A, acc_dare_B)
    
    # --- CPOS (Ours) ---
    alpha_val = 1.0 / np.sqrt(2)
    beta_val = 1.0 / np.sqrt(2)
    cpos_model = CPOSResNet(model_A, model_B, alpha=alpha_val, beta=beta_val).to(device)
    
    cpos_model.set_task(0) # Task A
    acc_cpos_A = evaluate_model(cpos_model, loader_A)
    
    cpos_model.set_task(1) # Task B
    acc_cpos_B = evaluate_model(cpos_model, loader_B)
    results["CPOS (Ours)"] = (acc_cpos_A, acc_cpos_B)
    
    # 5. Output results
    print("\nResults Summary:")
    print(f"| Method | {task_A.upper()} Acc (%) | {task_B.upper()} Acc (%) | Average Acc (%) |")
    print(f"|---|---|---|---|")
    print(f"| Oracle Experts | {acc_A_expert:.2f} | {acc_B_expert:.2f} | {(acc_A_expert+acc_B_expert)/2.0:.2f} |")
    for method, (acc_A, acc_B) in results.items():
        print(f"| {method} | {acc_A:.2f} | {acc_B:.2f} | {(acc_A+acc_B)/2.0:.2f} |")
        
    # 6. Run Activation Variance Analysis
    vars_expert, vars_wa, vars_cpos = measure_activation_variances(model_A, model_B, merged_wa, cpos_model, loader_A)
    print("\nActivation Variance Comparison across layers (Stem, Block 1-8):")
    print("| Layer | Expert A Var | WA Var | CPOS Var | CPOS/WA Ratio |")
    print("|---|---|---|---|---|")
    for i, (v_exp, v_wa, v_cpos) in enumerate(zip(vars_expert, vars_wa, vars_cpos)):
        ratio = v_cpos / (v_wa + 1e-8)
        print(f"| Layer {i+1} | {v_exp:.4f} | {v_wa:.4f} | {v_cpos:.4f} | {ratio:.2f}x |")
        
    return results, (vars_expert, vars_wa, vars_cpos)


def run_three_task_experiment():
    print("\n=======================================================")
    print("RUNNING THREE-TASK JOINT EXPERIMENT: CIFAR10 + SVHN + FMNIST")
    print("=======================================================")
    
    # 1. Load experts
    model_A = load_expert("cifar10")
    model_B = load_expert("svhn")
    model_C = load_expert("fmnist")
    
    # Load base model
    base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    base_model.fc = nn.Linear(512, 10)
    base_state = base_model.state_dict()
    
    # Extract FC states
    fc_A_state = {k.replace("fc.", ""): v for k, v in model_A.state_dict().items() if k.startswith("fc.")}
    fc_B_state = {k.replace("fc.", ""): v for k, v in model_B.state_dict().items() if k.startswith("fc.")}
    fc_C_state = {k.replace("fc.", ""): v for k, v in model_C.state_dict().items() if k.startswith("fc.")}
    
    # 2. Dataloaders
    loader_A = get_test_loader("cifar10", max_samples=MAX_SAMPLES)
    loader_B = get_test_loader("svhn", max_samples=MAX_SAMPLES)
    loader_C = get_test_loader("fmnist", max_samples=MAX_SAMPLES)
    
    # 3. Individual Experts (Oracle)
    acc_A_expert = evaluate_model(model_A, loader_A)
    acc_B_expert = evaluate_model(model_B, loader_B)
    acc_C_expert = evaluate_model(model_C, loader_C)
    print(f"Oracle Expert CIFAR-10: {acc_A_expert:.2f}%")
    print(f"Oracle Expert SVHN: {acc_B_expert:.2f}%")
    print(f"Oracle Expert FMNIST: {acc_C_expert:.2f}%")
    
    # 4. Evaluate multi-task baselines
    results = {}
    
    # --- Multi-Task Weight Averaging ---
    merged_wa = merge_multi_weight_averaging([model_A.state_dict(), model_B.state_dict(), model_C.state_dict()])
    model_wa_A = build_merged_real_model(merged_wa, fc_A_state, device)
    model_wa_B = build_merged_real_model(merged_wa, fc_B_state, device)
    model_wa_C = build_merged_real_model(merged_wa, fc_C_state, device)
    acc_wa_A = evaluate_model(model_wa_A, loader_A)
    acc_wa_B = evaluate_model(model_wa_B, loader_B)
    acc_wa_C = evaluate_model(model_wa_C, loader_C)
    results["Weight Averaging"] = (acc_wa_A, acc_wa_B, acc_wa_C)
    
    # --- Multi-Task Task Arithmetic ---
    merged_ta = merge_multi_task_arithmetic(base_state, [model_A.state_dict(), model_B.state_dict(), model_C.state_dict()], lambda_val=1.0/3.0)
    model_ta_A = build_merged_real_model(merged_ta, fc_A_state, device)
    model_ta_B = build_merged_real_model(merged_ta, fc_B_state, device)
    model_ta_C = build_merged_real_model(merged_ta, fc_C_state, device)
    acc_ta_A = evaluate_model(model_ta_A, loader_A)
    acc_ta_B = evaluate_model(model_ta_B, loader_B)
    acc_ta_C = evaluate_model(model_ta_C, loader_C)
    results["Task Arithmetic"] = (acc_ta_A, acc_ta_B, acc_ta_C)
    
    # --- Multi-Task TIES-Merging ---
    merged_ties = merge_multi_ties(base_state, [model_A.state_dict(), model_B.state_dict(), model_C.state_dict()], trim_percent=0.20, lambda_val=0.4)
    model_ties_A = build_merged_real_model(merged_ties, fc_A_state, device)
    model_ties_B = build_merged_real_model(merged_ties, fc_B_state, device)
    model_ties_C = build_merged_real_model(merged_ties, fc_C_state, device)
    acc_ties_A = evaluate_model(model_ties_A, loader_A)
    acc_ties_B = evaluate_model(model_ties_B, loader_B)
    acc_ties_C = evaluate_model(model_ties_C, loader_C)
    results["TIES-Merging"] = (acc_ties_A, acc_ties_B, acc_ties_C)
    
    # --- Multi-Task DARE-Merging ---
    merged_dare = merge_multi_dare(base_state, [model_A.state_dict(), model_B.state_dict(), model_C.state_dict()], drop_prob=0.5, lambda_val=0.4)
    model_dare_A = build_merged_real_model(merged_dare, fc_A_state, device)
    model_dare_B = build_merged_real_model(merged_dare, fc_B_state, device)
    model_dare_C = build_merged_real_model(merged_dare, fc_C_state, device)
    acc_dare_A = evaluate_model(model_dare_A, loader_A)
    acc_dare_B = evaluate_model(model_dare_B, loader_B)
    acc_dare_C = evaluate_model(model_dare_C, loader_C)
    results["DARE-Merging"] = (acc_dare_A, acc_dare_B, acc_dare_C)
    
    # --- Q-CPOS (Ours) ---
    w1 = 1.0 / np.sqrt(3)
    w2 = 1.0 / np.sqrt(3)
    w3 = 1.0 / np.sqrt(3)
    qcpos_model = QCPOSResNet(model_A, model_B, model_C, w1=w1, w2=w2, w3=w3).to(device)
    
    qcpos_model.set_task(0)
    acc_qcpos_A = evaluate_model(qcpos_model, loader_A)
    
    qcpos_model.set_task(1)
    acc_qcpos_B = evaluate_model(qcpos_model, loader_B)
    
    qcpos_model.set_task(2)
    acc_qcpos_C = evaluate_model(qcpos_model, loader_C)
    
    results["Q-CPOS (Ours)"] = (acc_qcpos_A, acc_qcpos_B, acc_qcpos_C)
    
    # Output results
    print("\nResults Summary (3-Task Joint Merging):")
    print(f"| Method | CIFAR-10 Acc (%) | SVHN Acc (%) | FMNIST Acc (%) | Average Acc (%) |")
    print(f"|---|---|---|---|---|")
    print(f"| Oracle Experts | {acc_A_expert:.2f} | {acc_B_expert:.2f} | {acc_C_expert:.2f} | {(acc_A_expert+acc_B_expert+acc_C_expert)/3.0:.2f} |")
    for method, (acc_A, acc_B, acc_C) in results.items():
        print(f"| {method} | {acc_A:.2f} | {acc_B:.2f} | {acc_C:.2f} | {(acc_A+acc_B+acc_C)/3.0:.2f} |")


def run_cpos_parameter_sweep(task_A="cifar10", task_B="fmnist"):
    print("\n=======================================================")
    print(f"RUNNING CPOS PARAMETER SWEEP: {task_A.upper()} + {task_B.upper()}")
    print("=======================================================")
    
    model_A = load_expert(task_A)
    model_B = load_expert(task_B)
    
    loader_A = get_test_loader(task_A, max_samples=MAX_SAMPLES)
    loader_B = get_test_loader(task_B, max_samples=MAX_SAMPLES)
    
    print("\nWeight Averaging Sweep:")
    print("| w_A | w_B | Task A Acc (%) | Task B Acc (%) | Avg Acc (%) |")
    print("|---|---|---|---|---|")
    for w in np.linspace(0.0, 1.0, 11):
        w_A = 1.0 - w
        w_B = w
        merged_wa = merge_weight_averaging(model_A.state_dict(), model_B.state_dict(), alpha=w_B)
        fc_A_state = {k.replace("fc.", ""): v for k, v in model_A.state_dict().items() if k.startswith("fc.")}
        fc_B_state = {k.replace("fc.", ""): v for k, v in model_B.state_dict().items() if k.startswith("fc.")}
        model_wa_A = build_merged_real_model(merged_wa, fc_A_state, device)
        model_wa_B = build_merged_real_model(merged_wa, fc_B_state, device)
        acc_A = evaluate_model(model_wa_A, loader_A)
        acc_B = evaluate_model(model_wa_B, loader_B)
        print(f"| {w_A:.2f} | {w_B:.2f} | {acc_A:.2f} | {acc_B:.2f} | {(acc_A+acc_B)/2.0:.2f} |")
        
    print("\nCPOS Magnitude-Normalization Constraint Sweep:")
    print("| alpha | beta | Task A Acc (%) | Task B Acc (%) | Avg Acc (%) |")
    print("|---|---|---|---|---|")
    for alpha in np.linspace(0.0, 1.0, 11):
        beta = np.sqrt(1.0 - alpha**2 + 1e-8)
        cpos_model = CPOSResNet(model_A, model_B, alpha=alpha, beta=beta).to(device)
        cpos_model.set_task(0)
        acc_A = evaluate_model(cpos_model, loader_A)
        cpos_model.set_task(1)
        acc_B = evaluate_model(cpos_model, loader_B)
        print(f"| {alpha:.2f} | {beta:.2f} | {acc_A:.2f} | {acc_B:.2f} | {(acc_A+acc_B)/2.0:.2f} |")


def run_noise_robustness_experiment(task_A="cifar10", task_B="fmnist"):
    print("\n=======================================================")
    print(f"RUNNING NOISE PERTURBATION ROBUSTNESS EXPERIMENT: {task_A.upper()} + {task_B.upper()}")
    print("=======================================================")
    
    model_A = load_expert(task_A)
    model_B = load_expert(task_B)
    
    loader_A = get_test_loader(task_A, max_samples=MAX_SAMPLES)
    loader_B = get_test_loader(task_B, max_samples=MAX_SAMPLES)
    
    # Set up models
    # 1. Weight Averaging Model
    merged_wa = merge_weight_averaging(model_A.state_dict(), model_B.state_dict(), alpha=0.5)
    fc_A_state = {k.replace("fc.", ""): v for k, v in model_A.state_dict().items() if k.startswith("fc.")}
    fc_B_state = {k.replace("fc.", ""): v for k, v in model_B.state_dict().items() if k.startswith("fc.")}
    wa_model_A = build_merged_real_model(merged_wa, fc_A_state, device)
    wa_model_B = build_merged_real_model(merged_wa, fc_B_state, device)
    
    # 2. CPOS Model
    alpha_val = 1.0 / np.sqrt(2)
    beta_val = 1.0 / np.sqrt(2)
    cpos_model = CPOSResNet(model_A, model_B, alpha=alpha_val, beta=beta_val).to(device)
    
    # Noise levels to evaluate
    sigmas = [0.0, 0.05, 0.1, 0.2]
    
    print("\nResults:")
    print(r"| Noise Sigma (\sigma) | WA Task A Acc (%) | CPOS Task A Acc (%) | WA Task B Acc (%) | CPOS Task B Acc (%) |")
    print("|---|---|---|---|---|")
    for sigma in sigmas:
        # WA Task A
        wa_acc_A = evaluate_model(wa_model_A, loader_A, noise_sigma=sigma)
        # CPOS Task A
        cpos_model.set_task(0)
        cpos_acc_A = evaluate_model(cpos_model, loader_A, noise_sigma=sigma)
        
        # WA Task B
        wa_acc_B = evaluate_model(wa_model_B, loader_B, noise_sigma=sigma)
        # CPOS Task B
        cpos_model.set_task(1)
        cpos_acc_B = evaluate_model(cpos_model, loader_B, noise_sigma=sigma)
        
        print(f"| {sigma:.2f} | {wa_acc_A:.2f} | {cpos_acc_A:.2f} | {wa_acc_B:.2f} | {cpos_acc_B:.2f} |")


def run_phase_angle_sweep(task_A="cifar10", task_B="fmnist"):
    print("\n=======================================================")
    print(f"RUNNING NOVEL PHASE-ANGLE CONTINUUM SWEEP: {task_A.upper()} + {task_B.upper()}")
    print("=======================================================")
    
    model_A = load_expert(task_A)
    model_B = load_expert(task_B)
    
    loader_A = get_test_loader(task_A, max_samples=MAX_SAMPLES)
    loader_B = get_test_loader(task_B, max_samples=MAX_SAMPLES)
    
    print("\nPhase-Angle Sweep (\\theta):")
    print("| Theta (rad) | Theta (deg) | cos(\\theta) | Task A Acc (%) | Task B Acc (%) | Avg Acc (%) |")
    print("|---|---|---|---|---|---|")
    
    alpha = 1.0 / np.sqrt(2)
    beta = 1.0 / np.sqrt(2)
    
    thetas = np.linspace(0.0, np.pi/2.0, 7)
    for theta in thetas:
        cos_val = np.cos(theta)
        deg = np.degrees(theta)
        
        g_model = GeneralizedCPOSResNet(model_A, model_B, alpha=alpha, beta=beta, theta=theta).to(device)
        
        g_model.set_task(0)
        acc_A = evaluate_model(g_model, loader_A)
        
        g_model.set_task(1)
        acc_B = evaluate_model(g_model, loader_B)
        
        print(f"| {theta:.4f} | {deg:.1f}° | {cos_val:.4f} | {acc_A:.2f} | {acc_B:.2f} | {(acc_A+acc_B)/2.0:.2f} |")


def run_channel_wise_cpos_experiment(task_A="cifar10", task_B="fmnist"):
    print("\n=======================================================")
    print(f"RUNNING CHANNEL-WISE PHASE-INTERLEAVED CPOS EXPERIMENT: {task_A.upper()} + {task_B.upper()}")
    print("=======================================================")
    
    model_A = load_expert(task_A)
    model_B = load_expert(task_B)
    
    loader_A = get_test_loader(task_A, max_samples=MAX_SAMPLES)
    loader_B = get_test_loader(task_B, max_samples=MAX_SAMPLES)
    
    alpha = 1.0 / np.sqrt(2)
    beta = 1.0 / np.sqrt(2)
    
    distributions = ["binary", "linear", "sinusoidal", "random"]
    
    print("\nChannel-Wise Distribution Sweep:")
    print("| Distribution | Task A Acc (%) | Task B Acc (%) | Avg Acc (%) |")
    print("|---|---|---|---|")
    
    for dist in distributions:
        cw_model = ChannelWiseCPOSResNet(model_A, model_B, alpha=alpha, beta=beta, distribution=dist).to(device)
        
        cw_model.set_task(0)
        acc_A = evaluate_model(cw_model, loader_A)
        
        cw_model.set_task(1)
        acc_B = evaluate_model(cw_model, loader_B)
        
        print(f"| {dist.capitalize()} | {acc_A:.2f} | {acc_B:.2f} | {(acc_A+acc_B)/2.0:.2f} |")


def run_dpr_cpos_experiment(task_A="cifar10", task_B="fmnist"):
    print("\n=======================================================")
    print(f"RUNNING DYNAMIC PHASE ROUTING CPOS EXPERIMENT: {task_A.upper()} + {task_B.upper()}")
    print("=======================================================")

    model_A = load_expert(task_A)
    model_B = load_expert(task_B)

    loader_A = get_test_loader(task_A, max_samples=MAX_SAMPLES)
    loader_B = get_test_loader(task_B, max_samples=MAX_SAMPLES)

    alpha = 1.0 / np.sqrt(2)
    beta = 1.0 / np.sqrt(2)

    # Load DPR-CPOS model
    dpr_model = DPRCPOSResNet(model_A, model_B, alpha=alpha, beta=beta).to(device)
    
    # Evaluate on Task A
    dpr_model.clear_recorded_thetas()
    dpr_model.set_task(0)
    acc_A = evaluate_model(dpr_model, loader_A)
    avg_theta_A = np.mean(dpr_model.recorded_thetas)
    
    # Evaluate on Task B
    dpr_model.clear_recorded_thetas()
    dpr_model.set_task(1)
    acc_B = evaluate_model(dpr_model, loader_B)
    avg_theta_B = np.mean(dpr_model.recorded_thetas)
    
    print("\nDynamic Phase Routing Results:")
    print("| Metric | Task A (CIFAR-10) | Task B (FashionMNIST) | Average |")
    print("|---|---|---|---|")
    print(f"| Accuracy (%) | {acc_A:.2f} | {acc_B:.2f} | {(acc_A+acc_B)/2.0:.2f} |")
    print(f"| Avg Phase Angle (rad) | {avg_theta_A:.4f} | {avg_theta_B:.4f} | {((avg_theta_A+avg_theta_B)/2.0):.4f} |")
    print(f"| Avg Phase Angle (deg) | {np.degrees(avg_theta_A):.1f}° | {np.degrees(avg_theta_B):.1f}° | {np.degrees((avg_theta_A+avg_theta_B)/2.0):.1f}° |")


if __name__ == "__main__":
    pairs = [("cifar10", "svhn"), ("cifar10", "fmnist"), ("svhn", "fmnist")]
    for t_A, t_B in pairs:
        try:
            run_experiment_pair(t_A, t_B)
        except Exception as e:
            print(f"Error evaluating pair {t_A} + {t_B}: {e}")
            
    try:
        run_three_task_experiment()
    except Exception as e:
        print(f"Error evaluating three-task experiment: {e}")
        
    try:
        run_cpos_parameter_sweep()
    except Exception as e:
        print(f"Error evaluating parameter sweep: {e}")
        
    try:
        run_noise_robustness_experiment()
    except Exception as e:
        print(f"Error evaluating noise robustness experiment: {e}")

    try:
        run_phase_angle_sweep()
    except Exception as e:
        print(f"Error evaluating phase angle sweep: {e}")

    try:
        run_channel_wise_cpos_experiment()
    except Exception as e:
        print(f"Error evaluating channel-wise CPOS: {e}")

    try:
        run_dpr_cpos_experiment()
    except Exception as e:
        print(f"Error evaluating DPR-CPOS: {e}")

