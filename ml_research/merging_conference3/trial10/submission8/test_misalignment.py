import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Configuration for CNN
L = 12
D = 128
K = 4
TASKS = ["MNIST", "FashionMNIST", "CIFAR-10", "SVHN"]
NOISE_SCALES = [0.05, 0.15, 0.40, 1.20]

def generate_task_signatures(D, K):
    signatures = torch.zeros(K, D)
    S = D // K
    for k in range(K):
        signatures[k, k * S : (k + 1) * S] = 1.0 / np.sqrt(S)
    return signatures

def generate_samples(num_samples_per_task, signatures, noise_scales):
    K, D = signatures.shape
    X = []
    y = []
    for k in range(K):
        sig = signatures[k]
        noise_scale = noise_scales[k]
        samples = sig.unsqueeze(0) + torch.randn(num_samples_per_task, D) * noise_scale
        X.append(samples)
        y.extend([k] * num_samples_per_task)
    return torch.cat(X, dim=0), torch.tensor(y, dtype=torch.long)

def generate_rotation_matrices(D, K, strength=0.3):
    rotations = []
    for k in range(K):
        A = torch.randn(D, D)
        Q, R = torch.linalg.qr(A)
        d = torch.diag(R)
        ph = d.sign()
        Q = Q * ph.unsqueeze(0)
        
        # Interpolate with identity
        M = (1 - strength) * torch.eye(D) + strength * Q
        QM, RM = torch.linalg.qr(M)
        dm = torch.diag(RM)
        phm = dm.sign()
        QM = QM * phm.unsqueeze(0)
        rotations.append(QM.unsqueeze(0))
    return torch.cat(rotations, dim=0)

def propagate_representation(h0, L, K, D, signatures, alpha, gamma_schedule, rotation_matrices=None):
    h = h0.clone()
    for l in range(1, L):
        alpha_l = alpha[l, :]
        if rotation_matrices is not None:
            rotated_sigs = torch.zeros_like(signatures)
            for k in range(K):
                rotated_sigs[k] = torch.matmul(rotation_matrices[k], signatures[k])
            v_term = torch.matmul(alpha_l, rotated_sigs)
        else:
            v_term = torch.matmul(alpha_l, signatures)
        alpha_sum = alpha_l.sum()
        pull = v_term - alpha_sum * h
        h = h + gamma_schedule[l] * pull
    return h

def compute_accuracy_and_loss(h_L, y, signatures, temp=0.05, eval_mode=False):
    batch_size = h_L.shape[0]
    distances = torch.sum((h_L.unsqueeze(1) - signatures.unsqueeze(0)) ** 2, dim=2)
    logits = -distances / temp
    logit_noise_scales = torch.tensor([0.25, 0.65, 1.85, 3.25])
    
    if eval_mode:
        state = torch.random.get_rng_state()
        torch.manual_seed(12345)
        noise = torch.randn(batch_size, K) * logit_noise_scales.unsqueeze(0)
        torch.random.set_rng_state(state)
    else:
        noise = torch.randn(batch_size, K) * logit_noise_scales.unsqueeze(0)
        
    noisy_logits = logits + noise
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(noisy_logits, y)
    preds = torch.argmax(noisy_logits, dim=1)
    cat_acc = (preds == y).float().mean().item()
    return cat_acc, loss

# Let's try running Static Uniform vs. RB-FTM/RB-DCTM under different misalignment strengths
signatures = generate_task_signatures(D, K)
gamma_schedule = torch.linspace(0.05, 0.15, L)

X_cal, y_cal = generate_samples(10, signatures, NOISE_SCALES)
X_test, y_test = generate_samples(500, signatures, NOISE_SCALES)

from run_experiments import FourierTrajectoryModule, DCTTrajectoryModule

for strength in [0.0, 0.2, 0.4, 0.6]:
    print(f"\n--- Misalignment Strength: {strength} ---")
    rotations = generate_rotation_matrices(D, K, strength) if strength > 0 else None
    
    # 1. Static Uniform
    alpha_static = torch.full((L, K), 1.0 / K)
    h_test_static = propagate_representation(X_test, L, K, D, signatures, alpha_static, gamma_schedule, rotations)
    acc_static, _ = compute_accuracy_and_loss(h_test_static, y_test, signatures, eval_mode=True)
    print(f"Static Uniform Cat. Acc: {acc_static:.2%}")
    
    # 2. RB-FTM F=1
    module_ftm = FourierTrajectoryModule(L, K, F=1)
    opt = optim.Adam(module_ftm.get_parameters(), lr=0.1)
    for step in range(30):
        opt.zero_grad()
        alpha = module_ftm.synthesize()
        h_cal = propagate_representation(X_cal, L, K, D, signatures, alpha, gamma_schedule, rotations)
        _, loss = compute_accuracy_and_loss(h_cal, y_cal, signatures)
        loss = loss + 0.01 * module_ftm.get_spectral_norm()
        loss.backward()
        opt.step()
        
    with torch.no_grad():
        alpha_ftm = module_ftm.synthesize()
        h_test_ftm = propagate_representation(X_test, L, K, D, signatures, alpha_ftm, gamma_schedule, rotations)
        acc_ftm, _ = compute_accuracy_and_loss(h_test_ftm, y_test, signatures, eval_mode=True)
    print(f"RB-FTM (F=1) Cat. Acc: {acc_ftm:.2%}")

    # 3. RB-DCTM F=1
    module_dct = DCTTrajectoryModule(L, K, F=1)
    opt = optim.Adam(module_dct.get_parameters(), lr=0.1)
    for step in range(30):
        opt.zero_grad()
        alpha = module_dct.synthesize()
        h_cal = propagate_representation(X_cal, L, K, D, signatures, alpha, gamma_schedule, rotations)
        _, loss = compute_accuracy_and_loss(h_cal, y_cal, signatures)
        loss = loss + 0.01 * module_dct.get_spectral_norm()
        loss.backward()
        opt.step()
        
    with torch.no_grad():
        alpha_dct = module_dct.synthesize()
        h_test_dct = propagate_representation(X_test, L, K, D, signatures, alpha_dct, gamma_schedule, rotations)
        acc_dct, _ = compute_accuracy_and_loss(h_test_dct, y_test, signatures, eval_mode=True)
    print(f"RB-DCTM (F=1) Cat. Acc: {acc_dct:.2%}")
