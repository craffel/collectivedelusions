import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from torchvision import datasets, transforms
from models import SimpleCNN
import numpy as np

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoints
checkpoint_std_mnist = torch.load('checkpoints/standard_mnist.pt', map_location=device, weights_only=False)
checkpoint_std_fashion = torch.load('checkpoints/standard_fashion.pt', map_location=device, weights_only=False)

model1 = SimpleCNN(is_cosface=False).to(device)
model2 = SimpleCNN(is_cosface=False).to(device)
model1.load_state_dict(checkpoint_std_mnist['state_dict'])
model2.load_state_dict(checkpoint_std_fashion['state_dict'])
model1.eval()
model2.eval()

# Let's get a clean batch of MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_test = datasets.MNIST(root="data", train=False, download=False, transform=transform)
loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=False)
x_clean, _ = next(iter(loader))
x_clean = x_clean.to(device)

# Noisy batch
noise = torch.randn_like(x_clean) * 0.6
x_noisy = torch.clamp(x_clean + noise, -1.0, 1.0)

def compute_sens(x, loss_type="entropy"):
    # Merged model
    merged_model = SimpleCNN(is_cosface=False).to(device)
    w1, w2 = 0.5, 0.5
    
    init_state = {}
    for name, param in model1.named_parameters():
        init_state[name] = w1 * param + w2 * dict(model2.named_parameters())[name]
        init_state[name].requires_grad_ = True
        
    init_buffers = {}
    for name, buf in model1.named_buffers():
        if "running_mean" in name or "running_var" in name:
            var_name = name.replace("running_mean", "running_var") if "running_mean" in name else name
            mean_name = name.replace("running_var", "running_mean") if "running_var" in name else name
            mu1, mu2 = dict(model1.named_buffers())[mean_name], dict(model2.named_buffers())[mean_name]
            sig1, sig2 = dict(model1.named_buffers())[var_name], dict(model2.named_buffers())[var_name]
            mu_fused = w1 * mu1 + w2 * mu2
            sig_fused = w1 * (sig1 + (mu1 - mu_fused)**2) + w2 * (sig2 + (mu2 - mu_fused)**2)
            if "running_mean" in name:
                init_buffers[name] = mu_fused
            else:
                init_buffers[name] = sig_fused
        else:
            init_buffers[name] = buf
            
    outputs = functional_call(merged_model, {**init_state, **init_buffers}, x)
    
    if loss_type == "entropy":
        loss = -torch.mean(torch.sum(F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1), dim=1))
    elif loss_type == "pseudo_ce":
        pseudo_labels = outputs.argmax(dim=1)
        loss = F.cross_entropy(outputs, pseudo_labels)
    elif loss_type == "uniform_ce":
        # CE with uniform distribution (or random labels)
        random_labels = torch.randint(0, 10, (x.size(0),), device=device)
        loss = F.cross_entropy(outputs, random_labels)
    
    grads = torch.autograd.grad(loss, init_state.values(), allow_unused=True)
    
    sens = {}
    total_sens = 0.0
    for (name, _), grad in zip(init_state.items(), grads):
        if grad is not None:
            sens_val = torch.sum(grad**2).item()
            sens[name] = sens_val
            total_sens += sens_val
        else:
            sens[name] = 1e-4
            
    for name in sens.keys():
        sens[name] /= (total_sens + 1e-8)
        
    return sens, total_sens

# Compute sensitivities under different settings
sens_clean_ent, tot_clean_ent = compute_sens(x_clean, "entropy")
sens_noisy_ent, tot_noisy_ent = compute_sens(x_noisy, "entropy")
sens_noisy_pce, tot_noisy_pce = compute_sens(x_noisy, "pseudo_ce")

# Also, let's look at offline sensitivities for comparison
# Let's compute a simple offline sensitivity on 256 clean samples
sens_offline = {}
total_offline = 0.0
# Just run compute_sens on a larger clean batch or average over multiple batches
loader_large = torch.utils.data.DataLoader(mnist_test, batch_size=256, shuffle=False)
x_large, _ = next(iter(loader_large))
x_large = x_large.to(device)
sens_offline, tot_offline = compute_sens(x_large, "entropy")

print(f"Total Sensitivity Magnitude:")
print(f"  Clean Entropy: {tot_clean_ent:.6f}")
print(f"  Noisy Entropy: {tot_noisy_ent:.6f}")
print(f"  Noisy Pseudo-CE: {tot_noisy_pce:.6f}")
print(f"  Offline Clean: {tot_offline:.6f}")
print("\nLayer Sensitivity Comparison (Selected Layers):")
layers_to_show = ["conv1.weight", "conv2.weight", "fc1.weight", "classifier.weight"]
print(f"{'Layer':<20} | {'Offline Clean':<15} | {'Clean Entropy':<15} | {'Noisy Entropy':<15} | {'Noisy Pseudo-CE':<15}")
print("-" * 90)
for layer in layers_to_show:
    print(f"{layer:<20} | {sens_offline[layer]:.6f}        | {sens_clean_ent[layer]:.6f}        | {sens_noisy_ent[layer]:.6f}        | {sens_noisy_pce[layer]:.6f}")
