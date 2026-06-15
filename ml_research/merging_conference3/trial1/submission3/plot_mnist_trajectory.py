import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import os

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# 1. Model Definition (Lightweight MLP)
# ==============================================================================
class MNISTMLP(nn.Module):
    def __init__(self):
        super(MNISTMLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(64, 10)
        
    def forward_encoder(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x
        
    def forward(self, x):
        features = self.forward_encoder(x)
        return self.classifier(features)

def copy_weights(src_model, dest_model):
    dest_model.load_state_dict(src_model.state_dict())

# ==============================================================================
# 2. Dataset Preparation
# ==============================================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

indices_all = np.arange(len(mnist_train))
labels = mnist_train.targets.numpy()

idx_04 = indices_all[(labels >= 0) & (labels <= 4)]
idx_59 = indices_all[(labels >= 5) & (labels <= 9)]

subset_04 = Subset(mnist_train, idx_04[:1000])
subset_59 = Subset(mnist_train, idx_59[:1000])

train_loader_all = DataLoader(Subset(mnist_train, indices_all[:1000]), batch_size=64, shuffle=True)
train_loader_04 = DataLoader(subset_04, batch_size=64, shuffle=True)
train_loader_59 = DataLoader(subset_59, batch_size=64, shuffle=True)

test_labels = mnist_test.targets.numpy()
test_idx_04 = np.arange(len(mnist_test))[(test_labels >= 0) & (test_labels <= 4)]
test_idx_59 = np.arange(len(mnist_test))[(test_labels >= 5) & (test_labels <= 9)]

test_loader_04 = DataLoader(Subset(mnist_test, test_idx_04[:1000]), batch_size=128, shuffle=False)
test_loader_59 = DataLoader(Subset(mnist_test, test_idx_59[:1000]), batch_size=128, shuffle=False)

# ==============================================================================
# 3. Training Base and Experts
# ==============================================================================
print("Training pre-trained base model...")
base_model = MNISTMLP()
optimizer = optim.Adam(base_model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

base_model.train()
for x, y in train_loader_all:
    optimizer.zero_grad()
    loss = criterion(base_model(x), y)
    loss.backward()
    optimizer.step()

print("Fine-tuning Expert 1 (Digits 0-4)...")
expert1 = MNISTMLP()
copy_weights(base_model, expert1)
opt1 = optim.Adam(expert1.parameters(), lr=0.005)
expert1.train()
for epoch in range(2):
    for x, y in train_loader_04:
        opt1.zero_grad()
        loss = criterion(expert1(x), y)
        loss.backward()
        opt1.step()

print("Fine-tuning Expert 2 (Digits 5-9)...")
expert2 = MNISTMLP()
copy_weights(base_model, expert2)
opt2 = optim.Adam(expert2.parameters(), lr=0.005)
expert2.train()
for epoch in range(2):
    for x, y in train_loader_59:
        opt2.zero_grad()
        loss = criterion(expert2(x), y)
        loss.backward()
        opt2.step()

# ==============================================================================
# 4. Merged Model Class
# ==============================================================================
class MergedModel(nn.Module):
    def __init__(self, base, exp1, exp2):
        super(MergedModel, self).__init__()
        self.base = base
        self.exp1 = exp1
        self.exp2 = exp2
        self.lambdas = nn.Parameter(torch.full((2, 2), 0.3))
        self.clf1 = nn.Parameter(exp1.classifier.weight.clone())
        self.clf2 = nn.Parameter(exp2.classifier.weight.clone())
        self.clf_bias1 = nn.Parameter(exp1.classifier.bias.clone())
        self.clf_bias2 = nn.Parameter(exp2.classifier.bias.clone())
        
    def forward_task(self, x, task_idx):
        tv1_fc1_w = self.exp1.fc1.weight - self.base.fc1.weight
        tv1_fc1_b = self.exp1.fc1.bias - self.base.fc1.bias
        tv1_fc2_w = self.exp1.fc2.weight - self.base.fc2.weight
        tv1_fc2_b = self.exp1.fc2.bias - self.base.fc2.bias
        
        tv2_fc1_w = self.exp2.fc1.weight - self.base.fc1.weight
        tv2_fc1_b = self.exp2.fc1.bias - self.base.fc1.bias
        tv2_fc2_w = self.exp2.fc2.weight - self.base.fc2.weight
        tv2_fc2_b = self.exp2.fc2.bias - self.base.fc2.bias
        
        l1_fc1, l2_fc1 = self.lambdas[0, 0], self.lambdas[0, 1]
        fc1_w = self.base.fc1.weight + l1_fc1 * tv1_fc1_w + l2_fc1 * tv2_fc1_w
        fc1_b = self.base.fc1.bias + l1_fc1 * tv1_fc1_b + l2_fc1 * tv2_fc1_b
        
        l1_fc2, l2_fc2 = self.lambdas[1, 0], self.lambdas[1, 1]
        fc2_w = self.base.fc2.weight + l1_fc2 * tv1_fc2_w + l2_fc2 * tv2_fc2_w
        fc2_b = self.base.fc2.bias + l1_fc2 * tv1_fc2_b + l2_fc2 * tv2_fc2_b
        
        x = x.view(-1, 784)
        x = nn.functional.linear(x, fc1_w, fc1_b)
        x = torch.relu(x)
        x = nn.functional.linear(x, fc2_w, fc2_b)
        features = torch.relu(x)
        
        if task_idx == 0:
            return nn.functional.linear(features, self.clf1, self.clf_bias1)
        else:
            return nn.functional.linear(features, self.clf2, self.clf_bias2)

# ==============================================================================
# 5. Execute Adaptation with Logging
# ==============================================================================
def run_adaptation_log(method_name):
    # Set random seeds
    torch.manual_seed(0)
    np.random.seed(0)
    import random
    random.seed(0)
    
    # Prepare adaptation data
    shuffled_04_idx = np.random.choice(test_idx_04, size=1000, replace=False)
    shuffled_59_idx = np.random.choice(test_idx_59, size=1000, replace=False)
    
    loader_04_shuffled = DataLoader(Subset(mnist_test, shuffled_04_idx), batch_size=128, shuffle=True)
    loader_59_shuffled = DataLoader(Subset(mnist_test, shuffled_59_idx), batch_size=128, shuffle=True)
    
    adaptation_data = []
    for (x1, _), (x2, _) in zip(loader_04_shuffled, loader_59_shuffled):
        adaptation_data.append((x1, x2))
        if len(adaptation_data) >= 8:
            break
            
    model = MergedModel(base_model, expert1, expert2)
    
    epochs = 5
    lr = 0.1
    temp = 0.002
    gamma = 0.9 if method_name == "ThermoMerge" else 0.85
    
    loss_history = []
    
    for epoch in range(epochs):
        for x1, x2 in adaptation_data:
            with torch.no_grad():
                t1 = expert1(x1).softmax(dim=1)
                t2 = expert2(x2).softmax(dim=1)
                
            out1 = model.forward_task(x1, 0)
            out2 = model.forward_task(x2, 1)
            
            loss1 = -torch.sum(t1 * out1.log_softmax(dim=1), dim=1).mean()
            loss2 = -torch.sum(t2 * out2.log_softmax(dim=1), dim=1).mean()
            total_loss = loss1 + loss2
            
            loss_history.append(total_loss.item())
            
            model.zero_grad()
            total_loss.backward()
            
            with torch.no_grad():
                # 1. Update lambdas (d_lambdas = 4)
                grad_lambdas = model.lambdas.grad
                if method_name == "ThermoMerge":
                    noise_lambdas = torch.randn_like(model.lambdas) * np.sqrt(2.0 * lr * temp / 4.0)
                    model.lambdas.copy_(model.lambdas - lr * grad_lambdas + noise_lambdas)
                else:
                    model.lambdas.copy_(model.lambdas - lr * grad_lambdas)
                    
                # 2. Update Classifiers (d_joint = 650)
                grad_clf1 = model.clf1.grad
                grad_bias1 = model.clf_bias1.grad
                if method_name == "ThermoMerge":
                    noise_clf1 = torch.randn_like(model.clf1) * np.sqrt(2.0 * lr * temp / 650.0)
                    noise_bias1 = torch.randn_like(model.clf_bias1) * np.sqrt(2.0 * lr * temp / 650.0)
                    model.clf1.copy_(model.clf1 - lr * grad_clf1 + noise_clf1)
                    model.clf_bias1.copy_(model.clf_bias1 - lr * grad_bias1 + noise_bias1)
                else:
                    model.clf1.copy_(model.clf1 - lr * grad_clf1)
                    model.clf_bias1.copy_(model.clf_bias1 - lr * grad_bias1)
                    
                grad_clf2 = model.clf2.grad
                grad_bias2 = model.clf_bias2.grad
                if method_name == "ThermoMerge":
                    noise_clf2 = torch.randn_like(model.clf2) * np.sqrt(2.0 * lr * temp / 650.0)
                    noise_bias2 = torch.randn_like(model.clf_bias2) * np.sqrt(2.0 * lr * temp / 650.0)
                    model.clf2.copy_(model.clf2 - lr * grad_clf2 + noise_clf2)
                    model.clf_bias2.copy_(model.clf_bias2 - lr * grad_bias2 + noise_bias2)
                else:
                    model.clf2.copy_(model.clf2 - lr * grad_clf2)
                    model.clf_bias2.copy_(model.clf_bias2 - lr * grad_bias2)
                    
        temp *= gamma
        
    return loss_history

# Run both methods
print("Running SyMerge...")
symerge_losses = run_adaptation_log("SyMerge")
print("Running ThermoMerge...")
thermomerge_losses = run_adaptation_log("ThermoMerge")

# Plotting the loss trajectories
plt.figure(figsize=(9, 5))
steps = np.arange(len(symerge_losses))

plt.plot(steps, symerge_losses, label="SyMerge (Deterministic Adam/SGD)", color='blue', linewidth=2, linestyle='--')
plt.plot(steps, thermomerge_losses, label="ThermoMerge (Ours with SGLD + SA)", color='green', linewidth=2)

# Shade Hot and Cold phases
plt.axvspan(0, 16, color='orange', alpha=0.1, label="Hot Phase (Exploration)")
plt.axvspan(16, 40, color='blue', alpha=0.05, label="Cold Phase (Crystallization)")

plt.title("MNIST Joint Test-Time Adaptation: Loss Trajectory Analysis", fontsize=14, fontweight='bold')
plt.xlabel("Adaptation Step (5 Epochs over 8 Batches)", fontsize=12)
plt.ylabel("Proxy Test-Time Loss (Self-Label CE)", fontsize=12)
plt.grid(True, which='both', linestyle=':', alpha=0.5)
plt.legend(fontsize=10, loc='upper right')

# Save the plot
plt.tight_layout()
os.makedirs("submission/results", exist_ok=True)
plt.savefig("submission/results/mnist_adaptation_trajectory.png", dpi=300)
print("Saved trajectory plot to submission/results/mnist_adaptation_trajectory.png")
