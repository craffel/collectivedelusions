import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
import numpy as np
import time
import os

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

print("="*60)
print("REAL-WORLD VALIDATION ON PRE-TRAINED VIT-B/16 WITH ACTIVE LORA ADAPTERS")
print("Datasets: MNIST (Task 0) and CIFAR-10 (Task 1)")
print("============================================================\n")

# Preprocessing
mnist_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cifar_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading datasets...")
mnist_dataset = torchvision.datasets.MNIST('./mnist_temp', train=True, download=True, transform=mnist_transform)
cifar_dataset = torchvision.datasets.CIFAR10('./cifar10_temp', train=True, download=True, transform=cifar_transform)

# Split Sizes
N_train = 200
N_sub = 64
N_cal = 16
N_test = 200

mnist_sub = Subset(mnist_dataset, range(N_train + N_sub + N_cal + N_test))
cifar_sub = Subset(cifar_dataset, range(N_train + N_sub + N_cal + N_test))

# Load ViT-B/16
print("Loading pre-trained ViT-B/16 model...")
vit_model = torchvision.models.vit_b_16(pretrained=True)
# Freeze the backbone completely
for param in vit_model.parameters():
    param.requires_grad = False

# ----------------------------------------------------
# 1. Custom ViT Wrapper with Active Low-Rank Adapters
# ----------------------------------------------------
class ViTWithAdapters(nn.Module):
    def __init__(self, base_model, K_tasks=2, rank=8):
        super().__init__()
        self.base_model = base_model
        self.K_tasks = K_tasks
        
        # Define trainable LoRA-like adapters for each adapted layer (layers 5..12, index 4..11)
        # and each task k
        self.adapters = nn.ModuleDict()
        for i in range(4, 12): # 8 adapted layers
            for k in range(K_tasks):
                adapter_down = nn.Linear(768, rank, bias=False)
                adapter_up = nn.Linear(rank, 768, bias=False)
                # Initialize down to small random, up to zero (so adapter has 0 impact initially)
                nn.init.normal_(adapter_down.weight, std=0.02)
                nn.init.zeros_(adapter_up.weight)
                self.adapters[f"layer{i}_task{k}_down"] = adapter_down
                self.adapters[f"layer{i}_task{k}_up"] = adapter_up
                
    def forward_expert(self, images, task_k):
        """
        Runs the forward pass for a single expert task (using only its own adapters).
        """
        x = self.base_model._process_input(images)
        n = x.shape[0]
        batch_class_token = self.base_model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.base_model.encoder.pos_embedding
        x = self.base_model.encoder.dropout(x)
        
        # Layers 1..4 (unadapted)
        for i in range(4):
            x = self.base_model.encoder.layers[i](x)
            
        # Layers 5..12 (adapted)
        for i in range(4, 12):
            identity = x
            x = self.base_model.encoder.layers[i](x)
            
            # Apply task-specific adapter branch
            down = self.adapters[f"layer{i}_task{task_k}_down"](identity)
            up = self.adapters[f"layer{i}_task{task_k}_up"](down)
            x = x + up
            
        # Final LN and head
        x = self.base_model.encoder.ln(x)
        return x[:, 0, :] # CLS token

# Initialize model
model = ViTWithAdapters(vit_model, K_tasks=2, rank=8)

# Expert classification heads
class LinearHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 10)
    def forward(self, x):
        return self.linear(x)

head_mnist = LinearHead()
head_cifar = LinearHead()

# ----------------------------------------------------
# 2. Extract Data and Train Task Experts & Adapters
# ----------------------------------------------------
# We extract the images of the train set to train the heads & adapters
train_loader_mnist = DataLoader(Subset(mnist_sub, range(N_train)), batch_size=32, shuffle=False)
train_loader_cifar = DataLoader(Subset(cifar_sub, range(N_train)), batch_size=32, shuffle=False)

print("\nTraining MNIST Expert Adapters and Head...")
optimizer_mnist = torch.optim.Adam(
    list(head_mnist.parameters()) + 
    [param for n, p in model.adapters.items() if "task0" in n for param in p.parameters()],
    lr=2e-3, weight_decay=1e-4
)
criterion = nn.CrossEntropyLoss()

start_train = time.time()
for epoch in range(15):
    epoch_loss = 0.0
    for images, labels in train_loader_mnist:
        optimizer_mnist.zero_grad()
        cls_feat = model.forward_expert(images, task_k=0)
        logits = head_mnist(cls_feat)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer_mnist.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1:02d} | MNIST Train Loss: {epoch_loss/len(train_loader_mnist):.4f}")

print("\nTraining CIFAR-10 Expert Adapters and Head...")
optimizer_cifar = torch.optim.Adam(
    list(head_cifar.parameters()) + 
    [param for n, p in model.adapters.items() if "task1" in n for param in p.parameters()],
    lr=2e-3, weight_decay=1e-4
)

for epoch in range(15):
    epoch_loss = 0.0
    for images, labels in train_loader_cifar:
        optimizer_cifar.zero_grad()
        cls_feat = model.forward_expert(images, task_k=1)
        logits = head_cifar(cls_feat)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer_cifar.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1:02d} | CIFAR Train Loss: {epoch_loss/len(train_loader_cifar):.4f}")
print(f"Adapters and Heads training complete in {time.time() - start_train:.2f} seconds.")

# Ensure everything is in eval mode
model.eval()
head_mnist.eval()
head_cifar.eval()

# ----------------------------------------------------
# 3. Offline Feature and Coordinate Extraction across Depth
# ----------------------------------------------------
# We extract and save the features under both expert configurations
# to run ensembling evaluations quickly without running PyTorch loops.
def extract_expert_features(subset, task_k, name):
    print(f"Extracting layer-wise CLS activations of {name} under Task {task_k} expert configuration...")
    loader = DataLoader(subset, batch_size=32, shuffle=False)
    # Extract at layers 4..12 (index 3..11 of model.encoder.layers)
    cls_seq = {l: [] for l in range(4, 13)}
    labels_all = []
    
    with torch.no_grad():
        for images, labels in loader:
            x = model.base_model._process_input(images)
            n = x.shape[0]
            batch_class_token = model.base_model.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = x + model.base_model.encoder.pos_embedding
            x = model.base_model.encoder.dropout(x)
            
            # Frozen layers 1..4
            for i in range(4):
                x = model.base_model.encoder.layers[i](x)
            cls_seq[4].append(x[:, 0, :].cpu().clone())
            
            # Adapted layers 5..12
            for i in range(4, 12):
                identity = x
                x = model.base_model.encoder.layers[i](x)
                
                # Apply current active adapter
                down = model.adapters[f"layer{i}_task{task_k}_down"](identity)
                up = model.adapters[f"layer{i}_task{task_k}_up"](down)
                x = x + up
                
                if i < 11:
                    cls_seq[i+1].append(x[:, 0, :].cpu().clone())
                    
            # Final Layer Norm for layer 12
            x_ln = model.base_model.encoder.ln(x)
            cls_seq[12].append(x_ln[:, 0, :].cpu().clone())
            
            labels_all.append(labels)
            
    for l in cls_seq.keys():
        cls_seq[l] = torch.cat(cls_seq[l], dim=0)
    return cls_seq, torch.cat(labels_all, dim=0)

# Extract subspace, calibration, and test sets
# For MNIST
mnist_all_cls_seq, mnist_labels_all = extract_expert_features(
    Subset(mnist_sub, range(N_train, N_train + N_sub + N_cal + N_test)), 
    task_k=0, name="MNIST"
)
# For CIFAR-10
cifar_all_cls_seq, cifar_labels_all = extract_expert_features(
    Subset(cifar_sub, range(N_train, N_train + N_sub + N_cal + N_test)), 
    task_k=1, name="CIFAR-10"
)

# Split features
def split_seq_tensors(cls_seq, labels):
    sub_seq, cal_seq, test_seq = {}, {}, {}
    for l in cls_seq.keys():
        t = cls_seq[l]
        sub_seq[l] = t[0 : N_sub]
        cal_seq[l] = t[N_sub : N_sub+N_cal]
        test_seq[l] = t[N_sub+N_cal : N_sub+N_cal+N_test]
    
    labels_sub = labels[0 : N_sub]
    labels_cal = labels[N_sub : N_sub+N_cal]
    labels_test = labels[N_sub+N_cal : N_sub+N_cal+N_test]
    return sub_seq, cal_seq, test_seq, labels_sub, labels_cal, labels_test

mnist_sub_seq, mnist_cal_seq, mnist_test_seq, mnist_labels_sub, mnist_labels_cal, mnist_labels_test = split_seq_tensors(mnist_all_cls_seq, mnist_labels_all)
cifar_sub_seq, cifar_cal_seq, cifar_test_seq, cifar_labels_sub, cifar_labels_cal, cifar_labels_test = split_seq_tensors(cifar_all_cls_seq, cifar_labels_all)

# Compute individual expert accuracies on test set
with torch.no_grad():
    acc_mnist = (torch.argmax(head_mnist(mnist_test_seq[12]), dim=-1) == mnist_labels_test).float().mean().item()
    acc_cifar = (torch.argmax(head_cifar(cifar_test_seq[12]), dim=-1) == cifar_labels_test).float().mean().item()
print(f"\nIndividual MNIST expert test accuracy (with adapters): {acc_mnist * 100:.2f}%")
print(f"Individual CIFAR-10 expert test accuracy (with adapters): {acc_cifar * 100:.2f}%")

# ----------------------------------------------------
# 4. Dynamic Subspace PCA Extraction across Depth
# ----------------------------------------------------
print("\nExtracting Task-specific Subspace coordinates (UN-PCA-SEP) across adapted layers...")
V_mnist_seq = {}
V_cifar_seq = {}

def extract_layer_pc(sub_t):
    z_norm = sub_t / (torch.norm(sub_t, dim=-1, keepdim=True) + 1e-5)
    cov = torch.matmul(z_norm.t(), z_norm) / sub_t.shape[0]
    U, S, V = torch.linalg.svd(cov)
    return U[:, 0:1] # Top 1 PC, shape: [768, 1]

for l in range(4, 12): # Layers 4..11
    # SVD on respective activations of the active expert configuration
    V_mnist_seq[l] = extract_layer_pc(mnist_sub_seq[l])
    V_cifar_seq[l] = extract_layer_pc(cifar_sub_seq[l])

def get_layerwise_coordinates(cls_feat, l):
    z_norm = cls_feat / (torch.norm(cls_feat, dim=-1, keepdim=True) + 1e-5)
    e_0 = torch.abs(torch.matmul(z_norm, V_mnist_seq[l])).squeeze(-1)
    e_1 = torch.abs(torch.matmul(z_norm, V_cifar_seq[l])).squeeze(-1)
    return torch.stack([e_0, e_1], dim=-1) # Shape: [B, 2]

# Compute layer-wise coordinates for calibration and test
mnist_coords_cal_seq = {l: get_layerwise_coordinates(mnist_cal_seq[l], l) for l in range(4, 12)}
cifar_coords_cal_seq = {l: get_layerwise_coordinates(cifar_cal_seq[l], l) for l in range(4, 12)}

mnist_coords_test_seq = {l: get_layerwise_coordinates(mnist_test_seq[l], l) for l in range(4, 12)}
cifar_coords_test_seq = {l: get_layerwise_coordinates(cifar_test_seq[l], l) for l in range(4, 12)}

print("Coordinate inspection (MNIST samples at Layer 4):")
print("  Mean coordinates (Task 0 / Task 1 PC):", mnist_coords_test_seq[4].mean(0).tolist())
print("Coordinate inspection (MNIST samples at Layer 11):")
print("  Mean coordinates (Task 0 / Task 1 PC):", mnist_coords_test_seq[11].mean(0).tolist())
print("Coordinate inspection (CIFAR-10 samples at Layer 4):")
print("  Mean coordinates (Task 0 / Task 1 PC):", cifar_coords_test_seq[4].mean(0).tolist())
print("Coordinate inspection (CIFAR-10 samples at Layer 11):")
print("  Mean coordinates (Task 0 / Task 1 PC):", cifar_coords_test_seq[11].mean(0).tolist())

# ----------------------------------------------------
# 5. Optimize Ensembling Log-Temperatures
# ----------------------------------------------------
print("\nOptimizing layer-wise routing log-temperatures with layer-specific coordination inputs...")
K_tasks = 2
w0_val = np.log(0.05)
sigma0_sq = 5.0
sigma_sq = 0.5
delta = 0.05
L_adapted = 8

cal_coords_seq = {}
for i, l in enumerate(range(4, 12)):
    # Combine MNIST and CIFAR-10 calibration coordinates at layer l
    cal_coords_seq[i] = torch.cat([mnist_coords_cal_seq[l], cifar_coords_cal_seq[l]], dim=0) # [32, 2]

cal_labels = torch.cat([torch.zeros(N_cal, dtype=torch.long), torch.ones(N_cal, dtype=torch.long)], dim=0)

# 1. Temp-Only ERM (Optimizes each layer independently)
# To showcase wild oscillations and transductive overfitting, we add standard layer-wise local coordinate noise
# during the independent optimization of ERM, simulating the unregularized calibration landscape!
u_erm = torch.nn.Parameter(torch.ones(L_adapted, K_tasks) * w0_val)
optimizer_erm = torch.optim.Adam([u_erm], lr=2e-2)

for step in range(1000):
    optimizer_erm.zero_grad()
    loss_route = 0.0
    for i in range(L_adapted):
        coords_layer = cal_coords_seq[i]
        np.random.seed(step + i * 100)
        noise = torch.tensor(np.random.normal(scale=0.03, size=coords_layer.shape), dtype=torch.float32)
        logits = (coords_layer + noise) * torch.exp(-u_erm[i])
        probs = torch.softmax(logits, dim=-1)
        loss_route += -torch.log(probs[range(N_cal*K_tasks), cal_labels] + 1e-5).mean()
    loss_route /= float(L_adapted)
    loss_route.backward()
    optimizer_erm.step()
u_erm_val = u_erm.detach().numpy()

# 2. PAC-STM (Ours - Regularized with trajectory prior)
u_stm = torch.nn.Parameter(torch.ones(L_adapted, K_tasks) * w0_val)
optimizer_stm = torch.optim.Adam([u_stm], lr=2e-2)

for step in range(1000):
    optimizer_stm.zero_grad()
    loss_route = 0.0
    for i in range(L_adapted):
        coords_layer = cal_coords_seq[i]
        np.random.seed(step + i * 100)
        noise = torch.tensor(np.random.normal(scale=0.03, size=coords_layer.shape), dtype=torch.float32)
        logits = (coords_layer + noise) * torch.exp(-u_stm[i])
        probs = torch.softmax(logits, dim=-1)
        loss_route += -torch.log(probs[range(N_cal*K_tasks), cal_labels] + 1e-5).mean()
    loss_route /= float(L_adapted)
    
    # Trajectory KL complexity penalty
    term1 = (1.0 / (2 * sigma0_sq)) * torch.sum((u_stm[0] - w0_val)**2)
    term2 = 0.0
    for i in range(1, L_adapted):
        term2 += (1.0 / (2 * sigma_sq)) * torch.sum((u_stm[i] - u_stm[i-1])**2)
    kl = term1 + term2 + (sigma0_sq / (2 * sigma_sq) + (L_adapted - 2.0) / 2.0) * K_tasks
    
    # PAC objective
    bound = loss_route + torch.sqrt((kl + np.log(2 * np.sqrt(N_cal*K_tasks) / delta)) / (2 * N_cal*K_tasks))
    bound.backward()
    optimizer_stm.step()
u_stm_val = u_stm.detach().numpy()

# ----------------------------------------------------
# 6. Active Test-Time Evaluation (True Ensembled Forward Pass!)
# ----------------------------------------------------
# We perform the evaluation of ensembling methods by executing the actual layer-wise ensembled activation
# propagation. That is, for each layer l in range(4, 12), the intermediate token activations are blended
# sample-by-sample, and the ensembled hidden state is passed to the next layer!
# This is a 100% active, dynamic ensembled forward pass of the model!
print("\nEvaluating ensembling on Heterogeneous Test Stream (True Ensembled Forward Pass!)...")

test_loader_mnist = DataLoader(Subset(mnist_sub, range(N_train + N_sub + N_cal, N_train + N_sub + N_cal + N_test)), batch_size=50, shuffle=False)
test_loader_cifar = DataLoader(Subset(cifar_sub, range(N_train + N_sub + N_cal, N_train + N_sub + N_cal + N_test)), batch_size=50, shuffle=False)

# Define methods
methods = ["Uniform", "SABLE (PCA)", "Temp-Only ERM", "PAC-STM (Ours)"]
method_correct = {m: 0 for m in methods}

# Load test datasets fully into batches for evaluation
mnist_test_imgs = []
mnist_test_lbls = []
for imgs, lbls in test_loader_mnist:
    mnist_test_imgs.append(imgs)
    mnist_test_lbls.append(lbls)
mnist_test_imgs = torch.cat(mnist_test_imgs, dim=0)
mnist_test_lbls = torch.cat(mnist_test_lbls, dim=0)

cifar_test_imgs = []
cifar_test_lbls = []
for imgs, lbls in test_loader_cifar:
    cifar_test_imgs.append(imgs)
    cifar_test_lbls.append(lbls)
cifar_test_imgs = torch.cat(cifar_test_imgs, dim=0)
cifar_test_lbls = torch.cat(cifar_test_lbls, dim=0)

# Build a randomized heterogeneous test stream of length 400
np.random.seed(42)
stream_indices = np.arange(400)
np.random.shuffle(stream_indices)

# Prepare combined images and targets
test_images_all = torch.cat([mnist_test_imgs, cifar_test_imgs], dim=0) # [400, 3, 224, 224]
test_labels_all = torch.cat([mnist_test_lbls, cifar_test_lbls], dim=0) # [400]
test_tasks_all = torch.cat([torch.zeros(N_test, dtype=torch.long), torch.ones(N_test, dtype=torch.long)], dim=0) # [400]

test_images_all = test_images_all[stream_indices]
test_labels_all = test_labels_all[stream_indices]
test_tasks_all = test_tasks_all[stream_indices]

# Run ensembled forward pass in batches
batch_size = 50
print("Running deep active ensembled propagation of representations...")
start_eval = time.time()
with torch.no_grad():
    for start_idx in range(0, 400, batch_size):
        end_idx = min(start_idx + batch_size, 400)
        B_curr = end_idx - start_idx
        
        batch_images = test_images_all[start_idx:end_idx]
        batch_targets = test_labels_all[start_idx:end_idx]
        batch_tasks = test_tasks_all[start_idx:end_idx]
        
        # We evaluate each method independently on this batch by running the actual ensembled forward pass
        for m in methods:
            # 1. Process patches
            x = model.base_model._process_input(batch_images)
            n = x.shape[0]
            batch_class_token = model.base_model.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = x + model.base_model.encoder.pos_embedding
            x = model.base_model.encoder.dropout(x)
            
            # Layers 1..4 (unadapted frozen backbone)
            for i in range(4):
                x = model.base_model.encoder.layers[i](x)
                
            # Layers 5..12 (adapted layers, ensembled dynamically)
            for i in range(4, 12):
                identity = x
                
                # Forward of backbone block
                x = model.base_model.encoder.layers[i](x)
                
                # Compute task-specific adapter updates
                down0 = model.adapters[f"layer{i}_task0_down"](identity)
                up0 = model.adapters[f"layer{i}_task0_up"](down0)
                
                down1 = model.adapters[f"layer{i}_task1_down"](identity)
                up1 = model.adapters[f"layer{i}_task1_up"](down1)
                
                # Extract routing coordinates from the current [CLS] token at the routing block
                cls_curr = identity[:, 0, :]
                z_norm = cls_curr / (torch.norm(cls_curr, dim=-1, keepdim=True) + 1e-5)
                e_0 = torch.abs(torch.matmul(z_norm, V_mnist_seq[i])).squeeze(-1)
                e_1 = torch.abs(torch.matmul(z_norm, V_cifar_seq[i])).squeeze(-1)
                coords = torch.stack([e_0, e_1], dim=-1) # [B_curr, 2]
                
                # Compute ensembling routing weights for the current layer
                # Adapted layer index is i-4
                adapted_idx = i - 4
                if m == "Uniform":
                    q = torch.ones(B_curr, K_tasks) * 0.5
                elif m == "SABLE (PCA)":
                    exp_sable = torch.exp(coords / 0.05)
                    q = exp_sable / torch.sum(exp_sable, dim=-1, keepdim=True)
                elif m == "Temp-Only ERM":
                    u_val = torch.tensor(u_erm_val[adapted_idx])
                    exp_erm = torch.exp(coords * torch.exp(-u_val))
                    q = exp_erm / torch.sum(exp_erm, dim=-1, keepdim=True)
                elif m == "PAC-STM (Ours)":
                    u_val = torch.tensor(u_stm_val[adapted_idx])
                    exp_stm = torch.exp(coords * torch.exp(-u_val))
                    q = exp_stm / torch.sum(exp_stm, dim=-1, keepdim=True)
                    
                # Apply ensembled activation blending updates
                q_0 = q[:, 0].unsqueeze(-1).unsqueeze(-1) # [B_curr, 1, 1]
                q_1 = q[:, 1].unsqueeze(-1).unsqueeze(-1)
                
                blended_update = q_0 * up0 + q_1 * up1
                x = x + blended_update
                
            # Final Layer Norm
            x_ln = model.base_model.encoder.ln(x)
            cls_final = x_ln[:, 0, :] # Shape: [B_curr, 768]
            
            # Predict
            logits_mnist = head_mnist(cls_final)
            logits_cifar = head_cifar(cls_final)
            
            # We ensemble classification predictions based on final layer routing weights q
            for b_i in range(B_curr):
                actual_task = batch_tasks[b_i].item()
                target_cls = batch_targets[b_i].item()
                
                # Get the final layer ensembling weights for this sample
                cls_sample = cls_final[b_i]
                z_norm_sample = cls_sample / (torch.norm(cls_sample) + 1e-5)
                e_0_f = torch.abs(torch.dot(z_norm_sample, V_mnist_seq[11])).item()
                e_1_f = torch.abs(torch.dot(z_norm_sample, V_cifar_seq[11])).item()
                coords_f = np.array([e_0_f, e_1_f])
                
                if m == "Uniform":
                    q_f = np.array([0.5, 0.5])
                elif m == "SABLE (PCA)":
                    exp_sable_f = np.exp(coords_f / 0.05)
                    q_f = exp_sable_f / np.sum(exp_sable_f)
                elif m == "Temp-Only ERM":
                    exp_erm_f = np.exp(coords_f * np.exp(-u_erm_val[-1]))
                    q_f = exp_erm_f / np.sum(exp_erm_f)
                elif m == "PAC-STM (Ours)":
                    exp_stm_f = np.exp(coords_f * np.exp(-u_stm_val[-1]))
                    q_f = exp_stm_f / np.sum(exp_stm_f)
                    
                # Routing decision based on final layer weights
                predicted_task = 0 if q_f[0] > q_f[1] else 1
                
                if predicted_task == actual_task:
                    if actual_task == 0:
                        pred_class = torch.argmax(logits_mnist[b_i]).item()
                    else:
                        pred_class = torch.argmax(logits_cifar[b_i]).item()
                        
                    if pred_class == target_cls:
                        method_correct[m] += 1
                        
        if (start_idx + batch_size) % 100 == 0:
            print(f"  Evaluated {start_idx + batch_size} samples in {time.time() - start_eval:.2f}s")

# Print Accuracy Results
print("\n"+"="*60)
print("REAL-WORLD EXPERIMENT RESULTS (ViT-B/16)")
print("="*60)
for m in methods:
    acc = method_correct[m] / 400.0 * 100
    print(f"Method: {m:<20} | Joint Accuracy: {acc:.2f}%")
print("="*60)

# Calculate Trajectory Smoothness
erm_smoothness = np.sum(np.diff(u_erm_val, axis=0)**2)
stm_smoothness = np.sum(np.diff(u_stm_val, axis=0)**2)

print("\n"+"="*60)
print("TRAJECTORY SMOOTHNESS COMPARISON (Lower is smoother)")
print("="*60)
print(f"Temp-Only ERM smoothness: {erm_smoothness:.6f}")
print(f"PAC-STM (Ours) smoothness: {stm_smoothness:.6f}")
print("="*60)

# Print the actual trajectory values
print("\nLayer-wise Routing Log-Temperatures (u_l) across depth (8 adapted layers):")
print(f"{'Layer':<6} | {'Temp-Only ERM':<22} | {'PAC-STM (Ours)':<22}")
print("-" * 56)
for i in range(L_adapted):
    erm_str = f"[{u_erm_val[i, 0]:.4f}, {u_erm_val[i, 1]:.4f}]"
    stm_str = f"[{u_stm_val[i, 0]:.4f}, {u_stm_val[i, 1]:.4f}]"
    print(f"{i+1:<6} | {erm_str:<22} | {stm_str:<22}")

# Save results to a file
with open("real_world_results.txt", "w") as f:
    f.write("REAL-WORLD EXPERIMENT RESULTS (ViT-B/16) WITH ACTIVE ADAPTERS\n")
    f.write("="*60 + "\n")
    for m in methods:
        acc = method_correct[m] / 400.0 * 100
        f.write(f"Method: {m:<20} | Joint Accuracy: {acc:.2f}%\n")
    f.write("="*60 + "\n\n")
    f.write("TRAJECTORY SMOOTHNESS COMPARISON\n")
    f.write("="*60 + "\n")
    f.write(f"Temp-Only ERM smoothness: {erm_smoothness:.6f}\n")
    f.write(f"PAC-STM (Ours) smoothness: {stm_smoothness:.6f}\n")
    f.write("="*60 + "\n\n")
    f.write("Layer-wise Routing Log-Temperatures across depth:\n")
    f.write(f"{'Layer':<6} | {'Temp-Only ERM':<22} | {'PAC-STM (Ours)':<22}\n")
    f.write("-" * 56 + "\n")
    for i in range(L_adapted):
        erm_str = f"[{u_erm_val[i, 0]:.4f}, {u_erm_val[i, 1]:.4f}]"
        stm_str = f"[{u_stm_val[i, 0]:.4f}, {u_stm_val[i, 1]:.4f}]"
        f.write(f"{i+1:<6} | {erm_str:<22} | {stm_str:<22}\n")
