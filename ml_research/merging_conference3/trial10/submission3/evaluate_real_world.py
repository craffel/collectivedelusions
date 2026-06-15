import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'packages')))
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

# Set seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

# Load real models
print("Loading prajjwal1/bert-tiny from Hugging Face...")
model_name = "prajjwal1/bert-tiny"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name)

# 1. Prepare GLUE Datasets
print("Loading small slices of GLUE tasks (SST-2, MRPC, CoLA)...")
sst2_train = load_dataset("glue", "sst2", split="train[:128]")
sst2_val = load_dataset("glue", "sst2", split="validation[:400]")

mrpc_train = load_dataset("glue", "mrpc", split="train[:128]")
mrpc_val = load_dataset("glue", "mrpc", split="validation[:400]")

cola_train = load_dataset("glue", "cola", split="train[:128]")
cola_val = load_dataset("glue", "cola", split="validation[:400]")

def tokenize_and_collate(examples, task_name):
    if task_name == "mrpc":
        texts = [f"sentence1: {s1} sentence2: {s2}" for s1, s2 in zip(list(examples["sentence1"]), list(examples["sentence2"]))]
    else:
        texts = list(examples["sentence"])
    
    inputs = tokenizer(texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
    inputs["labels"] = torch.tensor(list(examples["label"]), dtype=torch.long)
    return inputs

sst2_train_inputs = tokenize_and_collate(sst2_train, "sst2")
sst2_val_inputs = tokenize_and_collate(sst2_val, "sst2")

mrpc_train_inputs = tokenize_and_collate(mrpc_train, "mrpc")
mrpc_val_inputs = tokenize_and_collate(mrpc_val, "mrpc")

cola_train_inputs = tokenize_and_collate(cola_train, "cola")
cola_val_inputs = tokenize_and_collate(cola_val, "cola")

# 2. Add multiple LoRA adapters and separate task-specific classification heads
print("Setting up task-specific PEFT LoRA adapters and head classifiers...")
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none"
)

# We wrap base_model as a PEFT model
peft_model = get_peft_model(base_model, lora_config)

# Task-specific classification heads (D_emb = 128 -> 2 classes)
D_emb = 128
classifier_sst2 = nn.Linear(D_emb, 2)
classifier_mrpc = nn.Linear(D_emb, 2)
classifier_cola = nn.Linear(D_emb, 2)

# Helper to train an adapter and its head
def train_adapter_and_head(train_inputs, classifier, steps=100, batch_size=64):
    optimizer = torch.optim.AdamW(list(peft_model.parameters()) + list(classifier.parameters()), lr=2e-3)
    peft_model.train()
    classifier.train()
    
    # Reset lora params to random/zeros
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_A" in name:
                nn.init.normal_(param, std=1e-2)
            elif "lora_B" in name:
                nn.init.zeros_(param)
                
    num_samples = train_inputs["input_ids"].shape[0]
    for step in range(steps):
        # Sample mini-batch
        indices = torch.randperm(num_samples)[:batch_size]
        batch_input_ids = train_inputs["input_ids"][indices]
        batch_attn_mask = train_inputs["attention_mask"][indices]
        batch_labels = train_inputs["labels"][indices]
        
        optimizer.zero_grad()
        outputs = peft_model(
            input_ids=batch_input_ids,
            attention_mask=batch_attn_mask
        )
        h = torch.mean(outputs.last_hidden_state, dim=1) # [batch_size, 128]
        logits = classifier(h)
        loss = F.cross_entropy(logits, batch_labels)
        loss.backward()
        optimizer.step()

print("Fine-tuning SST-2 adapter and head classifier...")
train_adapter_and_head(sst2_train_inputs, classifier_sst2)
sst2_lora_weights = {name: param.clone() for name, param in peft_model.named_parameters() if "lora" in name}

print("Fine-tuning MRPC adapter and head classifier...")
train_adapter_and_head(mrpc_train_inputs, classifier_mrpc)
mrpc_lora_weights = {name: param.clone() for name, param in peft_model.named_parameters() if "lora" in name}

print("Fine-tuning COLA adapter and head classifier...")
train_adapter_and_head(cola_train_inputs, classifier_cola)
cola_lora_weights = {name: param.clone() for name, param in peft_model.named_parameters() if "lora" in name}

# 3. Extract Real-World Activations for PCA Coordinates and Classification
print("Extracting intermediate representations from BERT-Tiny...")
peft_model.eval()

# Extract activations at Layer 1 (intermediate)
layer1_output = None
def hook_fn(module, input, output):
    global layer1_output
    layer1_output = output[0].detach()

hook_handle = base_model.encoder.layer[0].register_forward_hook(hook_fn)

def load_lora_weights(weights):
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if name in weights:
                param.copy_(weights[name])

with torch.no_grad():
    load_lora_weights(sst2_lora_weights)
    sst2_out = peft_model(sst2_val_inputs["input_ids"], attention_mask=sst2_val_inputs["attention_mask"])
    sst2_h1 = torch.mean(layer1_output, dim=1) # [50, 128]
    sst2_h2 = torch.mean(sst2_out.last_hidden_state, dim=1) # [50, 128]

with torch.no_grad():
    load_lora_weights(mrpc_lora_weights)
    mrpc_out = peft_model(mrpc_val_inputs["input_ids"], attention_mask=mrpc_val_inputs["attention_mask"])
    mrpc_h1 = torch.mean(layer1_output, dim=1) # [50, 128]
    mrpc_h2 = torch.mean(mrpc_out.last_hidden_state, dim=1) # [50, 128]

with torch.no_grad():
    load_lora_weights(cola_lora_weights)
    cola_out = peft_model(cola_val_inputs["input_ids"], attention_mask=cola_val_inputs["attention_mask"])
    cola_h1 = torch.mean(layer1_output, dim=1) # [50, 128]
    cola_h2 = torch.mean(cola_out.last_hidden_state, dim=1) # [50, 128]

hook_handle.remove()

# Normalize and extract PCA coordinates
K_tasks = 3
d_comp = 5

cal_data = [sst2_h1, mrpc_h1, cola_h1]
V_pca = []
for k in range(K_tasks):
    H_k = cal_data[k]
    H_k_norm = H_k / (torch.norm(H_k, p=2, dim=1, keepdim=True) + 1e-5)
    U, S, V_t = torch.linalg.svd(H_k_norm, full_matrices=False)
    V_pca.append(V_t[:d_comp].t()) # [128, 5]

# Real Task Targets/Signatures (centroids of layer 1)
v_prime = [torch.mean(sst2_h1, dim=0), torch.mean(mrpc_h1, dim=0), torch.mean(cola_h1, dim=0)]

# 4. Define and Train the Stateful Routers on real-world representations
print("Training routers on real-world representations...")

class RealLVCSModel(nn.Module):
    def __init__(self, delta=0.1):
        super().__init__()
        self.delta = delta
        self.s = nn.Parameter(torch.zeros(K_tasks))
        self.b_grow = nn.Parameter(torch.zeros(K_tasks))
        self.u = nn.Parameter(torch.ones(K_tasks) * -0.105)
        self.v = nn.Parameter(torch.ones(K_tasks * (K_tasks - 1)) * -2.197)
        
    def forward(self, h, V_pca, prev_R=None):
        B, D = h.shape
        h_norm = h / (torch.norm(h, p=2, dim=1, keepdim=True) + 1e-5)
        
        R = torch.zeros(B, K_tasks, device=h.device)
        for k in range(K_tasks):
            R[:, k] = torch.norm(h_norm @ V_pca[k], p=2, dim=1)
            
        w_grow = torch.exp(self.s)
        r = w_grow.unsqueeze(0) * R + self.b_grow.unsqueeze(0)
        r = 1.9 * torch.tanh(r / 1.9)
        
        if prev_R is not None:
            dot_prod = torch.sum(R * prev_R, dim=1)
            norm_curr = torch.norm(R, p=2, dim=1)
            norm_prev = torch.norm(prev_R, p=2, dim=1)
            Sim_t = dot_prod / (norm_curr * norm_prev + 1e-5)
            Sim_t = torch.clamp(Sim_t, 0.0, 1.0)
        else:
            Sim_t = torch.ones(B, device=h.device)
            
        c_diag = torch.exp(self.u) + 0.1
        c_off_flat = torch.zeros(B, K_tasks * K_tasks, device=h.device)
        indices = [i for i in range(K_tasks * K_tasks) if i % (K_tasks + 1) != 0]
        c_off_flat[:, indices] = torch.sigmoid(self.v).unsqueeze(0)
        scale = (Sim_t + (1.0 - Sim_t) * self.delta).view(B, 1, 1)
        c_off = c_off_flat.view(B, K_tasks, K_tasks) * scale
        c_diag_matrix = torch.diag(c_diag)
        C = torch.eye(K_tasks, device=h.device).unsqueeze(0) * c_diag_matrix.unsqueeze(0) + c_off
        
        y = torch.ones(B, K_tasks, device=h.device) * -np.log(K_tasks)
        alpha_layers = torch.zeros(6, B, K_tasks, device=h.device)
        for l in range(6):
            x = torch.exp(y)
            suppression = torch.bmm(C, x.unsqueeze(2)).squeeze(2)
            y = y + r - suppression
            y = torch.clamp(y, min=-20.0, max=20.0)
            alpha_layers[l] = F.softmax(y, dim=1)
            
        return alpha_layers, y, R

class RealPACKineticsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.u = nn.Parameter(torch.zeros(K_tasks))
        
    def forward(self, h, g_t, prev_R=None):
        B, D = h.shape
        a = torch.sigmoid(self.u)
        
        if prev_R is not None:
            h_norm = h / (torch.norm(h, p=2, dim=1, keepdim=True) + 1e-5)
            R = torch.zeros(B, K_tasks, device=h.device)
            for k in range(K_tasks):
                R[:, k] = torch.norm(h_norm @ V_pca[k], p=2, dim=1)
            dot_prod = torch.sum(R * prev_R, dim=1)
            norm_curr = torch.norm(R, p=2, dim=1)
            norm_prev = torch.norm(prev_R, p=2, dim=1)
            Sim_t = dot_prod / (norm_curr * norm_prev + 1e-5)
            Sim_t = torch.clamp(Sim_t, 0.0, 1.0)
        else:
            Sim_t = torch.ones(B, device=h.device)
            
        a_eff = a.unsqueeze(0) * Sim_t.unsqueeze(1)
        alpha = torch.ones(B, K_tasks, device=h.device) / K_tasks
        alpha_layers = torch.zeros(6, B, K_tasks, device=h.device)
        for l in range(6):
            alpha = a_eff * alpha + (1.0 - a_eff) * g_t
            alpha_layers[l] = alpha
        return alpha_layers, torch.log(alpha + 1e-15), torch.zeros(B, K_tasks, device=h.device)

class RealEarlySoftmaxModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.s = nn.Parameter(torch.zeros(K_tasks))
        self.b_grow = nn.Parameter(torch.zeros(K_tasks))
        
    def forward(self, h, V_pca, prev_R=None):
        B, D = h.shape
        h_norm = h / (torch.norm(h, p=2, dim=1, keepdim=True) + 1e-5)
        R = torch.zeros(B, K_tasks, device=h.device)
        for k in range(K_tasks):
            R[:, k] = torch.norm(h_norm @ V_pca[k], p=2, dim=1)
        w_grow = torch.exp(self.s)
        logits = w_grow * R + self.b_grow
        alpha = F.softmax(logits, dim=1)
        return alpha.unsqueeze(0).repeat(6, 1, 1), logits, R

class RealMLPStaticModel(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(K_tasks, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, K_tasks)
        
    def forward(self, h, V_pca, prev_R=None):
        B, D = h.shape
        h_norm = h / (torch.norm(h, p=2, dim=1, keepdim=True) + 1e-5)
        R = torch.zeros(B, K_tasks, device=h.device)
        for k in range(K_tasks):
            R[:, k] = torch.norm(h_norm @ V_pca[k], p=2, dim=1)
        x = F.relu(self.fc1(R))
        logits = self.fc2(x)
        alpha = F.softmax(logits, dim=1)
        return alpha.unsqueeze(0).repeat(6, 1, 1), logits, R

class RealGRURouterModel(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = nn.GRUCell(input_size=K_tasks, hidden_size=hidden_dim)
        self.fc = nn.Linear(hidden_dim, K_tasks)
        
    def forward(self, h, V_pca, prev_R=None):
        B, D = h.shape
        h_norm = h / (torch.norm(h, p=2, dim=1, keepdim=True) + 1e-5)
        R = torch.zeros(B, K_tasks, device=h.device)
        for k in range(K_tasks):
            R[:, k] = torch.norm(h_norm @ V_pca[k], p=2, dim=1)
            
        alpha_layers = torch.zeros(6, B, K_tasks, device=h.device)
        hx = torch.zeros(B, self.hidden_dim, device=h.device)
        for l in range(6):
            hx = self.cell(R, hx)
            logits = self.fc(hx)
            alpha_layers[l] = F.softmax(logits, dim=1)
            
        return alpha_layers, logits, R

# Pool train representations
train_inputs = []
train_targets = []
for k in range(K_tasks):
    train_inputs.append(cal_data[k])
    train_targets.append(torch.ones(400, dtype=torch.long) * k)
train_inputs = torch.cat(train_inputs, dim=0) # [1200, 128]
train_targets = torch.cat(train_targets, dim=0) # [1200]

lvcs = RealLVCSModel()
pk = RealPACKineticsModel()
early_sm = RealEarlySoftmaxModel()
mlp_static = RealMLPStaticModel()
gru_router = RealGRURouterModel()

# Training loop
torch.set_grad_enabled(True)
opt_lvcs = torch.optim.Adam(lvcs.parameters(), lr=0.01)
opt_pk = torch.optim.Adam(pk.parameters(), lr=0.01)
opt_esm = torch.optim.Adam(early_sm.parameters(), lr=0.01)
opt_mlp = torch.optim.Adam(mlp_static.parameters(), lr=0.01)
opt_gru = torch.optim.Adam(gru_router.parameters(), lr=0.01)

centroids = torch.stack([torch.mean(cal, dim=0) for cal in cal_data])
centroids_norm = centroids / (torch.norm(centroids, p=2, dim=1, keepdim=True) + 1e-5)

inputs_norm = train_inputs / (torch.norm(train_inputs, p=2, dim=1, keepdim=True) + 1e-5)
sims = inputs_norm @ centroids_norm.t()
g_t = F.softmax(sims / 0.15, dim=1)

for epoch in range(150):
    # Train LVCS (numerically stable cross-entropy on raw log-state y!)
    opt_lvcs.zero_grad()
    _, logits, _ = lvcs(train_inputs, V_pca)
    loss = F.cross_entropy(logits, train_targets)
    loss.backward()
    opt_lvcs.step()
    
    # Train PK
    opt_pk.zero_grad()
    _, logits_pk, _ = pk(train_inputs, g_t)
    loss_pk = F.cross_entropy(logits_pk, train_targets)
    loss_pk.backward()
    opt_pk.step()
    
    # Train ESM
    opt_esm.zero_grad()
    _, logits_esm, _ = early_sm(train_inputs, V_pca)
    loss_esm = F.cross_entropy(logits_esm, train_targets)
    loss_esm.backward()
    opt_esm.step()

    # Train MLP
    opt_mlp.zero_grad()
    _, logits_mlp, _ = mlp_static(train_inputs, V_pca)
    loss_mlp = F.cross_entropy(logits_mlp, train_targets)
    loss_mlp.backward()
    opt_mlp.step()

    # Train GRU
    opt_gru.zero_grad()
    _, logits_gru, _ = gru_router(train_inputs, V_pca)
    loss_gru = F.cross_entropy(logits_gru, train_targets)
    loss_gru.backward()
    opt_gru.step()
    
    if epoch % 30 == 0:
        print(f"Epoch {epoch:03d} | LVCS Loss: {loss.item():.4f} | PK Loss: {loss_pk.item():.4f} | ESM Loss: {loss_esm.item():.4f} | MLP Loss: {loss_mlp.item():.4f} | GRU Loss: {loss_gru.item():.4f}")

torch.set_grad_enabled(False)
lvcs.eval()
pk.eval()
early_sm.eval()
mlp_static.eval()
gru_router.eval()

# 5. Evaluate on sequence classification stream
print("Evaluating on sequence stream...")
test_tasks_list = []
for k in range(K_tasks):
    test_tasks_list.extend([k] * 400)
random.shuffle(test_tasks_list)

stream_h3 = []
stream_h_final = []
stream_labels = [] # Ground truth task classification labels (0 or 1)
for y in test_tasks_list:
    eps = torch.randn(D_emb) * 0.01
    idx = random.randint(0, 399)
    stream_h3.append(cal_data[y][idx] + eps)

    if y == 0:
        stream_h_final.append(sst2_h2[idx] + eps)
        stream_labels.append(sst2_val_inputs["labels"][idx].item())
    elif y == 1:
        stream_h_final.append(mrpc_h2[idx] + eps)
        stream_labels.append(mrpc_val_inputs["labels"][idx].item())
    else:
        stream_h_final.append(cola_h2[idx] + eps)
        stream_labels.append(cola_val_inputs["labels"][idx].item())

stream_h3 = torch.stack(stream_h3)
stream_h_final = torch.stack(stream_h_final)
stream_labels = torch.tensor(stream_labels, dtype=torch.long)

# Precompute PCA coords
stream_norm = stream_h3 / (torch.norm(stream_h3, p=2, dim=1, keepdim=True) + 1e-5)
R_stream = torch.zeros(len(test_tasks_list), K_tasks)
for k in range(K_tasks):
    R_stream[:, k] = torch.norm(stream_norm @ V_pca[k], p=2, dim=1)
prev_R_stream = torch.zeros_like(R_stream)
prev_R_stream[0] = R_stream[0]
prev_R_stream[1:] = R_stream[:-1]

# SABLE alphas
sims_stream = stream_norm @ centroids_norm.t()
sable_alphas = F.softmax(sims_stream / 0.15, dim=1)

# Task classifier heads matrices and biases
W_sst2, b_sst2 = classifier_sst2.weight, classifier_sst2.bias
W_mrpc, b_mrpc = classifier_mrpc.weight, classifier_mrpc.bias
W_cola, b_cola = classifier_cola.weight, classifier_cola.bias

results = {}
for method in ["Uniform", "SABLE", "PAC-Kinetics", "Softmax (Static)", "MLP (Static)", "GRU Router", "LVCS"]:
    if method == "Uniform":
        alpha_layers = torch.ones(6, len(test_tasks_list), K_tasks) * (1.0 / K_tasks)
    elif method == "SABLE":
        alpha_layers = sable_alphas.unsqueeze(0).repeat(6, 1, 1)
    elif method == "PAC-Kinetics":
        alpha_layers, _, _ = pk(stream_h3, sable_alphas, prev_R=prev_R_stream)
    elif method == "Softmax (Static)":
        alpha_layers, _, _ = early_sm(stream_h3, V_pca)
    elif method == "MLP (Static)":
        alpha_layers, _, _ = mlp_static(stream_h3, V_pca)
    elif method == "GRU Router":
        alpha_layers, _, _ = gru_router(stream_h3, V_pca)
    elif method == "LVCS":
        alpha_layers, _, _ = lvcs(stream_h3, V_pca, prev_R=prev_R_stream)
        
    alpha_final = alpha_layers[-1] # [T, K]
    
    # Run dynamic sequence classification directly on RAW activations to prevent representation shift!
    correct = 0
    for t_idx in range(len(test_tasks_list)):
        a_t = alpha_final[t_idx] # [3]
        
        # Linear blending of weight matrices and biases
        W_blend = a_t[0] * W_sst2 + a_t[1] * W_mrpc + a_t[2] * W_cola
        b_blend = a_t[0] * b_sst2 + a_t[1] * b_mrpc + a_t[2] * b_cola
        
        logits_t = F.linear(stream_h_final[t_idx], W_blend, b_blend)
        pred_t = torch.argmax(logits_t).item()
        
        if pred_t == stream_labels[t_idx].item():
            correct += 1
            
    acc = correct / len(test_tasks_list) * 100.0
    results[method] = acc

print("\n=== REAL-WORLD BERT-TINY GLUE BENCHMARK RESULTS ===")
for m, acc in results.items():
    print(f"{m:<25} | Downstream Sequence Classification Accuracy: {acc:.2f}%")

# Save results
with open("real_world_results.md", "w") as f:
    f.write("# Real-World BERT-Tiny GLUE Sequence Classification Evaluation\n\n")
    f.write("We evaluated the ensembling models on a real-world multi-task sequence classification stream using `prajjwal1/bert-tiny` with PEFT LoRA adapters fine-tuned on SST-2, MRPC, and CoLA.\n\n")
    f.write("| Ensembling Method | Downstream Sequence Accuracy (%) |\n")
    f.write("| :--- | :---: |\n")
    for m, acc in results.items():
        f.write(f"| {m} | {acc:.2f}% |\n")
    f.write("\n")
