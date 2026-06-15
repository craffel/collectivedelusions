import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import copy
from torch.utils.data import DataLoader
from run_experiments_optimized import OFSTuneModel, ExpertHeadsWrapper, get_calibration_dataset, get_dataset, evaluate_stream, TensorDataset
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Load base model
print("Loading base model...")
base_model = timm.create_model('vit_tiny_patch16_224', pretrained=True).to(device)
base_model.reset_classifier(0)

# Load experts
datasets_list = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
experts = []
expert_heads = []

for ds in datasets_list:
    model_path = f'checkpoints/{ds.lower()}_expert.pth'
    expert = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    expert.head = nn.Linear(expert.head.in_features, 10)
    expert.load_state_dict(torch.load(model_path, map_location='cpu'))
    expert = expert.to(device)

    expert_heads.append(copy.deepcopy(expert.head))
    expert.reset_classifier(0)
    experts.append(expert)

# Load calibration dataset
cal_dataset = get_calibration_dataset()
cal_loader = DataLoader(cal_dataset, batch_size=16, shuffle=True)

# Create test loader
test_images = []
test_labels = []
test_task_indices = []
for task_idx, ds in enumerate(datasets_list):
    test_ds = get_dataset(ds, split='test')
    indices = list(range(len(test_ds)))
    random.seed(42)
    random.shuffle(indices)
    indices = indices[:50]  # Just 50 samples per task for quick evaluation
    for idx in indices:
        img, label = test_ds[idx]
        test_images.append(img)
        test_labels.append(label)
        test_task_indices.append(task_idx)

shuffled_indices = list(range(len(test_images)))
random.shuffle(shuffled_indices)
test_loader = DataLoader(
    TensorDataset(torch.stack(test_images)[shuffled_indices], torch.tensor(test_labels)[shuffled_indices], torch.tensor(test_task_indices)[shuffled_indices]),
    batch_size=16, shuffle=False
)

class OFSTuneLinear48(nn.Module):
    def __init__(self, base_linear, expert_linears, k_tasks, layer_idx, model_ref):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.bias = base_linear.bias
        self.layer_idx = layer_idx

        self.register_buffer('base_weight', base_linear.weight.data.clone())

        task_vectors = []
        for expert in expert_linears:
            task_vectors.append(expert.weight.data.clone() - self.base_weight)
        self.register_buffer('task_vectors', torch.stack(task_vectors, dim=0)) # [K, D_out, D_in]

        object.__setattr__(self, 'model_ref', model_ref)

    def forward(self, x):
        # Retrieve alphas, clamp them to [0, 1]
        alphas_clamped = torch.clamp(self.model_ref.alphas[self.layer_idx], 0.0, 1.0) # [K]
        delta_W = torch.einsum('k,koi->oi', alphas_clamped, self.task_vectors) # [D_out, D_in]
        W_merged = self.base_weight + delta_W # [D_out, D_in]
        return F.linear(x, W_merged, self.bias)

# Custom OFSTuneModel with 48 layer-wise parameters
class OFSTuneModel48(nn.Module):
    def __init__(self, base_model, experts, k_tasks):
        super().__init__()
        self.base_model = copy.deepcopy(base_model)
        self.k_tasks = k_tasks
        
        # 12 blocks, 4 linear layers each -> 48 linear layers in total
        self.alphas = nn.Parameter(torch.ones(48, k_tasks) * 0.3)
        
        # Replace linear layers
        self.replace_linear_layers(self.base_model, experts)
        
        # FREEZE base model parameters!
        for p in self.base_model.parameters():
            p.requires_grad = False
            
    def replace_linear_layers(self, model, experts, inside_blocks=False, target_count=None):
        if target_count is None:
            target_count = [0]
            
        for name, child in model.named_children():
            is_blocks = inside_blocks or (name == 'blocks')
            if isinstance(child, nn.Linear) and is_blocks:
                from run_experiments_optimized import get_module_by_name
                expert_linears = [get_module_by_name(exp, name) for exp in experts]
                new_linear = OFSTuneLinear48(child, expert_linears, self.k_tasks, target_count[0], self)
                setattr(model, name, new_linear)
                target_count[0] += 1
            else:
                self.replace_linear_layers(child, [getattr(exp, name) for exp in experts], is_blocks, target_count)

    def forward(self, x):
        return self.base_model(x)

ofs_model = OFSTuneModel48(base_model, experts, k_tasks=4).to(device)
wrapped_ofs = ExpertHeadsWrapper(ofs_model, expert_heads).to(device)

print("Check trainable parameters:")
for name, p in ofs_model.named_parameters():
    if p.requires_grad:
        print(f"  {name}: {p.shape}")

optimizer = torch.optim.Adam(ofs_model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()

ofs_model.train()
for step in range(100):
    for images, labels, task_indices in cal_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        features = ofs_model(images)
        outputs = []
        for b in range(images.size(0)):
            task_idx = task_indices[b].item()
            out = wrapped_ofs.heads[task_idx](features[b:b+1])
            outputs.append(out)
        outputs = torch.cat(outputs, dim=0)
        loss = criterion(outputs, labels) * images.size(0)
        loss.backward()
        optimizer.step()

# Evaluate
acc = evaluate_stream(wrapped_ofs, test_loader, device)
print(f"Frozen 48-parameter OFS-Tune accuracy: {acc:.2f}%")
print("Alphas mean after training:", ofs_model.alphas.mean().item(), "std:", ofs_model.alphas.std().item())
