import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm

# Robust CUDA & cuDNN configuration to prevent initialization errors
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False
    print("Disabled cuDNN to ensure stable CUDA execution.")

# Self-contained SAM optimizer implementation
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p.device)
                p.add_(e_w)  # climb to the local maximum
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # restore original parameters
        self.base_optimizer.step()  # actual gradient update
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        raise NotImplementedError("SAM requires separate first_step and second_step calls.")

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

# Custom self-contained LoRA layer for Conv2D
class LoRAConv2d(nn.Module):
    def __init__(self, original_conv, r=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.original = original_conv
        # Freeze original convolution weights
        for p in self.original.parameters():
            p.requires_grad = False
            
        in_channels = original_conv.in_channels
        out_channels = original_conv.out_channels
        kernel_size = original_conv.kernel_size
        stride = original_conv.stride
        padding = original_conv.padding
        
        # LoRA down-projection and up-projection
        self.lora_A = nn.Conv2d(in_channels, r, kernel_size, stride=stride, padding=padding, bias=False)
        self.lora_B = nn.Conv2d(r, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.scaling = lora_alpha / r
        self.dropout = nn.Dropout2d(lora_dropout)
        
        # Initialize weights (A: Kaiming uniform, B: Zeros)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        original_output = self.original(x)
        lora_output = self.lora_B(self.lora_A(self.dropout(x)))
        return original_output + lora_output * self.scaling

def apply_lora_to_resnet(model, r=8, lora_alpha=16, lora_dropout=0.1):
    # Apply LoRA to standard ResNet-18 Conv layers in blocks layer3 and layer4
    for layer_name in ["layer3", "layer4"]:
        layer = getattr(model, layer_name)
        for i in range(len(layer)):
            # Replace conv1 and conv2
            layer[i].conv1 = LoRAConv2d(layer[i].conv1, r, lora_alpha, lora_dropout)
            layer[i].conv2 = LoRAConv2d(layer[i].conv2, r, lora_alpha, lora_dropout)
    return model

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data Transformations
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load CIFAR-10
print("Loading CIFAR-10...")
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Task Splitting: Task A (classes 0-4), Task B (classes 5-9)
def get_task_subsets(dataset, classes):
    indices = [i for i, (_, label) in enumerate(dataset) if label in classes]
    return Subset(dataset, indices)

task_A_classes = list(range(5))
task_B_classes = list(range(5, 10))

train_subset_A = get_task_subsets(train_dataset, task_A_classes)
train_subset_B = get_task_subsets(train_dataset, task_B_classes)

test_subset_A = get_task_subsets(test_dataset, task_A_classes)
test_subset_B = get_task_subsets(test_dataset, task_B_classes)

train_loader_A = DataLoader(train_subset_A, batch_size=128, shuffle=True, num_workers=2)
train_loader_B = DataLoader(train_subset_B, batch_size=128, shuffle=True, num_workers=2)

test_loader_A = DataLoader(test_subset_A, batch_size=128, shuffle=False, num_workers=2)
test_loader_B = DataLoader(test_subset_B, batch_size=128, shuffle=False, num_workers=2)

print(f"Task A (0-4) - Train size: {len(train_subset_A)}, Test size: {len(test_subset_A)}")
print(f"Task B (5-9) - Train size: {len(train_subset_B)}, Test size: {len(test_subset_B)}")

def train_expert(task_name, train_loader, test_loader, num_epochs=3, use_sam=False):
    print(f"\n--- Training Expert for {task_name} (SAM={use_sam}) ---")
    
    # Load pre-trained ResNet-18
    model = models.resnet18(pretrained=True)
    # Apply custom LoRA
    model = apply_lora_to_resnet(model, r=8, lora_alpha=16, lora_dropout=0.1)
    
    # Set classification head to match 10 classes
    model.fc = nn.Linear(512, 10)
    model.to(device)
    
    # Set only LoRA parameters and classification head as trainable
    trainable_params = []
    for name, param in model.named_parameters():
        if "lora" in name or "fc" in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
            
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params if p.requires_grad)}")
    
    criterion = nn.CrossEntropyLoss()
    
    if use_sam:
        base_opt = torch.optim.AdamW
        optimizer = SAM(trainable_params, base_opt, rho=0.05, lr=5e-4, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=5e-4, weight_decay=1e-4)
        
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            if use_sam:
                # First pass (loss & gradient)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                # Second pass (perturbed gradient & update)
                outputs_perturbed = model(images)
                loss_perturbed = criterion(outputs_perturbed, labels)
                loss_perturbed.backward()
                optimizer.second_step(zero_grad=True)
                
                running_loss += loss.item()
            else:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100.0 * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        val_acc = 100.0 * val_correct / val_total
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(train_loader):.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")

    # Save checkpoint
    os.makedirs("./checkpoints", exist_ok=True)
    suffix = "_sam" if use_sam else "_standard"
    
    # Save the custom LoRA state dict (only the trainable parameters)
    lora_state_dict = {k: v for k, v in model.state_dict().items() if "lora" in k}
    torch.save(lora_state_dict, f"./checkpoints/lora_{task_name}{suffix}.pt")
    torch.save(model.fc.state_dict(), f"./checkpoints/head_{task_name}{suffix}.pt")
    print(f"Expert {task_name}{suffix} saved successfully!")

# Train experts
# We train Standard Experts
train_expert("A", train_loader_A, test_loader_A, num_epochs=3, use_sam=False)
train_expert("B", train_loader_B, test_loader_B, num_epochs=3, use_sam=False)

# We train SAM Experts
train_expert("A", train_loader_A, test_loader_A, num_epochs=3, use_sam=True)
train_expert("B", train_loader_B, test_loader_B, num_epochs=3, use_sam=True)
