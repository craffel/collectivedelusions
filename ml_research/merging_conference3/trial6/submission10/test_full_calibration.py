import torch
import os
import timm
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.expanduser("~/data")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

grayscale_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_task_dataset(name, train=True):
    if name == "MNIST":
        return datasets.MNIST(root=DATA_DIR, train=train, download=False, transform=grayscale_transform)
    elif name == "FashionMNIST":
        return datasets.FashionMNIST(root=DATA_DIR, train=train, download=False, transform=grayscale_transform)
    elif name == "CIFAR10":
        return datasets.CIFAR10(root=DATA_DIR, train=train, download=False, transform=transform)
    elif name == "SVHN":
        split = "train" if train else "test"
        return datasets.SVHN(root=DATA_DIR, split=split, download=False, transform=transform)

tasks = ["MNIST", "FashionMNIST", "CIFAR10", "SVHN"]
experts = {}

print("--- Step 1: Loading Task Experts ---")
for task in tasks:
    checkpoint_path = f"checkpoints/{task}_expert.pt"
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    experts[task] = model

print("Loading base pre-trained model...")
base_model = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=10)
with torch.no_grad():
    base_model.head.weight.zero_()
    base_model.head.bias.zero_()
base_model.to(DEVICE)

# Prepare calibration images and test images
calibration_images = {}
calibration_labels = {}
test_images = {}
test_labels = {}

for task in tasks:
    full_train_dataset = get_task_dataset(task, train=True)
    cal_indices = list(range(1000, 1016))
    
    cal_imgs_list = []
    cal_labels_list = []
    for idx in cal_indices:
        img, label = full_train_dataset[idx]
        cal_imgs_list.append(img)
        cal_labels_list.append(label)
        
    calibration_images[task] = torch.stack(cal_imgs_list).to(DEVICE)
    calibration_labels[task] = torch.tensor(cal_labels_list).to(DEVICE)
    
    full_test_dataset = get_task_dataset(task, train=False)
    test_indices = list(range(250))
    
    test_imgs_list = []
    test_labels_list = []
    for idx in test_indices:
        img, label = full_test_dataset[idx]
        test_imgs_list.append(img)
        test_labels_list.append(label)
        
    test_images[task] = torch.stack(test_imgs_list).to(DEVICE)
    test_labels[task] = torch.tensor(test_labels_list).to(DEVICE)

base_state = base_model.state_dict()
expert_states = {task: experts[task].state_dict() for task in tasks}

task_vectors = {task: {} for task in tasks}
for task in tasks:
    for key in base_state.keys():
        if base_state[key].shape == expert_states[task][key].shape:
            task_vectors[task][key] = expert_states[task][key] - base_state[key]
        else:
            task_vectors[task][key] = expert_states[task][key]

global_eval_model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10).to(DEVICE)
global_eval_model.eval()

def set_merged_weights(alphas):
    merged_state = {}
    for key in base_state.keys():
        merged_state[key] = base_state[key] + \
                            alphas[0] * task_vectors[tasks[0]][key] + \
                            alphas[1] * task_vectors[tasks[1]][key] + \
                            alphas[2] * task_vectors[tasks[2]][key] + \
                            alphas[3] * task_vectors[tasks[3]][key]
    global_eval_model.load_state_dict(merged_state)

def evaluate_alphas_on_calibration(alphas):
    set_merged_weights(alphas)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    with torch.no_grad():
        for task in tasks:
            imgs = calibration_images[task]
            labels = calibration_labels[task]
            outputs = global_eval_model(imgs)
            loss = criterion(outputs, labels).item()
            total_loss += loss
    return total_loss / 4.0

def evaluate_alphas_on_test(alphas):
    set_merged_weights(alphas)
    accuracies = {}
    with torch.no_grad():
        for task in tasks:
            correct = 0
            imgs = test_images[task]
            labels = test_labels[task]
            outputs = global_eval_model(imgs)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            accuracies[task] = 100.0 * correct / 250.0
    return accuracies

# Evaluate uniform first
print("\nEvaluating Uniform Merge [0.3, 0.3, 0.3, 0.3]:")
uniform_alphas = [0.3, 0.3, 0.3, 0.3]
cal_loss = evaluate_alphas_on_calibration(uniform_alphas)
test_accs = evaluate_alphas_on_test(uniform_alphas)
print(f"Calibration Loss: {cal_loss:.4f} | Test Accs: {test_accs} | Joint Mean: {np.mean(list(test_accs.values())):.2f}%")

# Let's run a simple random search / grid search or local coordinate descent to find the best alphas
# that minimize the calibration loss!
best_alphas = [0.3, 0.3, 0.3, 0.3]
best_loss = cal_loss

# Run simple coordinate search
print("\nRunning local coordinate search for optimal alphas...")
step_sizes = [0.1, 0.05, 0.01, 0.005]
for step in step_sizes:
    for i in range(4):
        # try increasing
        candidate = list(best_alphas)
        candidate[i] = min(0.3, candidate[i] + step)
        loss = evaluate_alphas_on_calibration(candidate)
        if loss < best_loss:
            best_loss = loss
            best_alphas = candidate
            
        # try decreasing
        candidate = list(best_alphas)
        candidate[i] = max(0.0, candidate[i] - step)
        loss = evaluate_alphas_on_calibration(candidate)
        if loss < best_loss:
            best_loss = loss
            best_alphas = candidate

test_accs_best = evaluate_alphas_on_test(best_alphas)
print(f"\nBest Alphas found on Calibration set: {best_alphas}")
print(f"Calibration Loss: {best_loss:.4f}")
print(f"Test Accs: {test_accs_best}")
print(f"Joint Mean: {np.mean(list(test_accs_best.values())):.2f}%")
