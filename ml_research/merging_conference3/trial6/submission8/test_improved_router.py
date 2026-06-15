import torch
import torch.nn as nn
import torch.optim as optim
from run_physical_validation import SimpleCNN, load_experts, build_data_splits, TASKS, blend_parameters_functional

class ImprovedPhysicalRoutingHead(nn.Module):
    def __init__(self, in_features=32, hidden_features=16, num_tasks=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, num_tasks)
        )
        
    def forward(self, x):
        return self.net(x)

def extract_improved_features(base_model, params, data):
    features = []
    for idx in range(len(data)):
        img = data[idx:idx+1]
        conv1_feat = torch.func.functional_call(base_model.features[0:2], {
            '0.weight': params['features.0.weight'],
            '0.bias': params['features.0.bias']
        }, img)
        mean_feat = conv1_feat.mean(dim=[2, 3]).view(-1)
        std_feat = conv1_feat.std(dim=[2, 3]).view(-1)
        combined_feat = torch.cat([mean_feat, std_feat], dim=0)
        features.append(combined_feat)
    return torch.stack(features)

def train_improved_router(experts, cal_data, cal_labels, cal_task_ids, num_epochs=300, lr=5e-3):
    base_model = SimpleCNN()
    uniform_params = blend_parameters_functional(experts, torch.tensor([0.25, 0.25, 0.25, 0.25]), k=0)
    
    with torch.no_grad():
        h_features = extract_improved_features(base_model, uniform_params, cal_data)
        
    mean = h_features.mean(dim=0, keepdim=True)
    std = h_features.std(dim=0, keepdim=True) + 1e-6
    norm_features = (h_features - mean) / std
    
    head = ImprovedPhysicalRoutingHead(in_features=32, hidden_features=16, num_tasks=4)
    optimizer = optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    head.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = head(norm_features)
        loss = criterion(out, cal_task_ids)
        loss.backward()
        optimizer.step()
        
    return head, mean, std

def evaluate_improved_routing(experts, head, mean, std, test_data, test_labels, k=4, T=0.1):
    head.eval()
    results = {}
    
    base_model = SimpleCNN()
    uniform_params = blend_parameters_functional(experts, torch.tensor([0.25, 0.25, 0.25, 0.25]), k=0)
    
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for task_id, task in enumerate(TASKS):
            imgs = test_data[task]
            lbls = test_labels[task]
            
            task_h_features = extract_improved_features(base_model, uniform_params, imgs)
            task_norm_features = (task_h_features - mean) / std
            
            logits = head(task_norm_features)
            alphas = torch.softmax(logits / T, dim=-1)
            mean_alphas = alphas.mean(dim=0)
            print(f"Task {task} mean_alphas: {mean_alphas.tolist()}")
            
            task_params = blend_parameters_functional(experts, mean_alphas, k=k)
            out = torch.func.functional_call(base_model, task_params, imgs)
            _, predicted = out.max(1)
            correct = predicted.eq(lbls).sum().item()
            acc = 100.0 * correct / len(lbls)
            
            results[task] = acc
            total_correct += correct
            total_samples += len(lbls)
            
    results['Joint Mean'] = 100.0 * total_correct / total_samples
    print(f"Joint Mean Accuracy with Improved Router: {results['Joint Mean']:.2f}%")
    return results

if __name__ == '__main__':
    experts = load_experts()
    cal_data, cal_labels, cal_task_ids, test_data, test_labels = build_data_splits(seed=42)
    head, mean, std = train_improved_router(experts, cal_data, cal_labels, cal_task_ids, num_epochs=300)
    
    print("\nEvaluating calibration split prediction:")
    with torch.no_grad():
        h_features = extract_improved_features(SimpleCNN(), blend_parameters_functional(experts, torch.tensor([0.25, 0.25, 0.25, 0.25]), k=0), cal_data)
        norm_features = (h_features - mean) / std
        logits = head(norm_features)
        _, preds = logits.max(dim=-1)
        cal_acc = preds.eq(cal_task_ids).sum().item() / len(cal_task_ids) * 100.0
        print(f"Calibration Accuracy: {cal_acc:.2f}%")
        
    print("\nEvaluating test splits:")
    evaluate_improved_routing(experts, head, mean, std, test_data, test_labels, k=4)
