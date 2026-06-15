import torch
from run_physical_validation import SimpleCNN, load_experts, build_data_splits, train_physical_router, TASKS, blend_parameters_functional

def test():
    experts = load_experts()
    cal_data, cal_labels, cal_task_ids, test_data, test_labels = build_data_splits(seed=42)
    head, mean, std = train_physical_router(experts, cal_data, cal_labels, cal_task_ids, num_epochs=250)
    
    head.eval()
    base_model = SimpleCNN()
    uniform_params = blend_parameters_functional(experts, torch.tensor([0.25, 0.25, 0.25, 0.25]), k=0)
    
    with torch.no_grad():
        task_h0_features = []
        for idx in range(len(cal_data)):
            img = cal_data[idx:idx+1]
            conv1_feat = torch.func.functional_call(base_model.features[0:2], {
                '0.weight': uniform_params['features.0.weight'],
                '0.bias': uniform_params['features.0.bias']
            }, img)
            task_h0_features.append(conv1_feat.mean(dim=[2, 3]).view(-1))
        task_h0_features = torch.stack(task_h0_features)
        task_norm_features = (task_h0_features - mean) / std
        
        logits = head(task_norm_features)
        _, preds = logits.max(dim=-1)
        correct = preds.eq(cal_task_ids).sum().item()
        print(f"Calibration accuracy: {correct}/{len(cal_task_ids)} ({100.0 * correct / len(cal_task_ids):.2f}%)")
        print(f"Predicted IDs: {preds.tolist()}")
        print(f"True IDs: {cal_task_ids.tolist()}")

if __name__ == '__main__':
    test()
