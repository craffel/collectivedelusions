import torch
from run_physical_validation import SimpleCNN, blend_parameters_functional, load_experts, build_data_splits, TASKS

def test_perfect():
    experts = load_experts()
    cal_data, cal_labels, cal_task_ids, test_data, test_labels = build_data_splits()
    
    base_model = SimpleCNN()
    
    # Perfect routing vectors (task-specific coefficients)
    # We use a scale of 1.0 first, then 1.2
    for scale in [1.0, 1.2]:
        print(f"\nEvaluating with scale = {scale}:")
        total_correct = 0
        total_samples = 0
        for task_id, task in enumerate(TASKS):
            # Create a one-hot vector scaled
            alphas = torch.zeros(4)
            alphas[task_id] = scale
            
            # Blend parameters
            task_params = blend_parameters_functional(experts, alphas, k=4)
            
            # Evaluate on test data
            imgs = test_data[task]
            lbls = test_labels[task]
            with torch.no_grad():
                out = torch.func.functional_call(base_model, task_params, imgs)
                _, predicted = out.max(1)
                correct = predicted.eq(lbls).sum().item()
                acc = 100.0 * correct / len(lbls)
                print(f"Perfect Router on {task} Test Accuracy: {acc:.2f}% (alphas: {alphas.tolist()})")
                total_correct += correct
                total_samples += len(lbls)
        print(f"Perfect Router Joint Mean Test Accuracy: {100.0 * total_correct / total_samples:.2f}%")

if __name__ == '__main__':
    test_perfect()
