import torch
from torch.utils.data import DataLoader
from dataset import load_datasets, get_calibration_set
from models import (
    MultiTaskResNet18, 
    merge_models_weight_averaging, 
    calibrate_model_ntaac,
    get_layer_stds
)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1. Load models
    print("Loading models...")
    task_mapping = {0: 'mnist', 1: 'fmnist', 2: 'cifar10'}
    experts = {}
    
    base_model = MultiTaskResNet18(pretrained=False)
    base_model.load_state_dict(torch.load("./checkpoints/base_model.pth", map_location=device))
    
    for task_id, task_name in task_mapping.items():
        expert = MultiTaskResNet18(pretrained=False)
        expert.load_state_dict(torch.load(f"./checkpoints/expert_{task_name}.pth", map_location=device))
        experts[task_name] = expert
        
    experts_list = [experts['mnist'], experts['fmnist'], experts['cifar10']]
    
    # 2. Perform Weight Averaging
    print("Performing Weight Averaging...")
    merged_model = MultiTaskResNet18(pretrained=False)
    merged_model = merge_models_weight_averaging(experts_list, merged_model)
    
    # 3. Load datasets and calibration sets (size N=128, seed=42)
    print("Loading calibration splits...")
    joint_cal_set, task_cal_sets = get_calibration_set(data_dir="./data", N=128, seed=42)
    joint_cal_loader = DataLoader(joint_cal_set, batch_size=len(joint_cal_set), shuffle=False)
    
    # 4. Clone merged model and calibrate via N-TAAC
    import copy
    calibrated_model = copy.deepcopy(merged_model)
    print("Calibrating model via N-TAAC...")
    calibrated_model = calibrate_model_ntaac(calibrated_model, joint_cal_loader, device, momentum=1.0)
    
    # 5. Measure standard deviations for task 0 (MNIST) calibration set
    task_id = 0
    task_name = 'mnist'
    mnist_cal_loader = DataLoader(task_cal_sets[task_name], batch_size=128, shuffle=False)
    
    print("\nExtracting layer standard deviations...")
    expert_stds = get_layer_stds(experts['mnist'], mnist_cal_loader, device, task_id)
    merged_stds = get_layer_stds(merged_model, mnist_cal_loader, device, task_id)
    calibrated_stds = get_layer_stds(calibrated_model, mnist_cal_loader, device, task_id)
    
    print("\nLayer-wise Standard Deviations (MNIST calibration set):")
    print(f"{'Layer':<10} | {'MNIST Expert':<15} | {'WA Merged (Uncal)':<20} | {'REDA (N-TAAC Cal)':<20} | {'Ratio (WA/Expert)':<20}")
    print("-" * 95)
    for l in range(len(expert_stds)):
        ratio = merged_stds[l] / expert_stds[l] if expert_stds[l] > 0 else 0.0
        print(f"Layer {l+1:<3}     | {expert_stds[l]:<15.4f} | {merged_stds[l]:<20.4f} | {calibrated_stds[l]:<20.4f} | {ratio:<20.2%}")

if __name__ == "__main__":
    main()
