import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# Set device to CPU
device = torch.device("cpu")

from test_merge import (
    get_dataset,
    load_experts,
    create_merged_model,
    calibrate_slr_wbc,
    evaluate_model
)

def run_sanity_check():
    print("Starting CPU sanity check...")
    
    # 1. Load a tiny calibration set (N=2)
    calibration_sets = {}
    for ds in ["mnist", "fashion", "cifar10"]:
        full_train = get_dataset(ds, train=True)
        calibration_sets[ds] = Subset(full_train, list(range(5000, 5002)))
        
    # 2. Load experts
    print("Loading experts on CPU...")
    experts = {}
    for ds in ["mnist", "fashion", "cifar10"]:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(512, 10)
        save_path = f"./checkpoints/{ds}_expert.pt"
        model.load_state_dict(torch.load(save_path, map_location=device))
        model.eval()
        experts[ds] = model
        
    # 3. Create base merged model
    print("Creating base merged model...")
    base_merged = create_merged_model(experts)
    
    # 4. Run SLR-WBC calibration
    print("Running SLR-WBC calibration on CPU...")
    calibrated_model = calibrate_slr_wbc(base_merged, experts, calibration_sets, rank=2, reg=1e-1)
    
    print("Calibration complete without errors!")
    
    # 5. Fast evaluation on a tiny test subset
    print("Evaluating calibrated model on tiny test subsets...")
    for ds in ["mnist", "fashion", "cifar10"]:
        test_dataset = get_dataset(ds, train=False)
        test_subset = Subset(test_dataset, list(range(0, 10)))
        loader = DataLoader(test_subset, batch_size=10, shuffle=False)
        correct = 0
        total = 0
        
        # Swapping classifier head
        original_fc = calibrated_model.fc
        calibrated_model.fc = experts[ds].fc
        calibrated_model.eval()
        
        with torch.no_grad():
            for inputs, targets in loader:
                outputs = calibrated_model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        calibrated_model.fc = original_fc
        print(f"{ds} tiny evaluation: {correct}/{total} correct ({100. * correct / total:.1f}%)")
        
    print("Sanity check completed successfully!")

if __name__ == "__main__":
    run_sanity_check()
