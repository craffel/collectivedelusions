import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from run_experiments import get_dataloaders, get_base_backbone, ExpertModel, evaluate_model, merge_task_arithmetic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def recalibrate_bn_exact(model, cal_loader):
    model.train()
    # For a simple and exact calibration, we set momentum = 1.0 so that the running stats
    # are completely replaced by the batch stats of the calibration data.
    # If we run multiple batches, we can do a cumulative average.
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            module.training = True
            module.momentum = 1.0
            
    with torch.no_grad():
        for i, (images, _) in enumerate(cal_loader):
            images = images.to(device)
            model(images)

def main():
    loaders = get_dataloaders()
    tasks = ['mnist', 'fashion', 'cifar']
    
    # Load experts
    expert_models = {}
    expert_backbone_states = []
    
    pretrained_backbone = get_base_backbone()
    pretrained_backbone_state = copy.deepcopy(pretrained_backbone.state_dict())
    
    for task in tasks:
        ckpt_path = f"expert_{task}.pt"
        if os.path.exists(ckpt_path):
            backbone = get_base_backbone().to(device)
            head = nn.Linear(512, 10).to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            backbone.load_state_dict(ckpt['backbone'])
            head.load_state_dict(ckpt['head'])
            model = ExpertModel(backbone, head)
            expert_models[task] = model
            expert_backbone_states.append(model.backbone.state_dict())
    
    # Merge with Lambda = 0.4 (best TA + TTBN lambda)
    ta_state = merge_task_arithmetic(pretrained_backbone_state, expert_backbone_states, 0.4)
    
    eval_backbone = get_base_backbone().to(device)
    
    print("\n--- Evaluating Recalibrated BN with Momentum=1.0 (Exact 128-sample calibration) ---")
    for task in tasks:
        eval_model = ExpertModel(eval_backbone, expert_models[task].head)
        eval_model.backbone.load_state_dict(copy.deepcopy(ta_state))
        
        # Create calibration loader from training set (e.g. 128 samples)
        cal_dataset = Subset(loaders[task]['train_dataset'], list(range(128)))
        cal_loader = DataLoader(cal_dataset, batch_size=128, shuffle=False)
        
        # Recalibrate
        recalibrate_bn_exact(eval_model, cal_loader)
        
        acc = evaluate_model(eval_model, loaders[task]['test'], bn_mode='eval')
        print(f"Recalibrated BN -> {task.upper()}: {acc:.2f}%")

if __name__ == '__main__':
    main()
