import os
import sys
import torch
import pickle
from pathlib import Path

# Add SyMerge and SyMerge/src to python path
sys.path.append('/fsx/craffel/collectivedelusions/ml_research/merging_conference3/trial2/submission9/SyMerge')
sys.path.append('/fsx/craffel/collectivedelusions/ml_research/merging_conference3/trial2/submission9/SyMerge/src')

from task_vectors import TaskVector
from eval import eval_single_dataset_preprocess_head_with_ece
from args import parse_arguments
from heads import get_classification_head

def patch_open_clip_model(model):
    for m in model.modules():
        if type(m).__name__ == 'Transformer':
            if not hasattr(m, 'batch_first'):
                m.batch_first = False
    return model

def main():
    # Setup args
    args = parse_arguments()
    args.config = '/fsx/craffel/collectivedelusions/ml_research/merging_conference3/trial2/submission9/SyMerge/configs/bpam.yaml'
    
    import yaml
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f) or {}
    for key, value in cfg.items():
        setattr(args, key, value)
        
    print("Args loaded:", args)
    
    checkpoints_root = Path(args.checkpoints_root)
    model_root = checkpoints_root / args.model
    pretrained_checkpoint = str(model_root / args.pretrained_checkpoint)
    args.save = str(model_root)
    
    exam_datasets = args.exam_datasets
    if isinstance(exam_datasets, str):
        exam_datasets = exam_datasets.split(",")
        
    print("Loading task vectors...")
    task_vectors = [TaskVector(pretrained_checkpoint, str(model_root / dataset_name / 'finetuned.pt')) for dataset_name in exam_datasets]
    
    # Task Arithmetic: sum the task vectors and scale by 0.3
    print("Merging task vectors...")
    merged_vector = {}
    for key in task_vectors[0].vector.keys():
        merged_vector[key] = sum(0.3 * tv.vector[key] for tv in task_vectors)
        
    # Apply to pre-trained model
    print("Loading pre-trained model...")
    pretrained_model = torch.load(pretrained_checkpoint)
    pretrained_model = patch_open_clip_model(pretrained_model)
    
    print("Applying merged task vector...")
    pretrained_state_dict = pretrained_model.state_dict()
    new_state_dict = {}
    for key in pretrained_state_dict:
        if key in merged_vector:
            new_state_dict[key] = pretrained_state_dict[key] + merged_vector[key].cpu()
        else:
            new_state_dict[key] = pretrained_state_dict[key]
            
    pretrained_model.load_state_dict(new_state_dict, strict=False)
    pretrained_model = pretrained_model.to(args.device)
    pretrained_model.eval()
    
    print("Evaluating Task Arithmetic...")
    Total_ACC = 0.
    results = {}
    for dataset_name in exam_datasets:
        classification_head = get_classification_head(args, dataset_name)
        classification_head = classification_head.to(args.device)
        classification_head.eval()
        
        metrics = eval_single_dataset_preprocess_head_with_ece(pretrained_model, classification_head, dataset_name, args)
        print(f"Dataset: {dataset_name} | ACC: {metrics['top1']*100:.2f}% | ECE: {metrics['ece']*100:.2f}%")
        results[dataset_name] = metrics['top1']
        Total_ACC += metrics['top1']
        
    print(f"Average Accuracy: {Total_ACC / len(exam_datasets)*100:.2f}%")
    print("Task Arithmetic results dict:", results)

if __name__ == '__main__':
    main()
