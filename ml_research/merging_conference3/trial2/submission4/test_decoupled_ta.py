import sys
sys.path.append('SyMerge')
sys.path.append('SyMerge/src')
sys.path.append('local_packages')

import os
import torch
import numpy as np

from task_vectors import TaskVector
from dataset.registry import get_dataset
from dataset.common import get_dataloader, maybe_dictionarize
from heads import get_classification_head
from modeling import ImageClassifier

def patch_model(model):
    for m in model.modules():
        if m.__class__.__name__ == 'Transformer':
            m.batch_first = False

def evaluate_model(pretrained_model, classification_heads, test_loaders, exam_datasets, max_eval_batches, device):
    accs = []
    for name in exam_datasets:
        classifier = ImageClassifier(pretrained_model, classification_heads[name])
        classifier.eval()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for b_idx, batch in enumerate(test_loaders[name]):
                if max_eval_batches is not None and b_idx >= max_eval_batches:
                    break
                batch = maybe_dictionarize(batch)
                x = batch['images'].to(device)
                y = batch['labels'].to(device)
                logits = classifier(x)
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(y.view_as(pred)).sum().item()
                total += y.size(0)
        accs.append(correct / total)
    return sum(accs) / len(accs)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']
    model_name = 'ViT-B-32'
    
    checkpoints_root = 'checkpoints_tint'
    checkpoints_root_path = os.path.abspath(checkpoints_root)
    model_root = os.path.join(checkpoints_root_path, model_name)
    pretrained_checkpoint = os.path.join(model_root, 'zeroshot.pt')
    
    class CustomArgs:
        pass
    
    c_args = CustomArgs()
    c_args.model = 'ViT-B-32'
    c_args.data_location = 'datasets'
    c_args.checkpoints_root = checkpoints_root_path
    c_args.save = model_root
    c_args.pretrained_checkpoint = 'zeroshot.pt'
    c_args.batch_size = 128
    c_args.device = device
    c_args.openclip_cachedir = os.path.expanduser('~/.cache/open_clip')
    c_args.cache_dir = None

    print("Loading pretrained base model...")
    pretrained_model = torch.load(pretrained_checkpoint, weights_only=False)
    patch_model(pretrained_model)
    pretrained_model = pretrained_model.to(device)
    pretrained_model.eval()

    print("Loading classification heads...")
    classification_heads = {}
    for name in exam_datasets:
        classification_heads[name] = get_classification_head(c_args, name).to(device)
        classification_heads[name].eval()

    print("Loading task vectors...")
    task_vectors = []
    for name in exam_datasets:
        finetuned_path = os.path.join(model_root, name, 'finetuned.pt')
        tv = TaskVector(pretrained_checkpoint, finetuned_path)
        task_vectors.append(tv)

    base_state_dict = {k: v.detach().clone().to(device) for k, v in pretrained_model.state_dict().items()}
    tv_state_dicts = [{k: v.detach().clone().to(device) for k, v in tv.vector.items()} for tv in task_vectors]

    print("Loading validation datasets...")
    test_loaders = {}
    for name in exam_datasets:
        dataset = get_dataset(name, pretrained_model.val_preprocess, location='datasets', batch_size=128)
        test_loaders[name] = get_dataloader(dataset, is_train=False, args=c_args)

    proj_key = 'model.visual.proj'
    if proj_key not in base_state_dict:
        for k in base_state_dict:
            if 'visual.proj' in k:
                proj_key = k
                break

    # Decoupled Task Arithmetic Grid Search
    print("\n--- Running Decoupled Task Arithmetic (DTA) Sweep ---")
    lmbda_statics = [0.15, 0.20, 0.25, 0.30]
    lmbda_projs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 3.0]

    best_dta_acc = -1.0
    best_dta_config = {}

    for l_static in lmbda_statics:
        for l_proj in lmbda_projs:
            merged_sd = {}
            for k in base_state_dict:
                if k == proj_key:
                    total_proj_tv = sum(tv_sd[proj_key] for tv_sd in tv_state_dicts)
                    merged_sd[k] = base_state_dict[k] + l_proj * total_proj_tv
                else:
                    total_tv = sum(tv_sd[k] for tv_sd in tv_state_dicts if k in tv_sd)
                    merged_sd[k] = base_state_dict[k] + l_static * total_tv
            
            pretrained_model.load_state_dict(merged_sd)
            pretrained_model.eval()
            acc = evaluate_model(pretrained_model, classification_heads, test_loaders, exam_datasets, max_eval_batches=8, device=device)
            
            if acc > best_dta_acc:
                best_dta_acc = acc
                best_dta_config = {
                    'lambda_static': l_static,
                    'lambda_proj': l_proj
                }
            print(f"Decoupled TA (l_static={l_static:.2f}, l_proj={l_proj:.2f}): {acc*100:.4f}%")

    print(f"\nBest Decoupled Task Arithmetic (DTA): {best_dta_acc*100:.4f}%")
    print(f"Config: {best_dta_config}")

if __name__ == '__main__':
    main()
