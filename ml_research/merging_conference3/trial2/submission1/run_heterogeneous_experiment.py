import os
import sys
import torch
import functools
import numpy as np
import tqdm
import json
import collections
import gc

# Patch torch.load globally to handle unpickling of custom classes
torch.load = functools.partial(torch.load, weights_only=False)

# Add AdaMerging/src to path
sys.path.insert(0, os.path.abspath('AdaMerging/src'))

from modeling import ImageEncoder, ImageClassifier

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, initial_weights=None):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.val_preprocess = getattr(model, 'val_preprocess', None)
        self.train_preprocess = getattr(model, 'train_preprocess', None)

        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        features = self.model(images)
        return features
from heads import get_classification_head
from datasets.registry import get_dataset
from datasets.common import get_dataloader, maybe_dictionarize, get_dataloader_shuffle
from task_vectors import TaskVector

class Args:
    def __init__(self):
        self.model = 'ViT-B-32'
        self.save = os.path.abspath('checkpoints/ViT-B-32')
        self.data_location = os.path.abspath('data')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.openclip_cachedir = os.path.abspath('openclip_cache')
        self.batch_size = 128
        self.cache_dir = None
        self.eval_datasets = ['MNIST', 'FashionMNIST', 'CIFAR10', 'SVHN']
        self.results_db = None

# Heterogeneous class capacities definition
class_capacities = {
    'MNIST': 3,          # limit to digits 0, 1, 2
    'FashionMNIST': 5,   # limit to apparel classes 0, 1, 2, 3, 4
    'SVHN': 8,           # limit to digits 0-7
    'CIFAR10': 10        # keep all 10 classes
}

# Helper functions for making the model functional
def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def make_functional(mod):
    orig_params = tuple(mod.parameters())
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names

def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)

class RegCalMerge(torch.nn.Module):
    def __init__(self, paramslist, model, names, exam_datasets, args, spatial_mean=False):
        super(RegCalMerge, self).__init__()
        self.paramslist = paramslist
        self.model = model
        self.names = names
        self.exam_datasets = exam_datasets
        self.args = args
        self.spatial_mean = spatial_mean

        # Coefficient initialization
        self.pretrain_lambdas = torch.ones(len(paramslist[0]), 1)
        prior = 0.3
        
        if self.spatial_mean:
            rlambdas = torch.ones(1, len(paramslist)-1) * prior
        else:
            rlambdas = torch.ones(len(paramslist[0]), len(paramslist)-1) * prior
            
        self.lambdas_raw = torch.nn.Parameter(rlambdas)

        # Build classifiers and load classification heads
        self.classifier = []
        for dataset_name in exam_datasets:
            classification_head = get_classification_head(args, dataset_name)
            layer_name = 'classifier_{}'.format(dataset_name)
            self.add_module(layer_name, classification_head.to(args.device))
            self.classifier.append(layer_name)

        # Stack parameters once for extremely fast vectorized operations on CPU
        self.stacked_params = []
        for j in range(len(paramslist[0])):
            stacked_j = torch.stack([paramslist[i][j].detach() for i in range(len(paramslist))], dim=0)
            self.stacked_params.append(stacked_j)

    def lambdas(self):
        task_lambdas = torch.clamp(self.lambdas_raw, min=0.0, max=1.0)
        if self.spatial_mean:
            task_lambdas = task_lambdas.expand(len(self.paramslist[0]), -1)
        lambdass = torch.cat((self.pretrain_lambdas, task_lambdas), 1)
        return lambdass

    def collect_trainable_params(self):
        return [self.lambdas_raw]

    def get_classification_head(self, dataset_name):
        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name)
        return classification_head

    def get_image_encoder(self):
        alph = self.lambdas()
        params = []
        for j, stacked_p in enumerate(self.stacked_params):
            w = alph[j].cpu()
            w_expanded = w.view(-1, *([1] * (stacked_p.dim() - 1)))
            merged_p = torch.sum(stacked_p * w_expanded, dim=0)
            params.append(merged_p)
        params = tuple(p.to(self.args.device) for p in params)
        load_weights(self.model, self.names, params)
        return self.model

    def forward(self, inp, dataset_name):
        alph = self.lambdas()
        params = []
        for j, stacked_p in enumerate(self.stacked_params):
            w = alph[j].cpu()
            w_expanded = w.view(-1, *([1] * (stacked_p.dim() - 1)))
            merged_p = torch.sum(stacked_p * w_expanded, dim=0)
            params.append(merged_p)
        params = tuple(p.to(self.args.device) for p in params)
        load_weights(self.model, self.names, params)
        feature = self.model(inp)
        # Classifiers load weights automatically
        head = self.get_classification_head(dataset_name)
        outputs = head(feature)
        return outputs

def softmax_entropy(x):
    probs = x.softmax(dim=-1)
    return -(probs * x.log_softmax(dim=-1)).sum(dim=-1)

# In-memory global caches
_dataloader_cache = {}
_calib_cache = {}

def get_cached_dataloader(dataset_name, preprocess, args):
    cache_key = (dataset_name, id(preprocess))
    if cache_key not in _dataloader_cache:
        print(f"Loading and caching evaluation batches for {dataset_name} (Heterogeneous)...")
        dataset = get_dataset(dataset_name, preprocess, location=args.data_location, batch_size=args.batch_size)
        dataloader = get_dataloader(dataset, is_train=False, args=args)

        batches = []
        limit_class = class_capacities[dataset_name]
        for i, batch in enumerate(dataloader):
            batch = maybe_dictionarize(batch)
            images = batch['images']
            labels = batch['labels']
            mask = labels < limit_class
            if mask.sum() > 0:
                batches.append({
                    'images': images[mask].clone(),
                    'labels': labels[mask].clone()
                })
            if len(batches) >= 2: # Keep at most 2 batches of filtered samples
                break
        _dataloader_cache[cache_key] = batches
        print(f"Successfully cached {len(batches)} evaluation batches for {dataset_name} with class limit {limit_class}.")
    return _dataloader_cache[cache_key]

def get_cached_calib_batch(dataset_name, preprocess, args):
    cache_key = (dataset_name, id(preprocess))
    if cache_key not in _calib_cache:
        print(f"Loading and caching calibration batch for {dataset_name} (Heterogeneous)...")
        dataset = get_dataset(dataset_name, preprocess, location=args.data_location, batch_size=128)
        dataloader = get_dataloader_shuffle(dataset)
        limit_class = class_capacities[dataset_name]
        for batch in dataloader:
            batch = maybe_dictionarize(batch)
            images = batch['images']
            labels = batch['labels']
            mask = labels < limit_class
            if mask.sum() >= 16:
                _calib_cache[cache_key] = images[mask][:16].clone()
                break
            elif mask.sum() > 0:
                _calib_cache[cache_key] = images[mask].clone()
                break
        print(f"Successfully cached calibration batch for {dataset_name} with class limit {limit_class} (shape: {_calib_cache[cache_key].shape}).")
    return _calib_cache[cache_key]


def eval_single_dataset(image_encoder, head, dataset_name, args):
    model = ImageClassifier(image_encoder, head)
    model.eval()
    
    # Retrieve in-memory evaluation batches
    batches = get_cached_dataloader(dataset_name, model.val_preprocess, args)
    
    test_correct = 0
    test_total = 0
    limit_class = class_capacities[dataset_name]
    
    with torch.no_grad():
        for batch in batches:
            x = batch['images'].to(args.device)
            y = batch['labels'].to(args.device)
            outputs = model(x)
            outputs_sliced = outputs[:, :limit_class]
            _, predicted = outputs_sliced.max(dim=1)
            test_total += y.size(0)
            test_correct += predicted.eq(y).sum().item()
            
    return test_correct / test_total


def evaluate_model(adamerging_model, exam_datasets, args):
    results = {}
    for dataset_name in exam_datasets:
        image_encoder = adamerging_model.get_image_encoder()
        head = adamerging_model.get_classification_head(dataset_name)
        acc = eval_single_dataset(image_encoder, head, dataset_name, args)
        results[dataset_name] = acc
    return results


def run_adam_optimization(adamerging_model, exam_datasets, args, beta=0.0, gamma=0.0, use_ccn=False, use_snew=False, steps=50):
    optimizer = torch.optim.Adam(adamerging_model.collect_trainable_params(), lr=1e-2)
    
    scale_weights = {dataset_name: 1.0 for dataset_name in exam_datasets}
    
    # Retrieve in-memory calibration batches
    calib_batches = {}
    for dataset_name in exam_datasets:
        calib_batches[dataset_name] = get_cached_calib_batch(dataset_name, adamerging_model.model.val_preprocess, args).to(args.device)

    if use_snew:
        # Compute baseline entropy at initial lambda (all 0.3)
        print("\n--- SNEW Initialization Stats ---")
        with torch.no_grad():
            for dataset_name in exam_datasets:
                x = calib_batches[dataset_name]
                outputs = adamerging_model(x, dataset_name)
                # Slice logits
                outputs_sliced = outputs[:, :class_capacities[dataset_name]]
                raw_ent = softmax_entropy(outputs_sliced).mean().item()
                ent = raw_ent
                if use_ccn:
                    ent /= np.log(class_capacities[dataset_name])
                scale_weights[dataset_name] = 1.0 / max(ent, 1e-5)
                print(f"Dataset: {dataset_name:<15} | Classes (C_k): {class_capacities[dataset_name]:<2} | Raw Entropy: {raw_ent:.4f} | Normalized Entropy: {ent:.4f} | SNEW Weight (w_k): {scale_weights[dataset_name]:.4f}")
        print("---------------------------------\n")
                    
    # Optimization Loop
    for step in range(steps):
        losses = 0.0
        
        for dataset_name in exam_datasets:
            x = calib_batches[dataset_name]
            outputs = adamerging_model(x, dataset_name)
            outputs_sliced = outputs[:, :class_capacities[dataset_name]]
            
            entropy_loss = softmax_entropy(outputs_sliced).mean()
            
            # Apply Class-Capacity Normalization (CCN)
            if use_ccn:
                entropy_loss = entropy_loss / np.log(class_capacities[dataset_name])
                
            # Apply Scale-Normalized Entropy Weighting (SNEW)
            entropy_loss = entropy_loss * scale_weights[dataset_name]
            
            losses += entropy_loss
                
        # Elastic Spatial Regularization (ESR)
        lambdas_raw = adamerging_model.lambdas_raw
        
        # Proximity Penalty
        proximity_penalty = torch.sum((lambdas_raw - 0.3) ** 2)
        
        # Spatial Deviation Penalty
        mean_lambdas = lambdas_raw.mean(dim=0, keepdim=True)
        spatial_dev_penalty = torch.sum((lambdas_raw - mean_lambdas) ** 2)
        
        total_loss = losses + beta * proximity_penalty + gamma * spatial_dev_penalty
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


def run_experiment_pipeline():
    args = Args()
    
    pretrained_checkpoint = os.path.join(args.save, 'zeroshot.pt')
    pretrained_model = torch.load(pretrained_checkpoint)
    pretrained_model_dic = pretrained_model.state_dict()
    
    model_wrapper = ModelWrapper(pretrained_model, args.eval_datasets)
    model_wrapper = model_wrapper.to(args.device)
    _, names = make_functional(model_wrapper)

    print("Loading task vectors...")
    task_vectors = []
    for dataset_name in args.eval_datasets:
        finetuned_path = os.path.join(args.save, dataset_name, 'finetuned.pt')
        task_vectors.append(TaskVector(pretrained_checkpoint, finetuned_path))

    # Pretrain params list
    paramslist = []
    paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in pretrained_model_dic.items())]
    paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in tv.vector.items()) for tv in task_vectors]

    seeds = [42, 43, 44]
    
    all_metrics = {}

    # 1. Task Arithmetic (Uniform static baseline)
    print("\nEvaluating Baseline: Task Arithmetic (Uniform)")
    ta_accs = collections.defaultdict(list)
    for seed in seeds:
        torch.manual_seed(seed)
        model_ta = RegCalMerge(paramslist, model_wrapper, names, args.eval_datasets, args)
        results = evaluate_model(model_ta, args.eval_datasets, args)
        for k, v in results.items():
            ta_accs[k].append(v)
        del model_ta
        gc.collect()
    
    all_metrics["task_arithmetic"] = {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in ta_accs.items()
    }
    print("Task Arithmetic Accuracies:", {k: f"{100*np.mean(v):.2f}%" for k, v in ta_accs.items()})

    # 2. Uncalibrated layer-wise AdaMerging (Adam GD) - no CCN, no SNEW
    print("\nEvaluating Baseline: Uncalibrated AdaMerging (Adam GD)")
    uncal_accs = collections.defaultdict(list)
    for seed in seeds:
        torch.manual_seed(seed)
        model_adam = RegCalMerge(paramslist, model_wrapper, names, args.eval_datasets, args)
        run_adam_optimization(model_adam, args.eval_datasets, args, beta=0.0, gamma=0.0, use_ccn=False, use_snew=False, steps=10)
        results = evaluate_model(model_adam, args.eval_datasets, args)
        for k, v in results.items():
            uncal_accs[k].append(v)
        del model_adam
        gc.collect()
            
    all_metrics["uncalibrated_adam"] = {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in uncal_accs.items()
    }
    print("Uncalibrated AdaMerging Accuracies:", {k: f"{100*np.mean(v):.2f}%" for k, v in uncal_accs.items()})

    # 3. Calibrated AdaMerging (CalMerge) - CCN + SNEW (beta=0.0, gamma=0.0)
    print("\nEvaluating Our Flagship Model: Calibrated AdaMerging (CalMerge)")
    calmerge_accs = collections.defaultdict(list)
    for seed in seeds:
        torch.manual_seed(seed)
        model_cm = RegCalMerge(paramslist, model_wrapper, names, args.eval_datasets, args)
        run_adam_optimization(model_cm, args.eval_datasets, args, beta=0.0, gamma=0.0, use_ccn=True, use_snew=True, steps=10)
        results = evaluate_model(model_cm, args.eval_datasets, args)
        for k, v in results.items():
            calmerge_accs[k].append(v)
        del model_cm
        gc.collect()
            
    all_metrics["calmerge"] = {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in calmerge_accs.items()
    }
    print("CalMerge (CCN+SNEW) Accuracies:", {k: f"{100*np.mean(v):.2f}%" for k, v in calmerge_accs.items()})

    # 4. Calibrated Spatial Mean (Cal-Mean) - spatial_mean=True + CCN + SNEW
    print("\nEvaluating Baseline: Calibrated Spatial Mean (Cal-Mean)")
    cal_mean_accs = collections.defaultdict(list)
    for seed in seeds:
        torch.manual_seed(seed)
        model_cmean = RegCalMerge(paramslist, model_wrapper, names, args.eval_datasets, args, spatial_mean=True)
        # Note: we optimize spatially collapsed parameters using Adam for fair comparison
        run_adam_optimization(model_cmean, args.eval_datasets, args, beta=0.0, gamma=0.0, use_ccn=True, use_snew=True, steps=10)
        results = evaluate_model(model_cmean, args.eval_datasets, args)
        for k, v in results.items():
            cal_mean_accs[k].append(v)
        del model_cmean
        gc.collect()
            
    all_metrics["cal_mean"] = {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in cal_mean_accs.items()
    }
    print("Cal-Mean Accuracies:", {k: f"{100*np.mean(v):.2f}%" for k, v in cal_mean_accs.items()})

    # 5. RegCalMerge with ESR (beta=1.0, gamma=1.0, CCN, SNEW)
    print("\nEvaluating Our Method: RegCalMerge (Adam GD with ESR, CCN, SNEW)")
    rcm_accs = collections.defaultdict(list)
    for seed in seeds:
        torch.manual_seed(seed)
        model_rcm = RegCalMerge(paramslist, model_wrapper, names, args.eval_datasets, args)
        run_adam_optimization(model_rcm, args.eval_datasets, args, beta=1.0, gamma=1.0, use_ccn=True, use_snew=True, steps=10)
        results = evaluate_model(model_rcm, args.eval_datasets, args)
        for k, v in results.items():
            rcm_accs[k].append(v)
        del model_rcm
        gc.collect()
            
    all_metrics["regcalmerge"] = {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in rcm_accs.items()
    }
    print("RegCalMerge Accuracies:", {k: f"{100*np.mean(v):.2f}%" for k, v in rcm_accs.items()})

    # Print summary table
    print("\n=== SUMMARY TABLE (Joint Mean Accuracy across Seeds) ===")
    print(f"{'Method':<35} | {'MNIST':<8} | {'Fashion':<8} | {'CIFAR10':<8} | {'SVHN':<8} | {'Joint Mean':<10}")
    print("-" * 85)
    
    for method, metrics in [
        ("Task Arithmetic (Uniform)", ta_accs),
        ("Uncalibrated AdaMerging (Adam GD)", uncal_accs),
        ("Calibrated Spatial Mean (Cal-Mean)", cal_mean_accs),
        ("Calibrated AdaMerging (CalMerge)", calmerge_accs),
        ("RegCalMerge (ESR + CCN + SNEW)", rcm_accs)
    ]:
        mnist_val = np.mean(metrics['MNIST']) * 100
        fash_val = np.mean(metrics['FashionMNIST']) * 100
        cifar_val = np.mean(metrics['CIFAR10']) * 100
        svhn_val = np.mean(metrics['SVHN']) * 100
        joint_val = np.mean([np.mean(v) for v in metrics.values()]) * 100
        print(f"{method:<35} | {mnist_val:6.2f}% | {fash_val:6.2f}% | {cifar_val:6.2f}% | {svhn_val:6.2f}% | {joint_val:8.2f}%")

    # Save to file
    with open('results_test/metrics_heterogeneous.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)

if __name__ == '__main__':
    os.makedirs('results_test', exist_ok=True)
    run_experiment_pipeline()
