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
            # Optimize only 1 scalar per task, expanded to all layers
            rlambdas = torch.ones(1, len(paramslist)-1) * prior
        else:
            # Optimize layer-wise (named parameter-wise) coefficients
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
            # Stack the same parameter across the pretrain model and K experts along dimension 0
            stacked_j = torch.stack([paramslist[i][j].detach() for i in range(len(paramslist))], dim=0)
            self.stacked_params.append(stacked_j)

    def lambdas(self):
        task_lambdas = torch.clamp(self.lambdas_raw, min=0.0, max=1.0)
        if self.spatial_mean:
            # Expand to matches the number of parameters
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

        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name)
        out = classification_head(feature)
        return out


def softmax_entropy(x):
    probs = x.softmax(dim=-1)
    return -(probs * x.log_softmax(dim=-1)).sum(dim=-1)


# In-memory global caches to completely bypass disk I/O and dataloader worker overhead during TTA
_dataloader_cache = {}
_calib_cache = {}

def get_cached_dataloader(dataset_name, preprocess, args):
    cache_key = (dataset_name, id(preprocess))
    if cache_key not in _dataloader_cache:
        dataset = get_dataset(dataset_name, preprocess, location=args.data_location, batch_size=args.batch_size)
        dataloader = get_dataloader(dataset, is_train=False, args=args)
        
        batches = []
        for i, batch in enumerate(dataloader):
            if i > 0: # Limit to 1 batch (128 images) for fast test
                break
            batch = maybe_dictionarize(batch)
            batches.append({
                'images': batch['images'].clone(),
                'labels': batch['labels'].clone()
            })
        _dataloader_cache[cache_key] = batches
    return _dataloader_cache[cache_key]

def get_cached_calib_batch(dataset_name, preprocess, args):
    cache_key = (dataset_name, id(preprocess))
    if cache_key not in _calib_cache:
        dataset = get_dataset(dataset_name, preprocess, location=args.data_location, batch_size=16)
        dataloader = get_dataloader_shuffle(dataset)
        for batch in dataloader:
            batch = maybe_dictionarize(batch)
            _calib_cache[cache_key] = batch['images'].clone()
            break
    return _calib_cache[cache_key]


def eval_single_dataset(image_encoder, head, dataset_name, args):
    model = ImageClassifier(image_encoder, head)
    model.eval()
    
    batches = get_cached_dataloader(dataset_name, model.val_preprocess, args)
    
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch in batches:
            x = batch['images'].to(args.device)
            y = batch['labels'].to(args.device)
            outputs = model(x)
            _, predicted = outputs.max(dim=1)
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
    class_capacities = {dataset_name: 10.0 for dataset_name in exam_datasets}
    
    if use_ccn:
        for dataset_name in exam_datasets:
            head = adamerging_model.get_classification_head(dataset_name)
            class_capacities[dataset_name] = float(head.weight.shape[0])
            
    calib_batches = {}
    for dataset_name in exam_datasets:
        calib_batches[dataset_name] = get_cached_calib_batch(dataset_name, adamerging_model.model.val_preprocess, args).to(args.device)

    if use_snew:
        with torch.no_grad():
            for dataset_name in exam_datasets:
                x = calib_batches[dataset_name]
                outputs = adamerging_model(x, dataset_name)
                ent = softmax_entropy(outputs).mean().item()
                if use_ccn:
                    ent /= np.log(class_capacities[dataset_name])
                scale_weights[dataset_name] = 1.0 / max(ent, 1e-5)
                    
    for step in range(steps):
        losses = 0.0
        for dataset_name in exam_datasets:
            x = calib_batches[dataset_name]
            outputs = adamerging_model(x, dataset_name)
            entropy_loss = softmax_entropy(outputs).mean()
            if use_ccn:
                entropy_loss = entropy_loss / np.log(class_capacities[dataset_name])
            entropy_loss = entropy_loss * scale_weights[dataset_name]
            losses += entropy_loss
                
        lambdas_raw = adamerging_model.lambdas_raw
        proximity_penalty = torch.sum((lambdas_raw - 0.3) ** 2)
        mean_lambdas = lambdas_raw.mean(dim=0, keepdim=True)
        spatial_dev_penalty = torch.sum((lambdas_raw - mean_lambdas) ** 2)
        
        total_loss = losses + beta * proximity_penalty + gamma * spatial_dev_penalty
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


def run_es_optimization(adamerging_model, exam_datasets, args, beta=0.0, gamma=0.0, use_ccn=False, use_snew=False, steps=50):
    best_lambdas = adamerging_model.lambdas_raw.data.clone()
    best_loss = float('inf')
    sigma = 0.1
    
    class_capacities = {dataset_name: 10.0 for dataset_name in exam_datasets}
    if use_ccn:
        for dataset_name in exam_datasets:
            head = adamerging_model.get_classification_head(dataset_name)
            class_capacities[dataset_name] = float(head.weight.shape[0])
            
    calib_batches = {}
    for dataset_name in exam_datasets:
        calib_batches[dataset_name] = get_cached_calib_batch(dataset_name, adamerging_model.model.val_preprocess, args).to(args.device)

    scale_weights = {dataset_name: 1.0 for dataset_name in exam_datasets}
    
    def evaluate_loss(lambdas_candidate):
        adamerging_model.lambdas_raw.data.copy_(lambdas_candidate)
        losses = 0.0
        with torch.no_grad():
            for dataset_name in exam_datasets:
                x = calib_batches[dataset_name]
                outputs = adamerging_model(x, dataset_name)
                entropy_loss = softmax_entropy(outputs).mean()
                if use_ccn:
                    entropy_loss /= np.log(class_capacities[dataset_name])
                entropy_loss *= scale_weights[dataset_name]
                losses += entropy_loss.item()
                    
            proximity_penalty = torch.sum((lambdas_candidate - 0.3) ** 2).item()
            mean_lambdas = lambdas_candidate.mean(dim=0, keepdim=True)
            spatial_dev_penalty = torch.sum((lambdas_candidate - mean_lambdas) ** 2).item()
            
            return losses + beta * proximity_penalty + gamma * spatial_dev_penalty

    if use_snew:
        with torch.no_grad():
            for dataset_name in exam_datasets:
                x = calib_batches[dataset_name]
                outputs = adamerging_model(x, dataset_name)
                ent = softmax_entropy(outputs).mean().item()
                if use_ccn:
                    ent /= np.log(class_capacities[dataset_name])
                scale_weights[dataset_name] = 1.0 / max(ent, 1e-5)

    best_loss = evaluate_loss(best_lambdas)
    
    for step in range(steps):
        mutation = torch.randn_like(best_lambdas) * sigma
        candidate = torch.clamp(best_lambdas + mutation, 0.0, 1.0)
        candidate_loss = evaluate_loss(candidate)
        if candidate_loss < best_loss:
            best_loss = candidate_loss
            best_lambdas.copy_(candidate)
            sigma = min(sigma * 1.1, 0.5)
        else:
            sigma = max(sigma * 0.9, 1e-4)
            
    adamerging_model.lambdas_raw.data.copy_(best_lambdas)


def run_experiment_pipeline():
    args = Args()
    
    pretrained_checkpoint = os.path.join(args.save, 'zeroshot.pt')
    pretrained_model = torch.load(pretrained_checkpoint)
    pretrained_model_dic = pretrained_model.state_dict()
    
    model_wrapper = ModelWrapper(pretrained_model, args.eval_datasets)
    model_wrapper = model_wrapper.to(args.device)
    _, names = make_functional(model_wrapper)

    task_vectors = []
    for dataset_name in args.eval_datasets:
        finetuned_path = os.path.join(args.save, dataset_name, 'finetuned.pt')
        task_vectors.append(TaskVector(pretrained_checkpoint, finetuned_path))

    paramslist = []
    paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in pretrained_model_dic.items())] # pretrain
    paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in tv.vector.items()) for tv in task_vectors] # task vectors

    seeds = [42] # single seed for fast verification
    all_metrics = {}

    # 1. Task Arithmetic
    print("Testing Stage 1: Task Arithmetic...")
    ta_accs = collections.defaultdict(list)
    for seed in seeds:
        torch.manual_seed(seed)
        model_ta = RegCalMerge(paramslist, model_wrapper, names, args.eval_datasets, args)
        results = evaluate_model(model_ta, args.eval_datasets, args)
        for k, v in results.items():
            ta_accs[k].append(v)
        del model_ta
        gc.collect()
    all_metrics["task_arithmetic"] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in ta_accs.items()}

    # 2. Unconstrained Adam GD
    print("Testing Stage 2: Unconstrained Adam GD...")
    unconstrained_adam_accs = collections.defaultdict(list)
    for seed in seeds:
        torch.manual_seed(seed)
        model_adam = RegCalMerge(paramslist, model_wrapper, names, args.eval_datasets, args)
        run_adam_optimization(model_adam, args.eval_datasets, args, beta=0.0, gamma=0.0, steps=2)
        results = evaluate_model(model_adam, args.eval_datasets, args)
        for k, v in results.items():
            unconstrained_adam_accs[k].append(v)
        del model_adam
        gc.collect()
    all_metrics["adam_opt"] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in unconstrained_adam_accs.items()}

    # 3. Unconstrained ES
    print("Testing Stage 3: Unconstrained ES...")
    unconstrained_es_accs = collections.defaultdict(list)
    for seed in seeds:
        torch.manual_seed(seed)
        model_es = RegCalMerge(paramslist, model_wrapper, names, args.eval_datasets, args)
        run_es_optimization(model_es, args.eval_datasets, args, beta=0.0, gamma=0.0, steps=2)
        results = evaluate_model(model_es, args.eval_datasets, args)
        for k, v in results.items():
            unconstrained_es_accs[k].append(v)
        del model_es
        gc.collect()
    all_metrics["es_opt"] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in unconstrained_es_accs.items()}

    # 4. Spatial Mean
    print("Testing Stage 4: Spatial Mean...")
    spatial_mean_accs = collections.defaultdict(list)
    for seed in seeds:
        torch.manual_seed(seed)
        model_sm = RegCalMerge(paramslist, model_wrapper, names, args.eval_datasets, args, spatial_mean=True)
        run_es_optimization(model_sm, args.eval_datasets, args, beta=0.0, gamma=0.0, steps=2)
        results = evaluate_model(model_sm, args.eval_datasets, args)
        for k, v in results.items():
            spatial_mean_accs[k].append(v)
        del model_sm
        gc.collect()
    all_metrics["spatial_mean_es"] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in spatial_mean_accs.items()}

    # 5. Shuffled Adam
    print("Testing Stage 5: Shuffled Adam GD...")
    shuffled_adam_accs = collections.defaultdict(list)
    for seed in seeds:
        torch.manual_seed(seed)
        model_sh_adam = RegCalMerge(paramslist, model_wrapper, names, args.eval_datasets, args)
        run_adam_optimization(model_sh_adam, args.eval_datasets, args, beta=0.0, gamma=0.0, steps=2)
        with torch.no_grad():
            lambdas_raw = model_sh_adam.lambdas_raw.data
            for k in range(lambdas_raw.shape[1]):
                perm = torch.randperm(lambdas_raw.shape[0])
                lambdas_raw[:, k] = lambdas_raw[perm, k]
        results = evaluate_model(model_sh_adam, args.eval_datasets, args)
        for k, v in results.items():
            shuffled_adam_accs[k].append(v)
        del model_sh_adam
        gc.collect()
    all_metrics["shuffled_adam"] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in shuffled_adam_accs.items()}

    # 6. Shuffled ES
    print("Testing Stage 6: Shuffled ES...")
    shuffled_es_accs = collections.defaultdict(list)
    for seed in seeds:
        torch.manual_seed(seed)
        model_sh_es = RegCalMerge(paramslist, model_wrapper, names, args.eval_datasets, args)
        run_es_optimization(model_sh_es, args.eval_datasets, args, beta=0.0, gamma=0.0, steps=2)
        with torch.no_grad():
            lambdas_raw = model_sh_es.lambdas_raw.data
            for k in range(lambdas_raw.shape[1]):
                perm = torch.randperm(lambdas_raw.shape[0])
                lambdas_raw[:, k] = lambdas_raw[perm, k]
        results = evaluate_model(model_sh_es, args.eval_datasets, args)
        for k, v in results.items():
            shuffled_es_accs[k].append(v)
        del model_sh_es
        gc.collect()
    all_metrics["shuffled_es"] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in shuffled_es_accs.items()}

    # 7. RegCalMerge
    print("Testing Stage 7: RegCalMerge...")
    rcm_accs = collections.defaultdict(list)
    for seed in seeds:
        torch.manual_seed(seed)
        model_rcm = RegCalMerge(paramslist, model_wrapper, names, args.eval_datasets, args)
        run_adam_optimization(model_rcm, args.eval_datasets, args, beta=1.0, gamma=1.0, use_ccn=True, use_snew=True, steps=2)
        results = evaluate_model(model_rcm, args.eval_datasets, args)
        for k, v in results.items():
            rcm_accs[k].append(v)
        del model_rcm
        gc.collect()
    all_metrics["regcalmerge_opt"] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in rcm_accs.items()}

    # 8. Ablation Sweep
    print("Testing Stage 8: Ablation Grid Sweep...")
    ablation_results = {}
    beta_sweep = [0.0, 1.0]
    gamma_sweep = [0.0, 1.0]
    for b in beta_sweep:
        for g in gamma_sweep:
            grid_key = f"beta_{b}_gamma_{g}"
            grid_accs = collections.defaultdict(list)
            for seed in seeds:
                torch.manual_seed(seed)
                model_grid = RegCalMerge(paramslist, model_wrapper, names, args.eval_datasets, args)
                run_adam_optimization(model_grid, args.eval_datasets, args, beta=b, gamma=g, use_ccn=True, use_snew=True, steps=2)
                results = evaluate_model(model_grid, args.eval_datasets, args)
                for k, v in results.items():
                    grid_accs[k].append(v)
                del model_grid
                gc.collect()
            ablation_results[grid_key] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in grid_accs.items()}
    all_metrics["ablation_sweep"] = ablation_results

    os.makedirs('results_test', exist_ok=True)
    with open('results_test/metrics_test.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print("Verification complete! Saved to results_test/metrics_test.json")

if __name__ == '__main__':
    run_experiment_pipeline()
