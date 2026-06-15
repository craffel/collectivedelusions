import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import DataLoader, Subset
import timm
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

TASKS = ["MNIST", "FashionMNIST", "CIFAR10", "SVHN"]

# Custom Gaussian Noise transform
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std

class FeatureExtractor:
    def __init__(self, device):
        print("Initializing vit_tiny_patch16_224 backbone model...")
        self.device = device
        self.model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
        self.model.eval()
        self.model.to(device)
        self.features = []
        self.hook = self.model.blocks[2].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.features.append(output[:, 0].detach().cpu().clone())

    def extract(self, dataloader):
        self.features = []
        with torch.no_grad():
            for imgs, _ in dataloader:
                imgs = imgs.to(self.device)
                _ = self.model(imgs)
        return torch.cat(self.features, dim=0)

    def close(self):
        self.hook.remove()

def get_dataloaders(noise_std=0.0):
    print(f"Preparing loaders for image-level noise_std = {noise_std}...")
    
    transform_rgb_clean = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_gray_clean = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if noise_std > 0.0:
        transform_rgb_noisy = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            AddGaussianNoise(0.0, noise_std),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transform_gray_noisy = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            AddGaussianNoise(0.0, noise_std),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform_rgb_noisy = transform_rgb_clean
        transform_gray_noisy = transform_gray_clean

    loaders = {}
    
    for task in TASKS:
        # Load test set (with potential noise)
        if task == "MNIST":
            test_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transform_gray_noisy)
        elif task == "FashionMNIST":
            test_dataset = dsets.FashionMNIST(root='./data', train=False, download=True, transform=transform_gray_noisy)
        elif task == "CIFAR10":
            test_dataset = dsets.CIFAR10(root='./data', train=False, download=True, transform=transform_rgb_noisy)
        elif task == "SVHN":
            test_dataset = dsets.SVHN(root='./data', split='test', download=True, transform=transform_rgb_noisy)
            
        test_indices = list(range(100)) # subset of 100 for fast CPU evaluation
        test_subset = Subset(test_dataset, test_indices)
        loaders[task] = DataLoader(test_subset, batch_size=32, shuffle=False)
        
    return loaders

# ShrunkGMM Implementation
class ShrunkGMM(GaussianMixture):
    def __init__(self, n_components=1, target_type='global_diagonal', random_state=42, **kwargs):
        super().__init__(n_components=n_components, covariance_type='diag', random_state=random_state, **kwargs)
        self.target_type = target_type
        
    def fit(self, X, y=None):
        super().fit(X, y)
        resp = self.predict_proba(X)
        n_samples, n_features = X.shape
        shrunk_covariances = np.zeros_like(self.covariances_)
        
        if self.target_type == 'global_diagonal':
            global_target = np.var(X, axis=0)
            global_target = np.clip(global_target, 1e-5, None)
            
        for m in range(self.n_components):
            sigmas = self.covariances_[m]
            if self.target_type == 'spherical':
                T = np.mean(sigmas) * np.ones_like(sigmas)
            elif self.target_type == 'global_diagonal':
                T = global_target
                
            w = resp[:, m]
            W = np.sum(w)
            
            diffs = (X - self.means_[m])**2
            var_of_vars = np.zeros(n_features)
            for j in range(n_features):
                var_of_vars[j] = np.sum(w**2 * (diffs[:, j] - sigmas[j])**2) / (W**2 + 1e-8)
                
            sum_var = np.sum(var_of_vars)
            sum_diff = np.sum((sigmas - T)**2)
            alpha_opt = sum_var / sum_diff if sum_diff > 1e-8 else 1.0
            alpha_opt = np.clip(alpha_opt, 0.0, 1.0)
            
            shrunk_covariances[m] = (1.0 - alpha_opt) * sigmas + alpha_opt * T
            
        self.covariances_ = shrunk_covariances
        self.precisions_cholesky_ = 1.0 / np.sqrt(shrunk_covariances)
        return self

def main():
    device = torch.device("cpu")
    print("Loading clean calibration features...")
    # Read the cached extracted_features.pt
    features = torch.load("extracted_features.pt", map_location=device)
    
    # Compute clean task centroids
    centroids = {}
    for task in TASKS:
        train_feats = features[task]["train"][:64] # N=64 calibration size
        centroids[task] = torch.mean(train_feats, dim=0)
        
    def map_coords(feats):
        coords_list = []
        for task in TASKS:
            centroid = centroids[task]
            norm_feats = torch.norm(feats, p=2, dim=1, keepdim=True)
            norm_centroid = torch.norm(centroid, p=2)
            sim = torch.mm(feats, centroid.view(-1, 1)) / (norm_feats * norm_centroid + 1e-8)
            coords_list.append(sim)
        return torch.cat(coords_list, dim=1).numpy()

    # Get calibration coordinates (clean)
    calib_coords = {}
    for task in TASKS:
        calib_coords[task] = map_coords(features[task]["train"][:64])
        
    extractor = FeatureExtractor(device)
    
    # Evaluate at multiple input-level noise standard deviations
    noise_stds = [0.0, 0.05, 0.1, 0.2]
    
    results = {m: [] for m in ["Raw Cosine", "Unreg GMM", "Ridge GMM", "SRC-DE"]}
    
    for std in noise_stds:
        print(f"\n==============================================")
        print(f"Running evaluation under image noise std = {std}")
        print(f"==============================================")
        
        # Load and extract test features under current image noise level
        loaders = get_dataloaders(std)
        test_coords = {}
        for task in TASKS:
            print(f"Extracting test features for {task} under image noise...")
            test_feats = extractor.extract(loaders[task])
            test_coords[task] = map_coords(test_feats)
            
        task_aucs = {m: [] for m in results}
        
        # OOD Rejection evaluation
        for task_idx, task in enumerate(TASKS):
            id_coords = test_coords[task]
            ood_coords_list = []
            for ood_task in TASKS:
                if ood_task == task:
                    continue
                ood_coords_list.append(test_coords[ood_task])
            ood_coords = np.concatenate(ood_coords_list, axis=0)
            
            y_true = np.concatenate([np.ones(len(id_coords)), np.zeros(len(ood_coords))])
            
            # 1. Raw Cosine
            scores_cos = np.concatenate([id_coords[:, task_idx], ood_coords[:, task_idx]], axis=0)
            task_aucs["Raw Cosine"].append(roc_auc_score(y_true, scores_cos))
            
            # 2. GMM Models (M=1)
            X_calib = calib_coords[task]
            
            # Unregularized
            gmm_unreg = GaussianMixture(n_components=1, covariance_type='diag', random_state=42, reg_covar=1e-5)
            gmm_unreg.fit(X_calib)
            id_unreg = gmm_unreg.score_samples(id_coords)
            ood_unreg = gmm_unreg.score_samples(ood_coords)
            task_aucs["Unreg GMM"].append(roc_auc_score(y_true, np.concatenate([id_unreg, ood_unreg])))
            
            # Ridge
            gmm_ridge = GaussianMixture(n_components=1, covariance_type='diag', random_state=42, reg_covar=1e-5)
            gmm_ridge.fit(X_calib)
            gmm_ridge.covariances_ = gmm_ridge.covariances_ + 1e-4
            gmm_ridge.precisions_cholesky_ = 1.0 / np.sqrt(gmm_ridge.covariances_)
            id_ridge = gmm_ridge.score_samples(id_coords)
            ood_ridge = gmm_ridge.score_samples(ood_coords)
            task_aucs["Ridge GMM"].append(roc_auc_score(y_true, np.concatenate([id_ridge, ood_ridge])))
            
            # SRC-DE
            gmm_shrunk = ShrunkGMM(n_components=1, random_state=42, reg_covar=1e-5)
            gmm_shrunk.fit(X_calib)
            id_shrunk = gmm_shrunk.score_samples(id_coords)
            ood_shrunk = gmm_shrunk.score_samples(ood_coords)
            task_aucs["SRC-DE"].append(roc_auc_score(y_true, np.concatenate([id_shrunk, ood_shrunk])))
            
        print(f"\nImage Noise {std:.2f} results:")
        for m in results:
            mean_auc = np.mean(task_aucs[m])
            results[m].append(mean_auc)
            print(f"  {m:12s}: Mean AUC = {mean_auc:.4f}")
            
    extractor.close()
    
    print("\n\n==============================================")
    print("FINAL SUMMARY - INPUT-LEVEL CORRUPTION AUDIT (M=1, N=64)")
    print("==============================================")
    print("Image Noise Std | Raw Cosine | Unreg GMM | Ridge GMM | SRC-DE")
    print("--------------------------------------------------------------")
    for idx, std in enumerate(noise_stds):
        print(f"{std:15.2f} | {results['Raw Cosine'][idx]:.4f}     | {results['Unreg GMM'][idx]:.4f}    | {results['Ridge GMM'][idx]:.4f}    | {results['SRC-DE'][idx]:.4f}")
    print("==============================================")

if __name__ == "__main__":
    main()
