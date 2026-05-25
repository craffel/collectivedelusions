import os
import sys
import numpy as np
import torch
from experiment import SimpleCNN, run_test_time_adaptation, set_seed
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def main():
    print("="*60)
    print("Starting Hyperparameter Sweep for TTMM Methods")
    print("="*60)
    sys.stdout.flush()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    sys.stdout.flush()
    
    set_seed(42)
    
    # 1. Setup Datasets & Stream
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    kmnist_imgs = np.load('kmnist-test-imgs.npz')['arr_0']
    kmnist_labels = np.load('kmnist-test-labels.npz')['arr_0']
    kmnist_imgs_t = torch.from_numpy(kmnist_imgs).float() / 255.0
    kmnist_imgs_t = (kmnist_imgs_t - 0.1307) / 0.3081
    kmnist_imgs_t = kmnist_imgs_t.unsqueeze(1)
    kmnist_labels_t = torch.from_numpy(kmnist_labels).long()
    kmnist_test = torch.utils.data.TensorDataset(kmnist_imgs_t, kmnist_labels_t)
    
    mnist_loader = DataLoader(mnist_test, batch_size=64, shuffle=True)
    fmnist_loader = DataLoader(fmnist_test, batch_size=64, shuffle=True)
    kmnist_loader = DataLoader(kmnist_test, batch_size=64, shuffle=True)
    
    mnist_iter = iter(mnist_loader)
    fmnist_iter = iter(fmnist_loader)
    kmnist_iter = iter(kmnist_loader)
    
    stream_batches = []
    for _ in range(10):
        stream_batches.append(next(mnist_iter))
    for _ in range(10):
        X, y = next(mnist_iter)
        noise = torch.randn_like(X) * 0.6
        stream_batches.append((X + noise, y))
    for _ in range(10):
        stream_batches.append(next(fmnist_iter))
    for _ in range(10):
        X, y = next(fmnist_iter)
        noise = torch.randn_like(X) * 0.6
        stream_batches.append((X + noise, y))
    for _ in range(10):
        stream_batches.append(next(kmnist_iter))
        
    # 2. Load Experts
    expert0 = SimpleCNN().to(device)
    expert1 = SimpleCNN().to(device)
    expert0.load_state_dict(torch.load("expert_mnist.pt", map_location=device, weights_only=True))
    expert1.load_state_dict(torch.load("expert_fmnist.pt", map_location=device, weights_only=True))
    
    # 3. Define Hyperparameter Grids
    etas = [0.01, 0.02, 0.05, 0.10]
    gamma_cs = [0.01, 0.02, 0.05, 0.10]
    alphas = [0.05, 0.15, 0.25]
    
    best_results = {
        "mog": {"acc": 0.0, "params": {}},
        "wb": {"acc": 0.0, "params": {}},
        "dwb": {"acc": 0.0, "params": {}},
        "sr_dwb": {"acc": 0.0, "params": {}}
    }
    
    # We open a results log file
    with open("sweep_results.txt", "w") as out_f:
        out_f.write("Method | eta | gamma_c | alpha | Clean MNIST | Noisy MNIST | Clean FMNIST | Noisy FMNIST | Novel KMNIST | Overall\n")
        out_f.write("---|---|---|---|---|---|---|---|---|---\n")
        out_f.flush()
        
        # MOG (BK-CoMerge) Sweep
        print("Sweeping MOG (BK-CoMerge)...")
        sys.stdout.flush()
        for eta in etas:
            for gamma_c in gamma_cs:
                accs, _ = run_test_time_adaptation(expert0, expert1, stream_batches, method="mog", device=device, eta=eta, gamma_c=gamma_c)
                overall = np.mean(accs)
                seg = [np.mean(accs[0:10]), np.mean(accs[10:20]), np.mean(accs[20:30]), np.mean(accs[30:40]), np.mean(accs[40:50])]
                out_f.write(f"mog | {eta} | {gamma_c} | - | {seg[0]*100:.2f}% | {seg[1]*100:.2f}% | {seg[2]*100:.2f}% | {seg[3]*100:.2f}% | {seg[4]*100:.2f}% | {overall*100:.2f}%\n")
                out_f.flush()
                if overall > best_results["mog"]["acc"]:
                    best_results["mog"] = {"acc": overall, "params": {"eta": eta, "gamma_c": gamma_c}, "seg": seg}
                print(f"  mog | eta: {eta} | gc: {gamma_c} | acc: {overall*100:.2f}%")
                sys.stdout.flush()
                    
        # WB-CoMerge Sweep
        print("Sweeping WB-CoMerge (Ours)...")
        sys.stdout.flush()
        for eta in etas:
            for gamma_c in gamma_cs:
                accs, _ = run_test_time_adaptation(expert0, expert1, stream_batches, method="wb", device=device, eta=eta, gamma_c=gamma_c)
                overall = np.mean(accs)
                seg = [np.mean(accs[0:10]), np.mean(accs[10:20]), np.mean(accs[20:30]), np.mean(accs[30:40]), np.mean(accs[40:50])]
                out_f.write(f"wb | {eta} | {gamma_c} | - | {seg[0]*100:.2f}% | {seg[1]*100:.2f}% | {seg[2]*100:.2f}% | {seg[3]*100:.2f}% | {seg[4]*100:.2f}% | {overall*100:.2f}%\n")
                out_f.flush()
                if overall > best_results["wb"]["acc"]:
                    best_results["wb"] = {"acc": overall, "params": {"eta": eta, "gamma_c": gamma_c}, "seg": seg}
                print(f"  wb | eta: {eta} | gc: {gamma_c} | acc: {overall*100:.2f}%")
                sys.stdout.flush()
                    
        # D-WB-CoMerge Sweep
        print("Sweeping D-WB-CoMerge (Ours, Dynamic)...")
        sys.stdout.flush()
        for eta in etas:
            for gamma_c in gamma_cs:
                for alpha in alphas:
                    accs, _ = run_test_time_adaptation(expert0, expert1, stream_batches, method="dwb", device=device, eta=eta, gamma_c=gamma_c, alpha=alpha)
                    overall = np.mean(accs)
                    seg = [np.mean(accs[0:10]), np.mean(accs[10:20]), np.mean(accs[20:30]), np.mean(accs[30:40]), np.mean(accs[40:50])]
                    out_f.write(f"dwb | {eta} | {gamma_c} | {alpha} | {seg[0]*100:.2f}% | {seg[1]*100:.2f}% | {seg[2]*100:.2f}% | {seg[3]*100:.2f}% | {seg[4]*100:.2f}% | {overall*100:.2f}%\n")
                    out_f.flush()
                    if overall > best_results["dwb"]["acc"]:
                        best_results["dwb"] = {"acc": overall, "params": {"eta": eta, "gamma_c": gamma_c, "alpha": alpha}, "seg": seg}
                    print(f"  dwb | eta: {eta} | gc: {gamma_c} | alpha: {alpha} | acc: {overall*100:.2f}%")
                    sys.stdout.flush()

        # SR-DWB-CoMerge Sweep
        print("Sweeping SR-DWB-CoMerge (Ours, Self-Referential)...")
        sys.stdout.flush()
        for eta in etas:
            for gamma_c in gamma_cs:
                for alpha in [0.01, 0.05, 0.15, 0.25]:
                    accs, _ = run_test_time_adaptation(expert0, expert1, stream_batches, method="sr_dwb", device=device, eta=eta, gamma_c=gamma_c, alpha=alpha)
                    overall = np.mean(accs)
                    seg = [np.mean(accs[0:10]), np.mean(accs[10:20]), np.mean(accs[20:30]), np.mean(accs[30:40]), np.mean(accs[40:50])]
                    out_f.write(f"sr_dwb | {eta} | {gamma_c} | {alpha} | {seg[0]*100:.2f}% | {seg[1]*100:.2f}% | {seg[2]*100:.2f}% | {seg[3]*100:.2f}% | {seg[4]*100:.2f}% | {overall*100:.2f}%\n")
                    out_f.flush()
                    if overall > best_results["sr_dwb"]["acc"]:
                        best_results["sr_dwb"] = {"acc": overall, "params": {"eta": eta, "gamma_c": gamma_c, "alpha": alpha}, "seg": seg}
                    print(f"  sr_dwb | eta: {eta} | gc: {gamma_c} | alpha: {alpha} | acc: {overall*100:.2f}%")
                    sys.stdout.flush()
                        
    print("\n" + "="*50)
    print("Sweep Completed!")
    print("="*50)
    for m, res in best_results.items():
        print(f"Best {m.upper()} accuracy: {res['acc']*100:.2f}% with params {res['params']}")
        if "seg" in res:
            print(f"  Segment accuracies: MNIST: {res['seg'][0]*100:.2f}%, Noisy MNIST: {res['seg'][1]*100:.2f}%, FMNIST: {res['seg'][2]*100:.2f}%, Noisy FMNIST: {res['seg'][3]*100:.2f}%, KMNIST: {res['seg'][4]*100:.2f}%")
    sys.stdout.flush()
        
if __name__ == "__main__":
    main()
