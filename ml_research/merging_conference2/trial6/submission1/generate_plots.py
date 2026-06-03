import json
import matplotlib.pyplot as plt
import numpy as np

# Set matplotlib style for publication-quality plots
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

def plot_svd_decay():
    # Singular values for layer4.1.conv2 from svd_decay_results.json
    mnist_sv = [0.7292, 0.6692, 0.6420, 0.6046, 0.5561, 0.5298, 0.4999, 0.4872, 0.4275, 0.3627, 0.2862, 0.2737, 0.2453, 0.2351, 0.2289]
    fmnist_sv = [0.7984, 0.6878, 0.6441, 0.6035, 0.5756, 0.5374, 0.5113, 0.4810, 0.4704, 0.3667, 0.3056, 0.2884, 0.2588, 0.2531, 0.2345]
    cifar_sv = [0.5840, 0.5546, 0.5124, 0.4829, 0.4570, 0.4362, 0.4269, 0.4184, 0.3999, 0.3819, 0.3619, 0.3434, 0.3383, 0.3372, 0.3269]
    
    # Compute normalized squared singular values (variance explained)
    mnist_var = np.array(mnist_sv)**2
    fmnist_var = np.array(fmnist_sv)**2
    cifar_var = np.array(cifar_sv)**2
    
    # We plot the cumulative variance explained (simulated to up to rank 15)
    # The actual values in Table 7 show cumulative explained variance for Rank 1, 2, 4, 8:
    # e.g., MNIST: 9.9%, 18.2%, 32.7%, 52.7%
    # We will reconstruct the exact cumulative curves based on Table 7 SVD decay profiles
    ranks = [1, 2, 4, 8]
    mnist_cum = [9.9, 18.2, 32.7, 52.7]
    fmnist_cum = [11.2, 19.6, 33.3, 52.9]
    cifar_cum = [9.9, 18.1, 31.0, 50.8]
    
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(ranks, mnist_cum, marker='o', color='#1f77b4', linewidth=2, label='MNIST')
    ax.plot(ranks, fmnist_cum, marker='s', color='#ff7f0e', linewidth=2, label='Fashion-MNIST')
    ax.plot(ranks, cifar_cum, marker='^', color='#2ca02c', linewidth=2, label='CIFAR-10')
    
    ax.set_xlabel('Truncation SVD Rank ($r$)')
    ax.set_ylabel('Cumulative Explained Variance (%)')
    ax.set_title('Singular Value Decay (\\texttt{layer4.1.conv2})')
    ax.set_xticks(ranks)
    ax.grid(True)
    ax.legend(loc='lower right')
    
    # Highlight the 50% explained variance threshold
    ax.axhline(50, color='red', linestyle=':', alpha=0.7, label='50% Threshold')
    
    plt.tight_layout()
    plt.savefig('svd_decay.png', dpi=300)
    plt.close()
    print("SVD Decay plot saved as svd_decay.png")

def plot_hyperparameter_sensitivity():
    ranks = [1, 2, 4, 8]
    acc_reg_05 = [51.85, 58.10, 63.66, 67.94]
    acc_reg_01 = [52.61, 61.52, 67.45, 71.79]
    acc_reg_001 = [56.97, 64.42, 70.44, 74.15]
    
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(ranks, acc_reg_05, marker='o', linestyle='--', color='#d62728', linewidth=2, label='reg = 0.5')
    ax.plot(ranks, acc_reg_01, marker='s', linestyle='-', color='#1f77b4', linewidth=2, label='reg = 0.1')
    ax.plot(ranks, acc_reg_001, marker='D', linestyle='-.', color='#2ca02c', linewidth=2, label='reg = 0.01')
    
    ax.set_xlabel('SVD Rank ($r$)')
    ax.set_ylabel('Average Multi-Task Accuracy (%)')
    ax.set_title('SSR-Merge Hyperparameter Sensitivity ($N=128$)')
    ax.set_xticks(ranks)
    ax.grid(True)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig('hyperparam_sensitivity.png', dpi=300)
    plt.close()
    print("Hyperparameter sensitivity plot saved as hyperparam_sensitivity.png")

def plot_confusion_matrix():
    # Confusion matrix values from Table 5
    # Rows: MNIST, F-MNIST, CIFAR-10
    # Cols: MNIST, F-MNIST, CIFAR-10
    data = np.array([
        [947, 77, 0],
        [51, 973, 0],
        [0, 12, 1012]
    ])
    
    tasks = ['MNIST', 'Fashion-MNIST', 'CIFAR-10']
    
    # Normalize row-wise to get percentages
    row_sums = data.sum(axis=1, keepdims=True)
    normalized_data = data / row_sums * 100.0
    
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(normalized_data, cmap='Blues', interpolation='nearest', vmin=0, vmax=100)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_xticklabels(tasks)
    ax.set_yticklabels(tasks)
    
    # Rotate tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    for i in range(len(tasks)):
        for j in range(len(tasks)):
            acc_pct = normalized_data[i, j]
            count = data[i, j]
            text_color = "white" if acc_pct > 50 else "black"
            ax.text(j, i, f"{count}\n({acc_pct:.1f}%)",
                    ha="center", va="center", color=text_color, fontweight='bold')
            
    ax.set_xlabel('Routed (Predicted) Task')
    ax.set_ylabel('Ground Truth (GT) Task')
    ax.set_title('Routing Confusion Matrix ($N=128$, Manhattan)')
    
    fig.colorbar(im, ax=ax, label='Routing Rate (%)')
    plt.tight_layout()
    plt.savefig('routing_confusion.png', dpi=300)
    plt.close()
    print("Confusion matrix heatmap saved as routing_confusion.png")

if __name__ == '__main__':
    plot_svd_decay()
    plot_hyperparameter_sensitivity()
    plot_confusion_matrix()
