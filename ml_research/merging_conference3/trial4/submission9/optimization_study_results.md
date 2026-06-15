# Optimization Steps and Generalization Study (500 Steps)

We systematically sweep the number of optimization steps $T \in \{40, 100, 250, 500\}$ on the 128-sample-per-task validation split to evaluate the convergence and generalization behavior of high-dimensional lyer-group-wise tuning (AdaMerging, 56 continuous parameters; ZipMerge, 70 parameters) against our low-dimensional TLC-Tune global coefficient scaling (4 parameters).

### 1. AdaMerging (Dense, 56 parameters)
| Optimization Steps | Best Validation Score | Test Mean Accuracy | Test Accs (MNIST/F-MNIST/CIFAR-10/SVHN) |
|---|---|---|---|
| 40 | 0.1725 | 0.3219 | ['0.1374', '0.2777', '0.5818', '0.2907'] |
| 100 | 0.1725 | 0.3219 | ['0.1374', '0.2777', '0.5818', '0.2907'] |
| 250 | 0.1725 | 0.3219 | ['0.1374', '0.2777', '0.5818', '0.2907'] |
| 500 | 0.1725 | 0.3219 | ['0.1374', '0.2777', '0.5818', '0.2907'] |

### 2. ZipMerge (p=0.5, 70 parameters)
| Optimization Steps | Best Validation Score | Test Mean Accuracy | Test Accs (MNIST/F-MNIST/CIFAR-10/SVHN) |
|---|---|---|---|
| 40 | 0.1732 | 0.2574 | ['0.1446', '0.2730', '0.4715', '0.1405'] |
| 100 | 0.1732 | 0.2574 | ['0.1446', '0.2730', '0.4715', '0.1405'] |
| 250 | 0.1732 | 0.2574 | ['0.1446', '0.2730', '0.4715', '0.1405'] |
| 500 | 0.1732 | 0.2574 | ['0.1446', '0.2730', '0.4715', '0.1405'] |

### 3. TLC-Tune EPM (Dense, 4 parameters)
| Optimization Steps | Best Validation Score | Test Mean Accuracy | Test Accs (MNIST/F-MNIST/CIFAR-10/SVHN) |
|---|---|---|---|
| 40 | 0.2375 | 0.4482 | ['0.1803', '0.3709', '0.4984', '0.7433'] |
| 100 | 0.2375 | 0.4482 | ['0.1803', '0.3709', '0.4984', '0.7433'] |
| 250 | 0.2375 | 0.4482 | ['0.1803', '0.3709', '0.4984', '0.7433'] |
| 500 | 0.2375 | 0.4482 | ['0.1803', '0.3709', '0.4984', '0.7433'] |

### 4. TLC-Tune EPM (p=0.5, 4 parameters)
| Optimization Steps | Best Validation Score | Test Mean Accuracy | Test Accs (MNIST/F-MNIST/CIFAR-10/SVHN) |
|---|---|---|---|
| 40 | 0.3646 | 0.3433 | ['0.3271', '0.4167', '0.3001', '0.3293'] |
| 100 | 0.3646 | 0.3433 | ['0.3271', '0.4167', '0.3001', '0.3293'] |
| 250 | 0.3646 | 0.3433 | ['0.3271', '0.4167', '0.3001', '0.3293'] |
| 500 | 0.3646 | 0.3433 | ['0.3271', '0.4167', '0.3001', '0.3293'] |

