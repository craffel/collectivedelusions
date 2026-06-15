# GPU Memory and Latency Profiling Results

| Batch Size | Model | Inference Latency (ms) | Peak GPU Memory (MB) | Relative Memory Overhead |
| :--- | :--- | :---: | :---: | :---: |
| B=1 | OFS-Tune (Static) | 8.99 ms | 486.92 MB | - |
| B=1 | Linear Router | 11.74 ms | 486.36 MB | +-0.57 MB (-0.1%) |
| B=1 | EpiMerge (Ours) | 17.36 ms | 489.73 MB | +2.81 MB (0.6%) |
| B=8 | OFS-Tune (Static) | 9.01 ms | 502.34 MB | - |
| B=8 | Linear Router | 12.04 ms | 511.46 MB | +9.12 MB (1.8%) |
| B=8 | EpiMerge (Ours) | 19.42 ms | 521.11 MB | +18.77 MB (3.7%) |
| B=16 | OFS-Tune (Static) | 9.09 ms | 520.17 MB | - |
| B=16 | Linear Router | 12.13 ms | 538.49 MB | +18.32 MB (3.5%) |
| B=16 | EpiMerge (Ours) | 19.47 ms | 556.49 MB | +36.32 MB (7.0%) |
| B=32 | OFS-Tune (Static) | 9.11 ms | 557.39 MB | - |
| B=32 | Linear Router | 12.69 ms | 594.03 MB | +36.64 MB (6.6%) |
| B=32 | EpiMerge (Ours) | 19.57 ms | 629.88 MB | +72.49 MB (13.0%) |
| B=64 | OFS-Tune (Static) | 9.12 ms | 630.78 MB | - |
| B=64 | Linear Router | 12.65 ms | 702.83 MB | +72.05 MB (11.4%) |
| B=64 | EpiMerge (Ours) | 27.34 ms | 774.83 MB | +144.05 MB (22.8%) |

# Test-Time Adaptation and Routing Dynamics

| Input Task | MNIST Expert Weight | FashionMNIST Expert Weight | CIFAR-10 Expert Weight | SVHN Expert Weight |
| :--- | :---: | :---: | :---: | :---: |
| **MNIST** | 0.611 | 0.497 | 0.501 | 0.426 |
| **FashionMNIST** | 0.557 | 0.499 | 0.500 | 0.463 |
| **CIFAR10** | 0.505 | 0.500 | 0.500 | 0.497 |
| **SVHN** | 0.505 | 0.500 | 0.499 | 0.498 |


| Input Task | MNIST Row Gating | FashionMNIST Row Gating | CIFAR-10 Row Gating | SVHN Row Gating |
| :--- | :---: | :---: | :---: | :---: |
| **MNIST** | 0.516 | 0.513 | 0.515 | 0.498 |
| **FashionMNIST** | 0.509 | 0.507 | 0.509 | 0.499 |
| **CIFAR10** | 0.502 | 0.502 | 0.502 | 0.500 |
| **SVHN** | 0.502 | 0.502 | 0.502 | 0.500 |