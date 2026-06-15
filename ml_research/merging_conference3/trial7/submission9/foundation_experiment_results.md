# High-Dimensional ResNet-18 Foundation Experiment Results

This experiment extracts 512-dimensional real-world image representations from a pre-trained ImageNet ResNet-18 model and trains a 2-layer MLP classifier on top, evaluating SABLE's performance on real foundation features.

- **Expert Ceiling Joint Mean:** 74.80% (MNIST Expert: 75.00%, F-MNIST Expert: 74.60%)
- **Uniform Merging:** 51.10%
- **PFSR Homogeneous:** 69.40%
- **PFSR Heterogeneous:** 49.30% (Heterogeneity Collapse!)

## Standard Stream Accuracies (MNIST + FashionMNIST)

| Configuration | r=2 | r=4 | r=8 | r=16 |
| --- | --- | --- | --- | --- |
| SABLE Strict (Support-16) | 57.20% | 58.70% | 66.30% | 69.30% |
| SABLE Strict (Naive Zero) | 51.30% | 53.70% | 58.60% | 58.20% |
| SABLE Strict (Refined Zero) | 54.20% | 55.90% | 59.60% | 61.60% |
| SABLE Hybrid (Support-16) | 62.10% | 58.90% | 66.30% | 69.30% |
| SABLE Hybrid (Naive Zero) | 56.20% | 52.70% | 58.00% | 58.20% |
| SABLE Hybrid (Refined Zero) | 57.20% | 55.60% | 59.90% | 61.60% |

## Domain-Confounded Blended Streams (Recall@2 Joint Success)

- Uniform Merging: 19.00%
- PFSR Weight Merging: 18.00%
- SABLE Strict (r=2) [Support] Soft: 22.00% | Hard: 21.00%
- SABLE Strict (r=2) [Refined Zero] Soft: 25.00% | Hard: 23.00%
- SABLE Hybrid (r=2) [Support] Soft: 26.00% | Hard: 24.00%
- SABLE Hybrid (r=2) [Refined Zero] Soft: 26.00% | Hard: 27.00%
- SABLE Strict (r=8) [Support] Soft: 15.00% | Hard: 17.00%
- SABLE Strict (r=8) [Refined Zero] Soft: 18.00% | Hard: 16.00%
- SABLE Hybrid (r=8) [Support] Soft: 15.00% | Hard: 16.00%
- SABLE Hybrid (r=8) [Refined Zero] Soft: 18.00% | Hard: 16.00%
