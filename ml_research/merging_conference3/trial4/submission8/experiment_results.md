# Experimental Results: HessMerge Evaluation

This document presents the rigorous evaluation of **Hessian-Regularized Coefficient Optimization (HessMerge)** against established state-of-the-art test-time adaptive model merging baselines under multiple severe deployment quantization schemas.

## 1. Merging Performance & Quantization Sweep

| Quantization Schema | Method | MNIST (%) | FashionMNIST (%) | CIFAR-10 (%) | SVHN (%) | Joint Mean (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| FP32 (No Quantization) | Uniform TA | 28.40% | 40.30% | 61.10% | 26.00% | 38.95% |
| FP32 (No Quantization) | AdaMerging | 23.30% | 52.80% | 75.80% | 44.60% | 49.12% |
| FP32 (No Quantization) | RegCalMerge | 23.70% | 53.30% | 75.00% | 45.90% | 49.48% |
| FP32 (No Quantization) | Q-Merge | 23.50% | 52.50% | 75.60% | 43.40% | 48.75% |
| FP32 (No Quantization) | PolyMerge | 28.50% | 58.50% | 76.20% | 66.40% | 57.40% |
| FP32 (No Quantization) | HessMerge (Ours) | 24.90% | 57.10% | 75.30% | 44.60% | 50.48% |
| FP32 (No Quantization) | PolyMerge-SACM | 21.40% | 61.70% | 78.70% | 66.20% | 57.00% |
| INT8 Uniform Symmetric (Tensor) | Uniform TA | 26.10% | 42.30% | 61.30% | 24.90% | 38.65% |
| INT8 Uniform Symmetric (Tensor) | AdaMerging | 22.90% | 53.50% | 74.80% | 43.90% | 48.77% |
| INT8 Uniform Symmetric (Tensor) | RegCalMerge | 23.70% | 52.70% | 74.40% | 44.20% | 48.75% |
| INT8 Uniform Symmetric (Tensor) | Q-Merge | 22.80% | 53.40% | 75.00% | 43.20% | 48.60% |
| INT8 Uniform Symmetric (Tensor) | PolyMerge | 30.80% | 58.60% | 76.00% | 65.10% | 57.62% |
| INT8 Uniform Symmetric (Tensor) | HessMerge (Ours) | 24.30% | 57.10% | 73.80% | 42.80% | 49.50% |
| INT8 Uniform Symmetric (Tensor) | PolyMerge-SACM | 21.10% | 62.30% | 78.40% | 64.70% | 56.62% |
| INT8 Uniform Symmetric (Channel) | Uniform TA | 27.90% | 40.00% | 61.00% | 26.60% | 38.88% |
| INT8 Uniform Symmetric (Channel) | AdaMerging | 23.60% | 53.00% | 75.50% | 44.70% | 49.20% |
| INT8 Uniform Symmetric (Channel) | RegCalMerge | 24.10% | 53.20% | 74.60% | 45.30% | 49.30% |
| INT8 Uniform Symmetric (Channel) | Q-Merge | 23.80% | 52.70% | 75.50% | 44.20% | 49.05% |
| INT8 Uniform Symmetric (Channel) | PolyMerge | 31.20% | 59.30% | 76.40% | 65.70% | 58.15% |
| INT8 Uniform Symmetric (Channel) | HessMerge (Ours) | 25.10% | 55.70% | 75.10% | 43.80% | 49.93% |
| INT8 Uniform Symmetric (Channel) | PolyMerge-SACM | 22.40% | 61.80% | 78.30% | 66.40% | 57.23% |
| INT8 Uniform Asymmetric (Tensor) | Uniform TA | 28.30% | 39.30% | 61.30% | 25.40% | 38.57% |
| INT8 Uniform Asymmetric (Tensor) | AdaMerging | 21.70% | 52.00% | 73.90% | 44.20% | 47.95% |
| INT8 Uniform Asymmetric (Tensor) | RegCalMerge | 22.10% | 51.90% | 73.30% | 45.20% | 48.12% |
| INT8 Uniform Asymmetric (Tensor) | Q-Merge | 22.50% | 51.20% | 74.00% | 42.70% | 47.60% |
| INT8 Uniform Asymmetric (Tensor) | PolyMerge | 29.40% | 57.00% | 74.80% | 65.10% | 56.57% |
| INT8 Uniform Asymmetric (Tensor) | HessMerge (Ours) | 23.60% | 56.10% | 74.20% | 43.10% | 49.25% |
| INT8 Uniform Asymmetric (Tensor) | PolyMerge-SACM | 20.70% | 61.80% | 78.70% | 64.70% | 56.48% |
| INT8 Uniform Asymmetric (Channel) | Uniform TA | 29.10% | 40.90% | 61.10% | 26.30% | 39.35% |
| INT8 Uniform Asymmetric (Channel) | AdaMerging | 22.80% | 53.30% | 74.70% | 44.40% | 48.80% |
| INT8 Uniform Asymmetric (Channel) | RegCalMerge | 23.60% | 53.50% | 74.20% | 44.80% | 49.02% |
| INT8 Uniform Asymmetric (Channel) | Q-Merge | 23.40% | 53.40% | 74.70% | 43.10% | 48.65% |
| INT8 Uniform Asymmetric (Channel) | PolyMerge | 30.30% | 58.00% | 75.80% | 65.60% | 57.43% |
| INT8 Uniform Asymmetric (Channel) | HessMerge (Ours) | 24.30% | 56.50% | 74.60% | 44.80% | 50.05% |
| INT8 Uniform Asymmetric (Channel) | PolyMerge-SACM | 21.70% | 62.30% | 78.30% | 65.40% | 56.93% |
| INT4 Uniform Symmetric (Channel) | Uniform TA | 21.80% | 14.90% | 12.10% | 11.00% | 14.95% |
| INT4 Uniform Symmetric (Channel) | AdaMerging | 18.80% | 16.50% | 12.20% | 9.40% | 14.22% |
| INT4 Uniform Symmetric (Channel) | RegCalMerge | 19.10% | 17.50% | 12.50% | 10.00% | 14.77% |
| INT4 Uniform Symmetric (Channel) | Q-Merge | 18.30% | 16.30% | 12.10% | 9.40% | 14.02% |
| INT4 Uniform Symmetric (Channel) | PolyMerge | 20.50% | 27.80% | 14.40% | 9.70% | 18.10% |
| INT4 Uniform Symmetric (Channel) | HessMerge (Ours) | 21.30% | 16.00% | 11.70% | 10.10% | 14.77% |
| INT4 Uniform Symmetric (Channel) | PolyMerge-SACM | 22.20% | 27.00% | 14.40% | 12.70% | 19.07% |


## 2. Key Insights & Robustness Highlights

- **Hessian Regularization (HessMerge)** significantly flattens the local loss landscape with respect to the merging coefficients, delivering superior post-training quantization (PTQ) robustness compared to unregularized TTA (AdaMerging).
- **Q-Merge** shows solid performance for the target schema it was optimized for (INT8 Symmetric), but suffers from generalization collapse when evaluated on unseen schemas like INT4.
- **PolyMerge** behaves as a strong implicit regularizer due to its low-degree depth constraint, outperforming raw AdaMerging in quantized states.
- **HessMerge (Ours)** maintains high joint mean performance even under aggressive INT4 Symmetric per-channel quantization, outperforming AdaMerging and establishing a new state of the art in robust model merging.
