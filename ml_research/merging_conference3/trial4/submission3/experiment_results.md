# Re-Quantization Adapter Merging Audit Report

## Part 1: High-Precision Baselines
| Method | MNIST Accuracy | FashionMNIST Accuracy | CIFAR-10 Accuracy | SVHN Accuracy | Mean Accuracy |
|---|---|---|---|---|---|
| Unmerged FP16 Experts | 98.20% | 88.20% | 95.00% | 94.00% | 93.85% |
| Naive FP16 Merge | 45.40% | 72.40% | 89.40% | 59.40% | 66.65% |

## Part 2: Quantization Configuration - INT8 Symmetric Per-Channel
| Method | MNIST Accuracy | FashionMNIST Accuracy | CIFAR-10 Accuracy | SVHN Accuracy | Mean Accuracy |
|---|---|---|---|---|---|
| Naive-RQ | 44.60% | 72.40% | 89.60% | 58.80% | 66.35% |
| Q-then-M | 45.60% | 72.40% | 89.40% | 59.40% | 66.70% |
| AdaMerging (PH-Q) | 69.40% | 70.60% | 81.20% | 59.20% | 70.10% |
| SAWS [Proposed] | 47.20% | 70.60% | 90.20% | 71.00% | 69.75% |
| QA-ACS [Proposed] | 58.00% | 74.00% | 86.20% | 59.20% | 69.35% |

## Part 2: Quantization Configuration - INT4 Symmetric Per-Channel
| Method | MNIST Accuracy | FashionMNIST Accuracy | CIFAR-10 Accuracy | SVHN Accuracy | Mean Accuracy |
|---|---|---|---|---|---|
| Naive-RQ | 44.40% | 70.60% | 87.80% | 56.60% | 64.85% |
| Q-then-M | 51.20% | 70.40% | 87.00% | 53.60% | 65.55% |
| AdaMerging (PH-Q) | 70.40% | 66.20% | 79.60% | 59.00% | 68.80% |
| SAWS [Proposed] | 43.20% | 70.80% | 89.80% | 67.40% | 67.80% |
| QA-ACS [Proposed] | 60.60% | 65.80% | 81.80% | 63.80% | 68.00% |

## Part 2: Quantization Configuration - INT4 Asymmetric Per-Channel
| Method | MNIST Accuracy | FashionMNIST Accuracy | CIFAR-10 Accuracy | SVHN Accuracy | Mean Accuracy |
|---|---|---|---|---|---|
| Naive-RQ | 36.80% | 72.40% | 88.60% | 55.00% | 63.20% |
| Q-then-M | 39.00% | 72.20% | 87.80% | 55.40% | 63.60% |
| AdaMerging (PH-Q) | 68.00% | 68.40% | 81.00% | 55.60% | 68.25% |
| SAWS [Proposed] | 40.40% | 70.80% | 90.00% | 67.80% | 67.25% |
| QA-ACS [Proposed] | 42.20% | 65.80% | 84.40% | 66.60% | 64.75% |

## Part 2: Quantization Configuration - INT4 Symmetric Per-Tensor
| Method | MNIST Accuracy | FashionMNIST Accuracy | CIFAR-10 Accuracy | SVHN Accuracy | Mean Accuracy |
|---|---|---|---|---|---|
| Naive-RQ | 42.00% | 62.80% | 78.60% | 43.60% | 56.75% |
| Q-then-M | 44.40% | 68.40% | 81.60% | 44.00% | 59.60% |
| AdaMerging (PH-Q) | 63.20% | 56.00% | 69.80% | 40.00% | 57.25% |
| SAWS [Proposed] | 44.80% | 66.00% | 70.40% | 44.40% | 56.40% |
| QA-ACS [Proposed] | 37.80% | 66.20% | 79.00% | 45.00% | 57.00% |

