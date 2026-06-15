# 1. Summary of the Paper

## Main Topic and Approach
This paper addresses the problem of **representation scale mismatch** in training-free parameter-space model merging. When merging independent, fine-tuned expert models (sharing a common pretrained initialization) into a single multi-task model, standard linear parameter averaging (e.g., Task Arithmetic) suffers from severe task dominance. This happens because tasks with larger parameter updates or layers with disproportionate magnitudes overshadow others, leading to catastrophic interference. 

To resolve this, existing approaches rely on complex, high-overhead operations such as singular value decomposition (SVD) or active test-time optimization. In contrast, this work proposes a minimalist, linear-time $O(N)$ approach:
1. **Standard-Deviation Scaling (SD-Scale)** and its mathematically stable counterpart **Root-Mean-Square Scaling (RMS-Scale)**. 
2. **Layer-wise Normalization**: Normalizes each task vector layer-wise to unit standard deviation or RMS, establishing a balanced, isotropic update direction.
3. **Scale Calibration**: Projects the merged update back into the network's natural adaptation space by multiplying the average normalized direction by the average of the original task scales (std or RMS) at that layer.
4. **Parameter-Free RMS-Scale (PF-RMS)**: An analytical, tuning-free variant that counteracts the natural shrinkage of merged updates (due to parameter conflicts/orthogonality) by dynamically inverting the layer-wise alignment ratio $\alpha^l = \text{RMS}(\bar{\tau}_{\text{norm}}^l)$ at each layer.

## Key Findings
- **Rigorous Evaluation on SimpleCNN**: Under a multi-seed evaluation on MNIST, FashionMNIST, and KMNIST image classification with uncoordinated training schedules, tuned SD-Scale (73.23%) and RMS-Scale (73.22%) slightly outperform standard Task Arithmetic (72.50%) and Ties-Merging (71.77%) on average.
- **Tuning-Free Success of PF-RMS**: Out-of-the-box, the completely parameter-free PF-RMS variant achieves **72.23%** average accuracy, outperforming un-tuned Task Arithmetic (71.68%) and Ties-Merging (71.81%) without requiring a validation set or post-hoc grid search.
- **Physical Verification on OpenAI CLIP ViT-B/32 Weights**: Merging 36 projection layers from CLIP ViT-B/32 shows that RMS-Scale achieves the exact same activation-space cosine alignment and isotropic balance as cubic-complexity SVD Isotropic Merging, but with a massive **100$\times$ wall-clock speedup** (5.67ms vs 571.92ms per layer).
- **Ablation Studies**: Normalization and global scale calibration are shown to be essential, synergistic halves of a complete solution; omitting either catastrophically degrades accuracy.

## Explicitly Claimed Contributions and Evidence
1. **Demonstration of scale mismatch issues**: Provided empirical evidence that heterogeneous fine-tuning schedules lead to highly mismatched updates (e.g., FashionMNIST updates have up to 1.9$\times$ larger standard deviations than MNIST/KMNIST), causing dominance.
2. **Formulation of RMS-Scale and PF-RMS**: Outlined a clear, stable mathematical framework that avoids translation-invariance bias instabilities on small tensors like biases and derives analytical layer-wise scales.
3. **Quantitative multi-task evaluation with statistical rigor**: Reported mean and standard deviations across 3 independent seeds and data splits on a multi-task SimpleCNN benchmark.
4. **Mathematical proof of Frobenius Equivalence**: Proved that RMS normalization is mathematically equivalent to parameter-count-scaled Frobenius-norm normalization on weight matrices.
5. **Physical scalability verification on CLIP ViT-B/32**: Evaluated runtime and activation alignment on actual visual transformer projection weights, proving the cubic vs. linear complexity trade-off.
