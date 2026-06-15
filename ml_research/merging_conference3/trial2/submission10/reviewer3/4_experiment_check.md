# Experimental Setup and Empirical Evaluation

## Critical Evaluation of the Experimental Setup
The authors design a seed-controlled experimental protocol across three independent seeds, which is highly disciplined and appropriate for managing optimization variance. 
- **Data Splits**: 512 labeled images for training heads, 64 unlabeled images for test-time adaptation, and standard test splits for final evaluation. The dual-scale protocol (small calibration vs. huge test set) is mathematically sound and simulates test-time adaptation conditions well.
- **Optimizers**: They evaluate both a zero-order derivative-free method (1+1 ES) and a first-order gradient descent method (Adam GD), which covers the primary optimization paradigms in test-time adaptation.

## Datasets and Baselines

### 1. Choice of Datasets (High Skepticism from a Practitioner's Lens)
The multi-task setup consists of **MNIST, FashionMNIST, CIFAR-10, and SVHN**. 
- To a practitioner in 2026, these are highly simplified, legacy, toy datasets. MNIST and FashionMNIST consist of tiny $28 \times 28$ grayscale images, while CIFAR-10 and SVHN consist of tiny $32 \times 32$ color images.
- Evaluating a visual foundation model (CLIP ViT-B/32) solely on these small-resolution toy datasets severely limits the generalizability and real-world significance of the findings. Real-world applications involve high-resolution, complex visual domains (e.g., ImageNet, DomainNet, PACS, COCO) or massive instruction-following textual domains (LLMs).
- The severe gradient imbalance observed in the experiments is highly influenced by mixing these grayscale digit datasets (MNIST/FashionMNIST), which have trivial logit distributions, with color object recognition datasets (CIFAR-10/SVHN). In actual industrial multi-task settings, practitioners rarely merge such highly disparate and simple toy tasks.

### 2. Baselines
The paper includes appropriate baselines for weight-space model merging:
- **Static Baselines**: Task Arithmetic, TIES-Merging, and DARE-Merging.
- **Adaptive Baselines**: SOTA Layer-wise AdaMerging (Adam GD and 1+1 ES).
- **Oracle Static Sweep**: A complete grid sweep of Task Arithmetic scales ($\lambda \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$) to establish the upper bound of static merging.

This is a comprehensive set of baselines for weight-space merging. However, the performance of TIES-Merging ($77.54\%$) and DARE-Merging ($73.67\%$) is surprisingly low, collapsing severely on SVHN ($49.46\%$ and $40.61\%$). This indicates that hard pruning and coordinate masking heuristics are highly fragile when applied to heterogeneous domains, further highlighting that the baseline configurations might be sub-optimal or overly sensitive.

### 3. Backbone Model
The evaluations are restricted to **CLIP ViT-B/32** (an isotropic Vision Transformer with $86\text{M}$ parameters and patch size of 32). 
- While standard in academic literature, ViT-B/32 is a relatively small and outdated backbone. It remains unproven whether these exact optimization paradoxes and representational alignment (measured via CKA) generalize to larger models (ViT-L/14, ViT-H/14), hierarchical vision backbones (Swin Transformers), or convolutional architectures (ConvNeXt).

## Do the Results Support the Claims?
- **Claim: Spatial Averaging Paradox exists**: Yes. The results in Table 1 strongly support this. Direct Task-wise AdaMerging collapses to $81.19\%$ (Adam GD), while post-hoc Spatial Averaging maintains a much higher $84.96\%$ (Adam GD).
- **Claim: Multi-task gradient imbalance explains the failure**: Yes. Under direct Task-wise AdaMerging, MNIST and FashionMNIST accuracies remain high, while CIFAR-10 and SVHN collapse. This supports the theory that easy-task logit-scaling gradients dominate the low-dimensional weight updates.
- **Claim: Layer-wise coefficients capture neural hierarchy**: Yes. Intra-Task Layer Shuffling (Adam GD) drops average accuracy from $88.05\%$ to $78.61\%$, confirming that coefficients are structurally specialized and cannot be randomly permuted without destroying performance.
- **Claim: Layer-wise AdaMerging overfits**: This claim is **not** fully supported in a practical sense. If SOTA layer-wise AdaMerging achieves the highest generalization accuracy ($88.05\%$) on the massive, disjoint test splits of all four tasks, calling its local optimization a "harmful transductive overfitting pathology" is conceptually flawed. The local, high-dimensional adaptation simply generalizes better than the spatially regularized counterpart.
