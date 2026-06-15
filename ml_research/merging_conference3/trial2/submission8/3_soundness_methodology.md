# 3. Soundness and Methodology

## Clarity of the Description
The methodology is exceptionally well-structured and described with high mathematical precision. The paper sequentially lays out:
1. Task vector extraction and multi-task merging (Equations 1-2).
2. Flatness-aware expert fine-tuning using SAM (Equations 3-5).
3. The exact mathematical formulation of Uniform Pruning (Equations 6-7) and Adaptive Saliency-Based Budget Allocation (Equations 8-12).
4. The two scaling formulations (Global vs. Layer-wise Scaling) for Saliency-Based Pruning (Equation 13).
5. The final sparse weight fusion (Equation 14).

Every equation is fully defined, self-contained, and easy to follow.

## Appropriateness of Methods
The proposed methods are highly appropriate, elegant, and pragmatically grounded:
* **First-Order Magnitude Pruning:** Choosing magnitude-based pruning over second-order methods (like diagonal Hessian/Fisher matrices) is highly appropriate. Second-order curvature computation is extremely compute- and memory-intensive, especially for edge devices with tens of millions of parameters. First-order magnitude pruning has $O(d \log d)$ sorting complexity (taking milliseconds) and requires zero forward/backward passes during weight fusion, making it exceptionally practical.
* **Norm-Preserving Rescaling ($1/p$ and $1/p_l$ scaling):** This is a simple, highly effective heuristic. The ablation study proves its criticality: without rescaling, average accuracy collapses by 9.4% to 9.8% (dropping to ~80% at $p=0.10$). Rescaling restores the steering signal of the sparse task updates and is a brilliant, zero-overhead enabler.
* **Evaluation Setup:** Evaluating both standard AdamW and SAM experts allows a rigorous investigation of training-stage loss-landscape flatness and post-hoc sparsification resilience.

## Potential Technical Flaws / Limitations
* **"Norm-Preserving" Misnomer:** As the authors analytically derive in the Appendix, applying a $1/p$ scale factor to the deterministically pruned largest absolute parameters actually *amplifies* (rather than strictly preserves) the expected $L_1$ update norm by $2.58\times$ (Gaussian) to $3.30\times$ (Laplace). The authors are scientifically honest about this, characterizing it as a beneficial "Signal-Strength Boost" that is empirically superior to strict $L_1$-preserving scale factors (since coordinates are reduced by 90%). However, calling the method "Norm-Preserved" is slightly inaccurate mathematically, even if it is conceptually motivated by preventing norm shrinkage.
* **Academic Benchmark Scale:** The experiments are conducted on 4 classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) using disjoint splits of 1024 samples. While these represent diverse visual domains, they are relatively small-scale academic benchmarks. Evaluating on larger-scale datasets (e.g., ImageNet-scale, or text instruction datasets for LLMs) would strengthen the generalizability of the findings, though the authors do provide a scalability analysis in Appendix Section C.
* **Negative Result for NP-BTVP-S:** The proposed Saliency-Based Pruning (NP-BTVP-S) is slightly outperformed by global Uniform Pruning (NP-BTVP-U). While proposing a more complex adaptive allocation scheme only to find that the simpler uniform pruning is superior could be seen as a weakness, the authors turn this into a major strength by formally identifying and diagnosing the *Saliency Double-Bind*. This represents a valuable, high-signal contribution for practitioners.

## Reproducibility
The paper is exceptionally detailed, providing all necessary information for reproducibility:
* Exact model backbone and pre-trained weights (`laion2b_s34b_b79k` via `open_clip`).
* Specific fine-tuned target tensors (28.7M parameters: visual projection and all self-attention projection weights).
* Training hyperparameters (5 epochs, AdamW vs. SAM, learning rate $10^{-5}$, SAM perturbation radius $\rho = 0.002$).
* Precise dataset details and disjoint split sizes (1024 samples).
* Three independent random seeds (42, 100, 2026) for statistical verification (mean and standard deviation).

The methodology and experimental setup are fully transparent and highly reproducible.
