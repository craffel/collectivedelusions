# 3. Soundness and Methodology

## Clarity of the Description
The methodology is exceptionally well-described and mathematically clear. The authors lay out the core equations step-by-step, starting from task vector extraction and standard Task Arithmetic merging, moving to SAM optimization, and then presenting the formalisms for both global Uniform Pruning (NP-BTVP-U) and Adaptive Saliency-Based Pruning (NP-BTVP-S).

Furthermore, the mathematical rigor in **Appendix A** is impressive. It provides:
1. An analytical derivation of the expected $L_1$ norm ratio under a Laplace distribution of task vector updates (yielding $\frac{\mathbb{E}[\|\tilde{\tau}^{(p)}\|_1]}{\mathbb{E}[\|\tau\|_1]} = 1 - \ln p$).
2. A corresponding derivation under a Gaussian distribution (yielding $\frac{1}{p} e^{-\frac{t_p^2}{2\sigma^2}}$).
3. A formal analysis of the expected $L_2$ reconstruction error, highlighting the variance-bias trade-off (maintaining low $L_1$ bias but increasing $L_2$ parameter distortion).

This level of detailed formal analysis is rare and adds significant weight to the paper's theoretical foundations.

## Appropriateness of Methods
From a practical deployment standpoint, the selected methods are highly appropriate:
* **Magnitude-Based Pruning:** It is completely training-free, requires zero forward or backward passes during weight fusion, and runs in milliseconds (with $O(d \log d)$ sorting complexity). This avoids the heavy compute and memory overhead of second-order Hessian calculations (e.g., Fisher Information or Hessian diagonals), which would be completely unviable for resource-constrained edge hardware.
* **Deterministic Rescaling ($1/p$):** Highly appropriate for compensating for the zeroed-out coordinates. By amplifying the surviving parameters, it successfully steers the dense, frozen base pre-trained model.
* **Evaluation Backbone (CLIP ViT-B/32):** A standard, highly representative foundation model widely used in weight merging and model interpolation research.
* **Dataset Selection (MNIST, FashionMNIST, CIFAR-10, SVHN):** These 4 diverse, disjoint classification tasks are highly standard and allow for a clear, multi-task merging validation across distinct visual domains.

## Potential Technical Flaws and Limitations
While the methodology is sound, there are a few minor limitations and areas that warrant critical observation:

1. **"Norm-Preserved" Naming Clarification:**
   As the authors transparently acknowledge, the name "Norm-Preserved" is conceptually a slight misnomer. Since they apply $1/p$ to the *largest* (deterministically selected) updates, they mathematically amplify the expected $L_1$ update norm ($3.30\times$ for Laplace, $2.58\times$ for Gaussian at $p=0.10$). Thus, the method acts as a **"Signal-Strength Boost"** rather than a strict norm-preservation identity. Although this over-scaling is highly beneficial in practice, the nomenclature could lead to initial confusion. The authors' upfront honesty about this is highly commendable and mitigates this concern.
2. **Setup Scale Constraint:**
   The empirical validation is focused on a low-data fine-tuning setup (1024 samples per dataset) and a relatively modest model size (CLIP ViT-B/32, fine-tuning 28.7 million parameters). While this setup is scientifically rigorous (featuring 3 random seeds), the generalizability of the "Saliency Double-Bind" and the coordinate-aligned resilience should eventually be verified on larger architectures (such as LLaMA-7B or ViT-L) and larger-scale datasets. The authors discuss this limitation in Appendix B, which is satisfying, but a larger-scale benchmark would make the empirical findings even more definitive.

## Reproducibility
The paper exhibits an exceptionally high bar for reproducibility:
* All training and optimization hyperparameters are clearly reported (learning rate $10^{-5}$, AdamW, 5 epochs, SAM perturbation radius $\rho=0.002$, 1024-sample training/test splits).
* The datasets are standard and open-source.
* The baseline setups for TIES-Merging and DARE-Merging are comprehensively detailed (including the specific hyperparameter sweeps and thresholds used).
* The algorithms (especially the binary search for layer-wise budget allocation) are mathematically defined down to the exact index, enabling any expert reader to easily reimplement the framework.
