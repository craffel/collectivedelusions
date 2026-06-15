# Soundness and Methodology Evaluation

An evaluation of the clarity, appropriateness of methods, potential technical flaws, and reproducibility of the proposed framework.

## Clarity of the Description
The description of the methodology is **excellent**. The paper is highly structured and clearly explains each step of the process:
* **Mathematical Formalization:** The task-vector extraction, SAM training formulation, Uniform Pruning (NP-BTVP-U), and Adaptive Saliency-Based Pruning (NP-BTVP-S) are defined mathematically with clear notation.
* **Rescaling Framing:** The authors are highly transparent and scientifically precise in clarifying that their "Norm-Preserving Rescaling" ($1/p$ or $1/p_l$) is actually a "Signal-Strength Boost" that amplifies the expected $L_1$ norm of the updates rather than keeping it strictly constant. This distinction is mathematically derived in Appendix A and helps prevent any confusion about what the scaling factor actually does.
* **Pseudocode and Heuristics:** The binary search method used to solve for the layer-specific budget allocation scaling factor $\alpha$ is clearly described. The selection of $L_1$ update magnitude as a first-order magnitude-based heuristic is well-justified for edge-deployment constraints, as it avoids expensive second-order Hessian or Fisher Information calculations.

---

## Appropriateness of Methods
The methods chosen are highly appropriate for the target problem of post-hoc task-vector compression:
* **Magnitude-Based Pruning:** Highly suitable for isolating the most important parameter updates in fine-tuned models, as verified by extensive prior work (e.g., TIES-Merging).
* **Rescaling Correction:** Crucial for counteracting the update norm shrinkage that occurs when zeroing out a large portion of parameters. Without this rescaling, the task vectors lose their steering capability, causing the merged model's performance to collapse toward the pre-trained base model.
* **Baselines and Backbone:** The choice of a pre-trained CLIP ViT-B/32 backbone is standard for model merging and task arithmetic research. The comparison against TIES-Merging and DARE-Merging is highly appropriate, as these are the leading state-of-the-art baselines for weight-space sparsification.

---

## Potential Technical Flaws and Limitations
The authors are careful and honest in evaluating the limitations and potential failure modes of their work:
1. **The Saliency Double-Bind:** The authors identify a significant technical limitation in their layer-wise budget allocation scheme (NP-BTVP-S). Under global rescaling, low-saliency layers are silenced while high-saliency layers dominate. Under layer-wise rescaling, low-saliency layers undergo extreme local noise amplification because their tiny budget $p_l$ results in a massive scaling factor $1/p_l \approx 100\times$, completely disrupting the scale harmony across the network. This is a deep and valuable finding that explains why the more complex Saliency-Based method fails to outperform the simpler Uniform method.
2. **Quantization Collapse:** In Appendix E, the authors show that under Saliency-Layer pruning combined with INT8 quantization, the extreme scaling factors of low-saliency layers magnify quantization rounding noise, leading to total model collapse.
3. **Variance Blowup ($L_2$ Reconstruction Error):** The mathematical derivation in Appendix A.3 honestly points out that while the $1/p$ rescaling preserves the $L_1$ update signal strength, it quadratically increases the $L_2$ reconstruction error/variance by a factor of $(\frac{1}{p}-1)^2 \approx 81$ for $p=0.10$. This theoretical trade-off is a high-signal observation that explains the small remaining performance gap between the sparse model and the dense baseline.

One minor point of concern is that the evaluation is limited to a relatively small CLIP ViT-B/32 backbone (28.7 million active parameters) and classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) with 1024 samples per task. While this is sufficient for a proof-of-concept, it is unclear if the results and the scale dynamics under the "Saliency Double-Bind" would shift when scaling up to extremely large language models (such as LLaMA-7B or 70B) or much larger downstream datasets.

---

## Reproducibility
The reproducibility of this work is **very high**:
* **Explicit Hyperparameters:** The training details (such as the AdamW and SAM learning rates, epochs, and especially the SAM perturbation radius $\rho = 0.002$) are clearly specified.
* **Detailed Formulas:** The exact mathematical formulations for mask generation and rescaling are provided.
* **Statistical Rigor:** The experiments are averaged over 3 independent random seeds with reported standard deviations, ensuring that the results are not due to random fluctuations.
* **Appendix Information:** The appendix contains a detailed sensitivity analysis of the SAM perturbation radius $\rho$ (Appendix D) and a preliminary quantization integration (Appendix E), providing all necessary implementation details for an expert reader to reproduce the results.
