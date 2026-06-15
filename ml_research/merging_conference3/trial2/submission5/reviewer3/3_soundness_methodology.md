# 3. Soundness and Methodology Evaluation

This section critiques the mathematical formulation, physical motivations, and overall soundness of the methodology proposed in the paper.

## Technical and Mathematical Soundness
The mathematical formulation of Norm-Equalized Task Arithmetic (NETA) is exceptionally clear, rigorous, and technically sound. 

* **Clear Notation**: The problem setup (Section 3.1) clearly defines parameters, layers, task vectors, and active parameter scopes (e.g., visual encoder parameters being active while text encoder parameters are frozen in CLIP).
* **Closed-Form Formulation**: The equations for Frobenius norm calculation, mean norm, and NETA weight computation (Equations 3, 4, and 5) are standard, correct, and easy to parse.
* **Isotropic and Cumulative Norm Proofs**: The proofs in Section 3.4 mathematically verify that NETA achieves perfect magnitude isotropy (i.e., $\|\hat{\tau}_k^l\|_F \approx \mu^l$ as $\beta \rightarrow 0$) and preserves the cumulative sum of individual update norms at each layer (i.e., $\sum_{k=1}^K \|\hat{\tau}_k^l\|_F \approx \sum_{k=1}^K \|\tau_k^l\|_F$).
* **Scientific Honesty regarding Norm Contraction**: The authors deserve high praise for their scientific honesty regarding *directional norm contraction* (Equation 10). They explicitly note that preserving individual norms does not mathematically guarantee the preservation of the final merged update vector norm, which depends on the directional alignment (cosine similarity) between individual task vectors. They explain that scaling down dominant updates (like SVHN) can contract the overall merged vector, which explains the performance drops on SVHN and CIFAR-10. This is a very deep, mature, and rigorous observation.
* **Scale-Compensation Factor ($\gamma^l$)**: To resolve the norm contraction issue, they formulate an elegant closed-form compensation factor $\gamma^l$ (Equation 11) that rescales the merged NETA update vector to match the norm of standard Task Arithmetic. This is a brilliant, zero-shot, and mathematically sound extension that resolves the contraction without any optimization loops.

## Appropriateness of Methods and Physical Motivations
The engineering and design choices in NETA are highly appropriate and physically motivated:

1. **Layer-Wise vs. Model-Wide Normalization**: The paper provides a compelling justification for layer-wise scaling. Standard deep networks store general features in early layers and highly specialized features in deep layers. A model-wide normalization would let massive early-layer updates of complex tasks (like SVHN) continue to dwarf early-layer updates of easy tasks (like MNIST), preserving representation dominance in early layers. Layer-wise normalization guarantees that at each level of feature extraction, tasks contribute equally.
2. **Noise-Damping Stabilizer ($\beta$)**: In intermediate layers where an expert undergoes near-zero updates, standard normalization would inflate the scaling coefficient catastrophically, amplifying noise. Generalizing NETA to include a tunable stabilizer $\beta \in [10^{-3}, 10^{-2}]$ as a soft-thresholding filter is highly appropriate and prevents noise amplification.
3. **Composite Visual Input Grouping (Group 0)**: Normalizing extremely low-dimensional or structurally isolated parameters (like positional or class embeddings) independently yields highly unstable scaling factors because their absolute updates are very small. Grouping them jointly with the first Transformer block is an excellent and practical engineering decision that prevents early-stage spatial and positional distortions.

## Reproducibility
The reproducibility of the proposed methodology is **excellent**.
* **Algorithm Description**: Algorithm 1 provides a precise, step-by-step pseudo-code of NETA that is straightforward to implement in PyTorch or JAX.
* **Architecture and Parameters**: The paper specifies the active visual parameters in CLIP ViT-B/32, the grouping details (mapping positional embeddings, class embeddings, conv1, and pre-layernorm together with the first block parameters), and the exact parameter keys used.
* **Calibration/TTA Settings**: For the AdaMerging baselines, the paper specifies the learning rate ($5 \times 10^{-3}$), batch size (32), training epochs (20), and calibration set size (256 images).
* **Source Code/Checkpoints**: The authors state they use independently fine-tuned expert checkpoints and classification heads from the public Hugging Face hub, making it easy for external researchers to download the exact weights and reproduce the findings.
* **No Unlabeled Hacks**: The authors do not employ any hidden logic, reflection, or shortcuts. The methods are explicit and fully described.
