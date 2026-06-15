# 3. Soundness and Methodology Check

## Clarity of Description
The methodology of the paper is exceptionally clear and well-written.
* **Formulations:** The classical parametric routers (Softmax and independent Sigmoid), Maximum-Entropy Zero-Initialization prior, and L2 regularized calibration objective are mathematically defined with precise notation.
* **Stress Test:** The covariance injection mechanism via a Toeplitz matrix parameterized by entanglement coefficient $\rho$ is clearly formulated, making the representation anisotropy stress test highly transparent.
* **Sandbox Architecture:** The 14-layer PyTorch-based Analytical Coordinate Sandbox (ICS) is meticulously described, detailing layer dimensions, the frozen boundary layer, the coordinate-based attraction dynamical system, and the unified distance-based classification head.

## Appropriateness of Methods
The evaluation methodology is highly appropriate for a scientific audit:
* **Analytical Sandbox (ICS):** Using a synthetic coordinate sandbox is an excellent methodological choice. It allows the authors to perform exhaustive parameter sweeps (sweeping $\rho \in [0.0, 0.5]$ and $N_{\text{cal}}$ from $32$ to $4000$ across multiple random seeds) that would be computationally prohibitive and structurally difficult to isolate in multi-billion parameter foundation models.
* **Real-World Validation:** Validating the sandbox insights on real pre-trained BERT-Tiny weights with LoRA adapters on actual GLUE datasets (SST-2 and QQP) is crucial to ensure that the findings are not mere artifacts of the synthetic coordinate simulation.
* **Ablations:** Conducting a $2\times2$ factorial initialization ablation, a layer-wise vs. layer-invariant ablation, and introducing EMA-SABLE are highly rigorous methods to isolate individual confounding factors.

## Potential Technical Flaws and Limitations (Practitioner's Perspective)
While the paper is methodologically strong, there are several key technical limitations and assumptions that a practitioner must keep in mind:

1. **Scale of the Foundation Model Validation:** The real-world validation is restricted to **BERT-Tiny** (4 encoder layers, hidden size 128) evaluated on custom LoRA adapters. This is an extremely compact toy model. It does not capture the high-dimensional manifolds, complex representations, or hardware compute bottlenecks of modern multi-billion parameter foundation models (such as LLaMA, Mistral, or ViT-B/16). Whether these results hold for large-scale generative models remains unproven.
2. **Direct Classifier Logit Blending:** In the BERT-Tiny experiments, the joint multi-task serving model blends classifier logits directly: $\text{logits} = \alpha_0 \cdot \text{classifier}_0(\text{pooled}) + \alpha_1 \cdot \text{classifier}_1(\text{pooled})$. This design assumes that all task-specific classifiers output logits of the exact same dimensionality. If we were to serve tasks with mismatched label spaces (e.g., combining 2-class sentiment analysis with a 10-class image classification task), this direct blending equation would crash immediately due to a shape mismatch. In realistic, heterogeneous multi-task serving, a general-purpose routing framework must instead route input samples directly to their respective task classifier heads without logit-level blending, or implement dynamic label space adapters.
3. **Under-fitted Experts:** The LoRA experts in the BERT-Tiny experiments are under-fitted, achieving standalone test accuracies of only $58.80\%$ on SST-2 and $65.60\%$ on QQP. Under-fitted experts generate noisier activation manifolds, which might distort ensembling behavior.
4. **Architectural Asymmetry in Gating Evaluation:** For BERT-Tiny, the classical parametric router is evaluated as a stateless, embedding-level gating model (at Layer 0) for efficiency and simplicity, whereas SABLE and ChemMerge compute routing decisions dynamically layer-by-layer. This structural asymmetry complicates direct comparison in the real-world validation.
5. **Absence of Train/Test Distribution Shift in Sandbox:** The Analytical Coordinate Sandbox assumes that the calibration (train) set and downstream test set are drawn from the same distribution. In realistic edge deployments, downstream test streams are characterized by temporal drift, task-order non-stationarity, or heteroscedastic noise. In such environments, an unregularized classical router would overfit to the clean calibration split, making proper regularization (or training-free priors) even more critical.

## Reproducibility
The paper is highly reproducible. The authors provide exact mathematical formulations, network dimensions, layer configurations, and optimization hyperparameters (Adam optimizer, learning rate of $10^{-3}$, 100 training epochs, ODE step size $\Delta t = 1.5$, reaction decay rate $K_{\text{decay}} = 0.3$, etc.). This detailed specification ensures that other researchers and practitioners can easily replicate the findings.
