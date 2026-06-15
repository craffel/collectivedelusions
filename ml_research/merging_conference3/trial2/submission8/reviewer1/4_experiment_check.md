# 4. Experiment Check

## Critical Evaluation of the Experimental Setup
The experimental setup in this paper is highly robust and scientifically rigorous:
* **Target Parameters:** The authors target 28.7 million parameters (visual projection layers and all self-attention projection weights in the visual encoder). This is a highly representative subset of the visual backbone, focusing on the layers that undergo the most significant task adaptation during downstream fine-tuning.
* **Hyperparameter Specification:** Every key parameter—learning rate, optimizer, epochs, data volume, and the SAM perturbation radius $\rho=0.002$—is explicitly provided. The sensitivity analysis of $\rho$ in **Appendix C** is highly valuable, explaining why a smaller radius is required for pre-trained backbones on low-data regimes to prevent feature destabilization.

## Baselines and Evaluation Fairness
The authors avoid "baseline-handicapping" and ensure complete evaluation fairness:
* **Modern Baselines:** They compare against the two most prominent weight-space sparsification and merging methods: **TIES-Merging** and **DARE-Merging**.
* **Individual Hyperparameter Optimization:** Rather than using default hyperparameters, the authors swept and optimized the merging coefficient $\lambda \in [0.1, 1.0]$ with a step size of $0.1$ for *each baseline on each configuration*. This ensures that both TIES and DARE are evaluated at their absolute peak performance.
* **Statistical Power:** All experiments are conducted over **3 independent random seeds** (42, 100, 2026), and both the mean and standard deviation are reported for every configuration.

## Do the Results Support the Claims?
Yes, the results exceptionally support every primary claim made in the paper:

1. **Claim: Standalone expert optimization is highly successful.**
   * **Support:** Table 1 shows that standard AdamW and flatness-aware SAM experts specialize exceptionally well, achieving an average accuracy of 93.49% and 93.60%, respectively, vastly outperforming the initial zero-shot base model (60.36%).

2. **Claim: Loss landscape flatness (via SAM) does not inherently buffer task vectors against unstructured, coordinate-aligned magnitude pruning compared to AdamW.**
   * **Support:** Table 2 shows that at $p=0.10$ (90% sparsity), AdamW Uniform achieves **90.34%** while SAM Uniform achieves **90.32%** (nearly identical performance). At $p=0.05$ (95% sparsity), AdamW actually slightly leads (**89.62%** vs **89.49%**). 
   * **Statistical Validation:** To prove this equivalence rigorously, the authors perform a two-tailed paired $t$-test. Under AdamW at $p=0.10$, the $p$-value comparing Uniform and Saliency-Global is $0.96 > 0.05$. Under SAM, the $p$-value is $0.68 > 0.05$. This statistical indistinguishability conclusively validates their claim and challenges common landscape assumptions.

3. **Claim: Global Uniform Pruning is practically superior to layer-wise Saliency-Based Pruning due to the "Saliency Double-Bind."**
   * **Support:** Saliency-Global and Saliency-Layer are statistically indistinguishable from Uniform Pruning in terms of accuracy, yet Saliency is computationally far more complex. 
   * **The Double-Bind Evidence:** Table 4 in Appendix E shows that under 8-bit quantization (INT8), Saliency-Layer completely collapses back to the base zero-shot level (**60.36%**) because its high layer-wise scale factors ($1/p_l \approx 100\times$) amplify rounding noise. Conversely, Uniform Pruning holds strong at **90.20%** accuracy, demonstrating superior scale stability.

4. **Claim: Competitive performance against TIES and DARE.**
   * **Support:** At 90% sparsity ($p=0.10$), our Uniform Pruning (90.32% under SAM) performs competitively with DARE-Merging (**90.95%** under SAM) which operates at a significantly looser sparsity of 80% ($p_{\text{drop}}=0.80$). It also completely crushes TIES-Merging (**86.51%** under SAM) by over **3.81%** average accuracy while using only half the parameter budget (10% vs 20%).

5. **Claim: Immediate real-world deployment viability and storage compression.**
   * **Support:** Section 4.5 details that a CLIP expert is compressed from 114.8 MB to **22.96 MB** in sparse formats. When combined with INT8 quantization, it is compressed to merely **5.74 MB (a 40$\times$ compression)** while maintaining **90.20%** average accuracy under SAM (only a 0.12% drop). This is a stellar result of immediate practical value.
