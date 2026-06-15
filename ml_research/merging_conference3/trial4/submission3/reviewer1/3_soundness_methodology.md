# Soundness & Methodology

## Clarity of Description
The mathematical and algorithmic description in the paper is exceptionally clear and rigorous:
- **Formulation:** The paper details the exact mathematics of QLoRA adapter merging, dequantization, uniform symmetric/asymmetric quantization (Section 3.1), and the magnitude discrepancy (Section 3.2).
- **SAWS:** The closed-form derivation of the scale factor $\gamma^l$ and the weight alignment factor $c^l$ via quadratic optimization is highly elegant and easy to follow.
- **QA-ACS:** The test-time entropy minimization objective and its optimization through the Straight-Through Estimator (STE) are well-articulated, and Algorithm 1 provides a step-by-step pseudo-code.
- **Related Work and Contextualization:** The paper positions itself very clearly among PEFT, model merging, and post-training quantization, outlining clear gaps in existing literature.

## Appropriateness of Methods
The evaluation methodology is highly rigorous and appropriate:
- **Multi-Axial Audit:** Examining four distinct quantization configurations (INT8/INT4, symmetric/asymmetric, per-tensor/per-channel) ensures a robust, multi-dimensional assessment.
- **Relative Frobenius Error:** Auditing the double-quantization format-shift noise using weight-space reconstruction error (Table 1) is an excellent way to isolate representational degradation.
- **Individual Expert Auditing Control Experiment:** This is a methodologically brilliant design. By quantizing unmerged experts individually, the authors successfully decouple pre-existing task interference from quantization-induced degradation, which has been a major confounding variable in prior literature.
- **Ecosystem Baselines:** The paper compares against appropriate baselines, including Naive FP16, Naive-RQ, Q-then-M (co-existence), and AdaMerging (PH-Q).

## Potential Technical Flaws & Limitations
1. **Inefficacy of SAWS under Per-Tensor Constraints:** Under the aggressive 4-bit per-tensor configuration (where Naive-RQ actually collapses, losing 10%), SAWS achieves $56.40\%$ mean accuracy, which is slightly *worse* than the unmitigated Naive-RQ baseline ($56.75\%$). Thus, in the exact scenario where "Re-Quantization Silence" is a severe issue, SAWS fails to offer any mitigation. The authors are honest about this, explaining that a uniform global multiplier cannot adapt to the highly non-uniform rounding boundaries of per-tensor grids, but this remains a clear technical limitation of the proposed method.
2. **Complexity and Fragility of QA-ACS (Entropy Collapse):** QA-ACS is an unsupervised test-time optimization method. Under severe discretization noise (4-bit per-tensor grids), the optimization is fragile and prone to **entropy collapse** (e.g., MNIST accuracy drops from 42% down to 31.60% or 37.80% under QA-ACS optimization). To stabilize this, the authors must resort to further complexity (supervised tuning or coefficient regularization), highlighting that unsupervised test-time optimization through the discrete rounding operator is highly fragile.
3. **Scale Limitation:** The empirical evaluation is restricted to a very small Vision Transformer backbone (`vit_tiny`, 5.7M parameters) and simple image classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). While the authors justify this with rapid multi-axial profiling and present base-weight reconstruction errors for `vit_base` (86M parameters), it remains highly questionable whether these findings and mitigations will translate to multi-billion parameter LLMs (e.g., 7B+ parameter models under 4-bit group-wise or block-wise quantization).

## Reproducibility
The paper achieves a **high standard of reproducibility**. The mathematical equations are self-contained and complete, and Algorithm 1 provides a clear, implementable description of the QA-ACS optimization loop.

## Minimalist Evaluation
From a **Minimalist** perspective, the paper deserves praise for its extreme transparency and self-critical honesty, but the proposed mitigations themselves suffer from over-engineering:
- **Unnecessary Algorithmic Complexity:** QA-ACS is highly convoluted, introducing test-time adaptation, continuous coefficient optimization, Straight-Through Estimators (STE), gradient mismatch, and tracking adaptive second moments (Adam), only to suffer from entropy collapse under severe noise.
- **The Simple Baseline is Sufficient:** The paper's most profound finding is that under standard per-channel quantization (which is simple, standard, and already widely adopted), naive re-quantization is nearly lossless (dropping only 0.15% to 1.8% accuracy), and the individual experts suffer almost zero degradation. Thus, the complex engineering of SAWS and QA-ACS is largely unnecessary in practice.
- **Conclusion:** We highly appreciate that the authors did not try to hide these findings under an artificial SOTA claim. By laying bare the limitations of their own proposed methods and proving that the simple, standard per-channel baseline is highly effective without any added complexity, they perform a highly valuable service to the ML community.
