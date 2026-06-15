# Critical Evaluation of the Experimental Setup and Results

## Experimental Setup Critique
The primary weakness of the paper's experimental setup is its **extreme reliance on over-simplified, toy sandboxes**:
1. **Split-MNIST Dataset:** The authors criticize prior work for validating their theories in "simplified linear representation-space sandboxes." Yet, they replace it with another sandbox: **Split-MNIST**, a dataset of 28x28 grayscale handwritten digits. Split-MNIST is a notoriously trivial and saturated toy benchmark that completely lacks the rich semantic hierarchies, texture variations, and rotational complexities of real-world datasets.
2. **Under-Parameterized, Miniature Architectures:**
   * **DeepMLP-12:** A multi-layer perceptron with only 12 layers and a hidden dimension of **64 units** (comprising a measly ~100k parameters).
   * **TinyCNN-4:** A convolutional network with only 4 layers (3 conv layers and a linear head).
   These models are extremely small and far removed from modern high-capacity deep architectures, Vision Transformers (ViTs), or Large Language Models (LLMs) where model merging is actually applied in practice.

## Evaluation of Baselines
1. **Consistently Outperformed by simple Static Baselines (OFS-Tune):**
   On the convolutional backbone (TinyCNN-4) in Table 2, the **static, 4-parameter baseline OFS-Tune consistently and significantly outperforms the proposed high-capacity dynamic Layer-wise Router across all task-conflict suites**:
   * *Low-Conflict:* OFS-Tune **82.85% ± 11.52%** vs. Layer-wise **78.70% ± 14.56%** ($+4.15\%$ delta).
   * *High-Conflict:* OFS-Tune **90.75% ± 1.58%** vs. Layer-wise **81.30% ± 9.69%** ($+9.45\%$ delta).
   * *Cross-Domain:* OFS-Tune **53.40% ± 7.16%** vs. Layer-wise **52.52% ± 5.95%** ($+0.88\%$ delta).
   The authors explain this away as a "Capacity-Variance Trade-off" where static baselines act as robust regularizers under small calibration budgets. However, Figure 4 shows that even when scaling the calibration split to 1024 samples per task, the dynamic router's crossover improvement is a marginal ~1% (climbing from 53.4% to ~54.5%). Introducing $L \times (d \cdot K + K)$ parameters and complex dynamic gating for a negligible ~1% improvement under large data budgets (which violates the "few-shot" calibration premise) indicates a severe lack of practical utility.

2. **Weak / Dismissed Baselines:**
   The authors do not compare against prominent weight-space alignment and merging baselines like **ZipIt!** or **TIES-Merging**. They claim that because the experts are fine-tuned from a shared initialization, these advanced alignment techniques "mathematically collapse to standard arithmetic interpolation." 
   * **The Flaw:** This is false. TIES-Merging trims parameter deltas based on magnitude (e.g., keeping only the top 20% of parameter updates) and resolves sign conflicts. Even when starting from a shared base initialization, different experts will update parameters in different directions (some positive, some negative), and trimming small parameter changes can remove optimization noise. Thus, TIES-Merging does *not* collapse to standard averaging and should have been included as a baseline. Bypassing these methods is an attempt to avoid comparing with stronger baselines.

3. **Catastrophic Performance on DeepMLP-12:**
   On DeepMLP-12 under Cross-Domain task conflict (Table 1), the proposed router achieves only **16.15% ± 5.60%** accuracy, while standard Uniform gets **11.80%**, and L1-Global gets **11.68%**. While their method is "statistically superior," all merged models perform below or near the **12.5% random guessing threshold**. A classification accuracy of 16% on digit recognition is a complete and catastrophic functional collapse. Drawing any architectural conclusions from a completely non-functional merged model is methodologically meaningless.

## Do the results support the claims?
* **Claim:** "layer-wise dynamic model merging is not a redundant over-parameterization, but an expressive, depth-specialized framework capable of achieving superior multi-task consensus."
* **Reality:** No, the results show that the dynamic router is highly over-parameterized, extremely sensitive to parameter-variance, consistently underperforms a simple 4-parameter static baseline (OFS-Tune) on CNNs, and fails catastrophically on dense MLPs. The claimed benefits are highly overstated.
