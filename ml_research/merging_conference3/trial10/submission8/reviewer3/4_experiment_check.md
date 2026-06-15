# Experimental Evaluation Check

A critical assessment of the experimental setup, baselines, and empirical results.

## 1. The Synthetic Analytical Coordinate Sandbox (ACS)
The majority of the quantitative analysis in the main body is conducted inside the **Analytical Coordinate Sandbox (ACS)**, which is simulated as a linear recurrence system without non-linearities, convolutions, or self-attention blocks.
* **The "Static Uniform Dominance Paradox"**: In this sandbox, the parameter-free, zero-tuning **Static Uniform baseline consistently and significantly outperforms all proposed spectral trajectory methods** (85.10% vs 70.70% on CNN, and 83.75% vs 72.70% on CLIP). This indicates that the sandbox assumes perfect structural symmetry and coordinate alignment, creating an environment where any trajectory-based adaptation is mathematically counterproductive due to "anisotropic representation shearing."
* **Coordinate Misalignment Sweep**: To resolve this paradox, the authors sweep coordinate rotation misalignment ($\eta \in [0.0, 0.6]$). While our proposed methods show relative resilience (e.g., RB-DCTM achieving 74.60% at $\eta = 0.4$), the **Static Uniform baseline still outperforms all adaptive methods across the entire sweep** (achieving 83.40% at $\eta = 0.4$ and 82.05% at $\eta = 0.6$).
* **Critical Takeaway**: While the ACS is a clean and mathematically tractable toy model for studying trajectory shapes and proving the theoretical boundary properties of spectral trajectories, it is a poor proxy for real-world model merging. It does not provide empirical evidence supporting the superiority of adaptive merging over static ensembling, since the static baseline dominates across all sandbox settings. The sandbox's design makes it a "worst-case scenario" for adaptive methods, which limits its utility as a primary experimental benchmark.

## 2. Real-World Proof-of-Concept Validation (ViT-B/16)
The small-scale real-world experiment on actual Vision Transformers is the most critical and high-value portion of the empirical evaluation, as it successfully resolves the Static Uniform Dominance Paradox and validates the practical utility of the proposed method.
* **The Setup**: Merging two actual ViT-B/16 experts fine-tuned on CIFAR-10 and CIFAR-100 using ZipIt! permutation alignment and a 10-sample per task calibration set.
* **The Results**:
  * Static Uniform (ZipIt! Aligned) gets **71.30%** joint average.
  * Globally-Scaled Task Arithmetic gets **72.50%** joint average.
  * Offline Unconstrained gets **69.80%** joint average (degrading below Static Uniform due to overfitting).
  * RBPM ($d=2$) gets **70.70%** joint average (degrading below Static Uniform due to polynomial boundary runaway).
  * **RB-FTM (Ours, F=2)** gets **74.20%** joint average.
  * **RB-DCTM (Ours, F=2)** gets **74.90%** joint average.
* **Key Observations**:
  * **Superiority Over Baselines**: RB-DCTM ($F=2$) outperforms the Static Uniform baseline by **+3.60%** absolute accuracy, Globally-Scaled by **+2.40%**, Offline Unconstrained by **+5.10%**, and its direct trajectory competitor RBPM ($d=2$) by **+4.20%**.
  * **Boundary Runaway Validation**: The fact that the quadratic polynomial competitor (RBPM) degrades below the Static Uniform baseline (70.70% vs. 71.30%) strongly validates the authors' thesis. It empirically demonstrates that polynomial trajectories suffer from severe boundary runaway oscillations that disrupt representations in deep neural networks. By contrast, the proposed spectral trajectories (RB-FTM and RB-DCTM) successfully stabilize early feature extraction and final classification layers, translating into substantial accuracy gains.
  * **Parameter Efficiency**: RB-DCTM ($F=2$) achieved these gains using only $F+1 = 3$ parameters per task (a total of 6 parameters), demonstrating exceptional compactness and lightweight optimization (completing in under 15 seconds).

## 3. Scope and Scale of Evaluation
While the real-world results are highly encouraging and validate the core theoretical claims, the **scale of the real-world validation is extremely limited**:
* **Few Tasks**: Only two task experts are merged ($K=2$: CIFAR-10 and CIFAR-100). Modern model ensembling and weight merging literature routinely evaluates multi-task merging across 5 to 10 distinct task experts (e.g., DomainNet streams, VTAB benchmarks).
* **Toy Datasets**: CIFAR-10 and CIFAR-100 are low-resolution, relatively simple visual classification tasks. The paper would be significantly stronger if the authors evaluated their spectral trajectory models on more complex, large-scale visual benchmarks (e.g., ImageNet-variants) or modern Large Language Models (LLMs) (such as ensembling multiple LLaMA-based or Mistral-based experts across distinct reasoning or coding tasks), where layer-wise ensembling is highly relevant and active.
* **Dual-Dataset Footprint**: As disclosed, the ZipIt! permutation alignment step required an unlabelled footprint of 100 samples per task strictly to compute stable activation covariance statistics. While this is a practical engineering choice, it technically increases the sample complexity of the ensembling process from a pure 10-shot setup to 10-shot labeled + 100-shot unlabelled. This should be more clearly highlighted in the main text when comparing against unaligned baselines.

## Conclusion
The empirical results on actual deep networks fully support the core claims of the paper: spectral trajectories successfully mitigate the boundary runaway pathology of polynomial trajectories and avoid transductive overfitting on few-shot calibration datasets. However, the heavy reliance on the synthetic sandbox (which is dominated by the static baseline) and the narrow, small-scale nature of the real-world Vision Transformer experiments leave a significant gap between the mathematically rich theory and its practical verification on large-scale, modern deep learning benchmarks.
