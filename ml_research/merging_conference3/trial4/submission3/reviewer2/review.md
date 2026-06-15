# Peer Review of Conference Submission

## 1. Summary of the Paper
The paper investigates the behavior of merged parameter-efficient fine-tuned (PEFT) models—specifically Low-Rank Adaptation (LoRA) adapters—under post-training quantization (PTQ) constraints. The authors introduce the concept of "Re-Quantization Silence," asserting that when merged models are quantized to low bit-widths (e.g., 4-bit), the small-magnitude, task-specific adapter updates ($\Delta W_{\text{merged}}$) are rounded to zero because the quantization step size is dominated by the much larger dynamic range of the pre-trained base model weights ($W_0$). 

To analyze and mitigate this phenomenon, the authors present a Multi-Axial Re-Quantization Auditing (RQA) framework to evaluate performance across granularities (per-tensor vs. per-channel), bit-widths (4-bit vs. 8-bit), and formats (symmetric vs. asymmetric). They propose two mitigations:
1.  **Scale-Adaptive Weight Shifting (SAWS):** A data-free scaling method that boosts adapter weights prior to merging using a layer-wise norm ratio $\gamma^l$ and applies a scalar weight alignment factor $c^l$ at inference.
2.  **Quantization-Aware Adapter Coefficient Search (QA-ACS):** A test-time optimization method that minimizes prediction entropy over a small calibration set ($N=16$) with gradients flowing through the quantizer via the Straight-Through Estimator (STE).

The methods are evaluated using a Vision Transformer backbone (`vit_tiny`, 5.7M parameters) fine-tuned on four toy classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN).

---

## 2. Strengths
The paper exhibits several noteworthy strengths:
1.  **Exemplary Self-Criticism and Transparency:** The authors are exceptionally honest and analytical about the limitations of their work. They proactively expose the "representation scale preservation dilemma" in SAWS, deconstruct the fragility of QA-ACS (entropy collapse) under severe noise, and openly acknowledge the toy-scale nature of their backbone and datasets. This level of self-critique is rare and highly commendable.
2.  **Rigorous and Brilliant Scientific Control Experiments:** 
    - The **Individual Expert Auditing** control experiment (Table 6) is outstanding. By applying quantization directly to unmerged experts (which have zero task interference in high-precision), the authors decoupled quantization-induced degradation from pre-existing task-interference conflicts. This is a model design for scientific validation.
    - The **Double Quantization Noise** format-shift analysis (Table 1) provides a highly interesting empirical characterization of how moving from non-linear, density-based NF4 to uniform INT4/8 grids introduces severe representation-space errors.
3.  **High-Quality Presentation:** The writing is elegant, the mathematical notation is formal and precise, and the narrative flow is easy to follow.

---

## 3. Major Weaknesses and Logical Flaws
Despite its polished writing, a rigorous and critical evaluation of the paper's core claims, methodology, and empirical results reveals several fundamental flaws that severely undermine its scientific contributions.

### Weakness 1: "Re-Quantization Silence" is a Rebranded Artifact of General Quantization Collapse
The authors frame "Re-Quantization Silence" as a dangerous, previously uncharacterized methodological blindspot unique to model merging. However, the paper's own data refutes this:
*   **Per-Channel configurations are virtually lossless:** In standard per-channel configurations (Tables 2, 3, and 4), naive re-quantization (Naive-RQ) suffers almost zero degradation (dropping only 0.30% in INT8 and 1.80% in INT4 from the FP16 ceiling). Since per-channel and group-wise quantization are the industry standards for edge deployment, "Re-Quantization Silence" is a non-issue in any practical, deployable scenario.
*   **Collapse is caused by general quantization noise, not merging:** Catastrophic collapse only occurs under the aggressive INT4 Symmetric Per-Tensor configuration (Table 5), where Naive-RQ drops by **9.90%** (falling to 56.75%). However, Table 6 shows that the *individual unmerged experts* (with zero task interference) suffer an even larger **10.90%** drop (falling to 82.95%) under the exact same INT4 per-tensor constraints. 
*   **Logical Implication:** The catastrophic performance drop in Table 5 is simply a reflection of the well-known fact that 4-bit per-tensor uniform quantization of a tiny Vision Transformer causes severe representation collapse across the board, affecting unmerged and merged models alike. There is no unique, merging-specific "silencing" of the adapters; the base model representations themselves collapse. Framing this general quantization failure as a novel, model-merging-specific phenomenon is mathematically and empirically incorrect.

### Weakness 2: Strict Underperformance of the Proposed QA-ACS Method
The paper proposes QA-ACS as a novel quantization-aware test-time optimization. However, a close inspection of the tables reveals a devastating result: **QA-ACS is strictly and systematically outperformed by the existing AdaMerging (PH-Q) baseline in every single tested configuration**:
*   **INT8 Symmetric Per-Channel:** AdaMerging PH-Q (**70.10%**) vs. QA-ACS (69.35%) — QA-ACS is **0.75% worse**.
*   **INT4 Symmetric Per-Channel:** AdaMerging PH-Q (**68.80%**) vs. QA-ACS (68.00%) — QA-ACS is **0.80% worse**.
*   **INT4 Asymmetric Per-Channel:** AdaMerging PH-Q (**68.25%**) vs. QA-ACS (64.75%) — QA-ACS is **3.50% worse**.
*   **INT4 Symmetric Per-Tensor:** AdaMerging PH-Q (**57.25%**) vs. QA-ACS (57.00%) — QA-ACS is **0.25% worse**.

This means that the entire complexity of proposing a new "Quantization-Aware" search—incorporating Straight-Through Estimators (STE) to propagate gradients through non-differentiable operators—is mathematically and empirically unjustified, as a simpler, existing baseline optimized in full-precision (AdaMerging) and quantized post-hoc performs superiorly across the board.

### Weakness 3: Total Failure of SAWS in the Only Quantization-Collapse Regime
The proposed data-free method, SAWS, is designed to protect adapter updates from being crushed to zero by quantization scales. However, in the only configuration where there is actually a catastrophic collapse (INT4 Symmetric Per-Tensor, Table 5), **SAWS is completely ineffective, performing worse than naive re-quantization** (achieving **56.40%** vs. Naive-RQ's **56.75%**). 
Under per-channel configurations where SAWS does improve performance (e.g., 67.80% in Table 3 vs. 64.85% Naive-RQ), there is no catastrophic collapse to begin with (Naive-RQ drops only 1.80%). This means SAWS is ineffective where it is actually needed, and its improvements in per-channel configurations are simply due to tuning the adapter scaling factors (selective task-vector boosting), which would be better done in full precision.

### Weakness 4: Overcomplicated Mathematical Distraction in SAWS
The authors introduce an "elegant, closed-form alignment factor" $c^l$ derived from a quadratic objective. However, they admit that because the weight tensors are heavily dominated by the base weights, $c^l \approx 1.0$ (typically $0.99$ in practice). They also show that true scale preservation collapses the base representations, and thus they do not apply any inverse scale.
Consequently, SAWS is functionally just multiplying the adapter weights by a large global scalar ($\gamma^l \approx 10$ to $100$) during merging. Scaling task vectors during merging is a standard hyperparameter. The mathematical framework of SAWS is an overcomplicated way of describing a simple heuristic: "multiply the adapter weights by a large scalar."

### Weakness 5: Toy-Scale Evaluation with Severe Baseline Task Interference
The empirical results are based entirely on `vit_tiny` (5.7M parameters) evaluated on four toy classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN).
*   **Severe Baseline Collapse:** The full-precision merged model (Naive FP16 Merge) already suffers from catastrophic task interference, achieving only **66.65%** mean accuracy compared to the **93.85%** unmerged expert ceiling (with MNIST dropping from 98.20% to 45.40%). 
*   **Confounding Noise:** Studying post-training quantization on top of a model that is already severely collapsed in full precision introduces massive confounding noise. The severe task interference in weight space means the representations are highly unstable, and any slight discretization noise can cause unpredictable shifts. Drawing broad methodological conclusions about post-training quantization behavior from such a degraded toy setup is highly unreliable.
*   **Lack of Generalizability:** Modern model merging and PEFT are predominantly applied to LLMs (7B+ parameters) or large diffusion models. A 5.7M parameter ViT on MNIST has fundamentally different representation dynamics, layer shapes, and quantization sensitivity than modern LLMs.

---

## 4. Soundness Evaluation
*   **Rating: Poor**
*   **Justification:** The paper contains several critical methodological flaws. First, the reported "Double Quantization Noise" of 30.40% in INT8 (Table 1) is anomalously high and is a direct artifact of using a highly naive, unclipped symmetric quantizer. In a standard industry-grade PTQ pipeline with basic percentile clipping or calibration, mapping a 4-bit NF4 distribution (16 values) to an 8-bit uniform grid (256 bins) would be virtually lossless. Second, the proposed QA-ACS method is systematically outperformed by the existing AdaMerging (PH-Q) baseline, making its core technical contribution empirically invalid. Third, the central thesis of "Re-Quantization Silence" is shown to be a mischaracterization of general, non-merging-specific per-tensor quantization collapse.

---

## 5. Presentation Evaluation
*   **Rating: Excellent**
*   **Justification:** The paper is exceptionally well-written, clearly structured, and mathematically precise. The authors have done a fantastic job of organizing the paper, presenting tables, and writing detailed, self-critical descriptions of their methods.

---

## 6. Significance Evaluation
*   **Rating: Poor**
*   **Justification:** The practical significance of this work is highly limited. In standard per-channel configurations, naive re-quantization is nearly lossless (~1.8% drop in INT4), meaning the problem the authors seek to solve is a non-issue. In the per-tensor INT4 regime where catastrophic collapse actually occurs, both proposed methods (SAWS and QA-ACS) fail to improve performance. Finally, the toy-scale Vision Transformer setup evaluated on MNIST limits the generalizability and relevance of these findings to modern deep learning deployment (which focuses on LLMs and Diffusion models).

---

## 7. Originality Evaluation
*   **Rating: Fair**
*   **Justification:** While the multi-axial audit framework (RQA) and the decoupling analysis (Table 6) are original and valuable contributions to the evaluation literature, the proposed technical methods (SAWS and QA-ACS) are highly incremental. SAWS is functionally a standard task-vector scaling heuristic wrapped in redundant mathematical formulations, and QA-ACS is an incremental modification of AdaMerging that performs worse than the original baseline.

---

## 8. Questions for the Authors
1.  **Regarding Table 1:** Why is the relative Frobenius reconstruction error of INT8 Symmetric Per-Channel on `vit_base` so high (**30.395%**)? An 8-bit symmetric quantizer has 256 bins, which should easily represent a 4-bit NF4 distribution (16 discrete values). Is this massive error caused by a single weight outlier squashing the uniform scale because you did not use any outlier clipping or percentile calibration? If so, does this mean your "Double Quantization Noise" is primarily an artifact of an extremely naive quantizer implementation rather than an inherent limitation of format shift?
2.  **Regarding QA-ACS vs. AdaMerging:** Why does QA-ACS perform strictly worse than AdaMerging (PH-Q) in every single tested configuration (Tables 2, 3, 4, 5)? If optimizing layer-wise coefficients in full-precision (FP16) and then quantizing post-hoc performs superiorly across the board, what is the scientific or practical utility of introducing the complexity of Straight-Through Estimators and quantization-aware test-time search in QA-ACS?
3.  **Regarding SAWS under Per-Tensor constraints:** Why does SAWS perform worse than Naive-RQ under the aggressive INT4 Symmetric Per-Tensor configuration (Table 5: **56.40%** vs. **56.75%**)? Since this is the only configuration where Naive-RQ actually collapses, and since SAWS is ineffective here, doesn't this mean SAWS fails to mitigate the collapse where it is actually needed?
4.  **Regarding the scaling baseline for SAWS:** Since $c^l \approx 1.0$ and scale-preservation is not applied, SAWS is functionally just multiplying the adapter task vectors by $\gamma^l$. What would be the performance of a baseline in full precision (FP16) where the task vectors are scaled by a manually tuned grid search factor $\lambda$? Does the Frobenius norm ratio in SAWS provide any representation-space benefit over standard, manual task-vector scaling?
5.  **Regarding Toy-Scale Evaluation:** Given that PEFT and model merging are almost exclusively used for LLMs and large-scale Diffusion models, why was the evaluation restricted to a 5.7M parameter ViT on toy datasets like MNIST, FashionMNIST, CIFAR-10, and SVHN, where the full-precision baseline already suffers from catastrophic task interference (66.65%)? Can you provide any preliminary results on a larger LLM or Diffusion model to prove that these dynamics generalize?

---

## 9. Overall Recommendation and Rating
*   **Recommendation: 2: Reject**
*   **Justification:** 
    While the paper is beautifully written, exceptionally transparent about its limitations, and contains a highly commendable control experiment design (Table 6), its core technical contributions fall short of the standard required for acceptance. 
    1.  The central premise of "Re-Quantization Silence" as a merging-specific failure mode is empirically disproven by their own control experiment, which shows that the collapse under per-tensor INT4 quantization is a general symptom of ViT representations collapsing, affecting unmerged models even more severely.
    2.  The proposed QA-ACS method is systematically outperformed by the existing AdaMerging baseline in all tested configurations, rendering its technical complexity unjustified.
    3.  The proposed SAWS method fails to outperform naive re-quantization in the only regime where collapse actually occurs (per-tensor INT4) and is an overcomplicated scaling heuristic in lossless per-channel regimes.
    4.  The experimental evaluation is restricted to an extremely toy-scale vision setup with severe baseline task interference, introducing substantial confounding noise and limiting any broader impact or generalizability.

For these reasons, the major weaknesses of the paper heavily outweigh its merits, and I must recommend a Reject.
