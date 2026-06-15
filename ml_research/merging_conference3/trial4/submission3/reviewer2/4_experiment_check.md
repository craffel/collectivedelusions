# Experimental Evaluation and Claims Check

This document provides a rigorous, data-driven analysis of the experimental results presented in the paper, questioning whether the empirical evidence actually supports the authors' claims.

## 1. Lack of a Optimized Full-Precision Baseline
A major logical flaw in the evaluation is the comparison of optimized/scaled quantized models against a *naive, unoptimized* full-precision baseline (Naive FP16 Merge, 66.65%).

*   **Analysis of the Tables:**
    In **Table 2** (INT8 Symmetric Per-Channel), the proposed methods and the optimized baseline achieve:
    - AdaMerging (PH-Q): **70.10%**
    - SAWS: **69.75%**
    - QA-ACS: **69.35%**
    All three of these methods **outperform the unquantized FP16 ceiling (66.65%)** by a significant margin (+2.7% to +3.4%).
    
*   **The Confounding Variable:**
    This improvement is *not* because these methods have any magical "quantization noise mitigation" capabilities. Rather, the full-precision merged model (Naive FP16 Merge) is severely unoptimized, suffering from massive task interference (dropping from a 93.85% expert ceiling to 66.65%). 
    By searching for layer-wise coefficients (AdaMerging/QA-ACS) or boosting the adapter scaling factors (SAWS), these methods are simply optimizing the *model-merging weights* to reduce task interference. 
    To make a fair comparison, the authors **must** report the performance of these optimization methods in full-precision (FP16). For example, what is the accuracy of AdaMerging in FP16? It is highly likely to be around 71% to 72%. Comparing the quantized optimized models against the naive unoptimized FP16 merge is a highly misleading baseline comparison that inflates the apparent "mitigation" efficacy of the proposed methods.

## 2. Strict Underperformance of the Proposed QA-ACS Method
The paper proposes QA-ACS as a novel quantization-aware optimization method. However, a close inspection of the tables reveals that **QA-ACS is strictly worse than the existing AdaMerging (PH-Q) baseline in every single tested configuration**:
- **INT8 Symmetric Per-Channel:** AdaMerging PH-Q (**70.10%**) vs. QA-ACS (69.35%) — QA-ACS is **0.75% worse**.
- **INT4 Symmetric Per-Channel:** AdaMerging PH-Q (**68.80%**) vs. QA-ACS (68.00%) — QA-ACS is **0.80% worse**.
- **INT4 Asymmetric Per-Channel:** AdaMerging PH-Q (**68.25%**) vs. QA-ACS (64.75%) — QA-ACS is **3.50% worse**.
- **INT4 Symmetric Per-Tensor:** AdaMerging PH-Q (**57.25%**) vs. QA-ACS (57.00%) — QA-ACS is **0.25% worse**.

*   **Implication:**
    The entire premise of proposing a complex "Quantization-Aware" test-time optimization (using STE gradients and Adam updates on a calibration set) is empirically invalidated by their own results. A simple, standard test-time optimization performed in full precision (AdaMerging), followed by standard post-hoc quantization, yields superior results across the board. The extra complexity of QA-ACS is completely unjustified.

## 3. Total Failure of SAWS in the Only Quantization-Collapse Regime
The authors argue that "Re-Quantization Silence" causes a catastrophic performance collapse of merged models under low-bit quantization. However, Section 4.3 (Point 1) admits that under 3 out of 4 evaluated configurations, naive re-quantization (Naive-RQ) is nearly lossless:
- INT8 Symmetric Per-Channel: Naive-RQ is 66.35% (only **0.30%** drop from 66.65% ceiling).
- INT4 Symmetric Per-Channel: Naive-RQ is 64.85% (only **1.80%** drop from 66.65% ceiling).
- INT4 Asymmetric Per-Channel: Naive-RQ is 63.20% (only **3.45%** drop from 66.65% ceiling).

The only configuration where catastrophic collapse is actually observed is **INT4 Symmetric Per-Tensor** (Table 5), where Naive-RQ drops by **9.90%** (falling to 56.75%).

*   **Critical Evaluation:**
    In this single failure regime (INT4 Symmetric Per-Tensor), the proposed data-free method, **SAWS, is completely ineffective**, achieving **56.40%** mean accuracy, which is **worse than Naive-RQ (56.75%)**.
    Thus, in the only scenario where "Re-Quantization Silence" actually causes a significant collapse, the proposed method fails to improve performance. The only method that provides any benefit under per-tensor quantization is the dual-path "Quantize-then-Merge" (Q-then-M) co-existence baseline (59.60%), which the authors dismiss due to inference latency.

## 4. Decoupling Analysis Reveals the "Re-Quantization Silence" is a Myth
The authors' control experiment in **Table 6** (Individual Unmerged Quantized Experts) provides the final blow to their main thesis.
Under **INT4 Symmetric Per-Tensor** quantization:
- Unmerged FP16 expert ceiling: 93.85%
- Unmerged INT4 Symmetric Per-Tensor experts: 82.95% (a **10.90%** drop!)
- Merged Naive FP16 ceiling: 66.65%
- Merged Naive-RQ INT4 Symmetric Per-Tensor: 56.75% (a **9.90%** drop!)

*   **Critical Evaluation:**
    The relative drop caused by per-tensor INT4 quantization is actually *smaller* for the merged model (9.90% drop) than it is for the individual, unmerged single-task models (10.90% drop). 
    This empirically proves that the collapse under per-tensor quantization has **nothing to do with model merging or "Re-Quantization Silence"**. It is simply a reflection of the fact that 4-bit per-tensor uniform quantization of a tiny Vision Transformer (ViT-Tiny) is highly destructive to representations in general. There is no specific "silencing" of the adapters; rather, the entire model's representation space collapses under naive per-tensor quantization.
    Under standard per-channel configurations (where single-task models do not collapse, dropping only ~0.7%), the merged model (Naive-RQ) also does not collapse (dropping only 1.80%). This means "Re-Quantization Silence" does not exist in any practical, deployable scenario (since per-channel/group-wise quantization is universally used).

## Summary of Experimental Critique
The empirical evaluation fails to support the paper's core claims. The "Re-Quantization Silence" is shown to be a non-issue under standard per-channel configurations, and the collapse under per-tensor configurations is simply due to standard, general quantization noise. Furthermore, both proposed methods are highly questionable: SAWS is ineffective in the only failure regime (per-tensor) and acts as an overcomplicated scale tuner in per-channel regimes, while the proposed QA-ACS is strictly outperformed by the existing full-precision optimized baseline (AdaMerging) in all evaluated scenarios.
