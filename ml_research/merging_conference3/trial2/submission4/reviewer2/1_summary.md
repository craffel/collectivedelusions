# Comprehensive Summary of EdgeMerge

## 1. Main Topic
The paper introduces **EdgeMerge**, a training-free, forward-only adaptive model merging framework designed to combine multiple task-specific expert models into a single multi-task network without joint training or test-time backpropagation. The primary target application is resource-constrained edge systems (e.g., mobile devices, IoT nodes) where gradient tracking, backpropagation latency, and high memory footprints make existing server-grade adaptive methods (like SyMerge or AdaMerging) impractical.

## 2. Methodology & Technical Approach
EdgeMerge restricts its adaptive channel-wise merging to a single strategic "choke-point" layer—the visual projection bottleneck layer (`model.visual.proj` in CLIP)—while merging the remaining layers using standard static Task Arithmetic. It operates in a single forward pass over a tiny unlabeled calibration batch (e.g., $B=32$ samples per task) through three main stages:

1. **Forward-Only Activation Sampling (FOAS):** Samples activations from the pre-trained base model and all $K$ task-specific experts using calibration inputs. To avoid the memory and latency overhead of running inputs through $K$ separate visual encoders, it reuses the base encoder's intermediate features $X_k^{base}$ to evaluate all expert projection activations.
2. **Scale-Normalized Delta Activation Salience (SNDAS):** Calculates representational shifts (activation deltas) between experts and the base model, normalizing them using the Frobenius norm to eliminate bias from varying task-specific activation scales.
3. **Channel-Wise Softmax Gating (CWSG):** Normalizes salience scores across tasks using a softmax function with a temperature hyperparameter $\tau$ to determine fine-grained, localized channel-wise merging coefficients.

The paper also proposes **Decoupled Scale Routing (DSR)** to address a mathematical scaling mismatch. In gated layers, softmax normalization forces the composed updates to behave as a weighted average (coefficients sum to 1), whereas in static layers, updates sum unnormalized. DSR decouples the gated projection scale ($\lambda_{proj}$) from the static scale ($\lambda_{static}$).

## 3. Key Findings & Performance Metrics
- **Performance vs. Baselines:** On an 8-task visual classification benchmark using CLIP ViT-B/32:
  - Standard EdgeMerge achieves **68.69%** average accuracy (at optimal $\tau=0.50$, $\lambda=0.30$), which matches the peak optimized Task Arithmetic baseline of **68.74%** (at $\lambda=0.20$).
  - Decoupled EdgeMerge (DSR) under Regime 2 ($\lambda_{static}=0.25, \lambda_{proj}=0.20, \tau=0.10$) achieves a peak average accuracy of **69.58%**, outperforming standard Task Arithmetic by **+0.84%** absolute points.
  - Server-grade, gradient-based SyMerge achieves **89.74%**, representing a massive **20.16%** absolute accuracy advantage over Decoupled EdgeMerge.
- **Resource Footprint:** EdgeMerge reduces merge preparation latency from 10 minutes (SyMerge) to **11.95 seconds** (a $50\times$ speedup) and requires negligible training GPU RAM (~100 MB, forward-only) compared to the high memory required for backpropagation. It introduces exactly zero inference latency or architectural overhead.
- **Hyperparameter Stability:** Standard Task Arithmetic exhibits a fragile, narrow peak in performance at $\lambda=0.20$ and collapses rapidly at other scales. EdgeMerge provides a broader, more stable scaling plateau (e.g., preserving high performance at $\lambda=0.30$).

## 4. Explicitly Claimed Contributions (with Evidence)
- **Contribution 1: A training-free, forward-only model merging framework.** 
  - *Evidence:* The authors implement FOAS, SNDAS, and CWSG and show that they can compute channel-wise coefficients in 11.95 seconds without backpropagation or gradient tracking.
- **Contribution 2: Decoupled Scale Routing (DSR).**
  - *Evidence:* The authors identify the representational dampening caused by softmax normalization and show that decoupling $\lambda_{proj}$ from $\lambda_{static}$ improves average accuracy from 68.69% to 69.58% (+0.89% improvement).
- **Contribution 3: Practical heuristics for non-CLIP architectures.**
  - *Evidence:* The paper provides a decision flowchart and guidelines for targeting compression/projection layers or SwiGLU FFN intermediate layers in larger architectures.
- **Contribution 4: Plateau preservation and hyperparameter stabilization.**
  - *Evidence:* The authors perform grid sweeps over scaling factors and present visual evidence (Figure 3) showing that EdgeMerge retains high accuracy over a wider range of $\lambda$ than Task Arithmetic.
- **Contribution 5: Invariance to representational drift (Encroached Encoder Fallacy).**
  - *Evidence:* The paper provides an ablation study (Table 5) showing that using base features $X_k^{base}$ instead of expert features $X_k^{expert}$ yields identical results to three decimal places (69.580% vs. 69.580%), justifying the $K\times$ computational savings.
