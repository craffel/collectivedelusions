# Peer Review

## 1. Summary of the Paper
The paper addresses the vulnerability of unsupervised Test-Time Adaptation (TTA) in model merging (specifically, layer-wise task blending via entropy minimization) when subjected to real-world environmental and sensor noise. Under these corrupted streams, standard TTA methods suffer from **Noise-Entropy Collapse** (due to the **Overfitting-Optimizer Paradox**), where the optimizer overfits high-frequency transductive noise and degrades out-of-distribution (OOD) generalization. 

To resolve this, the paper proposes **FlatMerge**, a dual-regularization test-time model merging framework. FlatMerge combines:
1. A **low-degree polynomial subspace constraint (PolyMerge)** of normalized layer depth to filter high-frequency spatial parameter noise across layers.
2. **Zeroth-Order (gradient-free) flatness-aware randomized smoothing** directly inside the compact polynomial coefficient space. This guides TTA to broad, flat entropy valleys using only forward passes, completely bypassing backpropagation and activation memory caching.

The framework is evaluated inside a simulated 12-layer Vision Transformer (ViT-B/32) weight-merging landscape and physically validated on real, small-scale MLP and CNN architectures fine-tuned on MNIST, FashionMNIST, and KMNIST.

---

## 2. Strengths of the Paper

* **Elegant Conceptual Angle (Sharpness-Aware Coefficient Space):** The most compelling strength of this paper is the idea of applying **Sharpness-Aware Minimization (SAM) principles directly to the compact mixing-coefficient space** of model merging, rather than the high-dimensional weight space of the model. By utilizing a Zeroth-Order (gradient-free) randomized smoothing formulation in this 12-parameter subspace, FlatMerge achieves flatness-aware regularization using only forward passes. This is a clever and computationally trivial conceptual shortcut that addresses the core limitation of standard TTA (activation memory and backpropagation overhead).
* **True Zero-Activation Memory Caching:** Because the optimization completely bypasses the backward pass through the network backbone, FlatMerge requires **exactly zero activation caching** during adaptation, maintaining peak memory identical to standard forward inference. This is a massive, highly practical benefit for SRAM-constrained edge accelerators.
* **Exceptional Clarity and Transparency:** The paper is beautifully written, mathematically rigorous, and very easy to follow. The authors deserve commendation for their intellectual honesty in Section 3.5, where they profile the DRAM-to-SRAM weight-reconstruction bandwidth bottleneck and detail practical, asynchronous periodic adaptation mitigations to reduce step latency.
* **Comprehensive Baselines:** The empirical comparison is highly thorough, comparing against uniform Task Arithmetic, unconstrained AdaMerging, conventional regularizers (TV and $L_2$), RegCalMerge, and PolyMerge across different degrees.

---

## 3. Weaknesses of the Paper

* **The "Adaptation-Decline" Paradox on Physical weights:** 
  A highly critical look at the physical 5-layer CNN model-merging experiments (Table 4) reveals a major empirical weakness: **unadapted, static uniform Task Arithmetic consistently and significantly outperforms FlatMerge across all noise levels (including clean conditions)**.
  * **At $\gamma = 0.0$ (Clean):** Task Arithmetic achieves **$58.20\%$** accuracy, while ZO-FlatMerge only achieves **$48.57\%$** (a drop of nearly $10\%$ absolute).
  * **At $\gamma = 1.0$ (Moderate):** Task Arithmetic achieves **$40.67\%$**, while ZO-FlatMerge achieves **$29.20\%$** (a drop of over $11\%$ absolute).
  * **At $\gamma = 2.0$ (Heavy):** Task Arithmetic achieves **$24.60\%$**, while ZO-FlatMerge achieves **$19.77\%$**.
  * **At $\gamma = 3.0$ (Extreme):** Task Arithmetic achieves **$17.77\%$**, while ZO-FlatMerge achieves **$16.07\%$**.

  While ZO-FlatMerge is indeed much more robust than standard first-order AdaMerging and PolyMerge (which catastrophically collapse to near random guessing, $\approx 14\% - 17\%$, due to the constant-prediction failure), **it still fails to outperform doing nothing (Task Arithmetic)** on real physical weights. This heavily undermines the practical utility and justification of the entire proposed active adaptation framework: Why would an edge practitioner deploy an active TTA loop (requiring repeated weight reconstructions, forward-pass perturbations, and latency overhead) if they can achieve $10\% - 11\%$ higher classification accuracy by simply using a static uniform merging coefficient?
* **Heavy Reliance on Simulated Loss Landscapes:** The primary results for the 12-layer Vision Transformer (ViT-B/32) are conducted entirely within an analytical, simulated loss landscape sandbox (Model I and Model II) rather than on real pre-trained weights. While the simulation is multi-seeded and continuous, there is an inevitable gap between a 12-parameter mathematical surrogate landscape (modeled using non-convex Rastrigin functions) and actual, high-dimensional deep learning weight spaces.
* **Limited Scope of Physical Validation:** To bridge the simulation gap, the authors conduct physical experiments on real CPU weights. However, these are restricted to very small, toy networks (a 108K parameter MLP and a 250K parameter CNN) fine-tuned on toy datasets (MNIST, FashionMNIST, KMNIST). The authors do not evaluate FlatMerge on actual, physical Vision Transformer weights, even though pre-trained models like CLIP ViT-B/32 (86M parameters) are widely available and can be merged and adapted on standard CPU/GPU hardware.
* **Omission of the Most Promising Conceptual Feature (Adaptive Perturbation Radius):** In Section 3.3, the authors propose an exceptionally clever, novel concept: an **Adaptive Perturbation Radius** ($\sigma(X)$) that dynamically scales with input noise/entropy. This would be a highly original, adaptive mechanism to handle non-stationary noise. However, the authors completely omit this from their experiments, stating, *"We leave the empirical exploration of this adaptive formulation to future work."* By leaving this feature as a purely theoretical formula, the paper misses a major opportunity for a truly bold, adaptive breakthrough, rendering the actual implemented algorithm more incremental.
* **Broken Baseline Bibliography Entry (Major Manuscript Bug):** The paper extensively builds its subspace constraint upon a prior baseline called **PolyMerge** (cited as `\cite{polymerge}` and `\citet{polymerge}`). However, there is **no bibliographic entry for `polymerge` in `references.bib`**. This is a major technical oversight that causes undefined citation warnings and prevents readers from verifying the exact "delta" and relation to prior art.

---

## 4. Evaluation of Dimensions

### Soundness
* **Rating:** Fair
* **Justification:** Mathematically, the formulation of FlatMerge is rigorous and sound. However, the empirical soundness is heavily weakened by two major issues: (1) the primary Vision Transformer experiments are simulated on analytical mathematical sandboxes, and (2) on the physical CNN weights, the proposed active adaptation actually hurts performance compared to doing nothing (unadapted Task Arithmetic), which contradicts the paper's core claim of "pragmatic utility" on physical deployments.

### Presentation
* **Rating:** Good
* **Justification:** The paper is extremely clear, logical, and beautifully structured. The hardware analysis in Section 3.5 is outstanding. However, there are minor presentation flaws: a notational disconnect between Section 3 ($\sigma$) and Section 4 ($\rho$), missing learning rate ($\eta$) hyperparameter details, and the broken bibliographic entry for the core baseline `polymerge`.

### Significance
* **Rating:** Fair
* **Justification:** In its current form, the practical significance is limited because the primary evaluation is simulated, and the physical validation is restricted to toy networks where adaptation actually degrades performance compared to the static Task Arithmetic baseline. 

### Originality
* **Rating:** Good
* **Justification:** The core concept of doing Sharpness-Aware Minimization in the mixing coefficient space via Zeroth-Order randomized smoothing is highly original and represents a very fresh conceptual angle. However, the originality is somewhat tempered because (1) the subspace constraint is inherited from prior work (PolyMerge), and (2) the authors omitted the implementation and evaluation of their most original proposed mechanism, the Adaptive Perturbation Radius ($\sigma(X)$), which would have elevated the paper's conceptual novelty significantly.

---

## 5. Overall Recommendation
* **Rating:** 3: Weak Reject
* **Justification:** This paper has clear merits, specifically its highly elegant conceptual shortcut (Mixing-Space SAM via Zeroth-Order randomized smoothing) and its excellent, transparent analysis of hardware bottlenecks. However, the weaknesses currently outweigh the merits. The primary results are simulated, the physical validations are restricted to toy architectures, and crucially, the adaptation process actually performs worse than standard static Task Arithmetic on real convolutional weights. Additionally, a core baseline citation (`polymerge`) is entirely missing from the bibliography. The paper requires key revisions—specifically, evaluating on actual pre-trained Vision Transformer weights, resolving the "Adaptation-Decline" Paradox on physical weights, and implementing the proposed adaptive perturbation radius—before it can be meaningfully built upon by others.

---

## 6. Questions and Constructive Suggestions for the Authors

1. **Resolve the "Adaptation-Decline" Paradox:** On the physical 5-layer CNN model, standard static Task Arithmetic (uniform blending of 0.3) outperforms ZO-FlatMerge by up to $11\%$ absolute. Why is this the case? Is the predictive entropy objective fundamentally misaligned with classification accuracy on real convolutional weights under noise, even when constrained to flat regions? Can you find an alternative test-time objective (or additional regularizer) that allows FlatMerge to actually beat the static Task Arithmetic baseline on real physical weights?
2. **Bridge the Simulation-to-Real Gap with physical ViTs:** Since the paper's primary focus is Vision Transformers, why did you not evaluate FlatMerge on actual pre-trained CLIP ViT-B/32 weights? Pre-trained CLIP models are readily available, and TTA can be executed easily on standard hardware. Replacing the continuous mathematical simulation with real CLIP ViT-B/32 physical weight merging would dramatically enhance the soundness and significance of your work.
3. **Implement and Evaluate the Adaptive Perturbation Radius:** Section 3.3 presents a very clever formulation for a dynamic, entropy-proportionate perturbation radius $\sigma(X)$. Why was this omitted from the experiments? Implementing this and demonstrating that it outperforms a fixed perturbation scale under non-stationary noise would provide a major boost to your paper's originality and significance.
4. **Fix the Bibliography and Notation:** 
   * Please add the missing bibliographic entry for `polymerge` in `references.bib` to resolve the undefined citation warnings.
   * Please unify the symbol for the perturbation scale throughout the paper. It is defined as $\sigma$ in Section 3 and Algorithm 1, but referred to as $\rho$ in Section 4.
   * Please specify the exact learning rate ($\eta$) used in your simulated and physical TTA experiments.
