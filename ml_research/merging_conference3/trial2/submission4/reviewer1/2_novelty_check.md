# Novelty and Originality Assessment

This document evaluates the key novel aspects of the proposed submission, its conceptual "delta" from prior work, and the characterization of its scientific novelty from the perspective of a reviewer who values significant, original, and potentially paradigm-shifting contributions.

---

## 1. Characterization of the Claimed Novelty
The paper positions **EdgeMerge** as an ambitious and conceptually original framework. The core conceptual leap is the introduction of **training-free, forward-only activation-guided channel-wise weight routing**. 

Rather than treating model merging as a purely static parameter-average problem (like Task Arithmetic or TIES-Merging) or as a resource-heavy test-time optimization problem (like AdaMerging or SyMerge), EdgeMerge proposes to use a tiny, unlabeled calibration dataset in a single forward pass to dynamically map individual output channels (neurons) of a bottleneck projection layer to the specific task experts that find them most salient. 

This is an elegant and conceptually appealing idea: it frames post-hoc model composition as localized, data-driven channel-routing, mimicking dynamic Mixture-of-Experts (MoE) but embedding the routing directly into the static parameters with zero inference-time latency or structural overhead.

---

## 2. Assessment of the 'Delta' from Prior Work
The paper positions its contributions relative to three main literature branches:
1. **Static Parameter-Space Merging:** (e.g., Task Arithmetic, Model Soups, TIES, ZipIt!). The delta is that EdgeMerge is *dynamic* and localized at the channel level based on actual model behavior (activations), rather than static or uniform.
2. **Adaptive Merging and Test-Time Adaptation:** (e.g., AdaMerging, SyMerge, Tent). The delta is that EdgeMerge is completely *training-free and forward-only*, requiring zero backpropagation, zero gradient tracking, and taking only seconds (11.95s) compared to minutes (10+ min) for gradient-based optimization.
3. **Activation-Based Saliency and Compression:** (e.g., channel pruning, CBAM). The delta is that EdgeMerge applies activation saliency *post-hoc directly to weight composition* rather than network pruning or training-time attention routing.

Additionally, the paper introduces **Decoupled Scale Routing (DSR)** to address the scale dampening of softmax normalization in gated layers versus summation in static layers.

---

## 3. Critical Novelty Deconstruction (The Redundant Gating Paradox)
While the paper is beautifully written, highly transparent, and mathematically rigorous, a deep look at the empirical results—specifically the ablation studies in Section 4.3.4—reveals a major conceptual and scientific paradox that severely undermines the originality and significance of the core "activation-guided channel-routing" contribution.

The ablation studies show the following:
- **Optimal Decoupled EdgeMerge (full FOAS + SNDAS + CWSG + DSR):** **69.58%** average multi-task accuracy.
- **No Frobenius Scale Normalization (No SNDAS + DSR):** **69.58%** average multi-task accuracy.
- **Layer-wise Gating (DSR):** **69.59%** average multi-task accuracy.
- **Uniform Gating (No activation sampling, flat $\alpha_k = 1/K$ + DSR):** **69.58%** average multi-task accuracy.

This is a startling scientific revelation. The entire elaborate, highly-advertised machinery of Forward-Only Activation Sampling (FOAS), Scale-Normalized Delta Activation Salience (SNDAS), and Channel-Wise Softmax Gating (CWSG) performs **exactly the same** (69.58%) as a completely static, uniform weighting of the projection layer ($1/K$ scaling). 

### Implications on Conceptual Novelty:
1. **The Activation-Guided Routing is Conceptually Inert:** From an engineering and scientific standpoint, the data-driven activation statistics extracted from the calibration batches are doing **no functional work**. The models gain absolutely nothing from the fine-grained channel routing compared to simply scaling down the task vectors in the projection layer by $1/K$ uniformly.
2. **The Real Utility is Highly Incremental:** Since the elaborate activation gating machinery is redundant, the actual performance improvement (from the standard Task Arithmetic baseline of 68.74% to the decoupled peak of 69.58%) is driven **entirely** by **Decoupled Scale Routing (DSR)**. DSR simply decouples the weight-scaling factor of the visual projection layer ($\lambda_{proj} = 0.20$) from the static layers of the transformer ($\lambda_{static} = 0.25$). 
3. **DSR is an Incremental Scaling Tweak:** Decoupling or optimizing layer-specific scales is a well-known, highly standard technique in model merging and parameter optimization. While practical, a minor hyperparameter-tuning adjustment (tuning two scalar $\lambda$'s instead of one) represents a very modest, incremental, and unoriginal contribution compared to the ambitious "weight-space routing" paradigm claimed in the introduction and abstract.
4. **Standard Coupled EdgeMerge Underperforms:** Under standard, single-scale (coupled) merging, the best accuracy EdgeMerge achieves is **68.69%**, which is actually *inferior* to standard Task Arithmetic (**68.74%**). This further highlights that the core activation-routing algorithm, when not saved by the decoupled scale tuning, does not outperform the simplest possible static weight average.

---

## 4. Overall Novelty Characterization
In light of these findings, the scientific novelty of this submission must be characterized as **highly incremental**. 

The ambitious, big, and bold idea of "forward-only activation-based channel routing" is shown by the authors' own experiments to have no functional advantage over a simple uniform scaling baseline. The working part of the framework is simply independent layer-wise hyperparameter tuning (DSR). While the authors deserve great credit for their outstanding intellectual honesty in reporting these ablations, the paper ultimately proposes a complex and elaborate mechanism to achieve a result that can be obtained via a simple static scaling trick. As a result, the conceptual leaps and the originality of the core proposed paradigm are minimal.
