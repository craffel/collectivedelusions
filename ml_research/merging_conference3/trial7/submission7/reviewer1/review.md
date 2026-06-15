# Conference Peer Review

## Overall Recommendation
**Rating: 5 (Accept)**  
*Justification:* This paper presents an exceptionally elegant, training-free, and parameter-free "one-pass" dynamic weight-merging framework (ELATI) that elegantly resolves the severe systems-level "two-pass latency penalty" of state-of-the-art penultimate routing networks. By shifting routing decisions to Layer 2 and projecting activations against unsupervised task centroids computed from a hyper-sparse 16-sample calibration split, the authors achieve near single-pass inference latency while maintaining robust task-specific capabilities. The paper exhibits outstanding scientific transparency, rigorous mathematical grounding, and comprehensive empirical validation on both a high-fidelity simulated sandbox and physical, pre-trained Vision Transformer and GPT-2 models. The core idea of leveraging non-parametric, training-free geometric centroids as a "statistical safety net" for dynamic weight blending is a superb example of how architectural and systems-level complexity can be successfully resolved through structural simplification.

---

## Key Strengths and Weaknesses

### Strengths
1. **Architectural Elegance and Simplification:** Shifting the routing decision from the penultimate layer to Layer 2 and using unsupervised geometric activation centroids represents a beautifully simple, intuitive, and highly effective design. It solves a complex systems bottleneck without adding parametric routing layers or requiring complex training.
2. **Computational Complexity Reduction:** Bypassing class-head projections in favor of task-level centroids reduces the mathematical complexity of the routing projection step from $O(B \cdot K \cdot C \cdot D)$ to $O(B \cdot K \cdot D)$, completely eliminating the class-head bottleneck and delivering an outstanding 3.33$\times$ vectorized speedup.
3. **Rigorous and Honest Validation:** The authors provide a highly detailed and honest validation profile. They clearly separate their high-fidelity simulated sandbox from physical pre-trained ViT-Tiny and GPT-2 models. Running physical evaluations on real image pixels and textual sequences provides solid proof of real-world generalizability.
4. **Superior Generalizability and Robustness (Figure 9):** The Out-of-Distribution (OOD) noise sweep demonstrates that ELATI's unsupervised geometric centroids heavily outperform trained, parametric linear classifiers under severe domain shifts, validating that non-parametric simplicity is inherently more robust to overfitting.
5. **Practical Scaling Analysis:** The paper includes a mature systems scaling analysis (Section 4.4.2), profiling full weight materialization against low-rank on-the-fly PEFT serving on LLaMA-7B, illustrating the specific deployment regimes where each paradigm dominates.

### Weaknesses (Unnecessary Over-Engineering)
1. **Redundant Complexity in Online Adaptation:** The "Hybrid Online Centroid Adaptation" (Equations 3-5) and its associated stabilizers (Centroid Anchoring, Dynamic Margin Filtering, Periodic Recalibration) introduce significant hyperparameter tuning and stateful complexity. Given that the static offline centroids already perform exceptionally well and display superior OOD robustness (Figure 9), this self-updating online updating framework is an unnecessary "just-in-case" addition that should be simplified or removed to preserve the elegant, stateless, and training-free spirit of the core design.
2. **Superfluous Mathematical Obfuscation in Sequence Pooling:** The proposed "Attention-Weighted Sequence Pooling" ($\Psi_{\text{attn}}$) requires computing dynamic query scaling. The sequence pooling simulations show that simple Global Mean-Pooling ($\Psi_{\text{mean}}$) and Causal Mean-Pooling are highly robust and already perform within ~1% of other configurations without requiring any unoptimized query vectors or self-attention pooling layers. Global Mean-Pooling is far simpler, more elegant, and completely sufficient.

---

## Detailed Evaluation Criteria

### 1. Soundness
**Rating: Excellent**  
*Justification:* The submission is technically and methodologically flawless. All mathematical formulations are rigorously stated, and the sequential propagation of residual representations is precisely modeled. Claims are thoroughly backed by an exhaustive set of sweeps over 10 independent seeds, subspace entanglement, routing layer indices, calibration split sizes, active expert pruning thresholds, and out-of-distribution noise. The physical end-to-end downstream classification experiments on ViT-Tiny verify that early-layer routing coefficients can successfully guide downstream dynamic weight merging under real representation flows without causing representational collapse, recovering over 82% of the Expert Oracle ceiling under extreme data-scarcity (16 training samples).

### 2. Presentation
**Rating: Excellent**  
*Justification:* The paper is written with outstanding clarity, precision, and logical flow. The figures (Figures 1-14) and tables (Tables 1-5) are exceptionally informative, detailed, and beautifully support the main text. The authors do an excellent job of contextualizing their work relative to static merging, MoE-LoRA, dynamic merging (PFSR), and early exit networks. Furthermore, the scientific honesty in explicitly disclosing the PyTorch sandbox simulations, CPU benchmarking limitations, and GPU scaling model details is highly commendable and sets a high standard for research transparency.

### 3. Significance
**Rating: Excellent**  
*Justification:* Bypassing the "two-pass latency penalty" of dynamic weight-space model merging is a highly significant advance for deep learning deployments. By proving that early-layer, training-free, unsupervised centroids can drive dynamic micro-batch homogenization, ELATI achieves a 1.40$\times$ physical end-to-end speedup, representing a major step toward making dynamic model merging practically viable for low-latency cloud streaming and resource-constrained edge-AI hardware.

### 4. Originality
**Rating: Excellent**  
*Justification:* The paper introduces a highly original and elegant paradigm. Shifting routing to intermediate layers and utilizing unoptimized, non-parametric activation centroids computed offline as "frozen projection keys" represents a highly creative and original combination of representation probing and weight-space model ensembling.

---

## Constructive Suggestions for Improvement
1. **De-emphasize the Online Adaptation Mechanism:** To streamline the narrative, the authors should move the "Hybrid Online Centroid Adaptation" (Section 3.2, Equations 3-5) and its associated stabilizers to the Appendix. This would keep the main text focused on the core, beautifully clean, non-parametric static centroids which are the paper's absolute strongest and most robust contribution.
2. **Champion Simplicity as the Primary Advantage:** The authors should use Figure 9 and Figure 11 to actively argue *for* the benefits of non-parametric simplicity over parametric complexity. Highlighting that unoptimized, training-free centroids are inherently more robust to overfitting and domain shifts than trained classifiers would make the paper's core thesis significantly stronger and highly aligned with practical systems engineering guidelines.
3. **De-obfuscate sequence pooling:** Consider presenting Global Mean-Pooling and Causal Mean-Pooling as the primary sequence pooling methods, and framing Attention-Weighted Sequence Pooling ($\Psi_{\text{attn}}$) as a speculative extension, since simpler spatial averages are completely sufficient and require zero query optimization.
