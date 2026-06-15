# Peer Review of "HyperMerge: Hyperbolic Space Activation Routing and Fusion for Modular Deep Learning"

## 1. Summary of the Paper
The paper proposes **HyperMerge** (*Hyperbolic Space Activation Routing and Fusion*), a novel non-Euclidean paradigm for dynamic, test-time model ensembling and activation-space routing. It challenges the fundamental "Euclidean assumption" of modern deep models, arguing that flat representation spaces suffer from "representation crowding" and "destructive inter-task cross-talk" near the origin, particularly when multiple task-specific expert adapters are ensembled under heterogeneous input streams.

To overcome these flat geometric limitations, HyperMerge projects intermediate adapter updates into the Poincaré Ball ($\mathbb{D}_c^D$) at each layer using exponential maps, performs non-linear ensembling via **Beltrami-Klein Symmetric Blending** (BKSB), and projects the fused activation back to Euclidean space using logarithmic maps. BKSB maps Poincaré coordinates to Beltrami-Klein coordinates ($\mathbb{K}_c^D$) to compute a permutation-invariant Einstein midpoint, resolving the order-dependence and non-associative flaws of sequential Möbius addition. Task reference centroids are calibrated using **Hyperbolic Centroid Alignment** (HCA). Dynamic routing and **Hyperbolic Out-of-Distribution Rejection** (HOR) are performed on Layer 0 embedding representations.

The authors evaluate HyperMerge inside a 14-layer synthetic "Analytical Coordinate Sandbox" designed to simulate representation flows. The paper reports performance across two regimes: an Orthogonal Subspace Sandbox (Standard) and a highly crowded Overlapping Subspace Sandbox.

---

## 2. Overall Recommendation

**Rating: 3 (Weak Reject)**

**Justification:**
HyperMerge presents an exceptionally creative, mathematically elegant, and highly original concept: shifting the geometric substrate of dynamic model merging and ensembling into negatively curved hyperbolic space. The introduction of Beltrami-Klein Symmetric Blending (BKSB) is a rigorous, permutation-invariant formulation that successfully resolves the "double-weighting flaw" of other ensembling schemes.

However, the paper suffers from severe empirical limitations and writing flaws that preclude publication in a top-tier machine learning venue (such as ICML, NeurIPS, or ICLR) in its current state. Crucially:
1. The empirical evaluation is restricted entirely to a toy, synthetic 14-layer identity-matrix simulator.
2. In BOTH the orthogonal and highly crowded sandbox regimes, flat-space ensembling methods (SABLE, SPS-ZCA) actually outperform HyperMerge, meaning the core claim that negative curvature resolves representation crowding is empirically refuted by the authors' own experiments.
3. The manuscript contains major, highly confusing internal numerical inconsistencies between the Abstract/Intro and the main results tables.

Revisions to include real-world transformer evaluations, resolve the performance deficit of BKSB, and align the numerical reporting are required before this work can be accepted.

---

## 3. Key Flaws (The "Big 3")

### Flaw 1: Exclusively Toy Synthetic Evaluation (No Real-World Backbones or Datasets)
While the "Analytical Coordinate Sandbox" serves as an excellent theoretical proof of concept, the empirical evaluation is completely artificial:
* The "models" are 14 layers of identity matrices with tiny random perturbations. This setup fails to capture the highly complex, non-linear representation manifolds and feature distributions of actual deep neural network backbones.
* To be considered a complete and publishable paper, HyperMerge must be validated on physical pre-trained models (e.g., ViT-B/16, CLIP, or RoBERTa-base) on actual image/text datasets (such as MNIST, CIFAR-10, SVHN, or GLUE) using actual trained LoRA adapters. Relying exclusively on a synthetic vector-space simulator severely limits the significance, generalizability, and impact of the empirical claims.

### Flaw 2: Core Hypothesis Refuted by Overlapping Subspace Performance Deficit
The central thesis and motivation of HyperMerge is that flat-space methods suffer from "representation crowding" and "cross-talk," and that negatively curved hyperbolic space physically separates task manifolds to resolve these issues. However, the empirical results directly contradict or fail to support this hypothesis:
* **In the Orthogonal Sandbox (Table 1):** SABLE (Early Routing) achieves **84.03% $\pm$ 5.15%** joint mean accuracy, outpacing HyperMerge (Ours) at **83.40% $\pm$ 5.15%**.
* **In the Overlapping Sandbox (Table 2):** Under the highly crowded regime specifically designed to showcase the benefits of negative curvature, SABLE (Early Routing) achieves **77.98% $\pm$ 2.12%** and SPS-ZCA achieves **77.32% $\pm$ 1.98%**, whereas HyperMerge ($c=0.1$) achieves only **76.62% $\pm$ 3.96%** (which degrades slightly to **76.50% $\pm$ 3.36%** when tuned).
* **The Implication:** Under both clean and crowded setups, flat Euclidean ensembling baselines outperform the hyperbolic formulation. This indicates that the non-linear radial compression and distortion introduced by repeated exponential and logarithmic projections ($\exp_{\mathbf{0}}^c, \log_{\mathbf{0}}^c$) at every layer outweigh any geometric separation benefits of negative curvature. Because the mathematical complexity of HyperMerge fails to yield any empirical gain over simpler flat-space alternatives, its practical utility remains unproven.

### Flaw 3: Severe Internal Numerical and Textual Inconsistencies
The manuscript contains a major internal discrepancy regarding its key performance numbers, suggesting extremely poor editing care:
* **discrepancy 1 (Orthogonal Sandbox):** 
  * The **Abstract, Introduction, and Figure 1 Caption** claim that HyperMerge achieves a joint mean accuracy of **89.30%**, SABLE (Early Routing) achieves **89.65%**, and SPS-ZCA achieves **88.55%**.
  * However, **Table 1 and Section 4.4 text** report the multi-seed averages over 3 random seeds: HyperMerge at **83.40% $\pm$ 5.15%**, SABLE (Early Routing) at **84.03% $\pm$ 5.15%**, and SPS-ZCA at **83.05% $\pm$ 4.95%**.
* **discrepancy 2 (Overlapping Sandbox):**
  * **Table 2 and Section 4.5 text** report: SABLE at **77.98% $\pm$ 2.12%**, SPS-ZCA at **77.32% $\pm$ 1.98%**, and HyperMerge at **76.62% / 76.50%**.
  * However, auxiliary logged files (such as `experiment_results.md` and `progress.md`) report completely different uncalibrated scores: SABLE at **75.35%**, SPS-ZCA at **74.95%**, and HyperMerge at **71.20% / 72.15%**.
* **Impact:** The authors completely failed to align the text in their abstract, introduction, and figures with their main results tables. This makes the manuscript highly confusing and compromises its empirical integrity.

---

## 4. Strengths
* **Highly Original Conceptual Paradigm:** Shifting the geometric substrate of dynamic test-time model merging and routing from flat Euclidean space to negatively curved hyperbolic space is an exceptionally creative and refreshing perspective in modular deep learning.
* **Mathematical Sophistication & Rigor:** The mathematical formulations of Poincaré Ball operations, conformal factors, geodesic distances, exponential/logarithmic maps, and Möbius primitives are clear and rigorous.
* **Resolution of the Double-Weighting Flaw:** The current formulation of BKSB correctly identifies and mathematically resolves the "double-weighting flaw" of other ensembling schemes. Projecting unscaled Poincaré updates directly to Klein space and computing the Lorentz-weighted Einstein midpoint ensures that routing weights are applied exactly once, preserving the correct magnitude of expert updates and making the fusion permutation-invariant.
* **Excellent Scientific Honesty and Transparency:** The authors are highly commended for reporting raw, uncalibrated experimental results and explicitly framing the evaluation within a synthetic "Analytical Coordinate Sandbox" (even when their method underperforms). The discussion in Section 4.4 on mapping distortion and radial compression is intellectually mature and scientifically rigorous.

---

## 5. Detailed Comments on Core Dimensions

### Soundness (Rating: Fair)
* **Critique of Tangent-Space Retraction Analogy:** In Section 3.8, the authors assert that adding a projected-and-blended update back to the Euclidean base activation is "mathematically analogous to retraction operations... guaranteeing geometric consistency." This is a conceptual stretch. Retractions map tangent vectors onto the manifold. Here, the opposite is done: manifold points are mapped back to the tangent space to perform a flat Euclidean addition ($h_b^{(l)} = h_{base, b}^{(l)} + E_{\text{merged}, b}^{(l)}$). Because the base model operates entirely in flat Euclidean space $\mathbb{R}^D$ and does not "see" the hyperbolic geometry, this addition is a flat heuristic. The "geometric consistency" is localized to the adapter updates, rather than a global Riemannian representation flow.
* **Fragile Shallow Centroid Routing & HOR Vulnerability:** Relying solely on Layer 0 embedding representations ($z_b^{\text{embed}}$) for dynamic routing and HOR assumes that task boundaries are easily separable in raw, shallow feature spaces. In complex, real-world tasks, shallow layers capture low-level features (e.g., lighting, resolution, textures) rather than semantic abstractions. This makes the routing and HOR mechanisms highly fragile to covariate shifts and low-level noise.
* **Lack of Statistical Significance Reporting in Abstract/Intro:** While the tables include standard deviations (e.g., 83.40% $\pm$ 5.15%), the performance differences between HyperMerge and the baselines are extremely tight and well within the margins of error. This makes reporting single point-estimates in the Abstract and Intro misleading.
* **Sharp Temperature and Gating Behavior:** The routing temperature $\tau = 0.05$ is extremely sharp, meaning the routing coefficients $\alpha_{k,b}$ are nearly one-hot. If routing is effectively a hard selection, then the complex non-linear ensembling of BKSB is rarely executed, and the system is acting as a hard router. The authors should report and analyze the entropy of the routing weights to prove that continuous ensembling actually takes place.
* **High-Dimensional Numerical Stability:** Clamping projected coordinates to $1/\sqrt{c} - \epsilon$ collapses representation differences near the boundary. As dimensions increase (e.g., $D=4096$ in LLMs), activation norms scale significantly, pushing representation vectors to the hyperbolic boundary and causing high distortion. The scalability of HyperMerge is therefore highly questionable.

### Presentation (Rating: Good)
* The notation and mathematical derivations are beautifully presented. The figures and tables are exceptionally clear.
* **The "Glossy Cover-up" Concern:** While the presentation is structurally excellent, the sophisticated terminology ("Analytical Coordinate Sandbox", "Beltrami-Klein Symmetric Blending", "Hyperbolic Centroid Alignment") and the labeling of simulated tasks as "MNIST, Fashion-MNIST, CIFAR-10, and SVHN" acts as a glossy cover-up. It risks misleading an average reader into believing that real pre-trained vision models were evaluated on actual image datasets. The manuscript must be much more transparent and explicit about the synthetic, non-functional, and subspace-partitioned nature of the underlying experiments.
* **Inconsistencies:** The massive numerical mismatch between the Abstract/Intro and Table 1/Section 4.4 degrades what would otherwise be an "Excellent" presentation rating.

### Significance (Rating: Poor)
* Because the evaluations are conducted entirely within an artificial 14-layer identity matrix simulator, the actual utility of HyperMerge on real-world deep learning backbones and downstream datasets is completely unverified. This makes the significance of the contribution extremely low in its current state, as there is no guarantee that these hyperbolic operations scale, remain numerically stable, or offer any benefit on actual pre-trained transformers.

### Originality (Rating: Good)
* The core idea of **shifting dynamic test-time model merging and activation-space ensembling into hyperbolic space** is highly original and could open up a new sub-field of non-Euclidean model ensembling.
* However, the mathematical primitives (conformal factors, geodesic distances, exponential/logarithmic maps, Möbius operations, Einstein midpoints) are directly adopted from prior hyperbolic deep learning literature (e.g., Ganea et al., 2018) without significant algebraic or geometric modifications. The originality lies in the pipeline ensembling wrapper rather than new mathematical primitives.

---

## 6. Path to Publication: Recommendations for Revision

To transition this paper into a strong, top-tier conference paper (e.g., ICML, NeurIPS, ICLR), the authors must address the critical issues through a thorough revision:

1. **Conduct Real-World Evaluations:**
   Completely replace or supplement the synthetic "Analytical Coordinate Sandbox" with real-world PEFT / LoRA ensembling experiments:
   * **Backbone Models:** Use popular pre-trained models (e.g., RoBERTa-base, ViT-B/16, or LLaMA-3-8B).
   * **Tasks & Datasets:** Evaluate on standard multi-task benchmarks like GLUE (for text understanding) or a suite of image classification datasets (MNIST, CIFAR-10, SVHN, Oxford Pets) using actual trained LoRA adapters.
   * **Ecosystem Tools:** Leverage standard libraries such as Hugging Face's `peft` or `mergekit` to ensure the results are robust, reproducible, and directly comparable to state-of-the-art baselines.

2. **Frame Results Honestly and Constructively (Resolve the Deficit):**
   Address the major empirical deficit where flat Euclidean methods (SABLE, SPS-ZCA) still outperform HyperMerge in both orthogonal and overlapping subspace sandboxes. The authors must either:
   * Refine the hyperbolic ensembling algebra (e.g., incorporating scale normalization, adaptive margins, or layer-wise top-$M$ masking) to ensure that negative curvature actually yields a performance gain under crowding, or
   * Honestly frame the hyperbolic ensembling as an order-independent, mathematically sound geometric alternative rather than claiming superior accuracy.

3. **Resolve Major Internal Numerical Inconsistencies:**
   Ensure complete textual and numerical consistency across the entire paper. The authors must replace the single-run or outdated numbers (**89.30% / 89.65% / 88.55%**) in the Abstract, Introduction, and Figure 1 Caption with the genuine multi-seed stats (**83.40% $\pm$ 5.15% / 84.03% $\pm$ 5.15% / 83.05% $\pm$ 4.95%**) reported in Table 1 and Section 4.4. Similarly, any logged result files should align perfectly with the multi-seed averages.

4. **Clarify the Motivation for Curvature (Task-level vs. Representation-level Hierarchy):**
   The paper motivates the use of hyperbolic space by asserting that negative curvature "naturally accommodates hierarchical task structures... and resolves taxonomic and power-law mismatch." However, the tasks being merged (MNIST, F-MNIST, CIFAR-10, SVHN) are flat and disjoint. Hyperbolic space is suited for tree-like hierarchical data. 
   To justify the geometric shift, the authors must clarify this and pivot their motivation to focus on the scale-invariant power-law distribution of activation features inside deep network layers, rather than vague claims about task-level taxonomies.

5. **In-Depth Analysis of Hybrid Projection Distortion:**
   Provide a theoretical or empirical analysis of the information loss and representation distortion introduced by projecting activations back and forth between $\mathbb{R}^D$ and $\mathbb{D}_c^D$ at every single layer (14 times).

6. **Address Temperature and Gating Behavior:**
   Evaluate the entropy of the routing weights across different routing temperatures $\tau$ to demonstrate that actual continuous ensembling is taking place, rather than simple hard selection of a single expert update.
