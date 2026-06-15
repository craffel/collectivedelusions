# Peer Review of Conference Submission: HyperMerge

## 1. Paper Summary
This paper addresses the problem of dynamic, test-time serving and ensembling of multiple Low-Rank Adaptation (LoRA) expert adapters under heterogeneous input streams. The authors identify a core limitation in existing dynamic ensembling methods: they operate on a flat Euclidean substrate ($\mathbb{R}^D$), which suffers from "representation crowding" and destructive "cross-talk" near the origin under high-dimensional or overlapping task manifolds.

To overcome this, the authors propose **HyperMerge** (Hyperbolic Space Activation Routing and Fusion), which projects intermediate adapter activation updates into the Poincaré Ball model of hyperbolic space ($\mathbb{D}_c^D$). Using negative curvature's exponential volume growth, HyperMerge aims to cleanly segregate task manifolds near the boundary. 

The paper introduces two core algorithms:
1. **Hyperbolic Centroid Alignment (HCA):** Projects calibration embeddings to Klein space ($\mathbb{K}_c^D$) to compute optimal task centroids via Einstein midpoints.
2. **Beltrami-Klein Symmetric Blending (BKSB):** Performs non-linear, permutation-invariant activation ensembling by computing Lorentz-weighted Einstein midpoints in Klein space before mapping back to Poincaré and Euclidean space.

The framework is evaluated in a simulated "14-layer Analytical Coordinate Sandbox" against static parameter merging and Euclidean dynamic ensembling baselines.

---

## 2. Strengths and Weaknesses

### Strengths
1. **Elegant Mathematical Formulations:** The derivation of HCA and BKSB using Lorentz-weighted Einstein midpoints in the Beltrami-Klein model is mathematically elegant. It successfully solves the non-associativity and order-dependency issues of sequential Möbius addition in hyperbolic space, delivering a fully permutation-invariant ensembling operation.
2. **Clear Conceptual Motivation:** The paper provides a well-reasoned motivation for adopting hyperbolic geometry, highlighting how representation crowding near the origin can introduce cross-talk during linear ensembling of overlapping coordinate partitions.
3. **Systems-Driven Design Choices:** Performing routing and Out-of-Distribution rejection (HOR) zero-shot at Layer 0 embeddings is a clever way to bypass the "Routing Paradox" (executing a full deep forward pass just to decide adapter weights), enabling single-pass, constant-latency edge serving.
4. **Detailed Ablation Studies:** The paper includes valuable parametric sensitivity analyses, examining the effects of curvature $c$, testing the robustness of the outlier detector (HOR), and evaluating under a crowded overlapping coordinate regime.

---

### Weaknesses

#### A. Originality and Literature Positioning (Scholar Critique)
1. **Overstated Novelty and Omission of Prior Work:**
   The paper claims that HyperMerge is *"the first work to introduce hyperbolic geometry and non-linear Möbius algebra to the domain of modular deep learning and dynamic model ensembling."* This claim is inaccurate and ignores significant prior literature that has already integrated hyperbolic manifolds with adapters, parameter-efficient fine-tuning (PEFT), and routing:
   * **Hyperbolic Adapters:** Works such as *HypLoRA: Hyperbolic Fine-Tuning for Large Language Models* (Yang et al., 2024/2025) and *Training-Free Dual Hyperbolic Adapters* (Zhang et al., 2024) have already explored applying hyperbolic geometry to LoRA and other adapters.
   * **Hyperbolic Routing & Mixture of Experts:** *MoSLoRA: Parameter-Efficient Fine-Tuning of LLMs with Mixture of Space Experts* (2025/2026) directly proposes using heterogeneous geometric spaces (hyperbolic, spherical, Euclidean) in an MoE-like LoRA framework with dynamic routing. Similarly, *HELM: Hyperbolic Large Language Models via Mixture-of-Curvature Experts* (2025) explores Mixture-of-Experts operating in distinct curvature spaces.
   
   While HyperMerge's focus on dynamic activation ensembling is distinct, the paper's failure to cite, discuss, or position itself relative to these highly relevant, established works represents a major literature gap and leads to false claims of absolute primacy.

2. **Uncited Core Baseline:**
   The authors present "Single-Pass Centroid Alignment (SPS-ZCA)" as a major state-of-the-art Euclidean ensembling baseline. However, there is no bibliographic citation or reference entry for SPS-ZCA in the entire manuscript. For a paper that places emphasis on rigorous comparison against existing work, omitting the citation for its primary state-of-the-art baseline is a severe scholarly error.

3. **Bibliography Clutter:**
   The file `references.bib` contains several highly relevant papers that are never cited in the text (e.g., `matena2021merging`, which discusses merging models with Fisher information). Conversely, there are numerous general-purpose, unused citations cluttering the bibliography. A scholarly submission should ensure its reference list is clean, accurate, and completely aligned with the text.

#### B. Soundness and Methodological Validity
1. **Severe Empirical Disconnection from Core Motivation:**
   The primary thesis of this work is that flat Euclidean space is fundamentally unsuited for ensembling due to representation crowding, and that hyperbolic geometry resolves this. However, in Section 4.5, when the authors simulate a highly crowded "Overlapping Subspace Sandbox Regime," the Euclidean baselines still **outperform** HyperMerge:
   * **SABLE (Early Routing - Euclidean):** **77.98% $\pm$ 2.12%**
   * **SPS-ZCA (SOTA Euclidean):** **77.32% $\pm$ 1.98%**
   * **HyperMerge (Ours, $c=0.1$):** **76.62% $\pm$ 3.96%**
   * **HyperMerge (Ours, Tuned $c=0.2$):** **76.50% $\pm$ 3.36%**
   
   If hyperbolic space is the optimal solution for representation crowding, it should comfortably outperform Euclidean space when crowding is introduced. Instead, it performs worse. The authors attribute this to "localized mapping distortions" from exponential and logarithmic projections. This creates a logical contradiction: the very mathematical machinery required to project activations into hyperbolic space introduces distortions that wipe out the geometric benefits of negative curvature. If flat Euclidean ensembling (like SABLE) is simpler, cheaper, and more accurate under both clean and crowded regimes, there is no practical or theoretical justification for practitioners to adopt HyperMerge.

2. **Severe Numerical Inconsistencies and Contradictions:**
   There are major contradictions between the baseline scores reported in the main results table (Table 1) and those cited in the ablation section (Section 4.5):
   * In **Table 1**, SABLE (Early) is reported at **84.03%** and SPS-ZCA at **83.05%**. But in the curvature ablation text (Section 4.5), SABLE is cited as getting **89.65%** and SPS-ZCA as getting **88.55%**.
   * In the curvature ablation text, "near-Euclidean" HyperMerge ($c=0.001$) is reported at **87.65%**. This near-Euclidean version scores higher than the default HyperMerge ($c=0.1$, **83.40%**) and the main Euclidean baselines from Table 1.
   * If setting $c=0.5$ yields a joint mean accuracy of **91.00%** (outperforming all baselines), why did the authors use a sub-optimal default curvature of $c=0.1$ (83.40%) for Table 1, where HyperMerge actually performs worse than SABLE (84.03%)?
   
   These inconsistencies suggest that the ablation study was evaluated under a completely different coordinate partition, seed, or dimensional split, and comparing these mismatched numbers in the same paper represents a major lack of scientific and empirical rigor.

3. **Suspect "Late Adaptation" Strawman:**
   In Table 1, `SABLE (Late Adaptation)` is shown to catastrophically collapse to **46.37% $\pm$ 5.95%**. The authors provide no explanation or diagnostic analysis for why allowing later layers to adapt routing weights would cause such a dramatic collapse, making it appear to be a poorly tuned strawman.

4. **Review Leak and Unprofessional Language:**
   Under Section 4.2 (Baselines), the authors describe SPS-ZCA as: 
   *"This is the top-performing baseline from Trial 7."*
   Referring to "Trial 7" (which is clearly an internal developmental iteration) is highly unscientific and unprofessional. It violates the standard of a polished, peer-review-ready manuscript.

#### C. Significance and Reproducibility
1. **Purely Synthetic Sandbox Evaluation:**
   The entire framework is evaluated within a custom, synthetic "Analytical Coordinate Sandbox." There are no real deep learning models (e.g., LLaMA, Mistral, ViTs, ResNets) and no actual datasets (e.g., GLUE, ImageNet, MNIST images) evaluated. The "experts" are merely simulated by partitioning dimensions. While synthetic environments are useful for proof-of-concept, presenting HyperMerge as a solved solution for edge AI and dynamic model merging without a single real-world neural network experiment is highly premature.
2. **Poor Reproducibility:**
   The "Analytical Coordinate Sandbox" is an ad-hoc, proprietary simulation environment. Because its exact generation process, dimensions, and coordinate splits are non-standard, and because no open-source code or repository link is provided, external researchers cannot reproduce these findings.

---

## 3. Detailed, Actionable Feedback for Improvement

1. **Recalibrate Novelty Claims and Address Literature Gaps:**
   * Remove the claims of absolute primacy in Section 1 and Section 2.
   * Add a dedicated subsection in Related Work (Section 2) discussing hyperbolic PEFT, adapters, and MoE routing. Cite and position HyperMerge directly relative to *HypLoRA* (Yang et al.), *MoSLoRA* (2025/2026), and *HELM* (2025). Clearly articulate how Beltrami-Klein Symmetric Blending is distinct from these works.
2. **Resolve All Numerical Inconsistencies:**
   * Ensure that all tables, figures, and text are evaluated under the exact same experimental configuration (same seeds, same dimensionality, same coordinate splits). 
   * Re-run and update Section 4.5 so that baseline scores (SABLE, SPS-ZCA) match those in Table 1, or explicitly explain why the setup is different.
   * If curvature $c=0.5$ truly achieves 91.00% under the main Table 1 configuration, update Table 1 to feature this optimal configuration.
3. **Conduct Real-World Deep Learning Experiments:**
   * Evaluate HyperMerge on a real-world multi-task learning setup. For example, merge task-specific LoRA adapters (e.g., trained on GLUE tasks or specialized vision datasets) using a pre-trained base model (like LLaMA-3-8B or ViT-B/16) and report actual validation accuracies.
4. **Address the Crowded Regime Deficit:**
   * Provide a thorough diagnostic analysis of why HyperMerge is outperformed by Euclidean baselines under coordinate crowding (Table 2). If exponential/logarithmic mapping distortions are the bottleneck, explore ways to mitigate this (e.g., adaptive curvature, curvature scaling, or learning the projection tangent space) to prove that negative curvature can actually outperform flat space in practice.
5. **Clean Up Presentation and Citations:**
   * Remove informal developmental language such as *"Trial 7"* in Section 4.2.
   * Add a proper bibliographic citation and reference entry for `SPS-ZCA`.
   * Prune unused references from `references.bib` and ensure all cited works in the text correspond to entries in the bibliography.

---

## 4. Evaluation Ratings

* **Soundness:** **Fair** (The mathematical formulations of HCA and BKSB are rigorous, but severe numerical inconsistencies in Section 4.5 and the failure to outperform Euclidean baselines under coordinate crowding undermine the methodological soundness).
* **Presentation:** **Fair** (While well-written and structured in Sections 1-3, the submission is severely marred by missing citations for the SOTA baseline, bibliography clutter, unscientific review leak language, and contradictory numbers between tables and text).
* **Significance:** **Poor** (Evaluating exclusively in a synthetic mathematical coordinate sandbox without real models or real datasets severely limits the significance and generalizability of the contribution).
* **Originality:** **Fair** (The mathematical use of Klein Einstein midpoints for activation blending is elegant, but the claims of primacy are heavily overstated due to the omission of key prior works in hyperbolic PEFT and routing).

---

## 5. Overall Recommendation

**Recommendation: Reject (2)**

**Justification:** 
While the mathematical formulation of Hyperbolic Centroid Alignment and Beltrami-Klein Symmetric Blending is elegant and theoretically interesting, the submission falls significantly short of conference standards on multiple fronts. 

Empirically, the paper fails to support its core motivation: under the highly crowded coordinate overlap regime (where hyperbolic space was supposed to resolve Euclidean representation crowding), the proposed method is outperformed by simpler flat Euclidean ensembling baselines. 

Furthermore, the paper contains severe scientific and scholarly flaws: major numerical contradictions between the main results and the ablation text, a complete omission of a citation for the state-of-the-art baseline SPS-ZCA, overblown novelty claims that ignore highly relevant prior work in hyperbolic adapters/PEFT, and unscientific review-leak language ("Trial 7"). 

Finally, the evaluation is entirely restricted to a synthetic custom sandbox, with no validation on real-world deep learning models or datasets. The authors are strongly encouraged to address these literature gaps, resolve the numerical inconsistencies, and validate their elegant mathematical ideas on real-world neural networks before re-submitting.
