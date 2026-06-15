# Peer Review

## Summary of the Paper
This submission proposes **Robust Linear Routing (RLR)**, a minimalist, training-free dynamic model merging framework, to deconstruct the escalating complexity of recent parameter fusion methods—specifically, the quantum-inspired **Quantum Wavefunction Superposition Merging (QWS-Merge)** (Vance, 2025). QWS-Merge claimed that classical linear routing suffers from structural limitations leading to catastrophic representation collapse on high-variance, out-of-distribution datasets (like SVHN), which motivated its wave-interference and phase-projection machinery.

Using the guiding principle of Occam's razor, the authors hypothesize that the reported SVHN collapse of classical linear routing is not a fundamental structural limitation of linear projections, but rather an unaddressed manifestation of overfitting and high-variance gating logits during optimization on small calibration sets. To prove this, they propose RLR, which regularizes a simple 768-parameter linear gating layer using standard $L_2$ weight decay and Softmax Temperature scaling. 

Empirically, on a unified Vision Transformer (ViT-Tiny) benchmark, the authors demonstrate that when trained using stable practices, the classical unregularized Linear Router is already highly robust (achieving $91.53\% \pm 0.41\%$ mean joint accuracy across 5 seeds and $94.87\%$ SVHN accuracy on seed 42), completely outperforming QWS-Merge. RLR acts as a specialized stabilizer, slightly trading off peak homogeneous performance to secure superior resilience and prevent batch-level coefficient collapse under mixed heterogeneous test streams (e.g., yielding a $+1.88\%$ absolute accuracy benefit over the unregularized baseline at batch size $B=256$).

---

## Strengths and Weaknesses

### Strengths
1. **Compelling Deconstruction of Complex Architectures:** Adopting a parsimonious "Occam's razor" approach to deconstruct the convoluted wave phase-interference projection of QWS-Merge is highly valuable. The paper serves as an excellent cautionary tale against unnecessary, over-engineered architectural complexity in deep learning.
2. **Unified, Local Re-Implementation for Fair Comparison:** Rather than merely comparing against cross-paper reported numbers, the authors locally re-implemented QWS-Merge and trained it under identical conditions on the exact same expert checkpoints. This ensures a fully controlled, high-signal comparison.
3. **Rigorous Empirical Verification:** The paper performs thorough verification, including multi-seed evaluations over 5 random calibration sets, a 2D hyperparameter sensitivity analysis over $\alpha$ and $T$, and layer representation ablation studies.
4. **Honest Discussion of Technical Trade-offs:** The authors do not over-claim. They provide a balanced, highly transparent analysis in Section 4.4 of the trade-offs between static methods (OFS-Tune) and dynamic methods (RLR) in the presence of test-stream heterogeneity.
5. **Practical Scaling Roadmap:** The discussion on scaling RLR to modern LLMs using sequence-level pooled routing signals over LoRA experts is computationally elegant and highly practical.

### Weaknesses
1. **Severe Citation and Bibliography Omissions (Critical Defect):**
   The paper has severe scholarship deficiencies in its citation and bibliography management. At least **8 unique works** cited in the body of the text are completely missing from the `references.bib` file, resulting in compilation errors and broken citations:
   - **`saim` (`\cite{saim}`):** Cited in Section 2.3 ("and SAIM [23]") to represent a baseline for model landscape regularization, but is completely missing from `references.bib`.
   - **`jin2022dataless` / `Jin2022dataless` (`\cite{jin2022dataless}`, `\cite{Jin2022dataless}`):** Cited in Section 2.1 and Section 1 to refer to RegMean, but is missing from `references.bib`.
   - **`goodfellow2016deep` (`\cite{goodfellow2016deep}`):** Cited in Section 2.3 to ground general deep learning regularization, but is missing from `references.bib`.
   - **`shwartz2014understanding` (`\cite{shwartz2014understanding}`):** Cited in Section 2.3 and Section 4.1 to represent the MNIST dataset reference, but is missing from `references.bib`.
   - **`srivastava2014dropout` (`\cite{srivastava2014dropout}`):** Cited in Section 2.3 to represent softmax temperature scaling, but is missing from `references.bib` due to a key mismatch (the bib file contains `Srivastava2014` but the text cites `srivastava2014dropout`).
   - **`liao2021are` (`\cite{liao2021are}`):** Cited in Section 2.2 to critique online test-time adaptation, but is missing from `references.bib`.
   - **`krizhevsky2009learning` (`\cite{krizhevsky2009learning}`):** Cited in Section 4.1 as the CIFAR-10 reference, but is missing from `references.bib`.
   - **`netzer2011reading` (`\cite{netzer2011reading}`):** Cited in Section 4.1 as the SVHN reference, but is missing from `references.bib`.
   This represents a significant oversight in basic academic scholarship and must be rectified.
2. **Logical Tension Regarding the Gating Collapse Hypothesis:**
   In Section 3.2, the authors hypothesize that "deep task-warped representation shift" (extracting representations from deep layers like Block 11) produces high-variance inputs that trigger catastrophic gating collapse for the unregularized router. 
   However, their own empirical evidence in Table 1 and Table 4 contradicts this: the **unregularized classical Linear Router** using Block 11 (Late) representations achieves **$94.87\%$** SVHN accuracy and **$95.46\%$** Joint Mean accuracy on seed 42, and **$95.41\%$** Joint Mean across Table 4, with **no collapse whatsoever**. 
   This indicates that deep representations are *not* a standalone trigger of collapse. The collapse in prior work was primarily driven by reckless optimization parameters (excessive learning rate $\eta > 0.1$ and over-optimization for $>1000$ steps). The authors need to refine their conceptual text to align precisely with their empirical findings.
3. **Small-Scale Vision Focus:**
   The experiments are conducted on a very compact Vision Transformer (ViT-Tiny; 5.7M parameters) and simple datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). Although this was appropriate to directly deconstruct Vance et al. (2025), a stronger paper would demonstrate the scalability of RLR on modern large-scale architectures (e.g., LLMs or larger Vision-Language Models) as described in their scaling section.

---

## Ratings

### Soundness: Good
The empirical evaluations are sound, featuring multi-seed sweeps, hyperparameter sensitivity sweeps, and a local re-implementation of the primary baseline under identical conditions. However, the rating is bounded at "Good" rather than "Excellent" due to the logical tension between the "deep task-warped representation collapse" hypothesis and the excellent empirical performance of the unregularized router on Block 11 representations.

### Presentation: Fair
The writing style, logical flow, and figure/table presentation are excellent. However, a paper cannot receive a high presentation score when its bibliography is severely incomplete, resulting in at least 8 broken or missing citations throughout the manuscript. This represents a substantial quality-control failure that degrades readability.

### Significance: Good
The conceptual significance of the paper is high, as it restores scientific rigor to the field and challenges the trend of escalating architectural complexity. The practical significance of establishing an elegant, 768-parameter baseline that outperforms state-of-the-art dynamic merging is also strong, though it is currently bounded by the small-scale nature of the vision benchmarks.

### Originality: Fair
The algorithmic originality is low, as the method applies decades-old, standard deep learning regularization techniques ($L_2$ weight decay and temperature scaling) to a simple linear layer. However, the scientific and deconstructive originality of debunking QWS-Merge is highly valuable.

---

## Overall Recommendation

**Score: 4 (Weak Accept)**

**Justification:** 
This paper provides an outstanding, highly necessary deconstruction of QWS-Merge, demonstrating that simple classical linear routing (when configured stably) easily outperforms complex quantum-inspired architectures. It advocates for the principle of Occam's razor in deep learning, which is highly valuable for the research community. 

However, my recommendation is strictly set to a **Weak Accept** (and would be a Weak Reject if the authors were not willing to revise) due to the **severe citation and bibliography omissions** in `references.bib`. A scholarly work must properly attribute its foundational ideas, textbooks, and datasets. Correcting these broken citations and addressing the logical tension regarding the deep-representation collapse hypothesis are absolute prerequisites for publication.

---

## Detailed Feedback and Questions for the Authors

1. **Urgent Bibliography Corrections:**
   Please ensure that all cited works are properly populated in `references.bib`. Specifically, provide complete bibtex entries for:
   - `jin2022dataless` / `Jin2022dataless` (RegMean)
   - `saim` (SAIM)
   - `goodfellow2016deep` (Deep Learning textbook)
   - `shwartz2014understanding` (Understanding Machine Learning textbook)
   - `liao2021are` (Test-time adaptation review)
   - `krizhevsky2009learning` (CIFAR-10)
   - `netzer2011reading` (SVHN)
   - Additionally, resolve the key mismatch for `srivastava2014dropout` in `02_related_work.tex` by aligning it with the bibtex key `Srivastava2014`.

2. **Resolving the Deep-Representation Collapse Hypothesis Tension:**
   In Section 3.2, you attribute gating collapse to "deep task-warped representation shift." Yet, in Table 1 and Table 4, your classical unregularized Linear Router achieves near-ceiling performance on SVHN and Joint Mean using Block 11 (Late) representations under Seed 42, without collapsing. 
   - Please reconcile this: if deep representation routing inherently triggers collapse, why does the unregularized router on Block 11 perform so exceptionally well?
   - As Table 2 indicates, isn't the collapse in Vance et al. (2025) primarily an artifact of high learning rates and massive over-optimization on a tiny dataset, rather than the deep representation layer itself? Please revise Section 3.2 to accurately reflect this distinction.

3. **Inconsistent Default Routing Layers:**
   Section 3.1 states that the default main model configuration extracts representations from Block 11 (Late), which matches Table 1's results. However, Table 2 lists the "Stable Configuration" as using the "First Patch Embedding (Early)" layer, which is also used for the multi-seed sweeps in Section 4.3. Please make the defaults consistent across sections or clearly state in each section which layer representation source was utilized.

4. **Evaluation Scale and LLM Validation:**
   The paper would be vastly stronger if the authors could provide even a single, small-scale experiment validating their elegant sequence-level pooled routing formulation on modern LLMs (e.g., blending two task-specific LoRAs on LLaMA-3-8B). This would demonstrate that RLR's advantages are not confined to compact vision models.
