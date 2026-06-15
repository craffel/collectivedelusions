# Peer Review

## Summary of the Paper
The paper addresses a highly pressing practical challenge in edge-based machine learning serving: the dynamic ensembling of multi-task parameter-efficient experts (adapters/LoRA) on sequential streams. At serving time, this process is corrupted by two distinct sources of noise: intra-sample depth-wise representation fluctuations (which cause layer-to-layer "routing jitter") and inter-sample temporal noise across sequential non-i.i.d. queries. 

Prior stateful serving frameworks (such as *ChemMerge* or *PAC-Kinetics*) attempt to smooth these trajectories using complex, heavily parameterized systems (e.g., continuous-time ordinary differential equations modeling biochemical kinetics, or PAC-Bayesian state-space optimization loops requiring online backpropagation). In contrast, this work applies Occam's razor to stateful serving. The authors show that the performance of prior complex frameworks does not stem from specialized biochemical formulas or learned matrices, but from the simple, underlying mathematical property of local recursive filtering.

To exploit this, the authors introduce **2D-STEM** (2D Spatio-Temporal Exponential Moving Average), a training-free, highly parameter-efficient, and analytically simplex-preserving 2D bilinear filter. By enforcing a simple linear inequality constraint ($\beta_{\text{depth}} + \beta_{\text{temp}, t} \le 1$), 2D-STEM is analytically proven to preserve the probability simplex at all layers and steps without requiring any online projection or re-normalization operations. To eliminate "transition lag" under abrupt task transitions, the authors propose **Adaptive Temporal Gating (ATG)** with **Power-Law Sharpening (ATG-PL)**, which measures stream similarity on-the-fly at an early frozen layer and uses a sharpening exponent (default $\gamma=3$) to collapse temporal momentum, resetting history instantly during transitions. The authors also solve a subtle mathematical flaw in prior works—first-layer spatial momentum cancellation—by formulating a task-specific **Coordinate-Prior Spatial Boundary Condition**.

---

## Strengths and Weaknesses

### Strengths
1. **Outstanding Practical Utility and Deployability:** The proposed 2D-STEM is completely training-free and introduces zero extra parameters. It requires only a microscopic runtime state of 240 bytes (for a 14-layer, 4-expert configuration), making it incredibly easy to compile and deploy on resource-constrained edge hardware.
2. **Analytical Simplex Preservation:** Proving that the ensembling weights are mathematically guaranteed to remain on the probability simplex via a simple, constant-overhead linear constraint is an elegant and useful contribution. It completely bypasses the need for costly online Softmax re-normalizations or Euclidean projections at each serving step.
3. **Robust Hardware-Level Benefits:** By reducing absolute routing jitter by up to $2.75\times$ in simulated sandboxes and over $5.23\times$ on physical, pre-trained Vision Transformer representations, 2D-STEM provides substantial hardware utility. Stabilizing ensembling trajectories prevents the constant DRAM transfers, bus congestion, and cache thrashing associated with rapidly loading and unloading different adapter weights on edge NPUs or TPUs, yielding significant energy and latency savings.
4. **Transition Lag Resolution via ATG-PL:** Identifying the upward bias of cosine similarity on non-negative spaces is a highly rigorous technical observation. Resolving this via a simple power-law exponent ($\gamma \ge 2$) to sharpen the transition response and collapse temporal momentum to zero is an incredibly elegant and low-overhead solution.
5. **Excellent Edge Efficiency:** Real-time CPU latency profiling shows that 2D-STEM runs in just $1,436.20\,\mu\text{s}$ per step, representing a massive **$49.5\%$ reduction in serving-time execution latency** compared to the continuous-time ChemMerge (Dynamic ODE) baseline, while adding only a minimal $1.24\times$ overhead relative to the completely stateless SABLE router.
6. **Data-Scarce Calibration Robustness:** The method is exceptionally robust to calibration data scarcity, retaining $94.88\%$ ensembling accuracy and $0.0087$ jitter even when the centroid calibration split is reduced to an extremely sparse **$N_{\text{cal}} = 5$ samples** per task.
7. **Exceptional Baseline and Evaluation Rigor:** The authors compare against the entire lineage of merging frameworks across 5 random seeds, conduct formal paired t-tests (showing highly significant $p < 0.01$ improvements), and validate their claims on physical representations extracted from a pre-trained ViT (`vit_tiny`).

### Weaknesses and Areas for Improvement
1. **Lack of Empirical Validation for OOD Fallback:** The authors outline an elegant Out-of-Distribution (OOD) Fallback Policy in Appendix B.1 involving uniform fallback or temporal bypass. Given that physical edge devices frequently encounter sensor noise or completely novel tasks, having a robust fallback is critical. While the theoretical formulation is highly sound, integrating and empirically evaluating this fallback under style drift or sensor noise would have made the paper's practical utility story even stronger.
2. **Minor Training Overhead in Fine-Grained Settings:** In extremely fine-grained multi-task serving where early-layer activations overlap heavily, the authors propose a 2-layer MLP coordinate mapper fallback (Appendix C). While highly appropriate, this fallback technically introduces $7,000$ trainable parameters and a 3-second training step during calibration, which slightly compromises the "completely training-free" and "parameter-free" claims. Discussing this fallback more prominently in the main body would improve transparency.
3. **Centroid Scaling on Deep Networks:** While centroid storage is microscopic for ViT-Tiny ($43$ KB), it scales linearly with layers, experts, and hidden dimensions. For a massive model like LLaMA-7B with 10 experts, it scales to $\approx 5.24$ MB. Although still highly negligible compared to the model's parameters, presenting this linear scaling calculation in the main body would benefit practitioners planning large-scale deployments.

---

## Detailed Evaluation

### Soundness
* **Rating: Excellent**
* **Justification:** The mathematical foundation of 2D-STEM is highly rigorous and correct. The analytical proof of simplex preservation via induction is flawless. The analysis of first-layer spatial momentum cancellation is a subtle and high-value mathematical find, and the proposed Coordinate-Prior boundary successfully resolves it. The empirical evaluation is incredibly thorough: the baseline comparisons are fair (testing both Constant and Dynamic formulations of ChemMerge), and the claims are validated on actual deep representations of a physical pre-trained ViT.

### Presentation
* **Rating: Excellent**
* **Justification:** The paper is exceptionally well-structured, logical, and easy to read. The authors do an outstanding job of spelling out the core signal-processing connections (discrete 2D IIR low-pass filtering) and deconstructing prior work.
* Crucially, the authors include a fully functional, compile-ready PyTorch implementation in Listing 1, which greatly aids practical deployment. Furthermore, Figure 1 (a & b) is beautifully designed, utilizing both distinct colors and highly contrasting line styles to ensure complete accessibility under grayscale compilation.

### Significance
* **Rating: Excellent**
* **Justification:** The problem of serving multi-task parameter-efficient experts on noisy edge streams is of high relevance to practitioners in industry and applied machine learning. Wild ensembling oscillations cause severe cache thrashing and latency spikes on resource-constrained devices due to constant DRAM memory transfers. By suppressing absolute routing jitter by $2.75\times$ to $5.23\times$ while matching or exceeding Oracle ensembling accuracy, 2D-STEM stabilizes edge-serving runtimes and dramatically improves overall energy and system-level efficiency. 

### Originality
* **Rating: Excellent**
* **Justification:** The concept of applying Occam's razor to show that continuous-time biochemical reaction kinetics or learned state-space models can be replaced by a simple, unified 2D discrete bilinear filter is highly creative and refreshing. It represents a major paradigm shift that challenges the prevailing trend of building increasingly complex and overparameterized systems. Formulating discrete 2D spatio-temporal filtering with analytical simplex preservation, power-law transition sharpening, and coordinate-prior boundary conditions is a highly original and high-signal contribution.

---

## Overall Recommendation

* **Recommendation: 6: Strong Accept**
* **Justification:** 
The paper is technically flawless and represents a major triumph of minimalist engineering. The authors have taken a highly complex, heavily parameterized state-of-the-art framework (ChemMerge/PAC-Kinetics) and stripped it down to a single-line, zero-parameter 2D discrete-time bilinear filter that runs **$49.5\%$ faster** while delivering superior noise filtering and transition accuracy. 

The mathematical soundness is exceptional (including induction-proved simplex preservation and boundary condition analyses), and the empirical evaluations are highly rigorous (incorporating formal paired t-tests, robustness sweeps over $N_{\text{cal}}$, and deep trajectory validation on pre-trained ViT representations). Given the pressing need for highly stable, low-latency multi-task edge serving in real-world environments, this work is of exceptional significance to machine learning practitioners. It provides a complete, deployable, and highly efficient solution that can be instantly compiled into popular runtime engines like TensorRT or ONNX. It sets an outstanding example of applying Occam's razor to deep learning systems and is a clear Strong Accept.

---

## Questions for the Authors

1. **Physical Validation of the OOD Fallback Policy:** The proposed Out-of-Distribution (OOD) Fallback Policy in Appendix B.1 is highly intuitive and mathematically elegant. Have you had the chance to empirically test either the "uniform fallback" or "temporal state bypass" strategies under physical sensor noise or style drift? If so, does freezing the temporal history under OOD inputs reliably prevent the contamination of sequence history?
2. **Scaling with Highly Overlapping Task Matrices:** If the system is scaled to a very large number of experts (e.g., $K \ge 50$ tasks) where cumulative background overlaps are substantial, does the Power-Law Gating exponent $\gamma = 3$ remain sufficient to squash the transition similarity bias, or is it beneficial to scale $\gamma$ dynamically based on the active pool size?
3. **Evaluation of the MLP Coordinate Mapper:** In Appendix C, you introduce a 2-layer MLP coordinate mapper to handle extremely fine-grained domains. Did you evaluate the performance of 2D-STEM with this MLP mapper on a physical fine-grained dataset (such as CUB-200 or Standford Dogs)? If so, does the addition of the MLP fully restore the transition-gating responsiveness under highly overlapping visual representations?
4. **Generalization to Token-Level MoE:** You suggest extending 2D-STEM to token-level routing in sparse Mixture-of-Experts (MoE) in Appendix E. In this token-level setting, representations across consecutive tokens fluctuate rapidly. To prevent temporal over-smoothing at syntactic boundaries (e.g., punctuation or start-of-sentence tokens), do you envision utilizing a lightweight syntax-parser or punctuation detector to dynamically reset the temporal momentum, and how would that integrate with the zero-overhead nature of 2D-STEM?
