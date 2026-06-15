# Impact, Presentation, and Overall Evaluation

## 1. Major Strengths
- **Refreshing Scientific Deconstruction (Occam's Razor):** The paper takes a bold, highly valuable stance against hyper-parameterization and unnecessarily convoluted routing metaphors (such as QWS-Merge). Proving that simple $L_2$ regularization on standard linear primitives easily replicates or outperforms "quantum" superposition routers is a high-signal contribution.
- **Strong Systems-ML Co-Design:** Unlike many model-merging papers that ignore deployment realities, this work explicitly co-designs the algorithm alongside systems constraints (GPU VRAM footprints, PCIe copy bandwidth, SGMV parallel kernels, and sequential-vs-parallel parameter materialization).
- **Statistical Rigor in Calibration:** The introduction of Unit-Norm Calibration (UNC) and Class-Size Scaling Calibration ($O(\sqrt{\log C_k / d})$) demonstrates high technical sophistication in resolving representation and output-space scale imbalances across asymmetrical experts.
- **Clear Analytical Intuition:** The first-order mathematical proof of Layer-Averaging Collapse provides an elegant, structured explanation of why layer-wise dynamic parameters are redundant under shared classification constraints.

## 2. Areas for Improvement
- **Eliminate the "Simulation" Gap:** The authors *must* perform genuine, live end-to-end evaluations of their framework on real fine-tuned Vision Transformers and LLaMA-7B weights. If computational constraints prevent this, the authors must tone down their claims of "large-scale validation on LLMs/ViTs" and explicitly highlight this simulation reliance as a major limitation in the abstract and introduction.
- **Acknowledge the Complexity-Shift Paradox:** The authors claim to simplify model-merging by removing parameters. However, they actually shift the complexity from the model architecture to the underlying data-serving infrastructure. Dynamic batch partitioning, sequential dispatching, index-based output re-assembly, and integration of SGMV kernels require a highly sophisticated, non-trivial serving layer. This engineering complexity-shift should be discussed openly.
- **Soften "Zero Calibration Data" Claims:** The abstract and introduction heavily emphasize a "completely non-parametric framework that contains zero trainable parameters and requires zero calibration split data." However, under GMM OOD rejection, non-classification clustering fallbacks, and representational drift MLP mitigations, the method explicitly requires a calibration split. The authors should reconcile these contradictory statements.
- **Validate GMM Rejection on Real Feature Spaces:** The Gaussian Mixture Model (GMM) density estimator for OOD rejection is only evaluated on the toy synthetic sandbox. Its effectiveness and stability in the complex, high-dimensional representation spaces of large-scale models (like LLaMA-7B) remain unverified.

## 3. Overall Presentation Quality
- **Rating: Excellent.**
- The paper is highly polished, professional, and mathematically rigorous. The writing is clear, the figures are clean, and the tables provide detailed quantitative breakdowns. The "Systems Deployment Decision Matrix" (Table 4) is exceptionally well-conceived and adds significant practical value for systems developers.

## 4. Potential Impact and Significance
- **Rating: Fair to Good.**
- While the philosophical critique is outstanding and could steer the model-merging community toward simpler, more transparent designs, the actual impact of the proposed PFSR + MBH method is limited by:
  1. The lack of genuine, live end-to-end empirical validation on real datasets, leaving its actual performance unproven.
  2. The strict co-design dependency on the PEFT (LoRA) paradigm, which restricts its applicability to full-parameter model-merging.
  3. The substantial systems overhead of MBH, which may discourage adoption in environments that prioritize pure serving simplicity over VRAM-saving trade-offs.
- Nonetheless, the paper provides a highly valuable bridge between systems-level request scheduling and parameter-level dynamic weight merging.
