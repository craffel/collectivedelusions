# 5. Impact and Presentation

## Major Strengths
1. **Critical and Highly Realistic Problem Focus:**
   The paper identifies and systematically addresses a critical deployment challenge: the vulnerability of test-time adaptive model merging (TTA) to physical input corruptions (sensor noise, weatherproof artifacts). This is a highly realistic outdoor edge-computing concern.
2. **Resource-Minded Engineering Design:**
   The proposed ZO-FlatMerge is explicitly designed around the hardware limits of edge accelerators. Operating flatness optimization exclusively inside the 12-parameter polynomial coefficient space via a gradient-free, zeroth-order approach is a highly creative formulation that achieves **true zero activation caching** (0 MB peak adaptation memory) and **zero weight-space backpropagation**.
3. **Rigorous and Transparent Hardware Profiling:**
   The paper is highly commendable for its thoroughness and transparency in analyzing technical trade-offs. In Section 3.5, the authors conduct a full hardware benchmark, openly discussing the weight-reconstruction DRAM bandwidth bottleneck and the $3.73\times$ latency ratio of ZO-FlatMerge. They offer practical, industry-standard engineering mitigations (e.g., fused CUDA kernels, asynchronous adaptation) to resolve it, which greatly enhances the engineering utility of the paper.
4. **Strong Statistical Setup for Simulated Experiments:**
   The main simulation evaluations are conducted across 15 independent random seeds with detailed standard deviations reported, ensuring high statistical rigor and reducing optimization fluke.

## Areas for Improvement
1. **Resolution of the "Task Arithmetic" Paradox:**
   The paper's most critical weakness is that the proposed FlatMerge method is consistently outperformed by static Task Arithmetic under clean, moderate, and heavy noise in physical MLP and CNN validations. For FlatMerge to have real-world utility, the authors must show that there are realistic physical neural network settings where FlatMerge actually beats Task Arithmetic under clean and moderate noise.
2. **Transition from Simulation to High-Capacity Physical Models:**
   Evaluating the main results on continuous mathematical simulations of Vision Transformers rather than actual Vision Transformer models is a major limitation. The authors should evaluate FlatMerge on actual physical Vision Transformers (such as CLIP ViT-B/32) fine-tuned on real downstream classification tasks. This would eliminate the simulation-to-real gap and provide a much more convincing empirical evaluation.
3. **Statistical Rigor in Physical Validation:**
   Unlike the simulation results, the physical MLP and CNN results (Tables 3 & 4) do not report standard deviations, confidence intervals, or multiple random seed trials. The authors should repeat their physical experiments across multiple seeds and report mean and standard deviation to match the rigor of the simulation section.
4. **Complete Comparative Baselines on Physical weights:**
   Key baselines such as RegCalMerge are missing from the physical validations. Additionally, PolyMerge is missing from the MLP validations. Including these is essential for a fair, comprehensive empirical comparison.
5. **Evaluating the Adaptive Perturbation Radius:**
   The authors introduce an elegant mathematical formula for a dynamic "Adaptive Perturbation Radius" $\sigma(X)$ (Eq. 8), but defer its evaluation to future work. It is highly recommended to actually implement and evaluate this dynamic radius, especially since they note that the fixed radius $\rho = 0.05$ underperforms PolyMerge under extreme noise ($\gamma = 3.0$).

## Overall Presentation Quality
The presentation quality of the paper is **excellent**:
- The paper is exceptionally well-structured, starting with a clear engineering philosophy and walking logically through the mathematical formulation, hardware profiling, and experiments.
- The writing style is direct, professional, and technically precise.
- The mathematical equations (Equation 1 through 8) are clean and correct.
- The figures (Fig. 1 through 7) are informative, highly detailed, and beautifully support the main arguments of the text (e.g., the layer-wise coefficient profiles in Fig. 6 and the latency scaling in Fig. 7).
- The paper successfully positions itself in the literature, clearly distinguishing its contributions from AdaMerging, RegCalMerge, PolyMerge, and standard SAM.

## Potential Impact and Significance
The concept of applying Zeroth-Order Sharpness-Aware Minimization (SAM) within highly compact hyperparameter/blending coefficient spaces to bypass backpropagation and activation caching is highly generalizable. If validated on larger models, this approach could have a significant impact:
- It could inspire a new class of backpropagation-free on-device test-time adaptors for edge-deployed Vision Transformers and Large Language Models (LLMs).
- It provides a highly valuable mathematical and conceptual template for optimizing hyperparameter networks under strict SRAM limits.
However, its potential impact is currently bottlenecked by the fact that it is outperformed by simple static Task Arithmetic on actual physical models. Resolving this performance paradox is critical for the work to achieve widespread practical adoption.
