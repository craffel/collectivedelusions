# Impact and Presentation Quality

## Major Strengths
1. **Mathematical Simplicity and Low Overhead:**
   - The formulation of 2D-STEM as a direct discrete-time 2D bilinear filter is highly elegant. By replacing online continuous-time ODE solvers and learned PAC-Bayesian state-space models with a single-line arithmetic update, it represents a highly practical, low-overhead solution for resource-constrained edge hardware.
2. **Analytical Simplex Preservation:**
   - The inductive proof that a simple linear inequality constraint ($\beta_{\text{depth}} + \beta_{\text{temp}, t} \le 1$) guarantees ensembling weights remain on the probability simplex is a standout contribution. It eliminates the need for expensive projection or re-normalization operations at every step, simplifying hardware compilation.
3. **Power-Law Gating (ATG-PL):**
   - The introduction of a sharpening exponent ($\gamma \ge 2$) directly addresses the geometric bias of cosine similarity over non-negative probability vectors. This is an intuitive and highly effective way to eliminate transition lag under overlapping task representations.
4. **Excellent Presentation:**
   - The writing is clear, logical, and easy to read. The equations are well-presented, and the visual ensembling trajectories in Figure 1 provide a highly intuitive qualitative verification of the noise-filtering and lag-suppression capabilities.

## Areas for Improvement
1. **Address the Real-Model Performance Discrepancy:**
   - The authors must address the critical performance gap on physical representations. In the pre-trained ViT simulation (Section 4.4), the baseline PAC-Kinetics (70.57% accuracy, 0.0063 jitter) and the ChemMerge Proxy (65.83% accuracy, 0.0419 jitter) both outperform 2D-STEM (63.70% accuracy, 0.0675 jitter) under stable homogeneous streams.
   - The authors attempt to brush this off by arguing that PAC-Kinetics isolates depth-wise noise, but to a practitioner, a method that achieves 10x lower jitter and 7% higher alignment accuracy is simply better, regardless of the underlying theoretical explanation. The authors must explain how they plan to bridge this gap or discuss it as an explicit limitation of 2D-STEM in real deep representation spaces.
2. **Conduct Real-World Downstream Classification Experiments:**
   - The evaluation relies entirely on simulated environment representations (ACS) and CLS-token trajectory simulations. To prove real-world utility and impact, the authors should fine-tune actual LoRA experts on real image or text classification datasets (e.g., CIFAR-100, SVHN, DomainNet) and evaluate the joint classification accuracy of the merged experts on edge streaming workloads.
3. **Verify Edge Hardware Latency and Energy Savings:**
   - The paper makes strong claims about 2D-STEM's low-overhead suitability for low-power edge devices. To support these claims, the authors should profile the actual inference latency, CPU/GPU utilization, and memory footprint of 2D-STEM compared to SABLE and ChemMerge on real edge platforms (such as an NVIDIA Jetson Nano or Raspberry Pi).
4. **Resolve the "Training-Free" Contradiction:**
   - The authors should clarify the role of the MLP coordinate-prior mapper introduced in Appendix B. If resolving fine-grained task boundaries requires training an MLP, the claim of 2D-STEM being a completely "training-free" and "parameter-free" method must be qualified.

## Potential Impact and Significance
If the authors can resolve the performance discrepancy on physical representation spaces and demonstrate robust classification accuracy on real downstream datasets, 2D-STEM has high potential impact for edge-based model serving. Its mathematical simplicity, projection-free nature, and low-overhead gating make it highly attractive for practical deployment in low-power, multi-task edge streaming applications. 

However, in its current state, the lack of realistic validation on actual task classification and the fact that simpler temporal-only baselines outperform 2D-STEM on physical representations severely limit its significance and persuasiveness to practitioners.
