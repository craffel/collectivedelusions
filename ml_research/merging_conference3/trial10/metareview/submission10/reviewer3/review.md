# Peer Review

## Summary of the Paper
The paper addresses the challenge of serving multi-task parameter-efficient fine-tuning (PEFT) experts, such as LoRA, on dynamic, sequential edge streams. These environments are corrupted by two orthogonal noise sources: (1) intra-sample depth-wise representation variations across layers (routing jitter), and (2) inter-sample temporal noise across consecutive queries. To resolve these, the authors apply Occam's razor to existing complex dynamical frameworks (such as biochemical reaction ODEs in ChemMerge or learned PAC-Bayesian state-space optimization in PAC-Kinetics). They propose **2D-STEM** (2D Spatio-Temporal Exponential Moving Average), a training-free, computationally lightweight, and discrete-time bilinear recursive filter that smooths routing trajectories across both backbone depth and sequence history simultaneously.

Furthermore, they prove that 2D-STEM analytically preserves the probability simplex at all layers and steps without the need for active projection or re-normalization operations, provided that the momentum hyperparameters satisfy a simple linear inequality constraint ($\beta_{\text{depth}} + \beta_{\text{temp}, t} \le 1$). To suppress the inertial transition lag (phase delay) inherent in stateful smoothing when tasks abruptly switch, they introduce **Adaptive Temporal Gating (ATG)**, which scales down temporal momentum on-the-fly. They refine this with **Power-Law Gating (ATG-PL)** to resolve the upward cosine-similarity bias under overlapping manifolds.

The authors evaluate their approach on a simulated 14-layer representation-space sandbox (Analytical Coordinate Sandbox, or ACS) and a CLS-token trajectory simulation on a pre-trained Vision Transformer with four synthetic visual domains.

---

## Main Contributions
1. **Minimalist Deconstruction:** Demonstrating that the noise-reduction capacity of complex dynamical model merging frameworks can be captured by a simple, unified 2D bilinear recursive filter, thereby removing unnecessary ODE and learning-theoretic complexity.
2. **2D-STEM Formulation:** Proposing a direct discrete-time 2D spatio-temporal EMA filter and proving that it analytically preserves the probability simplex under a simple linear constraint, avoiding projection overhead at serving time.
3. **Adaptive Temporal Gating (ATG-PL):** Introducing a similarity-based gating mechanism with Power-Law sharpening to dynamically collapse temporal memory and eliminate transition lag under abrupt task switches.
4. **Activation-Space Trajectory Validation:** Validating the routing trajectory behavior on a pre-trained Vision Transformer representation space under synthetic visual domain streams.

---

## Strengths
1. **Practical Design Philosophy and High Efficiency:** The paper’s application of Occam's razor is highly commendable. Replacing complex, online continuous-time ODE solvers (ChemMerge) or backpropagation-heavy offline training loops (PAC-Kinetics) with a single-line, discrete-time arithmetic update is highly valuable for deployment on resource-constrained edge devices (such as NVIDIA Jetson or mobile processors).
2. **Analytical Simplex Preservation:** The mathematical proof that a simple linear inequality constraint ($\beta_{\text{depth}} + \beta_{\text{temp}, t} \le 1$) is sufficient to guarantee that the ensembling weights reside on the probability simplex is an elegant and highly practical contribution. It completely eliminates the need for expensive projection or re-normalization steps at every layer and serving step, reducing inference latency.
3. **Intuitive Resolution of Cosine Gating Bias:** The analysis of the upward bias in cosine similarity coordinates over non-negative probability vectors is highly insightful. The introduction of Power-Law Gating (ATG-PL) with a sharpening exponent ($\gamma \ge 2$) is a highly elegant and parameter-free solution to eliminate transition lag while preserving stable block smoothing.
4. **Excellent Clarity and Presentation:** The paper is exceptionally well-written, clearly structured, and easy to read. The equations are well-formulated, and the visual ensembling trajectories in Figure 1 provide a highly intuitive qualitative verification of the filter's behavior.

---

## Weaknesses
Despite its elegant formulation and theoretical strengths, the paper has several critical weaknesses regarding its soundness, empirical validation, and the validity of its claims on real model representation spaces. These limitations are of primary concern for practitioners looking to deploy this method in actual edge-serving environments:

1. **Critical Performance Discrepancy on Physical Model Representations:**
   In Section 4.4, the authors present an activation-space trajectory validation on a pre-trained Vision Transformer CLS-token representation. However, the quantitative results in Table 5 reveal a severe performance gap. Under stable homogeneous streams, the baseline **PAC-Kinetics** (a temporal-only tracker) achieves an alignment accuracy of **$70.57\%$** and a routing jitter of **$0.0063$**. In contrast, the proposed **2D-STEM** only achieves **$63.70\%$** alignment accuracy, and its routing jitter is **$0.0675$** (which is over **10 times higher** than PAC-Kinetics). Even the constant-inertia **ChemMerge Proxy** baseline outperforms 2D-STEM on this task, achieving **$65.83\%$** alignment accuracy and **$0.0419$** jitter. Under heterogeneous streams, PAC-Kinetics also outperforms 2D-STEM in alignment accuracy ($67.08\%$ vs. $64.61\%$) and achieves nearly half the jitter ($0.0369$ vs. $0.0679$).
   This represents a major soundness issue. The authors argue in their abstract and introduction that 2D-STEM "surpasses" these highly parameterized frameworks. However, this empirical superiority is restricted to their synthetic Analytical Coordinate Sandbox (ACS). When evaluated on a real pre-trained ViT model, 2D-STEM is actually outperformed by simpler, temporal-only baselines. This indicates that local, layer-wise 2D bilinear filtering may not generalize well to physical deep representation spaces compared to static-depth, sequence-level temporal trackers that isolate the routing from localized propagation noise.

2. **Complete Absence of Real-World Downstream Task Evaluation:**
   A major limitation of the empirical evaluation is the lack of actual downstream classification experiments.
   - The authors do not fine-tune physical PEFT (e.g., LoRA) experts on real image datasets (such as CIFAR-100, SVHN, or DomainNet) and evaluate the joint classification accuracy of the merged experts on dynamic edge streaming workloads.
   - Instead, the entire evaluation is conducted in a highly simulated coordinate sandbox (ACS) and a CLS-token activation trajectory simulation on four programmatically generated visual noise patterns (Checkerboard, Sinusoidal Waves, etc.). 
   - Measuring "relative alignment accuracy" (the cosine similarity to CLS centroids) is a very indirect, surrogate-heavy evaluation. It does not prove that 2D-STEM actually preserves classification accuracy, avoids task interference, or stabilizes real-world predictions. Without actual downstream classification metrics, the practical utility of 2D-STEM remains unproven.

3. **Methodological Contradiction of the "Training-Free" Claim:**
   The authors strongly emphasize that 2D-STEM is a "training-free, highly parameter-efficient... with zero extra parameters and zero online backpropagation or optimization overhead."
   However, in Section 4.5 and Appendix B, they introduce a **2-layer MLP coordinate-prior mapper** to resolve representation overlaps in fine-grained domains. This MLP requires supervised training with task labels on the $N_{\text{cal}} = 64$ calibration samples and introduces approximately 7,000 parameters. While they argue it is trained offline, it directly contradicts the "training-free" and "parameter-free" claims. The authors must qualify their claims or clarify the necessity of this trained module for the physical ViT results.

4. **Inability to Verify Hardware Latency and Memory Gains:**
   The paper argues that 2D-STEM's $O(K \cdot L)$ complexity avoids the $O(K \cdot L \cdot N_{\text{ODE}})$ overhead of ChemMerge. However, because they did not implement 2D-STEM on actual edge hardware (such as an NVIDIA Jetson or Raspberry Pi) or profile the latency/energy consumption of a physical multi-expert serving pipeline, these claimed speedups remain purely theoretical. It is crucial to verify whether these computational savings are meaningful compared to the forward pass latency of the backbone model.

---

## Detailed Ratings

### Soundness: Fair
The mathematical formulation of 2D-STEM and the simplex-preservation proof are rigorous and correct. However, the empirical soundness is severely limited. The method's superior performance is demonstrated only in a highly synthetic coordinate sandbox (ACS). When evaluated on a physical pre-trained ViT model, 2D-STEM is outperformed by the simpler temporal-only baseline (PAC-Kinetics) and the constant-inertia proxy in both alignment accuracy and routing jitter. Furthermore, the lack of real downstream classification tasks and the reliance on CLS-token trajectory simulations undermine the soundness of the claims.

### Presentation: Good
The paper is exceptionally well-written and structured. The narrative is easy to follow, the equations are clear, and the visual trajectories in Figure 1 provide high-signal qualitative insights. The PyTorch implementation in Listing 1 is highly detailed and helpful. However, the authors must address the minor contradiction of claiming a "training-free" method while relying on a trained MLP mapper in the appendix.

### Significance: Fair
The potential significance of a low-overhead, projection-free 2D filter for edge serving is high. However, because the authors fail to demonstrate classification accuracy on real downstream datasets (e.g., CIFAR-100 or DomainNet) and are actually outperformed on physical representations by existing baselines, the practical significance of 2D-STEM is currently low. A practitioner looking to deploy this would choose PAC-Kinetics or a constant-inertia proxy based on the physical ViT results.

### Originality: Good
The application of a discrete-time 2D bilinear filter, the analytical simplex-preservation proof under a simple momentum constraint, and the Power-Law temporal gating to resolve cosine similarity bias represent a creative and elegant combination of signal processing and model merging.

---

## Overall Recommendation: 3 (Weak Reject)
The paper presents an elegant, mathematically pure application of Occam's razor to stateful dynamic model merging. The discrete-time 2D spatio-temporal filter, the simplex-preservation proof, and the Power-Law temporal gating are interesting and highly desirable for low-overhead edge serving.

However, the weaknesses currently outweigh the merits. The severe performance discrepancy on the pre-trained ViT representation space (where 2D-STEM is significantly outperformed by PAC-Kinetics in both accuracy and jitter), the complete absence of actual downstream classification experiments on physical LoRA experts, and the contradiction of using a trained MLP mapper while claiming to be "training-free" must be resolved. The paper requires revision to validate these claims in realistic representation spaces and actual classification tasks before it can be accepted.

---

## Questions and Constructive Feedback for the Authors

1. **Physical Representation Discrepancy:** Why does 2D-STEM perform significantly worse than PAC-Kinetics on the pre-trained ViT representation space (Table 5), achieving $63.70\%$ alignment accuracy and $0.0675$ jitter compared to PAC-Kinetics' $70.57\%$ accuracy and $0.0063$ jitter? If static-depth routing (which isolates the system from depth-wise layer noise) is the reason, doesn't this suggest that 2D-STEM's localized spatio-temporal coupling is disadvantageous in real deep networks? How do you plan to resolve this gap?
2. **Real Downstream Expert Merging:** Can you provide classification accuracy results of physical LoRA experts fine-tuned on real vision datasets (e.g., CIFAR-100, SVHN, DomainNet) merged dynamically using 2D-STEM? Demonstrating actual task accuracy on a streaming edge workload is crucial to establish the practical utility of the method.
3. **Training-Free Contradiction:** Please clarify the role of the 2-layer MLP coordinate mapper introduced in Appendix B. Is this mapper required to obtain the physical ViT results in Table 5? If so, the claim of being a completely "training-free" and "parameter-free" method must be qualified.
4. **Physical Latency Profiling:** Can you profile and report the actual execution latency and memory footprint of 2D-STEM compared to SABLE, PAC-Kinetics, and ChemMerge on real edge hardware (such as an NVIDIA Jetson or Raspberry Pi)? This would verify whether the $O(K \cdot L)$ complexity actually yields meaningful latency or energy savings compared to the forward pass of the backbone network.
5. **Overfitting Risk of MLP Mapper:** Since the MLP mapper in Appendix B is trained on a tiny calibration split ($N_{\text{cal}} = 64$), how do you prevent overfitting, and how robust is this mapper under test-time domain shifts or out-of-distribution inputs?
