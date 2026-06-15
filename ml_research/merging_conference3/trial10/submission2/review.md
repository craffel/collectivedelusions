# Comprehensive Peer Review

## Summary of the Paper
This submission addresses the problem of **dynamic model merging (test-time ensembling)** in multi-task sequential serving workloads, where task-specific low-rank adapters (LoRAs) are dynamically blended on-the-fly inside a shared backbone network to handle incoming queries from heterogeneous task domains. 

To overcome the **routing jitter paradox**—where stateless activation-space projections suffer from high-frequency ensembling weight oscillations—recent state-of-the-art frameworks employ stateful kinetics or state space modeling. However, existing methods assume **spatial homogeneity**, applying a single, global ensembling weight vector uniformly across all layers.

To challenge this assumption, the paper proposes **Layer-Decoupled Stateful Kinetics (LDS-Kinetics)**. This framework:
1. **Decouples ensembling dynamics** along network depth by partitioning layers into $M$ disjoint blocks (or individual layers), with each block maintaining its own independent temporal concentration state vector.
2. Incorporates a **unified PAC-Bayesian complexity penalty** based on Catoni’s $\beta$-mixing PAC bound to prevent transductive overfitting of the high-dimensional parameters under short calibration sequences.
3. Scales down state retention dynamically using the **rolling cosine similarity of incoming query coordinates** to minimize phase lag during abrupt workload switches.

The authors evaluate LDS-Kinetics inside a synthetic **14-layer, 192-dimensional Analytical Coordinate Sandbox simulator** across Orthogonal and Overlapping manifold layouts under Homogeneous and Heterogeneous sequential query streams over 5 independent random seeds. They perform a series of ablation studies, hyperparameter sweeps, and execution latency benchmarks.

---

## Strengths
1. **Compelling Conceptual Foundation:** Challenging the spatial homogeneity assumption of prior stateful routers is a highly logical and intellectually satisfying contribution. Modern deep networks are well-known to learn features at different semantic scales across depth, and adapting stateful ensembling tempos to this structure is a natural progression.
2. **Satisfying Physical Deconstruction of Depth Dynamics:** The paper's most impressive contribution is the empirical deconstruction of the learned parameters ($\tau^{(m)}_k, a^{(m)}_k$). Proving that early blocks learn high decay (short memory) to act as dynamic spatial aligners, while late blocks learn low decay (high retention) to act as stable low-pass decision filters, provides a highly satisfying physical explanation of network depth dynamics.
3. **Rigorous Statistical Verification:** Rather than reporting simple mean values, the authors evaluate their method across 5 independent random seeds, provide standard deviations, and conduct paired $t$-tests to assess the significance of the joint accuracy gains.
4. **Impressive Baseline Comparison:** The authors compare LDS-Kinetics against an extensive suite of baselines (Expert Oracle, Uniform Merging, SABLE, PAC-ZCA, ChemMerge, Global PAC-Kinetics, and Decoupled ERM). The inclusion of Decoupled ERM is a highly rigorous ablation that successfully isolates the impact of the proposed PAC-Bayesian regularization.
5. **High Presentation Quality and Transparency:** The paper is exceptionally well-written, clearly structured, and easy to follow. The mathematical equations are precise and consistent. The authors are also highly transparent about the work's boundaries, explicitly detailing several limitations.

---

## Weaknesses & Critical Flaws

Despite the paper's clear merits and polished presentation, there are **three critical flaws** regarding the experimental design and practical significance that must be addressed:

### 1. Lack of Evaluation on Real-World Datasets & Physical Backbones (Sandbox Limitation)
The single biggest weakness is that the entire empirical evaluation is restricted to a **synthetic, simulated 14-layer linear analytical coordinate sandbox**. 
* **Linearity Assumption:** The sandbox models activation propagation across layers as completely linear. Real-world deep networks (like Transformers or CNNs) contain highly non-linear activation functions (e.g., GeLU, SwiGLU, ReLU) and Layer Normalization. Non-linearities cause representational drift across layers, where small perturbations in ensembling weights at early layers can translate to highly non-linear, compounded trajectory shifts in deeper layers. A linear sandbox completely bypasses this critical challenge, making the findings highly idealized.
* **Synthetic Task Coordinates:** The task coordinates are simulated and projected onto synthetic PCA subspaces. In real-world serving workloads, extracting a clean, scale-free task coordinate signal at an early layer is highly challenging and prone to severe representation overlap and out-of-distribution noise.
* **Absence of Physical Backbones:** Without validating the framework on a physical vision model (e.g., ViT-B/L on VTAB) or a generative language model (e.g., LLaMA-3 or Mistral on GLUE) using real datasets, the claimed "deep layer-wise depth-dependent kinetics" cannot be verified as generalizable properties of physical neural networks.

### 2. Practically Negligible Accuracy Gains paired with Extreme Latency Overhead
While the authors show that their accuracy improvements are "statistically significant" via paired $t$-tests, a closer look at the absolute numbers reveals that the actual gains are practically imperceptible, while the computational penalty is severe:
* **Marginal Gains:** On Overlapping Heterogeneous workloads, LDS-Kinetics ($M=11$) achieves **66.84%** joint accuracy compared to **66.81%** for the global baseline (an absolute gain of only **0.03%**). On Orthogonal Heterogeneous, it achieves **66.79%** compared to **66.73%** for the global baseline (an absolute gain of only **0.06%**).
* **Stateless Superiority:** Stateless SABLE (Raw) actually outperforms LDS-Kinetics ($M=11$) in raw accuracy on heterogeneous workloads by **0.15% to 0.19%**, indicating that stateful ensembling still incurs a net accuracy penalty compared to stateless projection.
* **Jitter Penalty:** Decoupling the kinetics state recurrence across layers *increases* the temporal ensembling jitter by **6.3%** (overlapping) to **15.8%** (orthogonal) compared to the global baseline, suggesting a trade-off in local weight-blending stability.
* **Severe Latency Overhead:** Sequence execution latency increases from **5.94 ms** (Global $M=1$) to **65.75 ms** (Fully Decoupled $M=11$), which is a **1006.22% overhead**. Even the Tri-Block ($M=3$) incurs a **197.64% overhead**. 
* **No GPU Benchmarking:** The authors benchmark latency exclusively on CPU. In physical production servers, executing 11 sequential, independent tensor recurrences and Gibbs softmax operations on a GPU would introduce severe CUDA kernel launch overheads and device synchronization bottlenecks. The lack of GPU-based latency measurements is a major omission. A 1000% latency penalty for a $0.03\%$ absolute accuracy gain is an unfavorable and unrealistic trade-off for real-world practitioners.

### 3. Incomplete Empirical Sweeps (Missing Calibration Size $T$ Sweeps)
A core claim of the paper is that unregularized Decoupled ERM fails under "strict low-data regimes" ($T=32$) and collapses back to the global prior, while PAC-Bayesian regularization successfully prevents this. To provide a truly comprehensive empirical validation, a robust sweep over different calibration sizes ($T \in \{32, 64, 128, 256\}$) is missing. Mapping the generalization gap of both Decoupled ERM and regularized LDS-Kinetics as a function of the calibration length $T$ is critical to show:
* At what calibration size does Decoupled ERM naturally escape the lockstep collapse and begin to specialize?
* At what sequence length does the proposed PAC-Bayesian complexity penalty cease to be the primary driver of performance?
The omission of this multi-size sweep limits the depth of the empirical analysis.

---

## Detailed Ratings

### Soundness: Fair
The mathematical formulation of the decoupled kinetics recurrence and the joint PAC-Bayesian bound is theoretically sound, and the transparency regarding the violation of the stationarity assumption is commendable. However, the soundness of the empirical claims is heavily compromised by evaluating exclusively in an idealized linear simulator (sandbox) that ignores non-linear activation propagation and high-dimensional geometry.

### Presentation: Excellent
The paper is exceptionally well-written, clearly organized, and easy to follow. The mathematical notation is precise, the figures (especially Figure 1 and Figure 2) are of high-publication standard, and the description of the methodology provides sufficient detail.

### Significance: Fair
The conceptual idea of depth-decoupled stateful model merging is highly promising, and the analysis of depth-dependent ensembling tempos is intellectually valuable. However, because the framework is restricted to a linear sandbox simulator and incurs an extreme 1000% latency penalty on CPU for a negligible $0.03\%$ to $0.06\%$ absolute accuracy gain, its immediate utility and impact for real-world machine learning practitioners are very low.

### Originality: Good
Challenging the spatial homogeneity of kinetics routing is a highly creative direction. While the core mathematical mechanics (kinetics state recurrence, similarity-based retention scaling, and PAC-Bayesian bounds) are imported directly from PAC-Kinetics, applying them in a depth-decoupled manner and deconstructing the resulting depth-dependent tempos is a novel and interesting contribution.

---

## Overall Recommendation

**Recommendation: 3: Weak Reject**

**Justification:**
This submission has highly polished writing and a very compelling core hypothesis: that network depth should dictate the temporal scale of stateful ensembling. The empirical deconstruction of depth dynamics—showing that early layers act as fast spatial aligners while late layers act as stable low-pass decision filters—is highly interesting and valuable.

However, the weaknesses of the current paper outweigh its strengths. Evaluating exclusively inside a simplified 14-layer linear analytical coordinate sandbox simulator is a major limitation that leaves the generalizability of the findings to physical, non-linear deep neural networks completely unproven. Furthermore, the proposed fully decoupled router incurs an enormous 1000% execution latency penalty while yielding an absolute accuracy improvement of just $0.03\%$ to $0.06\%$ over the global baseline, and actually *increasing* routing jitter by up to $15.8\%$. 

Without validating the framework on physical backbones (such as ViT or LLaMA-3) on real-world datasets, providing actual GPU execution latencies, and executing a complete sweep over calibration sequence lengths, the paper falls short of the rigorous empirical standard required for publication.

---

## Constructive Feedback and Suggestions for Improvement
1. **Implement on Physical Models:** Transition the evaluation to physical, real-world backbones (e.g., LLaMA-3-8B or ViT-B) on standard sequential benchmarks (e.g., GLUE or VTAB task streams). Show that the depth-dependent ensembling tempos still emerge in the presence of non-linear activations and layer normalization.
2. **Conduct Calibration Size Sweeps:** Include a comprehensive sweep over the calibration sequence length ($T \in \{32, 64, 128, 256\}$) in Section 4.4, plotting the generalization gap and specialization trajectory of both LDS-Kinetics and Decoupled ERM.
3. **Address the GPU Latency Bottleneck:** Benchmark execution latency on a GPU (e.g., NVIDIA H100). Address the CUDA kernel launch overheads by exploring batched tensor updates or a fused Triton kernel, showing how the 1000% sequence latency overhead can be physically compressed in production servers.
4. **Optimize the Accuracy-Jitter-Latency Frontier:** Investigate if grouping the layers into a coarser, hierarchical block structure (such as the proposed Early-Heavy grouping) can achieve a much better trade-off, perhaps recovering the accuracy gains of the decoupled router with only a fraction of the computational and jitter penalty.
