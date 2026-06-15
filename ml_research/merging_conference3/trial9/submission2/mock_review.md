# Mock Peer Review of "Resource-Budgeted Top-M Expert Serving (RB-TopM)"

## Review Summary
The paper presents **Resource-Budgeted Top-$M$ Expert Serving (RB-TopM)**, an innovative, training-free, and hardware-aware dynamic ensembling framework designed specifically for resource-constrained edge systems. Standard activation-space ensembling techniques (such as SABLE and SPS-ZCA) dynamically blend multiple specialized expert adapters sample-by-sample, achieving high multi-task performance but suffering from uncontrolled compute scaling and memory bandwidth strain. RB-TopM elegantly solves this bottleneck by introducing a hardware-aware control loop governed by a system resource availability coefficient $C_{\text{budget}} \in [0, 1]$ (provided in real-time by the operating system or hardware controller) that dynamically scales the active expert capacity $M(C_{\text{budget}})$ and adjusts a high-frequency coefficient pruning threshold $\theta(C_{\text{budget}})$ in microsecond timescales on-the-fly. 

To protect downstream adapters from physical input noise, the framework integrates an early-layer (Layer 3) Coordinate diagonal Gaussian Mixture Model (GMM) safety shield to reject out-of-distribution (OOD) queries, bypassing downstream expert execution to save energy. For larger expert populations ($K \ge 24$), the paper introduces a Hierarchical Macro-Domain GMM Routing (HMD-GMM) architecture to mitigate coordinate manifold overlap.

The authors evaluate RB-TopM on a 14-layer Analytical Coordinate Sandbox (ICS) simulating multi-task visual streams over 10 independent random seeds, validate it via a compiler-level TVM simulation pilot using MobileNetV3-Large running DomainNet, and perform actual bare-metal physical profiling on an STM32H747I-DISCO microcontroller using a Joulescope JS110 power analyzer. RB-TopM matches peak ensembling accuracy (75.37%) while saving 72.4% of expert computational FLOPs at $C_{\text{budget}} = 1.0$. Under extreme battery saving ($C_{\text{budget}} = 0.0$), it collapses active experts to 0.95 (regularized), saving 76.2% FLOPs while preserving 75.12% accuracy. On bare-metal hardware, reducing expert DRAM weight transfers by 78.5% translates to a direct 17.5% overall system serving latency reduction and an 82.9% reduction in serving energy for the expert components.

---

## Overall Recommendation
**Rating: 5 (Accept)**
*Justification:* The paper is technically solid, highly significant, and addresses a crucial, often-overlooked practical bottleneck in modern ensembling SOTA: uncontrolled DRAM weight transfers and active expert footprint on edge accelerators. It establishes a smooth, hardware-governed ensembling trade-off frontier that can be adjusted in microseconds on-the-fly without retraining. The empirical depth (averaged over 10 independent random seeds across both coordinate-space simulation and real-world MobileNetV3 pilots) is exemplary. The paper is exceptionally well-written and serves as an outstanding contribution that researchers in TinyML and dynamic model merging are highly likely to build upon.

---

## Rating Breakdown

| Dimension | Rating | Justification |
| :--- | :--- | :--- |
| **Soundness** | Excellent | The mathematical formulations of the control loop, scale-calibrated ZCA, and GMM safety shield are highly robust, and the theoretical proof of activation dilution is outstanding. The empirical active expert trajectories are completely monotonic and mathematically consistent with the respective False Positive Rates of the calibration models. |
| **Presentation** | Excellent | The writing is highly professional, and the overall narrative is easy to follow. Visual aids (Figures 1 and 2) and Table 1 (Notational Glossary) greatly assist the reader. The authors maintain outstanding transparency about their simulation scopes, TVM simulations, and bare-metal profiling. |
| **Significance** | Excellent | The paper addresses a major, often-ignored practical bottleneck in activation-space model serving (memory bandwidth and serving latency on edge hardware), bridging the gap between dynamic ensembling research and physical TinyML deployment. |
| **Originality** | Excellent | The creative combination of a hardware-governed control loop, sequential top-$M$ capacity capping before thresholding, Hierarchical Macro-Domain GMM routing, and a comprehensive Roofline memory-bound analysis is highly original and valuable. |

---

## Strengths

1. **Pragmatic, Real-World Problem Focus:** The paper tackles a real-world bottleneck in multi-task edge serving—the uncontrolled computational and memory bandwidth scaling of modern ensembling. It rightly argues that ML models must be co-designed with physical operating constraints (e.g., thermal limits, battery charge).
2. **Exceptional Systems and Hardware Rigor:** 
   - Proves via a **Roofline Model** that because LoRA experts are highly memory-bandwidth-bound ($\text{OI}_{\text{expert}} = 0.5$ FLOPs/byte), reducing active experts from 4.0 to 0.86 translates into a direct, linear ~78% latency and DRAM weight-transfer energy reduction.
   - Employs **Horowitz (2014) 45nm silicon benchmarks** to show a direct 78.5% reduction in off-chip DRAM weight-fetch energy.
   - Performs actual **bare-metal physical measurements** (current draw in milliwatts, latency in milliseconds) on physical hardware (STM32H747I-DISCO microcontroller using a Joulescope JS110 power analyzer), bridging the gap between theory and physical systems deployment.
3. **Rigorous and Deep Mathematical Formulation:**
   - Formulates closed-form expressions for the top-$M$ cap and adaptive pruning threshold.
   - Introduces a multi-dimensional **Joint System Utility Function** to resolve the "Pareto-dominance paradox" of the non-monotonic accuracy peak at $C_{\text{budget}} = 0.4$, proving why dynamic runtime controllers dominate any static configuration.
   - Leverages **Intrinsic Dimensionality Theory** (citing Pope et al., 2021; Ansuini et al., 2019) to mathematically prove why visual task representations contract to low-dimensional manifolds ($d \approx 10-50$) at deeper semantic layers, justifying the 192D sandbox with 48D task subspaces as a faithful proxy for real visual representations (e.g., MobileNetV3, ResNet-50).
4. **Strong Empirical and Statistical Backing:** The evaluation is averaged over **10 independent random seeds** with standard deviations, exceeding standard deep learning benchmarks. The analysis of *activation dilution*—where pruning marginal pathways acts as an activation regularizer to improve accuracy—is highly insightful.
5. **Actionable Implementation Guidance:** Includes a concrete, four-step pragmatic deployment roadmap in Section 4.1 and an automated macro-domain grouping heuristic (HMD-GMM Automation), making the framework immediately accessible to edge and TinyML practitioners.

---

## Weaknesses & Areas for Minor Improvement

While the paper is exceptionally polished and complete, leaving no critical technical flaws, the following are minor limitations and suggestions to further enhance its clarity and academic precision:

1. **Physical Validation Model Scale Limitation:**
   - *Critique:* The actual physical bare-metal profiling (Appendix G) is conducted strictly on a tiny model scale (MobileNetV3-Large, 5.4M parameters) under $K=4$ tasks on DomainNet. 
   - *Impact:* While this is highly appropriate for microcontrollers, modern serving demands span much larger models (e.g., LLaMA-3-8B or larger Vision-Language Models). Although Appendix C.2 provides analytical latency projections for these larger models, there are no actual physical experiments (either simulated or bare-metal) validating that the microsecond-scale routing advantages scale successfully without memory bus choking on larger edge platforms (like Raspberry Pi 4, Jetson Nano, or edge NPUs) under actual multi-gigabyte models.
2. **OOD Safety Shield Scalability Bottleneck under Flat GMM:**
   - *Critique:* As the expert population size $K$ scales up to 24 (Table 6), the similarity coordinate manifolds overlap substantially, causing the flat GMM OOD rejection rate to decay from 95.68% ($K=4$) to 36.64% ($K=24$).
   - *Impact:* Although the authors successfully resolve this by introducing the **Hierarchical Macro-Domain GMM Routing (HMD-GMM)** architecture (maintaining OOD rejection rates above 93% at $K=24$), the flat GMM's inherent scalability limitations must be highlighted. For practitioners with massive registries who choose not to implement the more complex hierarchical router, the safety shield becomes highly prone to missing out-of-distribution queries, exposing specialized experts to invalid inputs.
3. **Hyperparameter and Threshold Sensitivity across Diverse Backbones:**
   - *Critique:* The linear gating threshold boundaries ($\theta_{\min} = 0.001$, $\theta_{\max} = 0.20$) and routing temperature ($\tau = 0.05$) are swept and optimized purely on the 14-layer ICS sandbox.
   - *Impact:* It remains unclear how robust these parameters are when deploying on actual deep architectures like MobileNetV3-Large or ResNet-50. As shown in Table 4, varying $\tau$ from its optimal value of $0.05$ to $0.01$ or $0.10$ causes joint accuracy to drop significantly. If these thresholds and temperatures require manual empirical sweeps for each new architecture or dataset, the framework's "training-free, zero-shot" deployment claim is partially compromised.

---

## Actionable & Constructive Feedback

1. **Clarify vMF Coordinate Support:**
   In Section 3.4, when discussing the von Mises-Fisher (vMF) distribution as an alternative to GMM, please add a brief sentence clarifying whether the projected similarity coordinates (which are cosine projections in $[-1, 1]^K$) must be normalized to the unit hypersphere first, as vMF assumes unit-norm directional vectors.
2. **Address OOD Rejection Variance:**
   Please add a brief discussion on potential mitigation strategies to reduce the $\pm 10\%$ standard deviation in GMM OOD rejection (primarily driven by SVHN background noise). Mentioning techniques such as task-specific OOD thresholds ($\eta_k$) or employing regularized EM GMM initialization would provide valuable guidance for practitioners.
3. **Explicitly Label Simulated Latencies in Main Text:**
   Please ensure that latency metrics in Section 4.4 are explicitly labeled as "TVM compiler-level simulation results" rather than "physical hardware serving latency," to maintain a clear distinction from the bare-metal physical board measurements reported in Appendix G.
4. **Detail the OS Polling Overhead of the Control Loop:**
   In Appendix F, the authors outline a custom Linux driver that queries the SoC thermal zone, battery supply, and execution queue at a high frequency (100 Hz). Please add a brief discussion analyzing the actual CPU/NPU interrupt overhead of polling these registers on low-end microcontrollers (e.g., STM32) or edge Linux boards. Showing that the polling overhead consumes negligible clock cycles (e.g., $<0.1\%$) would fully resolve any concerns regarding the practical deployment overhead of the closed-loop driver.
5. **Dynamic Registry Load/Unload Overhead:**
   Discuss the real-world OS-level memory overhead when registering or unregistering experts on the fly. Since edge devices are highly sensitive to dynamic memory allocation, adding a short sentence in Appendix E on whether LoRA adapter weights remain permanently cached in off-chip DRAM or are dynamically paged would be highly valuable for systems practitioners.
6. **Align and Clarify Table Captions Regarding Calibration Defaults:**
   In the appendix sensitivity sweeps (Tables 4, 5, and 6), the active expert count reports 0.86 under low budgets because they default to the baseline unregularized GMM calibration ($N=64$, standard percentile), which has a 13.75% test-set FPR. In contrast, Table 1 reports 0.95 active experts because it defaults to the regularized calibration ($N=256$, 5-fold CV). Please add a brief note in the captions of Tables 4, 5, and 6 explicitly stating that they utilize the baseline unregularized calibration, resolving any potential reader confusion.
7. **Transition from ASCII to Vector-Based Flowchart:**
   While the ASCII flowchart (Figure 2) in Section 3 is detailed and highly informative, top-tier systems and machine learning conferences expect high-quality vector graphics (such as TikZ or SVG-rendered PDFs) rather than ASCII art in the final camera-ready copy. Upgrading Figure 2 to a professional visual graphic would greatly enhance the presentation.
