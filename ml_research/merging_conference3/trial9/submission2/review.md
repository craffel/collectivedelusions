# Peer Review: Resource-Budgeted Top-M Expert Serving (RB-TopM)

## 1. Summary of the Paper
This paper presents **Resource-Budgeted Top-$M$ Expert Serving (RB-TopM)**, a training-free, hardware-aware framework for dynamic model ensembling on the edge. The paper addresses a critical and frequently neglected bottleneck of modern activation-space model ensembling methods (like SABLE or SPS-ZCA): they assume constant, infinite resources and execute multiple concurrent adapter pathways in parallel, which is highly inefficient and unsustainable under volatile edge conditions (e.g., low battery, thermal throttling).

RB-TopM introduces a system-level control loop governed by a dynamic resource budget coefficient $C_{\text{budget}} \in [0, 1]$. This parameter dynamically scales:
1. A **Resource-Budgeted Top-$M$ Cap ($M(C_{\text{budget}})$)** which restricts active ensembling capacity.
2. An **Adaptive Gating Threshold ($\theta(C_{\text{budget}})$)** which filters out marginal, low-contribution experts.

Additionally, the framework integrates an early-layer (Layer 3) Coordinate diagonal Gaussian Mixture Model (GMM) safety shield to reject out-of-distribution (OOD) queries, bypassing downstream expert execution to save energy. 

The authors evaluate RB-TopM on a 14-layer Analytical Coordinate Sandbox (ICS) simulating multi-task visual streams over 10 independent random seeds, validate it via a physical pilot using MobileNetV3-Large running DomainNet, and perform actual bare-metal physical profiling on an STM32H747I-DISCO microcontroller using a Joulescope JS110 power analyzer. RB-TopM matches peak ensembling accuracy (75.37%) while saving 72.4% of expert computational FLOPs at $C_{\text{budget}} = 1.0$. Under extreme battery saving ($C_{\text{budget}} = 0.0$), it collapses active experts to 0.95 (regularized), saving 76.2% FLOPs while preserving 75.12% accuracy. On bare-metal hardware, it achieves an outstanding **74.7% reduction in physical latency** and an **82.9% reduction in serving energy** for the expert components.

---

## 2. Reviewer Ratings

- **Soundness:** **Excellent**
- **Presentation:** **Excellent**
- **Significance:** **Excellent**
- **Originality:** **Excellent**
- **Overall Recommendation:** **5: Accept** (Technically solid paper, with high impact on at least one sub-area of AI or moderate-to-high impact on more than one area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.)

---

## 3. Strengths and Weaknesses

### Strengths:
1. **Highly Pragmatic & Realistic Framing:** The paper tackles a real-world bottleneck in multi-task edge serving—the uncontrolled computational and memory bandwidth scaling of modern ensembling. It rightly argues that ML models must be co-designed with physical operating constraints (e.g., thermal limits, battery charge).
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

### Weaknesses:

#### Weakness 1: Physical Validation Model Scale Limitation
- *Critique:* The actual physical bare-metal profiling (Appendix G) is conducted strictly on a tiny model scale (MobileNetV3-Large, 5.4M parameters) under $K=4$ tasks on DomainNet. 
- *Impact:* While this is highly appropriate for microcontrollers, modern serving demands span much larger models (e.g., LLaMA-3-8B or larger Vision-Language Models). Although Appendix C.2 provides analytical latency projections for these larger models, there are no actual physical experiments (either simulated or bare-metal) validating that the microsecond-scale routing advantages scale successfully without memory bus choking on larger edge platforms (like Raspberry Pi 4, Jetson Nano, or edge NPUs) under actual multi-gigabyte models.

#### Weakness 2: OOD Safety Shield Scalability Bottleneck under Flat GMM
- *Critique:* As the expert population size $K$ scales up to 24 (Table 6), the similarity coordinate manifolds overlap substantially, causing the flat GMM OOD rejection rate to decay from 95.68% ($K=4$) to 36.64% ($K=24$).
- *Impact:* Although the authors successfully resolve this by introducing the **Hierarchical Macro-Domain GMM Routing (HMD-GMM)** architecture (maintaining OOD rejection rates above 93% at $K=24$), the flat GMM's inherent scalability limitations must be highlighted. For practitioners with massive registries who choose not to implement the more complex hierarchical router, the safety shield becomes highly prone to missing out-of-distribution queries, exposing specialized experts to invalid inputs.

#### Weakness 3: Hyperparameter and Threshold Sensitivity across Diverse Backbones
- *Critique:* The linear gating threshold boundaries ($\theta_{\min} = 0.001$, $\theta_{\max} = 0.20$) and routing temperature ($\tau = 0.05$) are swept and optimized purely on the 14-layer ICS sandbox.
- *Impact:* It remains unclear how robust these parameters are when deploying on actual deep architectures like MobileNetV3-Large or ResNet-50. As shown in Table 4, varying $\tau$ from its optimal value of $0.05$ to $0.01$ or $0.10$ causes joint accuracy to drop significantly. If these thresholds and temperatures require manual empirical sweeps for each new architecture or dataset, the framework's "training-free, zero-shot" deployment claim is partially compromised.

---

## 4. Detailed Comments and Actionable Suggestions

- **Suggestion 1 (Align and Clarify Table Captions Regarding Calibration Defaults):**
  In the appendix sensitivity sweeps (Tables 4, 5, and 6), the active expert count reports 0.86 under low budgets because they default to the baseline unregularized GMM calibration ($N=64$, standard percentile), which has a 13.75% test-set FPR. In contrast, Table 1 reports 0.95 active experts because it defaults to the regularized calibration ($N=256$, 5-fold CV). While this is explained in Table 7 (labeled as `tab:app_gmm_regularization` in Appendix F.3), the authors should add a brief note in the captions of Tables 4, 5, and 6 explicitly stating that they utilize the baseline unregularized calibration, resolving any initial reader confusion.
- **Suggestion 2 (Transition from ASCII to Vector-Based Flowchart):**
  While the ASCII flowchart (Figure 2) in Section 3 is detailed and highly informative, top-tier systems and machine learning conferences expect high-quality vector graphics (such as TikZ or SVG-rendered PDFs) rather than ASCII art in the final camera-ready copy. Upgrading Figure 2 to a professional visual graphic would greatly enhance the presentation.
- **Suggestion 3 (Detail the OS Polling Overhead of the Control Loop):**
  In Appendix E, the authors outline a custom Linux driver that queries the SoC thermal zone, battery supply, and execution queue at a high frequency (100 Hz). The authors should add a brief discussion analyzing the actual CPU/NPU interrupt overhead of polling these registers on low-end microcontrollers (e.g., STM32) or edge Linux boards. Showing that the polling overhead consumes negligible clock cycles (e.g., $<0.1\%$) would fully resolve any concerns regarding the practical deployment overhead of the closed-loop driver.

---

## 5. Overall Recommendation

**Rating: 5 (Accept)**

**Justification:**
This is an exceptionally strong, technically solid, and highly polished paper. It addresses a critical and real bottleneck in on-device machine learning, bridging the gap between dynamic ensembling SOTA and physical deployment constraints. The mathematical formulations are complete and rigorous; the empirical evaluation across 10 random seeds is highly thorough; and the first-principles hardware modeling (Roofline, Horowitz silicon energy metrics) is outstanding. Crucially, the physical bare-metal results on STM32 using a Joulescope JS110 analyzer provide a robust, physical verification of the simulated savings. The paper is highly cohesive, beautifully written, and immediately actionable. The minor weaknesses identified do not detract from the overall quality and significance of this contribution. I strongly recommend this paper for acceptance!
