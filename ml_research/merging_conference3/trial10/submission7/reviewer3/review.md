# Peer Review

## 1. Summary of the Paper
The paper addresses a critical deployment barrier in test-time dynamic model merging (expert ensembling) for Parameter-Efficient Fine-Tuning (PEFT) adapters. Existing stateful and kinetic routers (like ChemMerge and PAC-Kinetics) use temporal low-pass filtering (first-order recurrences) to smooth ensembling weight trajectories and suppress high-frequency routing jitter. However, they assume a continuous, highly correlated single-user stream. In high-throughput cloud environments (such as LLM gateways or edge-compute nodes), workloads are multi-tenant and heavily interleaved across independent users. Under these interleaved workloads, standard stateful routers suffer from **state contamination (cross-talk)**, bleeding memory states across unrelated tenants, which leads to routing lag, incorrect adapter activation, representational bleeding, and severe accuracy drops (up to 9.0% absolute classification accuracy drop).

To resolve this deployment blocker, the paper introduces **Tenant-Decoupled Stateful Routing (TDSR)**, also known as **Slot-Kinetics**. TDSR maintains a highly compact pool of virtual routing state slots (microscale footprint of 64 bytes for 4 slots, 4 tasks) and supports two highly practical routing modes:
1. **Explicit Session Tagging:** A metadata-aware scheduler (like S-LoRA or Punica) supplies tenant session IDs, allowing the router to update the corresponding state slot directly with zero computational overhead.
2. **Implicit Tagless Clustering:** When metadata is unavailable, the router dynamically infers the session context on-the-fly. By computing online cosine similarity between the query coordinate vector and fixed orthogonal coordinate detector centroids ($\mathbf{c}_{m, m} = 1.0$), the assignment mathematically simplifies to a simple **coordinate-argmax assignment** (selecting the slot corresponding to the task with the maximum activation coordinate), which can be executed as a single, sub-nanosecond integer index lookup.

To prevent memory leaks and state washout, the paper details **Tenant-Specific Session-Step Decay** (holding inactive slots constant during interleaved steps) and a **Dual-Clock Decay** policy (evicting inactive slots after a specified physical wall-clock timeout).

The framework is evaluated inside a 14-layer synthetic environment called the **Analytical Coordinate Sandbox (ICS)**. Under interleaved multi-tenant streams, TDSR Explicit completely resolves state contamination, achieving up to **70.60%** classification accuracy on Orthogonal Manifolds (outperforming contaminated Global PAC-Kinetics by **+1.90%** absolute) and **70.85%** on Overlapping Manifolds (outperforming contaminated Global PAC-Kinetics by **+1.75%** absolute, within 0.50% of the clean-stream ceiling). TDSR Explicit also slashes intra-session routing jitter by up to **2.4$\times$** relative to stateless SABLE, recovering near-Oracle routing stability. Test-time scaling sweeps up to $M=256$ concurrent tenants and Dual-Clock timeout threshold sweeps validate the scalability and memory reclamation viability of the framework with negligible latency overhead.

---

## 2. Strengths and Weaknesses

### Strengths
- **Highly Practical Motivation:** The paper identifies and addresses a major deployment blocker (cross-talk and state contamination under interleaved streams) that has been neglected in academic literature but is a critical obstacle for hosting stateful ensembling routers in production.
- **Outstanding Computational & Memory Efficiency:** The Slot-Kinetics state updates require a microscopic memory footprint (e.g., $M \times K$ floats) and run in less than 1.5 microseconds, permitting in-register or fast L1 cache execution. This avoids database, disk, or network lookups.
- **Elegant Systems-Focused Optimizations:** 
  - Simplifying online cosine similarity against orthogonal centroids to a simple coordinate-argmax assignment is highly elegant, converting a floating-point bottleneck into a sub-nanosecond integer index lookup.
  - The **Virtual Task Caching** pattern (decoupling slot memory from user concurrency by specializing slots in task domains in implicit mode) is an exceptional systems design that scales to an arbitrary number of tenants with a tiny, fixed footprint of $K$ slots.
  - The **Dual-Clock Decay** policy provides a robust, self-cleaning memory reclamation mechanism that naturally purges obsolete states during idle periods to prevent memory leaks in production servers.
- **Rigorous Scaling & Memory Sweeps:** Conducting test-time scaling sweeps up to $M=256$ concurrent tenants and sweeping the physical eviction timeout threshold demonstrates robust empirical scaling and validates the safety of the memory reclamation policy.
- **Methodological and Statistical Rigour:** Evaluates all dynamic routing models across 5 independent seeds with standard deviations, ensuring findings are statistically sound. The distinction between inter-session and intra-session jitter is an important conceptual contribution that corrects how routing stability is analyzed under interleaved workloads.
- **High-Quality Presentation:** The writing is exceptionally clear, precise, and professional. The notation is consistent, and the pseudocode in Algorithm 1 is highly detailed and accessible.

### Weaknesses
- **Lack of Physical Hardware and Real-World Model Validation:** The paper's biggest weakness is that the entire empirical evaluation is conducted within a synthetic, 14-layer Analytical Coordinate Sandbox (ICS). There is no validation on a **real-world LLM** (e.g., LLaMA-3, Mistral) or **Vision model** (e.g., ViT) with **real PEFT/LoRA adapters** (using S-LoRA or Punica) running on physical GPU/TPU hardware on standard downstream benchmarks (e.g., GSM8K, CodeXGLUE, CIFAR, MMLU). This makes it difficult to verify if the model-side activations and coordinate projections generalize seamlessly to non-linear representation manifolds under non-Gaussian and non-stationary feature noise.
- **Task-Transition Stateful Tracking Failure in Implicit Mode:** Under the implicit tagless mode, because slots specialize in task domains rather than physical tenants, a single user's sequence that transitions across task boundaries is split across separate slots. As a result, the router has zero temporal memory across the task change, rendering it effectively stateless during transitions. This is a fundamental conceptual limitation of the implicit mode that is transparently exposed by the authors but remains unresolved.
- **Unimplemented Overlapping Manifold Mitigations:** Under overlapping manifolds (Table 2), TDSR Implicit trails TDSR Explicit by a minor 0.70% drop due to coordinate-space interference. While the authors propose highly promising future directions (such as Soft Slot Assignment, Online Centroid Learning, and Slot-Repelling Losses) to resolve this, none of them are empirically implemented or evaluated.
- **Missing Code Repository:** The paper does not provide a public link to an open-source repository containing the sandbox (ICS) code, which slightly limits reproducibility.

---

## 3. Ratings and Justifications

### Soundness: Good
The mathematical derivations, stateful recurrence formulations, and proposed algorithms are correct, well-designed, and logically sound. The use of load-balancing regularizers to prevent gating collapse and fixed orthogonal centroids to prevent clustering collapse are highly appropriate. However, because the evaluation is limited entirely to a synthetic, 14-layer sandbox rather than physical models with real-world workloads, the empirical soundness and generalization of the method under non-stationary and non-Gaussian representation noise remain unverified on real hardware.

### Presentation: Excellent
The paper is exceptionally well-written, logically organized, and highly transparent. Every step of the pipeline is mathematically formulated, and Algorithm 1 provides complete pseudocode that is highly accessible to systems developers. The tables are clearly laid out, and the trajectory plots are highly informative. The authors are also highly commendable for their honesty in exposing the limitations of the implicit mode during task transitions and under overlapping manifolds.

### Significance: Good
The paper addresses a highly important and relevant problem in parameter-efficient model serving. If validated on real-world models, TDSR could become a standard component of high-throughput PEFT ensembling gateways (like S-LoRA, Punica, vLLM) to enable highly stable, stateful dynamic expert ensembling. However, because the current significance is held back by the lack of physical model and physical hardware evaluation, it is rated "Good" rather than "Excellent" for now.

### Originality: Good
The combination of orthogonal coordinate assignment, virtual task caching, tenant-specific session-step decay, and dual-clock decay represents a highly novel, systems-focused architectural framework. The authors clearly distinguish their contribution from prior stateful routers (which maintain a single global state) and multi-tenant serving infrastructures (which are agnostic to temporal dynamics). The mathematical simplification of cosine similarity to coordinate-argmax lookup is a particularly original and elegant contribution.

---

## 4. Overall Recommendation: 4 (Weak Accept)
The paper is technically solid, exceptionally well-written, and introduces highly clever, systems-focused mechanisms (Virtual Task Caching, Dual-Clock Decay, coordinate-argmax lookup) to resolve a major deployment blocker in stateful dynamic model merging. The findings are statistically verified across 5 independent seeds, and the scaling sweeps are highly promising from a systems engineering perspective. 

The primary weakness that limits its immediate impact is that the evaluation is conducted *entirely* within a synthetic, 14-layer Analytical Coordinate Sandbox (ICS) without any physical-hardware or real-world LLM/PEFT framework validation. Proving that these representation dynamics generalize to real-world manifolds under non-Gaussian and non-stationary feature noise on physical hardware is necessary to fully confirm deployment readiness. Nonetheless, the paper makes a highly valuable, well-constructed contribution that others are likely to build on, making it a strong candidate for a Weak Accept.

---

## 5. Actionable Feedback and Questions for the Authors

1. **Physical Model Validation:** To demonstrate the real-world deployment viability of TDSR, do you have plans to evaluate this framework on a physical open-source LLM (e.g., LLaMA-3-8B or Mistral-7B) with a fleet of specialized LoRA adapters (e.g., math, coding, translation) on standard downstream benchmarks? Evaluating the actual generation quality and measuring the physical latency impact on a GPU-enabled serving gateway would significantly strengthen the paper's significance.
2. **Task-Transition Performance:** Under the implicit tagless mode, how does the router perform when a single user session frequently transitions across different task domains? Since these queries are routed to separate task-specialized slots, the router loses temporal memory at the moment of transition. Have you quantitatively measured the classification accuracy drop during these task transitions compared to TDSR Explicit, which preserves sequence-level history?
3. **Evaluating Proposed Overlapping Manifold Mitigations:** In Section 4.4, you discuss three highly promising future directions (Soft Slot Assignment / Gumbel-Softmax Routing, Online Centroid Learning, and Slot-Repelling Losses) to resolve the coordinate-interference drop in implicit mode under overlapping manifolds. Have you conducted any preliminary simulations or experiments with these techniques, and if so, what were the trade-offs (e.g., did soft assignment introduce state contamination across slots)?
4. **Calibration Data Sensitivity:** The diagonal state retention matrix $\mathbf{A}$, the injection coupling matrix $W$, and log-temperatures $\mathbf{w}$ are trained on a joint calibration stream of $N_{\text{cal}} = 100$ samples. How sensitive are these learned parameters to the diversity or ordering of the calibration stream? If the calibration stream lacks representation of specific tasks, does it lead to gating bias or gating collapse at scale?
5. **Open-Source Code:** Do you plan to open-source the Analytical Coordinate Sandbox (ICS) codebase alongside the paper? Making the simulation framework publicly available would greatly benefit reproducibility and allow the community to benchmark other routing architectures.
