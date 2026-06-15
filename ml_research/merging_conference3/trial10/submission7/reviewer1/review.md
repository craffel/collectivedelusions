# Peer Review

## Summary of the Paper
This paper addresses a critical and previously overlooked deployment bottleneck in test-time dynamic model merging and expert ensembling: **state contamination (cross-talk) in multi-tenant workloads**. Standard stateful ensembling methods (e.g., PAC-Kinetics) maintain a single global routing state. While highly effective under continuous, single-user streams, they fail catastrophically when query streams are multi-tenant and heavily interleaved. In such environments, memory states from unrelated tenants bleed into each other, leading to severe routing lag, representational bleeding, and sharp drops in serving accuracy.

To resolve this issue, the paper introduces **Tenant-Decoupled Stateful Routing (TDSR)**, also known as **Slot-Kinetics**. Rather than introducing a massive, over-parameterized neural router or a complex online clustering algorithm, the framework maintains a highly compact pool of virtual routing state slots to isolate temporal smoothing dynamics within respective tenant contexts. The authors propose two practical and elegant modes:
1. **Explicit Session Tagging:** When session/tenant metadata is provided by the serving infrastructure, the router directly indexes and updates the corresponding virtual state slot with zero computational overhead.
2. **Implicit Tagless Clustering:** When metadata is unavailable, the router dynamically infers the session context on-the-fly. By fixing slot centroids as orthogonal task coordinate detector vectors, the online cosine similarity calculation simplifies exactly to choosing the maximum coordinate activation (**coordinate-argmax assignment**). This elegant finding allows the slot assignment to run in sub-nanoseconds.

To prevent state washout in sparse streams, the paper formulates **Tenant-Specific Session-Step Decay** (local decay), where inactive slots hold state constant during global steps. For production deployments, they propose a **Dual-Clock Decay** policy that integrates physical wall-clock timers to evict obsolete sessions and prevent memory leaks.

Evaluations inside the Analytical Coordinate Sandbox (ICS) across 5 independent seeds demonstrate that TDSR Explicit completely resolves state contamination, achieving up to **70.60%** accuracy on Orthogonal Manifolds (outperforming contaminated Global PAC-Kinetics by **+1.90%** absolute) and **70.85%** on Overlapping Manifolds (outperforming Global PAC-Kinetics by **+1.75%** absolute, within 0.50% of the clean-stream Oracle). Furthermore, the paper correctly disentangles inter-session vs. intra-session routing jitter, demonstrating that TDSR Explicit slashes intra-session jitter by up to **2.4$\times$** relative to stateless SABLE.

---

## Strengths and Weaknesses

### Strengths
1. **Elegant and Simple Design:** The core strength of this paper lies in its remarkable design philosophy. Instead of adding uninterpretable, heavy neural network architectures or complex, slow clustering models to handle session tracking, the authors solve a major deployment bottleneck using fundamental systems-engineering and clean mathematical principles.
2. **Brilliant Coordinate-Argmax Simplification:** Exposing that online cosine similarity against fixed orthogonal centroids simplifies to coordinate-argmax assignment ($m^*_t = \arg\max_{m} e_{m, t}$) is a highlight of the paper. It shows exceptional scientific intuition, exploiting the geometric properties of the projection space to achieve a sub-nanosecond, zero-overhead assignment.
3. **Systems-Ready Pragmatism:** The introduction of logical session-step decay to prevent state washout under sparse traffic, combined with a background physical wall-clock timer (Dual-Clock Decay) to evict obsolete slots, is an outstanding design pattern. It shows deep systems awareness, ensuring a micro-scale memory footprint (64 bytes) without complex garbage-collection overhead.
4. **Statistical Jitter Disentanglement:** Exposing that global inter-session jitter must remain high under interleaved streams to track task context-switches, and proposing intra-session jitter to correctly evaluate temporal low-pass filtering stability, is a high-signal contribution that resolves a common misunderstanding in the literature.
5. **Rigorous and Transparent Evaluation:** Commendably, the authors broke the simplifying assumption of tenant-task conflation in their updated evaluation, evaluating on fully randomized streams. The evaluation across 5 seeds, the scaling sweep up to M=256 tenants, and the physical timeout threshold sweep provide thorough, empirical support for all central claims. The paper is also exceptionally transparent about the limitations of the implicit mode.

### Weaknesses
1. **Lack of End-to-End LLM Benchmarks:** The empirical validation is conducted entirely within the high-fidelity Analytical Coordinate Sandbox (ICS). While this sandbox is highly effective for isolating representational dynamics and eliminating confounding generation factors, evaluating on an end-to-end multi-tenant LLM gateway (e.g., S-LoRA with LLaMA models) would elevate the paper's impact even further. (Note: The authors do address this in Section 4.5, explaining the deployment integration, non-linear manifolds, and KV-cache scheduler integration, which heavily mitigates this weakness).
2. **Implicit Mode Limitations under Task Transitions:** In implicit tagless clustering, slots specialize in task domains rather than physical user sessions. Consequently, if a user transitions between tasks, the router cannot smooth states across the task boundary, rendering it stateless at the moment of switch. While the authors are refreshingly transparent about this, further exploration or implementation of one of the proposed mitigations (such as soft slot assignment) would have strengthened the tagless analysis.

---

## Soundness
**Rating: Excellent**

The paper is technically solid and methodologically sound. The mathematical formulations are clean, correct, and highly appropriate for a low-latency routing context. The first-order diagonal recurrence is the simplest and most elegant stateful tracking mechanism, introducing minimal computational overhead. The coordinate-argmax simplification is mathematically proven and robust. Adding a standard load-balancing entropy term successfully resolves the gating collapse issue observed during calibration. The evaluations are rigorous, using 5 random seeds and sweeping both high concurrency scales (up to 256 tenants) and physical decay timeouts.

---

## Presentation
**Rating: Excellent**

The paper is exceptionally well-written, structured, and easy to follow. The overall narrative is compelling, and the progression from identifying the bottleneck to the mathematical formulation, system-level analysis, and empirical results is highly logical. Algorithm 1 is incredibly clear and detailed, ensuring straightforward reproducibility. Figure 1, Figure 2, and the tables are informative, high-signal, and directly support the text. Section 4.5 ("Towards Real-World LLM and PEFT Deployment") is particularly outstanding, bridging the gap between theoretical sandbox simulation and actual systems engineering.

---

## Significance
**Rating: Excellent**

Stateful model ensembling represents a powerful technique for reducing routing jitter and stabilizing representational flow, but it was previously blocked from real-world production due to the state contamination bottleneck on interleaved workloads. By proposing a lightweight, zero-overhead slot decoupling framework, this paper completely unblocks stateful dynamic merging for real-world high-throughput cloud servers and edge gateways. Given the rapid growth of PEFT serving infrastructures (like S-LoRA and Punica), this zero-overhead, highly scalable router has high practical utility and is likely to influence future systems and serving research.

---

## Originality
**Rating: Good**

While virtual slotting and session tracking are standard systems concepts, their formalization and application to dynamic model merging to resolve state contamination is highly original. The mathematical derivation of the coordinate-argmax simplification is a beautiful and elegant contribution. The paper stands out by avoiding the trend of introducing bloated, complex routing networks, demonstrating instead how a simple, mathematically grounded design can achieve superior performance and efficiency.

---

## Overall Recommendation

**Choice: 5: Accept**

The paper is an outstanding, highly pragmatic, and beautifully executed piece of work. It addresses a critical systems-level blocker in stateful dynamic model merging with a solution of remarkable simplicity, elegance, and effectiveness. By leveraging the existing geometric properties of the activation projections, the authors bypass heavy online clustering or deep learning routing layers, achieving sub-nanosecond, register-level slot assignments. The paper is exceptionally clear, highly reproducible, and rigorously evaluated. This work is a shining example of how clean, thoughtful design and mathematical simplification can outperform bloated and over-engineered architectures, and it represents a high-impact contribution to the PEFT serving community. I strongly recommend its acceptance.
