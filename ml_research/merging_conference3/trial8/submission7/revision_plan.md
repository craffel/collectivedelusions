# Revision Plan - Addressing Mock Reviewer Critiques

Following the latest feedback from the Mock Reviewer (overall recommendation: **4 - Weak Accept**), we are systematically addressing the remaining 5 critical suggestions to make the manuscript flawless and elevate its score to a definitive Accept.

## Prioritized Weaknesses & Action Plan

### 1. Disclose Sandbox & Routing-Only Limitations Prominently (Area 1 / Soundness)
*   **Critique:** Table 1 contains simulated results inside our Analytical Coordinate Sandbox (ICS) representing MNIST, Fashion-MNIST, CIFAR-10, and SVHN, but they should be clearly and prominently warned in Table 1's main columns and captions. The ViT-B/16 section is a routing-only simulation on frozen activations and needs explicit clarification.
*   **Action Plan:**
    *   Make sure Table 1 caption, headers, and description explicitly emphasize that MNIST, Fashion-MNIST, CIFAR-10, and SVHN scores are simulated within the ICS sandbox rather than run on image pixels.
    *   Ensure Section 4.5.4 (or 4.5.3 as referenced by the reviewer) prominently and explicitly clarifies that the ViT-B/16 validation is a routing-only simulation on frozen pre-trained activations, and does not involve actual trained or loaded LoRA adapters or physical activation blending (CAB).

### 2. Reframe Narrative Around Routing Stability (Area 2 / Presentation)
*   **Critique:** Tone down any lingering "accuracy breakthrough" claims in the Abstract and Intro, framing the main contribution precisely around solving the accuracy-stability trade-off (reducing layer-to-layer ensembling jitter by up to 9.9$\times$) rather than presenting it as a major accuracy breakthrough.
*   **Action Plan:**
    *   Review `00_abstract.tex` and `01_intro.tex` to ensure any absolute accuracy claims are balanced with standard deviations and explicitly positioned as secondary to the primary victory of ensembling stability and routing jitter reduction.

### 3. Deepen Discussion on Cascading Drift and Temperature Volatility (Area 3 / Methodology)
*   **Critique:** Expand discussion on the tension between routing selectivity (small $\tau = 0.01$) and input rate volatility, and detail why active coupling feedback ($\eta > 0$) is counter-productive in heterogeneous streams due to cascading representational drift.
*   **Action Plan:**
    *   Expand Section 3.5 in `03_method.tex` and Section 4.5.1 in `04_experiments.tex` to explicitly detail the physical dynamics of cascading representational drift and how the ODE's continuous integration low-pass behavior successfully dampens rate input volatility under low reaction temperatures.

### 4. Provide a Concrete Roadmap for Real-World Adapter Ensembling (Area 4 / Significance)
*   **Critique:** Detail a concrete path for evaluating ChemMerge's full ensembling capabilities (specifically CAB) across actual trained adapters on standard multi-task benchmarks like VTAB or GLUE.
*   **Action Plan:**
    *   Elaborate on Section 5.2 (Future Research Horizons) in `05_conclusion.tex` to present a step-by-step roadmap for training, loading, and ensembling task-specific adapters on VTAB and GLUE under streaming serving workloads.

### 5. Acknowledge Hardware serving context (Area 5 / Generalization)
*   **Critique:** Clarify that while CPU-bound NumPy benchmarks show excellent computational efficiency, physical edge accelerator evaluations are needed for definitive serve-time latency.
*   **Action Plan:**
    *   Add an explicit paragraph in Section 4.5.2 (Expert Scaling) of `04_experiments.tex` acknowledging the hardware limitations of CPU-bound NumPy benchmarks and positioning edge accelerator evaluations as the next step.

---

## Status of Execution
*   **Sandbox and Routing-Only Disclosures:** **In Progress.**
*   **Reframing Narrative Around Routing Stability:** **In Progress.**
*   **Cascading Drift and Temperature Volatility Discussion:** **In Progress.**
*   **Real-World Adapter Roadmap:** **In Progress.**
*   **Hardware Accelerator Acknowledgment:** **In Progress.**
