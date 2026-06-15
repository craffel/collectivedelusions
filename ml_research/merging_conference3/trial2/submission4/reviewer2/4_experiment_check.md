# Experimental Evaluation and Claims Verification

This check evaluates the rigor of the experimental setup, the choice of baselines, and whether the empirical results actually support the paper's core motivational claims.

---

## 1. The Deployment Dilemma: The Motivation-Workflow Contradiction
The paper's entire motivation rests on a fundamental logical contradiction regarding how and where EdgeMerge is deployed:

1. **The Edge Constraint Motivation:** The authors argue that server-grade adaptive methods like SyMerge (89.74% accuracy) are "deeply impractical" because running test-time backpropagation on edge hardware is blocked by severe memory limits (OOMs) and latency constraints.
2. **The Checkpoint Storage Obstacle:** However, running *any* on-device adaptive merging requires having access to all $K=8$ task-expert checkpoints to perform the forward-pass calibration. Storing 8 independent checkpoints requires **1.2 GB of storage**, which the authors state "immediately triggers out-of-memory (OOM) faults" on edge hardware.
3. **The Offline Developer Workflow Solution:** To resolve this, the authors explicitly propose an **"Offline Developer Workflow"** in Section 3.3:
   > "A developer can run our 11.95-second calibration pass on a local workstation using small validation samples, reconstruct the single merged multi-task checkpoint, and ship it to edge hardware... This workflow completely bypasses the need for on-device checkpoint storage or test-time calibration..."

### The Critical Flaw:
If the calibration and weight reconstruction are performed **offline on a local workstation or staging server**, then the computational, latency, and memory constraints of the edge device **no longer apply**. 

A developer working on a staging server or workstation has unconstrained access to high-end GPUs, gradient tracking, and backpropagation. In this offline setting, there is **no logical reason** to choose EdgeMerge (69.58% accuracy) over SyMerge (89.74% accuracy). 
No sane engineering team would sacrifice **20.16% absolute accuracy** across all multi-task applications simply to save **10 minutes** of one-time offline compilation time on their workstation.

Thus, EdgeMerge is caught in a fatal motivational trap:
- **On-Device (Online) deployment is impossible** because edge devices cannot store or swap the $K$ expert checkpoints required for calibration.
- **Offline deployment makes the method obsolete** because the unconstrained workstation/server environment allows developers to run high-accuracy gradient-based methods like SyMerge, rendering EdgeMerge's 10-second speedup irrelevant.

---

## 2. Weak Baseline Analysis & Insufficient Comparisons
The paper presents several static baselines, but fails to include highly relevant, standard baselines that are standard in model-merging literature:

- **Missing Baseline - Fisher Weighted Averaging:** Fisher-weighted averaging (Matena & Raffel, 2022) is a training-free method that uses diagonal Fisher information matrices (which can be computed in closed-form or estimated from activations) to weight parameter merges. Fisher merging is highly relevant because it operates in the same training-free, low-compute regime as EdgeMerge. Its absence is a significant omission.
- **Inflated Static Baselines:** The authors claim that static alignment methods like Git Re-Basin (41.50%) and ZipIt! (49.30%) perform poorly, using this to argue for their method. However, they compare against an unoptimized static Task Arithmetic baseline (46.62% at $\lambda=0.50$) rather than the optimized one (68.74% at $\lambda=0.20$). When compared against the properly optimized Task Arithmetic baseline, the base EdgeMerge (68.69%) actually *underperforms* static weight averaging by $0.05\%$ absolute points.

---

## 3. Statistical Significance of the +0.84% DSR Improvement
The authors claim that Decoupled EdgeMerge (DSR) under Regime 2 achieves a significant performance breakthrough, reaching **69.58%** average accuracy (an improvement of **+0.84%** over optimized Task Arithmetic at 68.74%).

However, looking at the controls and ablations:
- **Decoupled TA (DTA, control):** This control, which simply uses decoupled scaling ($\lambda_{static}=0.25, \lambda_{proj}=0.20$) but **no adaptive gating**, achieves **69.45%** accuracy.
- **Uniform Gating under DSR:** This ablation achieves **69.58%** accuracy.

This indicates that:
1. The actual margin between doing complex adaptive routing (69.58%) and simple static decoupled scaling (69.45%) is a tiny **0.13% absolute points**. Given that the standard error of the evaluation is approximately $0.51\%$ (as derived by the authors in Section 4.1), a difference of $0.13\%$ is **statistically completely insignificant**. It is well within the margin of noise and cannot support any scientific claim of "active performance boosting."
2. The difference between EdgeMerge and Uniform Gating is **0.00%**. This proves that the activation-based routing contributes absolutely nothing over a flat, static average.

---

## 4. Conclusion on Experimental Validation
The experiments fail to validate the paper's core claims. The dynamic, activation-based routing does not outperform uniform gating, its performance is statistically indistinguishable from a simple decoupled static baseline, and the motivating deployment scenario (edge merging) is logically incoherent. The massive 20% accuracy gap compared to offline methods makes EdgeMerge entirely impractical for any real-world production system.
