# Comprehensive Peer Review

## Recommendation: 4: Weak Accept
*Reviewer Rating: 4 (Weak Accept) | Soundness: Good | Presentation: Excellent | Significance: Excellent | Originality: Good*

---

### 1. Summary of the Paper
This paper addresses the critical challenge of dynamic, sample-wise multi-task expert model ensembling for Low-Rank Adaptation (LoRA) on resource-constrained edge devices. The authors target two fundamental bottlenecks in the model merging and ensembling literature:
1. **The Routing Paradox and the Early-Feature Loss Trade-Off:** Non-parametric activation-blending methods like SABLE postpone task routing to late layers of the network to avoid redundant base-model forward passes (the "Routing Paradox"). However, this forces them to leave early-to-mid layers (e.g., Blocks 0 to 9) completely unadapted ("Late Adaptation"), discarding crucial early-stage specialized features learned during fine-tuning.
2. **Vectorization Collapse:** Low-data parametric gating networks (e.g., linear classification routers trained on small calibration splits) fail to generalize to individual samples under batch-independent, heterogeneous vectorized streaming ($B=1$), with performance collapsing to that of static uniform merging.

To resolve these limitations, the paper introduces **PEAR (Patch-Embedding Activation Routing)**, a training-free, parameter-free, closed-form ensembling framework that performs sample-wise routing immediately at the base model's frozen Patch Embedding layer (Layer 0) or early transformer blocks (Layer 1 or 2, via the *Early-Layer Routing Compromise*). PEAR operates via a sequence of non-parametric steps:
* **Zero-Shot Patch Centroids (ZPC):** Establishing reference anchor coordinates in the early manifold without any optimization parameters.
* **Cosine Similarity on the Unit Hypersphere (Unit-Norm Projection):** Evaluating cosine similarity on a unit-norm sphere to achieve strict scale invariance.
* **Intra-Task Dispersion Calibration (IDC):** Score normalization based on expected in-distribution calibration variance to standardize scales across diverse, asymmetric task manifolds.
* **Temperature-Scaled Softmax & OOD Rejection:** Softmax normalization with a calibrated temperature ($\tau=0.05$ default) and an Out-of-Distribution (OOD) rejection threshold ($\gamma_{\text{OOD}} = 0.05$), with mathematically sound fallbacks including a **Static Uniform Weight Merging Fallback** ($\alpha = 1/K$) or a **Hard Edge Rejection** fallback ($\alpha = 0$) to protect resource-constrained systems.
* **Dynamic Activation Blending (SPS):** Dynamic, sample-specific scaling of LoRA activations across up to 100% of the network depth with flat $O(1)$ latency complexity.

The method is evaluated in two distinct environments:
1. **12-Layer Vision Transformer Representation Sandbox (PyTorch):** Simulating standard visual tasks (MNIST, Fashion-MNIST, CIFAR-10, SVHN) using an *Overlapping Subspace Layout* (64-dimensional overlap) to capture realistic representation sharing. PEAR achieves a consistent **59.34%** Joint Mean accuracy across all batch sizes, outperforming SABLE SOTA (**55.30%**) by **+4.04%** absolute, and completely eliminating Vectorization Collapse under vectorized streaming ($B=1$), where the Linear Router collapses to **52.36%**. 
2. **Real-World Empirical Validation on actual images:** Evaluated on MNIST, Fashion-MNIST, CIFAR-10, and SVHN using a pre-trained ImageNet $\mathtt{vit\_tiny\_patch16\_224}$ backbone. While pure Layer 0 routing (PEAR L0) is constrained by a "Global-Average-Color Routing Paradox" (57.81% accuracy), shifting the routing boundary slightly deeper to Layer 1 or Layer 2 (the *Early-Layer Routing Compromise*) successfully resolves the paradox, achieving up to **95.31%** real-world routing accuracy, outperforming an explicitly trained pre-backbone Tiny CNN router (**91.02%**) with zero trainable parameters.

---

### 2. Key Strengths
*   **Absolute Conceptual Elegance:** Shifting the routing operation early (Layer 0/1/2) is an extremely elegant and minimalist solution to both the Routing Paradox and the Early-Feature Loss Trade-Off. It completely bypasses the need for complex, stateful multi-pass scheduling systems (like MBH) or freezing early blocks (like SABLE), enabling up to 100% depth adaptability in a single parallel forward pass.
*   **Frugal & Systems-Aware Design:** PEAR relentlessly applies Occam's razor—introducing zero new trainable parameters, requiring flat $O(1)$ sequential latency, and eliminating dynamic activation storage. The inclusion of a "Hard Edge Rejection" fallback demonstrates outstanding, systems-aware foresight for edge serving.
*   **Rigorous Empirical Sandbox Core:** The authors evaluate PEAR under a wide variety of conditions (linear, non-linear GeLU, highly optimized expert regimes, and routing boundary sweeps), providing a comprehensive, Random Seed-averaged (5 seeds) set of tables that establish stable ensembling gains.
*   **Bridging the Sim-to-Real Gap:** The addition of Section 4.3 evaluating actual real-world images on a pre-trained ImageNet ViT backbone is highly impressive. It successfully demonstrates the practical viability of the routing algorithm and empirically resolves the "Global-Average-Color Routing Paradox" using the "Early-Layer Routing Compromise".
*   **Outstanding Presentation & Visuals:** The manuscript is exceptionally well-written, logically structured, and professionally presented. The inclusion of high-quality embedded systems scaling plots (Figure 1) beautifully illustrates the constant-time $O(1)$ serving latency and batch size robustness of PEAR compared to competitors.

---

### 3. Key Weaknesses & Remaining Gaps (3 Critical Critiques)

Despite these significant strengths, a rigorous methodology audit reveals three key nuances and remaining empirical gaps:

#### Critique 1: Gated Routing Accuracy vs. Full Adapter Ensembling on Real Images
*   **The Issue:** In the real-world evaluation (Section 4.3), the authors evaluate the **routing (gating) accuracy** of PEAR (and the Tiny CNN router) on real images (Table 8). However, this experiment does *not* actually fine-tune actual task-specific LoRA adapters on these real datasets and measure the *final merged multi-task model classification accuracy* on real images. 
*   **The Critique:** The actual activation ensembling and final classification performance of the merged LoRA adapters is still evaluated solely within the 1D synthetic vector sandbox (Section 4.1 & 4.2). While validating routing accuracy on real images is a critical prerequisite, the complete ensembling performance (evaluating actual fine-tuned LoRA matrices and measuring output quality on actual images) remains unproven. This is a remaining empirical gap in the evaluation.

#### Critique 2: Clarification of Compute Overhead vs. Sequential Latency Delay in Deeper Routing
*   **The Issue:** The discussion in Section 4.3.3 refers to PEAR L2's 6.26 ms execution as a "20.78% computational overhead relative to a full backbone forward pass ($30.12$ ms)".
*   **The Critique:** This distinction should be clarified in the text. If blocks 0-1 are run as part of the standard forward pass and their activations are cached and re-used for block 2, the actual *FLOPs overhead* (computational overhead) is virtually zero (only the similarity calculation). The 6.26 ms does not represent extra execution time; rather, it represents the *latency elapsed before the routing decision is finalized* (during which execution is sequential) and the portion of the network (2 out of 12 blocks, or 16.7%) that must remain unadapted.

#### Critique 3: Low Baseline Ceiling on SVHN in Standard Sandbox
*   **The Issue:** In the standard sandbox experiments (Tables 1-3), the SVHN task is configured with an extremely high noise scale (noise factor $1.20$), resulting in an expert ceiling of only **19.68%** (nearly random for 10 classes). 
*   **The Critique:** While testing routing under noisy/degraded conditions is valuable, configuring a task to have a ~19.7% ceiling in the main experiments is an odd design choice that represents an unnaturally degraded baseline. The authors should explicitly clarify in the text that this was a specialized stress-test designed to evaluate routing robustness under highly degraded expert conditions.

---

### 4. Actionable Suggestions for Authors

To elevate the manuscript to a strong Accept, the authors are encouraged to address the following actionable suggestions:
1.  **Proof-of-Concept Adapter Ensembling on Real Images:** To address Critique 1, run a small, qualitative proof-of-concept using pre-trained task LoRAs (e.g., from Hugging Face or trained locally) on MNIST and CIFAR-10. This would confirm that the ensembled model actually yields correct final predictions on real images, completing the sim-to-real bridge.
2.  **Terminology Refinement in Section 4.3.3:** Clarify the difference between "sequential latency delay prior to ensembling" and "extra computational FLOPs" for PEAR L1 and L2, making sure the reader understands that caching activations avoids redundant execution.
3.  **Stress-Test Clarification:** Explicitly declare the high-noise SVHN setting in the sandbox as a specialized stress-test to manage reader expectations regarding the low baseline ceiling.
4.  **Minor Presentation Sugesstion:** In Section 3.6, ensure that the definition of "Hard Edge Rejection Fallback" includes a brief mention of the task-agnostic head architecture used to prevent logit nullification, building further systems credibility.

---

### 5. Categorical Ratings
*   **Soundness: Good.** The mathematical derivations are rigorous and the claims are highly supported by the extensive sandbox and real-world routing evaluations. The lack of actual real LoRA evaluations on real images is the only remaining gap preventing an "Excellent" rating.
*   **Presentation: Excellent.** The writing style is highly polished, the math is precise, the tables are complete and seed-averaged, and the embedded systems plots compile cleanly and look highly professional.
*   **Significance: Excellent.** Relocating routing early represents a valuable paradigm shift that early-layer routing is a highly accurate spectrum, offering highly practical training-free blueprints for multi-task edge ensembling.
*   **Originality: Good.** PEAR challenges long-held assumptions about routing boundaries, presenting an original combination of non-parametric steps (ZPC, UNC, IDC) to bypass systems scheduling and late adaptation bottlenecks.
