# Mock Review: Lyapunov-Stable Active Representation Coupling (L-ARC)

## Overall Recommendation
**Recommendation:** **5: Accept**  
*Justification:* The paper is an exceptionally high-quality, mathematically rigorous, and scientifically transparent contribution that bridges classical control theory with deep neural network serving. By framing active representation warping as a non-linear closed-loop control problem, the authors replace prior ad-hoc, fragile heuristics with a formally guaranteed stability framework. The mathematical derivations are mathematically sound, and the empirical evaluations under various system-level failures (network dropouts, systematic router bias) are exemplary in their rigor and honesty. While the quantitative benchmarks are primarily conducted within a high-fidelity Analytical Coordinate Sandbox, the authors' real-world pilot study on LLaMA-3-8B successfully bridges the gap to practical deployment. 

---

## Detailed Ratings

*   **Soundness:** **Excellent**  
    The mathematical formulation is exceptionally robust. The authors prove that their candidate Lyapunov function is unconditionally positive semi-definite, provide a formal error bound on the Layer-Identity Approximation (Theorems 3.2), and derive a rigorous Lagrange remainder bound to validate the Taylor linearization at finite feedback step sizes (Theorem 3.5). The theoretical integration between ECG-Reset and Lyapunov stability is elegant, and the dual-loop RASC mechanism is a highly sound, classic solution to circular feedback dependencies.
*   **Presentation:** **Excellent**  
    The paper is cohesive, well-structured, and easy to follow. The narrative moves logically from physical kinetics (NEKR) to Lyapunov-stable active coupling, followed by RASC and scaling designs (MNR and H-RASC). Crucially, the scientific transparency is incredibly refreshing; the authors openly discuss when their controller is redundant ($p = 0.0969$ under clean workloads), when active feedback is statistically marginal ($p = 0.3443$ under failures), and the trade-off of kinetics propagation lag.
*   **Significance:** **Excellent**  
    The practical utility of L-ARC is substantial for edge-device dynamic serving. By reducing absolute routing latency overhead to just $0.06$ ms per sample using Entropy-Triggered Gating (ET-L-ARC), the paper presents a highly practical serving solution. The data efficiency (requiring as few as 8 calibration samples per task) and scalability designs (MNR for deep networks and H-RASC for dense pools of adapters) make L-ARC an invaluable framework for deep learning systems engineers and control theorists alike.
*   **Originality:** **Excellent**  
    Rather than introducing another empirical deep learning heuristic, the paper introduces a fundamentally novel control-theoretic paradigm for stabilizing dynamic model ensembling. The combination of Lyapunov stability, state-space shielding, and dual-loop state correction represents a significant conceptual and technical leap over stateless (SABLE) and open-loop stateful (ChemMerge) ensembling.

---

## Strengths and Weaknesses

### Core Strengths
1.  **High Theoretical Rigor:** The paper features comprehensive mathematical proofs (Theorems 3.1–3.5) establishing stable representation propagation and linearization error bounds.
2.  **Robustness to System-Level Faults:** L-ARC is uniquely resilient to transient failures (Setting C) and persistent routing bias (Setting D), outperforming SABLE and Decay-ChemMerge by over **5.14%** and **5.32%** absolute accuracy, respectively.
3.  **Exceptional Scientific Transparency:** The authors clearly identify and profile all trade-offs, including the kinetics propagation lag in early layers and the non-significance of feedback warping under clean workloads.
4.  **Real-World Pilot Verification:** The small-scale pilot study on LLaMA-3-8B confirms high-dimensional coordinate orthogonality and Dissipation Guard stability, confirming the transferability of the theoretical model.

### Key Weaknesses / Areas of Improvement
1.  **Reliance on a Stylized Sandbox for Main Benchmarks:** Although the LLaMA-3-8B pilot study validates the geometric and control assumptions, the main quantitative results are evaluated inside the 14-layer Coordinate Sandbox (ICS). Full end-to-end task-accuracy evaluations on standard NLP or Vision benchmarks (e.g., GLUE, MMLU, GSM8K) would strengthen the empirical claims.
2.  **Kinetics Propagation Lag under Ideal Settings:** Under ideal layer-specific centroids (Setting B), the spatial inertia of the ODE kinetics introduces a propagation lag, causing L-ARC to underperform the instantly responsive SABLE by approximately $0.36\%$. While the authors discuss this trade-off excellently, it remains a performance constraint in clean settings.
3.  **Boundary Limitations of the MHSA-Only Restriction:** To preserve the Layer-Identity bound, representation warping is restricted to the input of Multi-Head Self-Attention (MHSA) blocks, completely bypassing highly non-linear FFN/MLP blocks. While this is mathematically justified, it prevents stabilizing ensembling representations inside MLP layers, which often specialize in factual task knowledge.

---

## Actionable Feedback & Suggestions for the Authors

To elevate this paper to its final, camera-ready form, the authors are encouraged to address the following minor suggestions:

1.  **Expand the Real-World Pilot into a Full-Scale Benchmark Evaluation:**  
    While the current LLaMA-3-8B pilot study on GSM8K, AG-News, and SST-2 is an excellent addition, the authors should report the final task-specific accuracies (e.g., SST-2 sentiment accuracy, AG-News accuracy, GSM8K math accuracy) rather than just perplexity, coordinate orthogonality, and gating rates. Providing these numbers would fully silence any skepticism regarding the sandbox-to-LLM performance gap.
2.  **Formally Discuss the Impact of Non-Linear Transformer Blocks on the Dissipation Bound:**  
    In Section 3.4 (Theorem 3.5), the Lagrange remainder bound assumes a linear projection subspace. The authors should include a brief discussion or appendix section analyzing how non-linear MLP blocks (e.g., SwiGLU or GeLU activations) scale the remainder error if feedback warping were to be extended beyond MHSA layers. This would provide a solid mathematical foundation for future works trying to expand closed-loop feedback across the entire transformer block.
3.  **Incorporate an Analysis of Online Centroid Adaptation:**  
    L-ARC currently relies on offline calibration to extract static centroids. In a real-world edge serving environment, task distributions can shift over time (non-stationary streams). The authors should discuss how the Dissipation Guard would behave under online, running centroid estimates (e.g., using exponential moving averages of activations) and whether the Lyapunov stability proof holds when centroids themselves are time-varying states.
