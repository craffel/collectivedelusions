# Revision Plan - Addressing Peer Review Critiques (Round 2)

## Identified Weaknesses & Action Plan

### 1. Flaw 1: Evaluation Confined Entirely to a Low-Dimensional Synthetic Sandbox (ICS)
- **Critique:** The entire evaluation is conducted within the synthetic 14-layer Analytical Coordinate Sandbox (ICS), which has 192 dimensions and relies on orthogonal PCA coordinate spaces and injected Gaussian noise. It lacks validation on real-world models (e.g., LLaMA, Mistral) and standard downstream PEFT benchmarks (e.g., GSM8k, GLUE, MMLU).
- **Plan:** We will add a dedicated, systems-minded discussion subsection titled **"Towards Real-World LLM and PEFT Deployment: Generalization, Architecture, and Non-Stationary Representation Manifolds"** in the paper (under Section 4). This section will detail:
  - A concrete deployment architecture showing how TDSR is integrated into real-world multi-tenant PEFT serving frameworks (e.g., S-LoRA, Punica, vLLM).
  - A detailed scientific discussion of how real-world LLMs (operating on complex, non-linear representation manifolds with non-Gaussian, non-stationary activation noise) would affect the PCA coordinate projection.
  - Explain how intermediate activation representations can be extracted at transformer layer boundaries (e.g., Layer 4 in LLaMA-3-8B), projected via robust online dynamic PCA, and mapped to virtual routing states.
  - Detail how pinning virtual slot states in fast CPU/GPU registers (or L1 cache) achieves sub-microsecond latency, completely bypassing slow disk or database accesses to maintain high serving throughput.

### 2. Flaw 2: Significant Underperformance of the Implicit Mode on Overlapping Manifolds
- **Critique:** Under Overlapping Manifolds, TDSR Implicit achieves only 69.00% classification accuracy, which represents a severe -3.25% absolute drop compared to the contaminated Global PAC-Kinetics baseline (72.25%). The paper previously glossed over this.
- **Plan:** We will add a detailed, transparent discussion paragraph titled **"The Overlapping Manifold Bottleneck for Implicit Tagless Clustering"** in Section 4. We will explain that in overlapping manifolds, coordinate projections are contaminated by shared dimensions, causing the fixed orthogonal centroids to select incorrect slots frequently, corrupting the decoupled state updates. We will propose concrete methodological solutions:
  - Deploying **Soft Slot Assignment / Gumbel-Softmax Routing** over slots to distribute updates across multiple slots based on similarity.
  - Using **Dynamic Online Centroid Learning** with maximum-entropy regularizers or slot-repelling losses.
  - Provide a practical systems recommendation: under high task overlap, practitioners should prefer the explicit metadata-tagged mode (TDSR Explicit), reserving TDSR Implicit for settings with low task overlap.

### 3. Flaw 3: Statistical Discrepancy in Abstract Claims vs. Main Body Results
- **Critique:** The reviewer identified a discrepancy in the abstract's statistical claims (claiming a +6.25% improvement over the Oracle when it was actually over SABLE).
- **Plan:** Although this was partially corrected in the `.tex` files in the previous round, we will conduct a comprehensive audit of all quantitative claims in `00_abstract.tex`, `01_intro.tex`, `04_experiments.tex`, and `05_conclusion.tex`. We will ensure that:
  - The +2.00% absolute accuracy improvement of TDSR Explicit (70.25%) over Global PAC-Kinetics (68.25%) is correctly attributed to Orthogonal Manifolds.
  - The +6.25% absolute improvement on Overlapping Manifolds is correctly attributed as being over Stateless SABLE (72.75% vs 66.50%).
  - The +7.75% absolute improvement on Overlapping Manifolds is correctly attributed as being over the isolated Oracle baseline (72.75% vs 65.00%).
  - All numbers in the abstract, introduction, body, and conclusion are mathematically and scientifically aligned with Tables 1 and 2.
