# Systematic Critique - Step 4: Empirical Evaluation and Integrity Check

## 1. Major Internal Numerical Inconsistencies
The manuscript contains a major internal discrepancy regarding its key performance numbers. 
* **The Discrepancy:**
  * The **Abstract, Introduction, and Figure 1 Caption** claim that HyperMerge achieves a joint mean accuracy of **89.30%** under both stream configurations, while SABLE is reported as **89.65%** and SPS-ZCA as **88.55%**.
  * However, **Table 1 and Section 4.4 text** (which report the actual multi-seed results over 3 random seeds) state that HyperMerge achieves **83.40% $\pm$ 5.15%**, SABLE (Early Routing) achieves **84.03% $\pm$ 5.15%**, and SPS-ZCA achieves **83.05% $\pm$ 4.95%**.
* **Impact on Integrity:** The authors completely failed to align the text in the abstract and introduction with the multi-seed statistical results in the main tables. This indicates extremely poor editing care and raises red flags regarding which results are indeed correct and which are from single uncalibrated runs or outdated iterations.

## 2. Core Hypothesis Refuted by Overlapping Subspace Results
The main motivation of HyperMerge is that flat-space methods suffer from "representation crowding" and "destructive inter-task cross-talk," and that negatively curved hyperbolic space resolves these issues. However, the results in the highly crowded **Overlapping Subspace Sandbox (Table 2 and Section 4.5 text)** directly contradict this:
* **The Results:** SABLE (Early Routing) achieves **77.98% $\pm$ 2.12%**, SPS-ZCA achieves **77.32% $\pm$ 1.98%**, while HyperMerge ($c=0.1$) achieves only **76.62% $\pm$ 3.96%** (improving slightly to **76.50% $\pm$ 3.36%** when tuned to $c=0.2, \tau=0.08$).
* **The Deficit:** Under the exact crowded regime designed to prove the superiority of negative curvature, flat Euclidean ensembling methods (SABLE, SPS-ZCA) still outperform HyperMerge.
* **Impact on Motivation:** This empirical finding completely refutes (or at least fails to support) the paper's core hypothesis. If flat-space methods still outperform the hyperbolic method under heavy crowding, then the practical utility of the high mathematical complexity of hyperbolic geometry in model merging is completely unproven. The authors' claim that HyperMerge "separates task manifolds" and "segregates cross-talk" does not translate into any empirical benefit over the simpler flat-space baselines.

## 3. Discrepancies in Overlapping Sandbox Reporting
There is a further numerical mismatch between the main text's multi-seed results for the Overlapping Sandbox and other project logs (e.g., `experiment_results.md` and `progress.md`):
* **The Mismatch:** 
  * Table 2 and Section 4.5 text report: SABLE at **77.98% $\pm$ 2.12%**, SPS-ZCA at **77.32% $\pm$ 1.98%**, and HyperMerge at **76.62%** / **76.50%**.
  * However, `experiment_results.md` and `progress.md` report: SABLE at **75.35%**, SPS-ZCA at **74.95%**, and HyperMerge at **71.20%** / **72.15%**.
* **Impact on Reproducibility:** While the files explain that these are raw, single-seed or uncalibrated scores, having multiple divergent sets of numbers across project files and manuscript sections makes verification difficult and compromises the reproducibility of the evaluation.

## 4. Highly Trivial, Synthetic "Analytical Coordinate Sandbox"
The empirical evaluation remains severely limited by its reliance on a highly artificial, synthetic simulator:
* The "Analytical Coordinate Sandbox" is a 192-dimensional vector space where tasks are manually partitioned into non-overlapping, orthogonal 48-dimensional coordinate subspaces (e.g., MNIST occupies dimensions 0–48, F-MNIST occupies dimensions 48–96, etc.).
* Because the task representations are placed in orthogonal subspaces, there is virtually no real representation crowding or complex cross-talk near the coordinate origin in the baseline setup.
* The "models" are 14 layers of identity matrices with tiny random perturbations.

Because this sandbox is extremely simplified and orthogonal, it fails to simulate the complex, non-linear, and heavily overlapping representation manifolds of actual deep neural network backbones (such as ViTs or LLMs) trained on real-world datasets. Consequently, the empirical results on this sandbox do not provide sufficient evidence to support the claim that negative curvature is necessary or beneficial for resolving representation crowding in real deep models.

## 5. Total Absence of Real-World Evaluation
To demonstrate the actual utility and scalability of HyperMerge, evaluations on physical, pre-trained neural networks are completely missing. A true validation of the method requires merging actual trained task experts (e.g., LoRA adapters) on standard backbones (such as ViT-B/16, CLIP, or RoBERTa) on real multi-task image/text benchmarks. Relying exclusively on a 14-layer identity simulator limits the significance of the empirical claims and prevents publication in top-tier machine learning conferences (ICML, NeurIPS, ICLR).
