# 4_experiment_check.md - Detailed Experimental Rigor and Validation Check

## Experimental Strengths
1. **Multi-Scale Real-World Validation:** Testing on multiple pre-trained BERT backbones (\texttt{bert-tiny}, \texttt{bert-mini}, \texttt{bert-medium}, and \texttt{bert-base-uncased}) is a commendable effort to scale the simulated results to real-world models.
2. **Replication Over Multiple Seeds:** Averaging results over 5 independent random seeds (and reporting mean $\pm$ standard deviation) is standard and correct.
3. **Thorough Ablation Studies:** The simulated experiments inside the Analytical Coordinate Sandbox (ICS) include detailed ablation studies (Ablation 1--4) covering subspace dimension ($d$), calibration split size ($N_{\text{cal}}$), prior temperature ($\tau_0$), and representation interference scale ($\eta$).

## Critical Experimental Flaws and Weaknesses

### 1. The "Worse-than-Uniform" Reality on Real-World Backbones
The most critical empirical weakness of the paper is that on the larger, more realistic BERT backbones, **the proposed Dirichlet-PAC and PEM-Div frameworks are outperformed by the simple, static Uniform Merging and SABLE (SEP-Block) Norm baselines:**
* **BERT-Medium:**
  * `Uniform Merging`: **96.00% ± 3.89%**
  * `SABLE (SEP-Block) Norm`: **96.00% ± 3.89%**
  * `Dirichlet-PAC (Ours)`: **89.33% ± 3.89%** (Lagging by **6.67%**)
  * `PEM-Div (Ours)`: **88.67% ± 3.40%** (Lagging by **7.33%**)
* **BERT-Base:**
  * `Uniform Merging`: **95.33% ± 4.52%**
  * `SABLE (SEP-Block) Norm`: **94.67% ± 3.40%**
  * `Dirichlet-PAC (Ours)`: **92.00% ± 8.59%** (Lagging by **3.33%**)
  * `PEM-Div (Ours)`: **91.33% ± 7.18%** (Lagging by **4.00%**)

**Why this is a major issue:** 
The authors claim in the abstract and intro that Dirichlet-PAC achieves "outstanding serving performance" and "improves average accuracy over standard unregularized ERM". However, they gloss over the fact that in real-world settings, **it fails to beat a simple static parameter average (Uniform Merging) or a static uncalibrated soft router (SABLE Norm).** 
Uniform Merging requires *zero* data splitting, *zero* training/calibration optimization, and *zero* runtime coordinate extraction. In contrast, Dirichlet-PAC requires partitioning the precious calibration split, extracting SVD bases, and running 100 epochs of Adam optimization—only to lose by up to 6.67% in accuracy. This seriously undermines the practical significance and utility of the proposed framework.

### 2. Discrepancy Between Simulated Sandbox (ICS) and Real-World Experiments
There is a massive, unaddressed contradiction in the empirical results:
* **In the Simulated Sandbox (ICS):** Uniform Merging is a terrible baseline, achieving only **46.26%** accuracy, while Dirichlet-PAC is highly superior at **77.88%** (Table 2).
* **In the Real-World BERT Experiments:** Uniform Merging is an extremely dominant baseline, achieving **94.00% to 99.33%** accuracy, outperforming Dirichlet-PAC on larger models (Table 3).

**What this discrepancy reveals:**
The 14-layer Analytical Coordinate Sandbox (ICS) simulation is **highly artificial and heavily biased against weight-space/static merging.** The simulation injects severe "representation interference" ($\eta = 0.05$) and task manifold entanglement that forces static merging to collapse. However, in physical, pre-trained transformer networks fine-tuned with LoRA, the adapters are highly compatible, and their simple linear average or uniform blend is incredibly robust and high-performing. By evaluating primarily on a custom sandbox that exaggerates representation clashing, the paper presents a highly distorted view of the necessity and performance advantages of dynamic routing.

### 3. Evaluation on Toy, Synthetic Datasets
The real-world BERT evaluation is performed on **highly simplistic, toy synthetic datasets** (sentiment classification with keywords, professional sports vs. science words, statements vs. wh-questions). 
* This is why the "Expert Ceiling" is 99% to 100% and Uniform Merging achieves 99.33% on BERT-Mini.
* These tasks do not represent realistic, challenging multi-task serving workloads (e.g., GLUE, SuperGLUE, translation, reasoning). On such simple tasks, the representations are so easily separable that dynamic routing is almost trivial, yet the proposed method still underperforms uniform blending due to overfitting and routing errors on the scarce calibration data.
* Evaluating on standard, established multi-task benchmarks (such as GLUE tasks with LoRA adapters) is essential to validate the method's real-world viability.
