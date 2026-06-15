# Review Step 4: Experimental Evaluation and Results Check

## 1. Assessment of Experimental Design
The experimental evaluation in this paper is exceptionally rigorous, comprehensive, and well-designed. The authors evaluate their framework across multiple independent dimensions, ensuring that their theoretical claims are empirically validated under controlled settings, physical networks, and real-world image datasets.

## 2. Key Strengths of the Experimental Evaluation

* **Multi-Axial and Multi-Scale Environments:**
  The paper does not rely on a single benchmark. Instead, it validates SpectralMerge across three distinct scales of execution:
  * **Calibrated Simulation Landscape (Model II):** A $12$-layer continuous Vision Transformer (ViT-B/32) simulation landscape calibrated on multi-task sensitivity statistics across MNIST, FashionMNIST, CIFAR-10, and SVHN, evaluated over 30 random seeds (seeds $42$ to $71$). This allows for statistically rigorous sweeps under controlled conditions.
  * **Physical PyTorch Multi-Layer Perceptron (MLP):** A physical 12-layer MLP with alternating projection-like and feedforward-like layers on a synthetic 3-task setup. This tests the block-wise / layer-type spectral decomposition.
  * **Real-World ResNet-18 on CIFAR-10:** An 18-layer pre-trained ResNet-18 model (including all convs, downsampling, BN, and classification head) fine-tuned on a 2-task split of CIFAR-10. This evaluates physical model-merging on actual parameters and real images.

* **Diverse Baselines:**
  The authors compare SpectralMerge against a wide array of relevant baselines:
  * Static baselines: Uniform (Task Arithmetic).
  * Parameterized online test-time adaptation (TTA) baselines: Online AdaMerging and Online RegCalMerge.
  * Parameterized continuous trajectory baselines: Online PolyMerge ($d=2$).
  * Offline few-shot validation tuning (OFS-Tune) baselines: OFS-Tune Layer-wise (Unconstrained) and OFS-Tune Poly-Val ($d=2$).

* **Thorough Stress-Testing (Adversarial Stream Conditions):**
  The authors evaluate online test-time adaptation under three severe stream corruptions: Extreme Label Shift, Bursty Task Streams, and Small Batch Size Noise. This is crucial because online methods like AdaMerging collapse catastrophically in these scenarios (e.g., AdaMerging drops from $79.15\%$ to $62.30\%$ under label shift). The evaluation shows that SpectralMerge-LP and SpectralMerge-Reg maintain high accuracy and that the offline OFS-Tune SpectralMerge-LP remains completely immune.

* **Rigorous Sample Complexity Analysis:**
  The paper addresses the Overfitting-Optimizer Paradox directly by sweeping the validation sample size $M \in \{5, 10, 20, 50\}$. It shows that while unconstrained Layer-wise search degrades to $82.77\%$ at $M=5$ (worse than the unoptimized Uniform baseline of $84.44\%$), SpectralMerge-Reg and SpectralMerge-LP achieve $86.20\%$ and $86.02\%$ accuracy, respectively, resolving the paradox.

* **Resilience to Validation Selection Bias and Domain Shift:**
  The paper sweeps Isotropic and Structured validation selection bias from 0.0 to 0.3. This simulates real-world conditions where validation sets do not match the target deployment distribution. Unconstrained layer-wise optimization degrades catastrophically as bias increases, while SpectralMerge exhibits graceful degradation and maintains high accuracy ($>85.2\%$).

* **Definitive Real-World ResNet-18 Blowout:**
  The pre-trained ResNet-18 CIFAR-10 experiments provide a dramatic and compelling validation of the proposed method. Under extreme data scarcity ($M=15$), both unconstrained spatial search and PolyMerge overfit catastrophically and collapse to $29.00\%$ accuracy (representing a majority-class collapse and an Expected Calibration Error ECE of $0.71$). In contrast, SpectralMerge-Reg ($\mu=1.0$) prevents this collapse, achieving $54.00\%$ multi-task test accuracy and a beautifully calibrated Expected Calibration Error of $0.18$. This is an absolute blowout of $+25\%$ over the baselines and $+13\%$ over the Uniform baseline.

* **Extensive Ablation and Sensitivity Analysis:**
  The paper provides deep ablation studies on:
  * **Hyperparameter Sensitivity:** Sweeping cutoff frequency $F \in \{1, \dots, 12\}$ and regularization strength $\mu \in [10^{-3}, 10^2]$, demonstrating clear, smooth, convex-like trajectories peaking at $F=3$ and $\mu=1.0$.
  * **Symmetric vs. Odd Boundary Extensions:** Confirming that DCT-II outperforms DST by $4.5\%$ in final accuracy.
  * **Block-wise Partitioning:** Showing that Block-wise SpectralMerge-LP outperforms Global SpectralMerge-LP by $+2.92\%$ accuracy on the heterogeneous MLP.
  * **Adaptive Bandwidth (LP-Adaptive):** Showing that dynamically expanding active bandwidth as optimization progresses improves performance over fixed cutoffs on the MLP (achieving $55.00\%$).
  * **Optimization Convergence Trajectories:** Showing that SpectralMerge-LP converges rapidly and smoothly at deep layers ($L \in \{48, 96\}$) due to perfect conditioning, while PolyMerge stalls or fluctuates due to ill-conditioning.

* **Methodological Justification of the Multi-Scale Spectral Framework (Global Task-Wise/DC-Only Baseline):**
  We highly commend the authors for including the fundamental **Global Task-Wise (DC-Only)** baseline in their evaluations. In this baseline, a single merging coefficient $\alpha_k$ is optimized per task and shared globally across all $L$ layers of the network (equivalent to optimizing only the zero-frequency spectral coordinate $c_{k,0}$ and setting all AC components $j > 0$ to zero).
  
  By comparing SpectralMerge against this baseline, the authors successfully isolate and prove the empirical benefit of allowing layer-wise variations (AC frequency components). Specifically, OFS-Tune SpectralMerge-LP ($F=3$) achieves $86.46\%$ accuracy, outperforming the Global Task-Wise (DC-Only) baseline ($85.42\%$) by a non-trivial and statistically significant $+1.04\%$ absolute improvement. This comparison mathematically and empirically justifies the multi-scale spectral framework, confirming that optimizing low-frequency AC coordinates captures critical localized layer-specific sensitivities that are vital for maximizing multi-task consolidation, while unconstrained spatial search overfits validation noise and collapses to $83.81\%$ accuracy.

## 3. Constructive Suggestions for Further Improvement (Minor Weaknesses)

While the experiments are outstanding, a rigorous reviewer must highlight two areas for further expansion and exploration:

* **Scaling to Larger Multi-Task Configurations ($K \ge 8$):**
  The empirical validation is thorough but limited to configurations with a small number of consolidated tasks: $K=4$ in the ViT simulation, $K=3$ in the physical MLP, and $K=2$ in the real ResNet-18. 
  
  In modern practical applications, model-merging algorithms are frequently deployed to consolidate larger pools of task-specific expert models (e.g., merging 8 to 12 task-specific LLMs or vision adapters). As the number of tasks $K$ scales, task interference and weight representation clashes grow exponentially. 
  
  It is crucial to understand if the spectral trajectory dynamics remain consistent, or if a larger task pool requires a larger spectral bandwidth $F$ to resolve high-dimensional multi-task conflicts. Discussing or evaluating SpectralMerge on a larger pool (e.g., $K \ge 8$ tasks) would greatly enhance the practical significance and impact of the empirical validation. Specifically, the authors' proposal of a **2D DCT-II** to map joint depth-and-task coordinates into 2D spectral coordinates is a highly promising direction that could be highlighted further.

* **Evaluation on Massive Pre-trained Foundation Models (LLMs/ViTs):**
  The physical experiments are evaluated on an MLP and a pre-trained ResNet-18 model fine-tuned on CIFAR-10. While these are highly appropriate and mathematically rigorous steps to verify physical weight-space and backpropagation validity, they do not fully capture the raw parameters of massive pre-trained foundation models like Llama-3 or ViT-B/16.
  
  To accelerate widespread adoption, the authors should discuss the practical guidelines for applying SpectralMerge to massive pre-trained foundation models. For instance, explaining how extracting layer-wise task vectors, applying the 1D DCT-II to partition homogeneous attention/feedforward projections, and using standard zero-shot datasets to optimize the low-frequency spectral coefficients would serve as a highly scalable alternative for massive-scale pre-trained model consolidation in practice.

## 4. Rating of Experimental Evaluation
**Excellent.** The experimental design is exceptionally thorough, multi-axial, and statistically sound. The use of multiple seeds, adversarial streams, selection biases, and physical networks (both MLP and pre-trained ResNet-18) provides incredibly robust and undeniable empirical support for the authors' claims.
