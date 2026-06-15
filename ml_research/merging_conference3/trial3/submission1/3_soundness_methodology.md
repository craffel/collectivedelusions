# 3. Technical Soundness and Methodology Check

This paper exhibits a high level of **technical soundness and mathematical rigor**. The formulations are precisely defined, and the experimental choices are logically structured. However, several methodological assumptions and scope choices warrant critical examination.

---

## 1. Key Strengths of the Methodology

* **Rigorous Quantization Mathematical Formalization:** The paper provides clean, mathematically complete definitions of Uniform Symmetric vs. Asymmetric quantization, including dynamic re-calculation of scales ($s$) and zero-points ($z$) at every optimization step. This is crucial because dynamic scaling introduces a highly non-linear feedback loop that complicates gradient tracking.
* **Deep Insight into PyTorch Autograd for Quantization:** The discussion on *Gradient Tracking of Quantization Parameters* is exceptionally sharp. The authors clearly explain that the scale factor $s$ is fully active in the PyTorch computational graph (as a continuous function of minimum and maximum weights), whereas the zero-point $z$ is detached due to the non-differentiable rounding operator. This asymmetry is a major source of Straight-Through Estimator (STE) gradient noise, and highlighting this detail demonstrates a profound understanding of quantization mechanics.
* **Derivative-Free Comparator (1+1 ES):** Introducing a 1+1 Evolution Strategy based on Rechenberg's 1/5th success rule is an excellent methodological choice. It provides a pure black-box baseline that is completely free of STE gradient bias, allowing the authors to isolate the effect of optimization style (gradient-based vs. derivative-free) on operator overfitting.
* **Ablations and Controls:** The authors did not stop at standard sweeps; they included key control experiments:
  * Running a hyperparameter sweep over the Adam learning rate to confirm that the instability of STE is not a simple tuning issue.
  * Starting the optimization from optimal unquantized coefficients (AdaMerging) to show that dynamic initialization does not solve cross-schema collapse.
  * Proposing and validating a **Supervised Calibration Baseline** using cross-entropy loss to isolate the transductive limits of the unsupervised entropy objective from pure data-scarcity limits.

---

## 2. Key Weaknesses & Methodological Concerns

* **The Extreme Task-Interference Regime:** In the experimental setup, the FP16 Task Arithmetic baseline yields an average accuracy of only **35.12%**, despite individual experts achieving $>90\%$ unmerged performance. This indicates an extremely high-conflict weight-space divergence between the independently fine-tuned experts. 
  * *Critique:* While testing in an extreme conflict scenario is highly valuable to stress-test these methods under non-convex landscapes, it represents a worst-case scenario. When expert checkpoints are highly divergent, the multi-task loss landscape is highly fractured. The paper briefly ablated "moderate task conflict" (simulated by pre-aligning expert checkpoints via joint training) and observed a smaller generalization gap ($-5.00\%$), but a more systematic, multi-degree analysis of expert alignment (e.g., as a continuous axis of variation) is missing.
* **Mathematical SVD Simulation of PEFT/LoRA:** In Section 4.5, the authors simulate Parameter-Efficient Fine-Tuning (PEFT) experts by projecting the full-parameter task vectors into a low-rank subspace (rank $r=4$) via Singular Value Decomposition (SVD).
  * *Critique:* Although mathematically elegant as a way to isolate low-intrinsic-dimension constraints, this projection approach is not identical to merging actual LoRA adapters ($BA$) trained natively. In native LoRA, adapters are localized to specific projection matrices (e.g., query, key, value projections) within self-attention blocks, and their interaction under quantization mismatches might be different from global full-parameter SVD projections.
* **Dataset Resizing Artifacts:** The experts are trained on MNIST, FashionMNIST, CIFAR-10, and SVHN, which natively range from $28\times 28$ to $32\times 32$ pixels. To process them with the ViT-Tiny backbone, all images are normalized and resized to $224\times 224$ pixels.
  * *Critique:* Resizing small-resolution digits (MNIST, SVHN) to $224\times 224$ and processing them through a transformer patch embedding can introduce substantial resizing artifacts and over-parameterization. While necessary to share a single pre-trained ViT-Tiny backbone, this setup is somewhat non-standard and could impact the generalizability of the findings to native high-resolution image tasks.
* **Scale of the Backbone:** The core evaluation is restricted to `vit_tiny_patch16_224` (5.7M parameters). While this small scale allows for highly exhaustive sweeps and complete scientific control (e.g., multiple seeds, multiple random streams), model merging and quantization are most widely deployed in billion-parameter scale Large Language Models (LLMs) or Vision-Language Models (VLMs). The paper's Section 5 discusses LLMs conceptually, but the lack of direct empirical verification on at least a moderate-sized model (e.g., `ViT-Base` with 86M parameters or a small LLM like Pythia-70M) limits the technical soundness of scaling claims.
