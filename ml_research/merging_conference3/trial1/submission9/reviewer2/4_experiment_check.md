# Evaluation Component 4: Experimental Evaluation and Rigor Check

## 1. Quality and Scope of Experimental Datasets
The empirical evaluation is structured across two distinct levels:
1. **Low-Dimensional CNN End-to-End Evaluation:**
   - **Setup:** A custom SimpleCNN (approx. 500k parameters) evaluated on a multi-task image classification benchmark consisting of MNIST, FashionMNIST, and KMNIST.
   - **Critique:** While the authors design a clever pretraining-then-finetuning workflow with heterogeneous schedules to simulate uncoordinated downstream updates (e.g., fine-tuning MNIST for 3 epochs at 1e-3, FashionMNIST for 2 epochs at 3e-3, and KMNIST for 1 epoch at 2e-3), the benchmark datasets are **highly simplistic and standard toy datasets**. Running a custom CNN on MNIST-style datasets is a major limitation in scope for a modern machine learning conference-level paper on model merging.
2. **High-Dimensional CLIP ViT-B/32 Real-Weight Simulation:**
   - **Setup:** To bridge the simulation gap, the authors evaluate the methods directly on the actual pretrained weight layers of the official OpenAI CLIP ViT-B/32 visual encoder (36 projection layers, including attention outputs, MLP intermediate layers, and MLP final projection layers). They simulate $K = 3$ task updates with severe scale mismatches (RMS scales of 0.1, 0.5, and 2.0) and project these weights onto real activation batches, measuring **Wall-Clock Time** and **Activation-Space Cosine Alignment**.
   - **Critique:** This high-dimensional evaluation is highly rigorous, elegant, and directly addresses the scale gap of the custom CNN. It provides physical proof of the SVD alignment parity (both RMS-Scale and SVD Isotropic achieve exactly **57.74%** average cosine alignment and a virtually identical **0.15%** alignment standard deviation across tasks). However, it remains an activation-level simulation rather than an end-to-end downstream classification task. Fully closing the evaluation scale gap (evaluating downstream classification accuracy of merged CLIP or LLM models on complex datasets like Stanford Cars, DTD, EuroSAT, etc.) is a critical next step that the authors honestly list as future work.

## 2. Selection and Rigor of Baselines
The paper compares the proposed methods against a solid set of contemporary model-merging baselines:
- **Task Arithmetic** (Ilharco et al., 2022)
- **Ties-Merging** (Yadav et al., 2023)
- **DARE** (Yu et al., 2024)
- **AdaMerging** (Yang et al., 2024)
- **SVD Isotropic / SAIM-like** (2025)

The evaluation of these baselines is highly transparent and scientifically honest:
- They search over a wide grid for the global scaling coefficient ($\lambda \in [0.3, 1.5]$) and report both **validation-tuned** and **default un-tuned** ($\lambda=1.0$) performance.
- They report the exact optimal hyperparameters found (e.g., pruning ratio of 60% and $\lambda = 0.90$ for Ties-Merging; drop rate of 40% for DARE).

## 3. Statistical Rigor
The paper demonstrates exemplary statistical rigor:
- All results on the CNN multi-task benchmark are aggregated across **3 independent random seeds and data splits**, reporting both the mean and standard deviation.
- High-dimensional CLIP ViT evaluation results are similarly averaged across 3 independent seeds and across all 36 projection layers.
- The reporting of standard deviations helps put the improvements into context. On the CNN benchmark, the average accuracies of validation-tuned SD-Scale (73.23 $\pm$ 2.19%) and RMS-Scale (73.22 $\pm$ 2.15%) lie within the seed-to-seed variance boundaries of Task Arithmetic (72.50 $\pm$ 1.17%) and Ties-Merging (71.77 $\pm$ 2.06%). The authors transparently discuss this variance, noting that the primary empirical advantage is the **Parameter-Free PF-RMS variant**, which achieves **72.23%** out-of-the-box (beating validation-tuned Ties-Merging and tuned DARE) without any validation data or grid searching.

## 4. Support for Claims
The reported results directly support all central claims in the paper:
- **Task dominance is resolved:** On the CNN benchmark, standard Task Arithmetic causes KMNIST accuracy to degrade to 56.00% (or 57.63% when tuned) because FashionMNIST's updates dominate due to its 1.9x larger standard deviation. Our proposed RMS-Scale recovers KMNIST accuracy to **61.57%** (a +3.94% boost), showing that scale calibration balances representation spaces effectively.
- **Complexity is redundant:** RMS-Scale outperforms SVD Isotropic (73.22% vs. 73.13%) on the CNN benchmark. On the CLIP ViT benchmark, RMS-Scale achieves identical activation alignment (57.74%) to SVD Isotropic but runs **over 100x faster** (5.67ms vs. 571.92ms per layer), physically validating the claims of linear time complexity.
- **Both components are necessary:** The ablation study shows that Normalization Only drops to 19.23% (representation distortion) and Calibration Only drops to 53.20% (scale mismatch), confirming they are synergistic halves of a complete solution.
- **Alternative estimators and parameters are robust:** The empirical comparison of scale estimators (Harmonic, Geometric, and Arithmetic means) and sensitivity analysis on stability constant $\epsilon$ and clipping threshold $\gamma$ are highly comprehensive, proving that the framework is robust.
