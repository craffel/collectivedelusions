# Evaluation Stage 4: Experimental Evaluation Check

## Experimental Setup & Datasets
- **Coordinate Sandbox (ICS):** The main evaluation environment is the Analytical Coordinate Sandbox (ICS). While ICS is a controlled simulator configured with $D=192$ and $L=14$ layers, it uses synthetic task signatures to represent MNIST, Fashion-MNIST, CIFAR-10, and SVHN datasets.
- **Empirical Limitation:** Crucially, the paper *does not* validate the proposed QA-Merge on a real-world deep learning backbone (such as LLaMA-3, Mistral, ResNet, or Vision Transformer) running on actual dataset streams. All main tables (Table 1 and Table 2) are simulated within the ICS coordinate-space environment. The provided PyTorch script (`toy_qamerge_lora.py`) is a toy demonstration on random input tensors. This represents a significant empirical limitation for a systems/edge-deployment paper.

## Baselines & Comparisons
- **Baselines evaluated:** The paper includes a thorough set of baselines: Expert Oracle (Float32), Uniform Merging (Float32 & Quantized), SABLE (Float32 & Quantized-Naive), ChemMerge (Float32 & Quantized-Naive), Momentum-Merge (Float32 & Quantized-Naive), and Parametric Router (Float32 & Quantized-Naive).
- **Tuning & Fairness:** The baselines appear to be well-tuned and compared fairly, with hyperparameters listed.

## Empirical Rigor, Random Seeds, and Confidence Intervals
- **Missing Statistical Metrics:** Across all main experimental tables (Table 1 and Table 2) and appendix tables, the authors report only single-point accuracies (e.g., "79.20%", "65.80%"). There are **no standard deviations, confidence intervals, or error bars** reported, despite these experiments being run within a synthetic coordinate-space simulator where representation covariance and task experts are simulated.
- **Empirical Standard:** For simulated evaluations, reporting results across multiple random seeds with confidence intervals is critical to ensure statistical significance. The lack of these metrics is a major empirical weakness.

## Supporting the Claims
- **Quantization Collapse Recovery:** Within the ICS simulator, the results strongly support the claim that QA-Merge successfully recovers full-precision ensembling gains (e.g., closing the gap to SABLE/ChemMerge Float32 performance).
- **Amdahl's Law and End-to-End Speedup:** In Appendix E, the authors disclose that their microcontroller speedup (5.2x latency reduction, 42% power reduction on STM32H753XI) is evaluated *exclusively on the coordinate propagation/ensembling loop itself*. Under Amdahl's Law, because the heavy backbone layers of a deep neural network consume the vast majority of execution time, accelerating the ensembling loop yields a negligible end-to-end speedup. Although the authors provide a reasonable systems justification (that avoiding dynamic float-to-int format conversions is crucial for maintaining a unified integer pipeline), the lack of end-to-end model latency profiling on the physical hardware represents another empirical gap.
