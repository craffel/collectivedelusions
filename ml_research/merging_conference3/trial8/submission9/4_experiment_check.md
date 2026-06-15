# 4. Experimental Evaluation Check

The empirical portion of this paper is exceptionally thorough, spanning synthetic sweeps, real-world embeddings, temperature ablations, physical profiling, and scalability sweeps. The authors have done a commendable job updating their evaluations. However, a deep review reveals several persistent gaps, methodological constraints, and areas for empirical improvement.

### 1. Strengths of the Experimental Design

- **Multi-Dimensional Sweeps:** The authors evaluate their methods across an impressive array of scenarios, including shuffling streams, continuous representation drift ($d=0.45$), heterogeneous vocabulary sizes, registry scaling ($K \in \{4, 8, 12\}$), and different Softmax temperatures ($\tau$).
- **Realistic Streaming Properties:** The inclusion of "temporal task locality" (coherent task blocks rather than a fully shuffled stream) to analyze the impact of **Amortized EER** is a highly practical and realistic addition, making the systems-ML analysis extremely convincing.
- **Physical CPU Latency Profiling:** Measuring actual wall-clock execution times (milliseconds per sample) and relative overhead on an AMD EPYC CPU bridges the gap between theoretical FLOP counts and real systems-level implementation.
- **Validation on Real Embeddings with Rigor:** In response to prior reviews, the paper now validates its methods on real 512-dimensional ResNet-18 embeddings across four image domains (MNIST, FashionMNIST, CIFAR-10, SVHN) **reporting Mean $\pm$ Standard Deviation over 5 independent random seeds** (Table 4). This adds immense statistical rigor to the real-world portion of the paper.

### 2. Experimental Gaps and Critique Points

#### A. Validation Against Test-Time Adaptation (TTA) Baselines
In response to previous critiques, the authors have added a quantitative evaluation of a gradient-based Test-Time Adaptation (TTA) baseline, specifically **TENT**, on the real ResNet-18 heterogeneous stream.
- The results are highly revealing: under the shuffled heterogeneous stream, TENT experiences catastrophic representation collapse, yielding an accuracy of only **20.00%**—which is significantly lower than even the Static Uniform Merging baseline (31.66%) and our proposed EER (35.38%).
- This empirical outcome strongly validates the authors' theoretical critique in the Introduction and Related Work: online gradient-minimization of prediction entropy is highly unstable under realistic shuffled streams on edge nodes. 
- By demonstrating that training-free, forward-pass ensembling is both far more stable (+15.38% absolute accuracy boost for EER) and requires zero backpropagation, the authors have substantially strengthened their "Pragmatist" positioning, providing a complete and convincing systems-ML argument.

#### B. The Severe Entropy Calibration Discrepancy on Real Features
While EER achieves an outstanding 71.38% Joint Mean accuracy in the synthetic sandbox, its accuracy drops precipitously to **35.38 ± 0.66%** on real ResNet-18 embeddings (Table 4).
- The authors explain this through the "Entropy Calibration Discrepancy and OOD Overconfidence": simpler domains like MNIST produce highly confident, low-entropy predictions even on out-of-distribution (OOD) tasks, leading EER to route 75.2% of samples to MNIST.
- While the proposed **Centroid-Gated EER (CG-EER)** resolves this, raising accuracy to **61.50 ± 0.18%**, CG-EER is no longer calibration-free as it relies on pre-computed task centroids. 
- This highlights a critical empirical limitation: **the pure calibration-free direct routing paradigm (EER) is highly fragile on real-world representation embeddings** due to uncalibrated OOD expert behaviors. The paper should discuss this fragility more prominently, as it represents a massive gap between synthetic sandbox performance and real-world deployment.

#### C. The SVHN Performance Anomaly
Across all synthetic experiments, SVHN performance is remarkably poor:
- **Expert Ceiling (SVHN):** $39.44 \pm 4.56\%$ (compared to 100% on MNIST/FashionMNIST and 90.88% on CIFAR-10).
- **SPS-ZCA (SVHN):** $19.20 \pm 2.49\%$.
- **EER (SVHN):** $21.12 \pm 3.76\%$.
The extremely low expert ceiling and routing accuracies on SVHN skew the Joint Mean accuracy downward. The authors state that SVHN has an "extremely high intrinsic noise scale ($0.56$)" in their sandbox. While this explains the low accuracy and serves as an aggressive stress-test, it raises questions about whether this noise scale is representative of real-world SVHN. If SVHN is modeled with such extreme noise that it is essentially a random classifier, its contribution to the Joint Mean is highly corrupted, and the ensembling results are dominated by the other three tasks.
