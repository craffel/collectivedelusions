# 4. Experiment Check

## Critical Evaluation of the Experimental Setup

### 1. The Simulation Gap: Evaluation Exclusively in a Synthetic Sandbox
The entire quantitative evaluation of Active Inference Routing (AIR) is conducted on the **Analytical Coordinate Sandbox (ACS)**, which is a synthetic 14-layer, 192-dimensional coordinate simulation. 
* **No Real-World Backbone Experiments**: The paper does not evaluate AIR on a single physical, pre-trained neural network (e.g., a Vision Transformer like ViT-B, or a Large Language Model like LLaMA-3).
* **No Real-World Workloads**: The "tasks" and "datasets" are simulated as coordinate projections rather than real sequential text generation, token classification, or image stream classification tasks. 
* **Unmeasured Systems Claims**: A core motivation of the paper is resolving physical systems-level bottlenecks, specifically **Hardware Cache Thrashing** and **Representational Instability** in GPU SRAM/HBM caused by SABLE's routing jitter. However, **the authors never run a physical model serving framework (such as S-LoRA, vLLM, or DeepSpeed-MInference) to measure actual hardware metrics**. There are no measurements of GPU memory bandwidth, SRAM cache misses, register allocation times, or physical wall-clock serving latency of a real LoRA-blending system. 

While Appendix G provides a "Real-World Roadmap" and Appendix H reports isolated microsecond-level PyTorch linear solver runtimes, the lack of actual downstream task metrics (like perplexity, BLEU, or ImageNet accuracy) and physical hardware metrics limits the scientific weight and systems-level validity of the claims.

---

## Evaluation of Baselines

The baseline comparison includes Oracle, Uniform, SABLE, Momentum-Merge, ChemMerge, and PAC-Kinetics. While this covers standard stateless and stateful approaches, there is a **significant omission of simple, standard adaptive temporal filters**:
* **Adaptive Exponential Moving Average (Adaptive EMA)**: Instead of rigid, static filters like Momentum-Merge (constant EMA), one can easily implement a simple adaptive EMA where the smoothing factor $\beta_t$ is dynamically adjusted based on input shift detection (e.g., if there is a large spike in sensory coordinate prediction error, set $\beta_t = 0$ to reset, otherwise keep $\beta_t$ high). 
* **Adaptive Kalman Filtering**: Applying a standard adaptive Kalman filter directly to the gating weights $\alpha_t$ with an online covariance update.

Such adaptive baselines are far simpler, computationally lighter, require no off-line parameter calibration, and would achieve the exact same noise-filtering and rapid-switching capabilities without the complexity of a full Active Inference framework with a linear solver. The failure to compare against simple adaptive temporal filters is a major methodological omission.

---

## Technical Discrepancies and Contradictions in the Results

### 1. Stateless SABLE Outperforming AIR on Heterogeneous Streams
In Table 1 (Orthogonal Manifolds) and Table 2 (Overlapping Manifolds), **stateless SABLE actually outperforms proposed AIR in Alignment Accuracy under Heterogeneous Streams**:
* **Table 1 Heterogeneous Accuracy**: SABLE achieves **66.30\%**, while AIR achieves **66.23\%** (Oracle ceiling is **66.32\%**).
* **Table 2 Heterogeneous Accuracy**: SABLE achieves **66.30\%**, while AIR achieves **66.22\%** (Oracle ceiling is **66.32\%**).

Furthermore, under Heterogeneous Streams, SABLE exhibits higher routing jitter (**1.4900** in Table 1, **1.4886** in Table 2) than AIR (**1.4202** in Table 1, **1.4169** in Table 2), which is closer to the Oracle's ensembling tracking speed (**1.4979**). This proves that SABLE actually tracks rapid switches *faster* and with *less lag* than AIR, achieving superior alignment accuracy. 

Under stable Homogeneous Streams, SABLE achieves the exact same alignment accuracy as AIR (**66.44\%**), albeit with higher jitter (**0.0860** vs. **0.0364**). This raises a critical question: is SABLE's routing jitter actually a performance bottleneck in real-world networks if it yields equal or superior representation alignment accuracy across both stream types?

### 2. Alignment vs. Categorical Accuracy Contradiction in the Nonlinear Stress Test
In Appendix F (Table 3, Nonlinear Manifold Stress Test), we observe a confusing mathematical discrepancy:
* Under Heterogeneous streams, **SABLE still achieves higher Representation Alignment Accuracy than AIR (60.33\% vs. 59.38\%)**.
* Yet, the authors state in the text (Point 1 in Appendix F) that SABLE's average categorical accuracy collapses to **93.99\%** under heterogeneous streams, compared to AIR's **98.83\%**.

This is highly surprising. Representation alignment accuracy is measured as the exponential negative Euclidean distance to the optimal task coordinates, meaning that large ensembling deviations (arising from SABLE's severe routing jitter) should heavily penalize the alignment score. If SABLE's high-frequency routing noise "directly disrupts task prediction accuracy," why does SABLE still achieve **higher** representation alignment accuracy than AIR? The paper fails to provide a mathematically rigorous or empirically clear explanation for this contradiction.

---

## Sensitivity and Overfitting Risks

As shown in Appendix E (Table 4), the performance of AIR is highly sensitive to the smoothness regularization hyperparameter $\lambda_{\text{smooth}}$:
* Disabling it ($\lambda_{\text{smooth}} = 0.0$) collapses AIR into stateless SABLE (high routing jitter).
* Over-penalizing ($\lambda_{\text{smooth}} = 1.0$) collapses AIR into a rigid temporal filter (representational lag and accuracy collapse).
The paper does not provide a robust, non-heuristic method for practitioners to select or tune $\lambda_{\text{smooth}}$ on a real-world serving stream without access to an oracle sequence. 

Additionally, calibration is performed over a tiny sequence of length $T_{\text{cal}} = 50-100$ steps. This introduces a high risk of "sequence-slicing" overfitting, where the learned prior precision $\mathbf{\Pi}_s$ and transition decay $\mathbf{A}$ are highly tuned to the specific transition frequency and noise levels of the calibration slice.
