# 4. Experiment & Empirical Verification Check

## Strengths of the Empirical Evaluation

The empirical evaluation is **exemplary** in its rigor, comprehensiveness, and scientific honesty. The authors analyze multiple serving conditions and explicitly profile trade-offs.

### 1. Robustness and Fault-Tolerance Benchmarking
The experiments are structured around four distinct serving environments:
*   **Setting A (Static Centroids)**: Simulates a highly practical, memory-constrained serving setup where task centroids are extracted once at Layer 3. This environment introduces significant representational drift across depth.
*   **Setting B (Layer-Specific Centroids)**: Simulates ideal centroid tracking across all layers, introducing substantial memory overhead.
*   **Setting C (Transient Routing Failures)**: Simulates transient sensor/network dropouts ($p_{\text{fail}} = 0.20$ at each layer), forcing the router to output uniform noise.
*   **Setting D (Confident Router Bias)**: Simulates persistent systematic routing bias toward an incorrect expert ($\text{bias\_scale} = 0.15$), with low routing entropy.

### 2. Empirical Performance Gains
*   **Setting A (Clean, Practical)**: L-ARC Feedback+ECG-Reset achieves **74.38% $\pm$ 0.31%** accuracy and **0.7937 $\pm$ 0.0059** semantic similarity, outperforming SABLE SOTA ($74.06\%$ accuracy, $0.7590$ similarity) and SPS-ZCA SOTA ($72.28\%$). More importantly, it completely prevents SABLE's semantic corruption (where similarity collapses to $0.7590$).
*   **Setting C (Failures)**: Stateful models normally propagate memory faults, dropping open-loop ChemMerge to $68.79\%$ and Decay-ChemMerge to $68.83\%$. L-ARC's ECG-Reset successfully freezes the stateful memory, achieving **73.97% $\pm$ 0.39%** accuracy and **0.7813 $\pm$ 0.0075** similarity. This is a massive **5.14%** absolute improvement over open-loop ChemMerge.
*   **Setting D (Systematic Router Bias)**: Stateful methods normally suffer from "state-locking failure", locking onto the incorrect expert and dropping to $\sim 68.25\% - 68.52\%$. L-ARC's RASC completely resolves this state-locking, achieving **73.59% $\pm$ 0.39%** accuracy and **0.7467 $\pm$ 0.0082** similarity. This is a massive **5.32%** absolute increase over Decay-ChemMerge (highly significant with a p-value of $0.0000$ and t-statistic of $17.9183$).

### 3. Deep Analysis of Heuristics & Trade-offs
*   **EMA-SABLE (Smoothing Heuristic)**: SABLE smoothed with a simple Exponential Moving Average ($\beta=0.5$) works exceptionally well in Setting B ($75.00\%$), but collapses to $73.87\%$ in Setting A due to representational backward-shift, proving that stateless smoothing is fragile when reference anchors are static.
*   **Decay-ChemMerge (Feedback Heuristic)**: Applying a linear decay to the feedback step size matches L-ARC in Setting A ($74.38\%$), but completely collapses in Setting C ($68.83\%$), proving that heuristic decaying feedback is fragile under system-level faults.
*   **The Responsibility vs. Inertia Trade-off**: The authors present a beautiful discussion on the trade-off between the *instant responsiveness* of stateless models (SABLE) and the *spatial consistency* of stateful models (L-ARC). Stateful models introduce physical inertia which dampens routing jitter but causes a *kinetics propagation lag* in initial layers. Under clean, layer-specific centroids (Setting B), SABLE's responsiveness allows it to slightly outperform stateful kinetics. Under practical static centroids (Setting A), SABLE's responsiveness becomes a liability as routing confusion propagates, allowing L-ARC's stateful kinetics to completely outperform it.
*   **The Accuracy vs. Representational Distortion Trade-off**: Under Setting C (failures), stateless SPS-ZCA achieves a superior semantic similarity of **0.8270** compared to L-ARC's **0.7813**. The authors explain that because SPS-ZCA is stateless and operates strictly at the early Layer 3, it completely avoids depth-wise kinetics propagation lag and does not execute active representation warping on corrupted signals across deeper layers. Conversely, L-ARC actively warps activations toward blended centroids, and any remaining routing noise that passes through its gating filters still introduces a minor, cumulative representational distortion relative to simple early-stage static routing.

### 4. Extreme Scientific Transparency
The authors are remarkably honest and transparent about their results:
*   They report a Relational Paired t-test over 10 random seeds in clean Setting A, showing a p-value of **0.0969** between L-ARC and open-loop ChemMerge. They explicitly write that **under clean, unperturbed serving workloads, active representation feedback warping is not statistically significant**, and advise edge-device practitioners that simply setting the coupling coefficient $\eta = 0$ is preferred under clean conditions due to lower latency.
*   They report that active representation feedback warping under transient failures (Setting C) does not yield a statistically significant accuracy improvement over pure state gating (L-ARC ECG-Reset Only, $\eta = 0$) with a p-value of **0.3443**, but provides a minor improvement in semantic representation-space alignment.
*   They explicitly profile latency overheads on a batch size of 1000 over 50 independent runs, proving that pre-normalization optimizations reduce vector execution by **36.0%**. They show that ET-L-ARC gating provides a **15.0% absolute speedup** (collapsing latency from $141.76$ ms to $120.50$ ms), reducing relative latency overhead vs. ChemMerge to only **99.85%** (adding a negligible **0.06 ms per sample**).

### 5. Extensive Robustness Sweeps
The authors perform extensive parameter sweeps, proving extreme resilience across multiple dimensions:
*   **Manifold Entanglement ($\rho$)**: Swept from $0.0$ to $0.5$. L-ARC maintains stable performance while nearest-centroid SPS-ZCA collapses.
*   **Downstream Non-Linear Threshold Effects**: They sweep the competency threshold $\theta \in [0.4, 0.9]$ under step-threshold activation mappings, showing a highly consistent absolute improvement of $+3.81\%$ over ChemMerge.
*   **Calibration Data Complexity**: Swept from $1$ to $64$ samples. L-ARC is highly data-efficient, achieving $70.05\%$ accuracy with just 8 calibration samples per task.
*   **Kinetics temperature ($\tau$)**: Swept from $0.005$ to $0.20$. RASC-equipped L-ARC is shown to decouple ensembling performance from temperature scaling fragility.

## Minor Limitations
*   **Stylized Sandbox**: While the Coordinate Sandbox is standard for dynamical system analysis and isolating control variables, the main quantitative results are evaluated within this simplified $D=192$ and $K=4$ simulation.
*   **Lack of End-to-End LLM Benchmark Evaluation**: Although the authors conducted an excellent small-scale pilot study using pre-trained LLaMA-3-8B (which successfully validates their coordinate orthogonality and Dissipation Guard gating rate assumptions), a large-scale evaluation of L-ARC's final task accuracy on full-scale transformers served on standard benchmarks (e.g., GLUE, MMLU) is still missing.
