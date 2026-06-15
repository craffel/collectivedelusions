# 4. Experiment Check

## Evaluation of Experimental Setup, Datasets, and Baselines
The paper's empirical evaluation suffers from several severe methodological limitations, weak baselines, and cases where the results actually contradict or fail to support the core claims:

### 1. Stylized and Custom Simulation Setup
* The primary evaluation is conducted in a custom **14-layer Analytical Coordinate Sandbox (ICS)**. activations reside in a $D=192$ dimensional space.
* The "datasets" are not actual ML datasets, but rather synthetic orthogonal coordinate subspaces with calibrated coordinate noise. 
* A 14-layer simulator with coordinate noise is a highly stylized environment that does not reflect the complex, non-linear representation spaces, semantic transformations, and vocabulary manifolds of modern deep learning serving pipelines (such as LLMs). Custom simulators are notoriously easy to tune and over-parameterize to produce any desired result.

### 2. Extremely Weak "Real-World" Pilot Study
* The authors attempt to claim generalizability through a "LLaMA-3-8B pilot study." 
* This study is evaluated on a microscopic test set of only **100 queries** with only **16 calibration samples** per task. 
* There is no linked repository, no open-source code, and no documentation of how the LoRA adapters were fine-tuned or how the evaluation was conducted. This tiny pilot study does not constitute a robust or scientifically rigorous evaluation of a real-world serving pipeline.

---

## Critical Deconstruction of the Results

A closer look at the results in Tables 1, 2, and 3 reveals that the core proposed method—the **Lyapunov-Stable Closed-Loop Feedback Controller**—provides almost zero actual benefit over simpler heuristics or baseline models, and in some cases performs worse:

### 1. Active Feedback is Practically Useless under Clean Workloads (Table 1)
* Under the practical Static Centroids setting (Setting A), full L-ARC (Ours) achieves **74.38% $\pm$ 0.31%** Joint Mean Accuracy. 
* However, **Decay-ChemMerge** (a simple, non-dynamic heuristic that linearly decays the feedback rate $\eta$ over depth) achieves an identical **74.38% $\pm$ 0.30%** accuracy and a virtually identical **0.7933 $\pm$ 0.0062** semantic similarity (compared to L-ARC's $0.7937 \pm 0.0059$).
* This means that under standard clean workloads, the complex online Lyapunov dissipation calculations, Dissipation Guard, and control loop add **absolutely zero** value over a simple, one-line linear decay heuristic on $\eta$.

### 2. Stateful Kinetics and Closed-Loop Control are Beaten by a Simple Stateless Heuristic
* Under Setting B (Layer-Specific Centroids), **EMA-SABLE** (a simple, low-latency layer-wise Exponential Moving Average of SABLE's stateless routing weights) achieves a Joint Mean Accuracy of **75.00% $\pm$ 0.33%** and Semantic Similarity of **0.8183 $\pm$ 0.0085**.
* Full L-ARC achieves only **74.46% $\pm$ 0.31%** accuracy and **0.8017 $\pm$ 0.0078** similarity in Setting B.
* This is a major failure: a simple, low-overhead stateless smoothing heuristic (EMA-SABLE) **directly outperforms** the highly complex, stateful kinetics and closed-loop control system of L-ARC. SABLE SOTA also beats L-ARC ($74.82\%$ vs. $74.46\%$). This completely undermines the claim that stateful kinetics and continuous-depth routing are superior.

### 3. Active Feedback warps representations poorly under failures (Table 2)
* Under transient routing failures (Setting C), full L-ARC (Ours) achieves **73.97% $\pm$ 0.39%** Joint Mean Accuracy.
* However, L-ARC with **ECG-Reset Only (and feedback disabled, $\eta = 0$)** achieves **73.93% $\pm$ 0.41%** Joint Mean Accuracy.
* The difference is a mere **0.04%** in accuracy, which the authors admit has a p-value of **0.3443** (highly statistically insignificant). 
* This means that under failures, the entire $+5.14\%$ accuracy improvement over ChemMerge is driven by **ECG-Reset** (the simple entropy check), and the elaborate closed-loop Lyapunov feedback controller and Dissipation Guard add **virtually nothing** to downstream classification accuracy.

### 4. Active Feedback causes Representational Distortion under Failures
* In Table 2, the stateless **SPS-ZCA SOTA** baseline achieves a final-layer Semantic Similarity of **0.8270 $\pm$ 0.0042**.
* Full L-ARC achieves a significantly lower Semantic Similarity of **0.7813 $\pm$ 0.0075**.
* This is a direct contradiction of the paper's narrative: active representation feedback is supposed to shield representations and prevent semantic corruption. But in reality, under failures, active warping pulls representations off-manifold, resulting in worse representational distortion than a simple stateless baseline.

### 5. Excessive Latency and Computational Overhead
* SABLE is **58.16 ms**, ChemMerge is **60.29 ms**, and L-ARC (with ET-L-ARC optimization) is **120.50 ms**.
* This represents a **100% relative latency overhead (doubling the routing latency)** compared to existing baselines.
* For a method designed for "resource-constrained edge devices," a 2x slowdown in routing overhead is a massive drawback. 
* The authors try to minimize this by claiming an "absolute overhead of 0.06 ms per sample," but this is calculated by profiling on a large batch size of 1000. In practical edge-serving environments, queries arrive in streams with a batch size of $B=1$. The relative 100% latency overhead remains, making L-ARC highly impractical for latency-sensitive edge deployment.
