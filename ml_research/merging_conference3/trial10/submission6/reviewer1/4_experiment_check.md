# Intermediate Evaluation 4: Experimental Evaluation

## Critique of Experimental Design
The experimental section is exceptionally well-designed, employing a dual-pronged validation strategy that combines:
1. **The Isolating Coordinate Sandbox (ICS):** A fast, highly controlled simulated environment. This sandbox allows the authors to model orthogonal and overlapping manifolds, inject noise with precise standard deviations, simulate homogeneous and heterogeneous stream transitions, and execute exhaustive grid-search sensitivity and scalability sweeps.
2. **Physical GPU Validation (Appendix A):** Physical deployment on a pre-trained GPT-2 Small (12-layer, 117M parameters) backbone on an NVIDIA A100 GPU. The authors fine-tune three distinct task-specific adapters (IMDB Sentiment, SAMSum Summarization, and WMT16 Translation) and measure actual task serving accuracies, depth-wise jitter, and real-world hardware latencies using GPU-side asynchronous profiling.

This combination of controlled simulation and real-world physical profiling provides a powerful and comprehensive empirical foundation.

## Evaluation of Baselines and Datasets
The authors compare PID-Merge against an extensive and representative suite of baselines:
- **Expert Oracle (Hypothetical Ceiling):** Re-routes $100\%$ of queries to the correct expert.
- **Uniform Merging (Static):** Merges all expert adapters with equal, static weights.
- **SABLE (Stateless Raw):** Represents state-of-the-art stateless dynamic routing.
- **ChemMerge (Kinetics ODE):** A SOTA stateful method that models weights via continuous ODE kinetics.
- **Momentum-Merge (EMA):** An open-loop stateful baseline using first-order Exponential Moving Average.
- **PAC-Kinetics:** A stateful router optimized via PAC-Bayesian complexity regularization.

The choice of datasets for the physical validation (IMDB, SAMSum, WMT16 English-to-German) is highly appropriate. These tasks represent three fundamentally distinct NLP domains with different vocabularies, sequence lengths, and generation behaviors, making the multi-task query stream a highly challenging test of dynamic ensembling.

## Do the Results Support the Claims?
Yes, the empirical results provide overwhelming and direct support for all the paper's core claims:
1. **Resolving the Stateful-Stateless Dilemma:** On overlapping heterogeneous streams, SABLE (Stateless) achieves high accuracy ($94.93\%$) but oscillates wildly. ChemMerge and Momentum-Merge smooth out noise but suffer from severe inertial drag, collapsing accuracy to $88.42\%$ and $86.17\%$, respectively. PID-Merge (Calibrated) achieves **94.82%** accuracy under the same chaotic stream, virtually matching the stateless ceiling while guaranteeing depth-wise convergence stability.
2. **Physical Jitter Reduction:** In physical GPT-2 experiments, SABLE incurs high depth-wise jitter ($0.724$). Calibrated PID-Merge slashes this jitter by **73.3%** (down to $0.193$) while keeping serving accuracy at $88.64\%$ (compared to SABLE's $89.14\%$).
3. **Imperceptible Latency Overhead:** Hardware latency profiling confirms that PID-Merge adds just **0.012 ms** of overhead per forward pass, running **40 times faster** than ChemMerge ($0.482$ ms).
4. **Computational Scalability:** When scaling the active expert pool $K$ from 4 to 64 (Appendix F), ChemMerge's latency explodes quadratically to a prohibitive $12.482$ ms due to coupled ODE solving. PID-Merge's latency scales sub-linearly, rising to only **0.022 ms** at $K=64$, yielding an outstanding **567.3$\times$ latency reduction** over ChemMerge.

## Empirical Gaps or Limitations
1. **Scale of Physical Validation:** The physical hardware validation is conducted on GPT-2 Small (117M parameters, 12 layers). While GPT-2 Small is an excellent proof-of-concept for testing layer-wise activation mixing and latency, modern multi-tenant workloads typically run on multi-billion parameter models (e.g., LLaMA-3 8B with 32 layers). The scaling challenges (like integrator windup) are analyzed theoretically in Appendix A, and the proposed conditional integration clamping is designed specifically to mitigate them. However, actually running physical validation on a multi-billion parameter model would make the empirical results even more compelling.
2. **Sanity Checking Simulation Noise:** As the authors transparently point out, the ICS sandbox has a methodological limitation where noise is only injected at the boundary layer, making stateless SABLE appear artificially stable depth-wise. This limitation is minor because the authors recognize it and address it thoroughly through their physical GPT-2 validation, where continuous independent layer-wise noise is naturally present. This transparent, self-aware approach is commendable.
