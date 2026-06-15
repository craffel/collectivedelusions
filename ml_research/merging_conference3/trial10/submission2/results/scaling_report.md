# Scaling Expert Sweeps Analysis (K=4 to K=16)

This table summarizes the performance of Global PAC-Kinetics (M=1) and Layer-Decoupled Stateful Kinetics (M=11) across various numbers of task experts K.

| Experts (K) | Model | Homo Acc (%) | Hetero Acc (%) | Hetero Jitter | Step Latency (us) |
| --- | --- | --- | --- | --- | --- |
| K=4 | Global M=1 | 66.22% ± 0.12% | 66.73% ± 3.90% | 0.8002 | 31.45 us |
| | LDS-Kinetics M=11 | 66.22% ± 0.12% | 66.79% ± 3.88% | 0.9269 | 355.47 us |
| K=8 | Global M=1 | 58.66% ± 0.09% | 56.17% ± 3.66% | 0.9236 | 29.92 us |
| | LDS-Kinetics M=11 | 58.66% ± 0.07% | 56.18% ± 3.66% | 0.9118 | 345.02 us |
| K=12 | Global M=1 | 54.76% ± 0.03% | 51.41% ± 3.29% | 0.9819 | 29.93 us |
| | LDS-Kinetics M=11 | 54.77% ± 0.03% | 51.45% ± 3.30% | 0.9385 | 347.26 us |
| K=16 | Global M=1 | 51.92% ± 0.03% | 48.31% ± 1.79% | 0.9987 | 29.93 us |
| | LDS-Kinetics M=11 | 51.91% ± 0.02% | 48.34% ± 1.78% | 0.9184 | 333.91 us |
