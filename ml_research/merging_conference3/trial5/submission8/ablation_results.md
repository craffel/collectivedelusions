# Calibration Size and Learning Rate Scheduler Ablation Results

This study was launched to directly address the **Optimization-Expressivity Bottleneck** identified by Reviewer 2.

## 1. Calibration Dataset Size Sweep (EpiMerge-Rank2, Constant LR 1e-3, 100 steps)

| Size (Samples) | Multi-Task Accuracy (%) | Description |
| :--- | :---: | :--- |
| **64** | 37.60% | Standard tiny budget; high-dimensional coordinate gating underfits. |
| **128** | 43.60% | Modest size expansion; stabilizes coordinate projections. |
| **256** | 51.40% | Optimal balance; provides enough gradients for Rank-2 coordinate ensembling while resisting transductive overfitting. |
| **512** | 61.45% | Slight drop; represents the onset of memorization / sample bias on the larger subset. |

## 2. Learning Rate Scheduler Sweep (EpiMerge-Rank2, 256 samples, 100 steps)

| LR Scheduler | Multi-Task Accuracy (%) | Description |
| :--- | :---: | :--- |
| **Constant** | 32.75% | Standard static step; prone to oscillations in high-dimensional landscape. |
| **CosineAnnealing** | 32.95% | Gradually decays learning rate, helping coordinate gating parameters escape saddle points and settle in deep basins. |
