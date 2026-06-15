# Physical Evaluation Results: MNIST & Fashion-MNIST on PyTorch MLP

This file documents the physical evaluation of PAC-Kinetics on real-world datasets using deep LoRA-blended neural networks. We pre-train a shared trunk MLP, freeze it, and fine-tune two active LoRA-style adapters on MNIST and Fashion-MNIST subsets. Then, we evaluate our stateful PAC-Kinetics router against 5 standard baselines over sequential query streams.

## Homo Stream serving (5 seeds)

| Method | Representation Alignment Acc. (%) | Actual Classification Acc. (%) | Routing Jitter |
| :--- | :---: | :---: | :---: |
| oracle | 100.00% &plusmn; 0.00% | 81.00% &plusmn; 2.28% | 0.0101 &plusmn; 0.0000 |
| uniform | 24.22% &plusmn; 3.77% | 54.90% &plusmn; 2.85% | 0.0000 &plusmn; 0.0000 |
| sable_raw | 30.74% &plusmn; 5.29% | 61.70% &plusmn; 2.86% | 0.1182 &plusmn; 0.0214 |
| pac_zca | 79.57% &plusmn; 4.97% | 71.20% &plusmn; 4.02% | 0.4891 &plusmn; 0.0864 |
| chemmerge | 30.17% &plusmn; 5.12% | 62.20% &plusmn; 3.08% | 0.0471 &plusmn; 0.0087 |
| pac_kinetics | 87.61% &plusmn; 9.68% | 76.40% &plusmn; 5.50% | 0.1888 &plusmn; 0.0511 |
| pac_kinetics_rand | 52.53% &plusmn; 2.66% | 43.71% &plusmn; 1.78% | 0.0228 &plusmn; 0.0428 |

**Pearson correlation coefficient** between intermediate representation blending error and final downstream classification accuracy success: **0.1704 &plusmn; 0.0931**

## Hetero Stream serving (5 seeds)

| Method | Representation Alignment Acc. (%) | Actual Classification Acc. (%) | Routing Jitter |
| :--- | :---: | :---: | :---: |
| oracle | 100.00% &plusmn; 0.00% | 81.00% &plusmn; 2.28% | 1.0151 &plusmn; 0.0744 |
| uniform | 24.22% &plusmn; 3.77% | 54.90% &plusmn; 2.85% | 0.0000 &plusmn; 0.0000 |
| sable_raw | 30.74% &plusmn; 5.29% | 61.70% &plusmn; 2.86% | 0.1718 &plusmn; 0.0232 |
| pac_zca | 79.57% &plusmn; 4.97% | 71.20% &plusmn; 4.02% | 0.9227 &plusmn; 0.0402 |
| chemmerge | 27.09% &plusmn; 4.49% | 58.60% &plusmn; 3.18% | 0.0695 &plusmn; 0.0116 |
| pac_kinetics | 68.50% &plusmn; 10.07% | 66.30% &plusmn; 7.79% | 0.5877 &plusmn; 0.0673 |
| pac_kinetics_rand | 52.45% &plusmn; 2.31% | 43.64% &plusmn; 1.52% | 0.0299 &plusmn; 0.0574 |

**Pearson correlation coefficient** between intermediate representation blending error and final downstream classification accuracy success: **0.4115 &plusmn; 0.0544**

