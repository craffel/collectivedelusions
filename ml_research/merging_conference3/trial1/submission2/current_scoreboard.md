# Current Experimental Results Scoreboard

| Optimizer           | Merging Strategy   | Accuracy (ACC) %   | Forgetting (BWT) %   | Duration (s)   |
|:--------------------|:-------------------|:-------------------|:---------------------|:---------------|
| adamw               | isotropic          | 56.38%             | -40.33%              | 208.9s         |
| adamw               | isotropic          | 70.98%             | -18.86%              | 178.8s         |
| adamw               | spectral_dampening | 24.69%             | -67.89%              | 211.9s         |
| adamw               | task_arithmetic    | 59.64%             | -38.61%              | 207.3s         |
| adamw               | task_arithmetic    | 62.58%             | -31.84%              | 169.0s         |
| sabcd_adam_gt       | isotropic          | 50.00%             | -45.91%              | 256.0s         |
| sabcd_adam_gt       | spectral_dampening | 22.44%             | -66.68%              | 256.7s         |
| sabcd_adam_gt       | task_arithmetic    | 57.66%             | -39.99%              | 257.7s         |
| sabcd_literal       | isotropic          | 4.28%              | -0.38%               | 255.5s         |
| sabcd_literal       | spectral_dampening | 4.19%              | -0.50%               | 259.0s         |
| sabcd_literal       | task_arithmetic    | 4.84%              | -0.23%               | 257.4s         |
| sabcd_standard_adam | isotropic          | 50.13%             | -46.30%              | 265.4s         |
| sabcd_standard_adam | spectral_dampening | 19.72%             | -67.55%              | 268.6s         |
| sabcd_standard_adam | task_arithmetic    | 62.85%             | -34.98%              | 279.9s         |
| sam                 | isotropic          | 76.00%             | -14.84%              | 239.3s         |
| sam                 | isotropic          | 62.46%             | -35.18%              | 237.6s         |
| sam                 | spectral_dampening | 26.85%             | -70.36%              | 239.6s         |
| sam                 | task_arithmetic    | 68.27%             | -29.64%              | 236.1s         |
| sam                 | task_arithmetic    | 73.52%             | -20.94%              | 239.5s         |
