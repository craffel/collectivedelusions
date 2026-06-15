# Critical Evaluation of the Experimental Setup and Results

A close and critical inspection of the experimental results in Section 4 and Table 1 reveals severe weaknesses, flawed assumptions, and highly underwhelming performance:

## 1. Catastrophically Low Absolute Performance
The absolute accuracies reported in Table 1 are exceptionally low and practically non-functional:
* On **MNIST** (a trivial 10-class dataset where simple models easily achieve $>99\%$ accuracy), the baseline Task Arithmetic gets **21.40%** and the proposed ThermoMerge gets **20.00%**. This is barely above random guessing ($10\%$) and represents a complete failure of the model to perform digit classification.
* On **FashionMNIST** (where simple models easily achieve $>92\%$), the proposed method gets **32.60%** (worse than the static Task Arithmetic baseline of **35.40%**).
* On **CIFAR-10** (standard $>85\%$), the proposed method gets **33.00%**.
* On **SVHN** (standard $>90\%$), the proposed method gets **30.60%**.

An average multi-task accuracy of **29.05%** across these four simple datasets is extremely poor. The merged model is essentially non-functional. Presenting a model that gets $20\%$ on MNIST and $33\%$ on CIFAR-10 as an "outstanding" success is an extreme overstatement of the results. 

## 2. Omission of Standalone Expert Performance (Critical Baseline)
The paper completely omits the standalone accuracies of the fine-tuned experts (i.e., the individual models before merging). This is a critical scientific flaw:
* Without knowing the upper bound (e.g., did the MNIST expert achieve 99% accuracy on MNIST?), we cannot quantify the actual "representation collapse" or "interference."
* If the experts themselves achieved high accuracies (which they should have, even on a subset), then a drop to $20\%$ on MNIST and $30\%$ on SVHN indicates that the merging process is extremely destructive. 
* If the experts themselves were weak (e.g., only getting $25\%$ standalone due to poor fine-tuning), then the entire experimental setup is flawed and invalid.

## 3. High Unlabeled Calibration Data Requirements
The test-time adaptation setting is highly unrealistic:
* The authors stream data sequentially for **100 steps** with a batch size of **128**, requiring a total of **12,800 unlabeled images** from the target domains.
* For context, the entire test sets of MNIST, FashionMNIST, CIFAR-10, and SVHN contain only **10,000 images** each. 
* Requiring 12,800 calibration images to adapt a handful of merging coefficients ($\boldsymbol{\Lambda}$ and $\boldsymbol{\tau}$) violates the fundamental assumption of test-time adaptation, which should operate online under severe data scarcity (e.g., single-sample or few-sample regimes). If 12,800 images are available, one could easily perform standard joint fine-tuning or supervised adaptation rather than relying on a complex test-time model merging optimization.

## 4. Performance Degradation on Grayscale Tasks
The proposed adaptive method actually performs **worse** than static Task Arithmetic on both **MNIST** (20.00% vs 21.40%) and **FashionMNIST** (32.60% vs 35.40%). The authors attempt to brush this off as a "minor representational drift" and an "intriguing trend." In reality, this demonstrates a fundamental flaw in their unsupervised objective: the optimization is dominated by the more complex color datasets, causing the shared representation to warp and actively destroy the performance of simpler grayscale tasks.

## 5. Marginal Improvement over Static Baselines
The proposed ThermoMerge achieves an average accuracy of **29.05%**, which is only a marginal **+1.80%** improvement over standard, zero-overhead static Task Arithmetic (**27.25%**). 
* To achieve this tiny $+1.80\%$ gain, ThermoMerge requires **100 steps of gradient descent backpropagation**, **12,800 target domain images**, and **$K=4$ parallel forward passes** through the frozen expert models at each step to compute the partition functions.
* This massive computational overhead and latency during inference are completely unjustifiable for such a negligible performance improvement.

## 6. Failure of SimpleCNN Backbone
Under the "From-Scratch SimpleCNN Backbone" group:
* ThermoMerge gets **11.40%** on CIFAR-10 and **16.20%** on SVHN (essentially random chance).
* This proves that the method is entirely fragile and fails completely without a massive, pre-trained ImageNet backbone to provide mode connectivity. It cannot align representations on its own, undermining the claim that the framework "resolves" the gray-to-color collapse.
