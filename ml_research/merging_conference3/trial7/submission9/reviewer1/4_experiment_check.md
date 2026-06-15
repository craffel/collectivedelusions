# 4. Experiment Check

## Evaluation of Experimental Setup

The paper evaluates SABLE across four distinct experimental environments, spanning from highly controlled synthetic sandboxes to high-dimensional physical foundation models:

1. **Analytical Coordinate Sandbox:** A 14-layer ($L=14$), 192-dimensional ($D=192$) coordinate-space sandbox simulating $K=4$ tasks calibrated to MNIST, FashionMNIST, CIFAR-10, and SVHN noise and difficulty profiles. It uses 1,000 samples per task for training, 16 for calibration, and 250 for testing.
2. **Physical CNN (Grayscale Image Data):** A 3-layer Convolutional Neural Network (two conv layers with max pooling, and a linear output projection layer) trained from scratch on MNIST and FashionMNIST grayscale images.
3. **Physical Deep Multi-Layer Perceptron (MLP):** A 4-layer Deep MLP ($\text{FC}_1 \in \mathbb{R}^{784 \times 128} \to \text{FC}_2 \in \mathbb{R}^{128 \times 64} \to \text{FC}_3 \in \mathbb{R}^{64 \times 32} \to \text{FC}_4 \in \mathbb{R}^{32 \times 10}$) trained on MNIST and FashionMNIST, evaluating multi-layer ensembling.
4. **High-Dimensional Foundation Feature Extractor (ResNet-18):** A frozen ResNet-18 pre-trained on ImageNet-1K, extracting 512-dimensional features from MNIST and FashionMNIST, topped with a 2-layer MLP adapter classification head ($\text{FC}_1 \in \mathbb{R}^{512 \times 128} \to \text{FC}_2 \in \mathbb{R}^{128 \times 10}$).

The multi-tiered experimental setup is highly commendable. It systematically bridges the gap between controlled, mathematically clean synthetic environments and actual high-dimensional physical computer vision representations.

## Choice of Baselines and Datasets
The baselines are comprehensive and highly appropriate for evaluating test-time model merging and streaming robustness:
- **Expert Ceiling (Oracle):** Establishes the performance upper bound.
- **Uniform Merging:** Establishes the static, non-adaptive parameter averaging baseline.
- **Linear Router (Unregularized, Parametric):** Represents classical calibrated routing.
- **PFSR (No MBH):** Represents the state-of-the-art in parameter-space parameter-free routing.
- **PFSR + MBH:** Represents the state-of-the-art systems-centric streaming wrapper.

The paper's discussion on the exclusion of PEFT-specific ensembling baselines (such as LoraHub or MoE-Adapters) is theoretically justified: LoraHub is static at test-time and requires target calibration data, while MoE-Adapters require heavy multi-task training phases, violating SABLE's non-parametric, calibration-free, and stateless constraints.

The choice of datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) is standard for model merging literature, representing a wide variety of domain shifts and task difficulties.

## Claims vs. Empirical Support

The empirical results provide exceptionally strong, direct support for all of the paper's central claims:

### 1. Robustness to Heterogeneous Streams (Immunity to Collapse)
- **Claim:** SABLE is natively immune to heterogeneity collapse under mixed-task batch environments.
- **Support:** In Table 5, under Heterogeneous Batching ($B=256$), SABLE Late Adaptation achieves **68.10%** accuracy, exhibiting exactly **0.00% collapse** compared to its homogeneous performance. In contrast, PFSR (No MBH) experiences a catastrophic **15.40% collapse** (dropping from 71.70% to 56.30%). In all physical CNN, MLP, and ResNet-18 experiments (Tables 1, 3, and 4), SABLE consistently maintains **0.00% collapse** across all ranks, centroid types, and configurations.

### 2. Bypassing Systems-Level Complexity
- **Claim:** SABLE achieves superior performance to the complex, stateful PFSR+MBH systems pipeline while completely stripping away systems-level serving dependencies.
- **Support:** In the Coordinate Sandbox, SABLE Late Adaptation achieves **68.10%** joint mean accuracy, outperforming PFSR+MBH (**67.20%**) under heterogeneous streams.
- **Systems Evidence:** End-to-end wall-clock serving latency benchmarks on an NVIDIA A100 GPU (Section 4.8) demonstrate that SABLE reduces average latency by **6.8$\times$** (12.4 ms vs 84.6 ms) and saves **36.4%** peak VRAM memory (412 MB vs 648 MB) compared to PFSR+MBH. This physical serving benchmark provides exceptionally high-signal, rigorous evidence of SABLE's systems-level superiority.

### 3. Layer-Dependent Hybrid-Rank Selection Protocol
- **Claim:** final classification heads suffer severely under low-rank constraints, which can be mitigated by keeping them full precision while hidden layers remain aggressive low-rank.
- **Support:** On ResNet-18 foundation features (Table 4), SABLE Strict at rank $r=2$ drops accuracy to 57.20% (Support-16) and 51.30% (Naive Zero-Data). Applying the hybrid-rank protocol (SABLE Hybrid) causes joint accuracy to surge to **62.10%** (+4.90% absolute gain) and **57.20%** (+5.90% absolute gain).
- **The Low-Rank Regularization Paradox:** SABLE Hybrid at $r=2$ consistently outperforms its $r=4$ counterpart (e.g. 62.10% vs 58.90% with Support-16). This empirically supports the theoretical claim that constraining intermediate hidden layers acts as a powerful regularizer, filtering out task-irrelevant noise and cross-task adapter interference.

### 4. Empirical Necessity of Soft Activation Blending
- **Claim:** Soft activation blending ($M \ge 2$) is strictly superior to hard routing ($M=1$) under overlapping task domains or highly ambiguous inputs.
- **Support:** Under 50-50 overlaid MNIST/FashionMNIST images (Table 2 and Table 4), SABLE Soft ($M=2$) dramatically and consistently outperforms SABLE Hard ($M=1$). In the physical CNN (Table 2), SABLE Soft at $r=10$ achieves **31.00%** joint class recall, while SABLE Hard collapses to **14.00%** because a hard single-expert selection is structurally incapable of retrieving both domains.
- **Destructive Representational Interference:** On ResNet-18 features (Table 4), the paper uncovers that soft ensembling must be paired with low-rank or highly regularized adapters ($r \le 2$) under confounded inputs, whereas high-capacity experts ($r=8$) require hard expert routing to prevent disjoint manifolds from colliding. This is a brilliant, highly nuanced empirical finding that perfectly bridges the gap between soft blending and hard routing.

### 5. Multi-Layer Scalability and Drift Control
- **Claim:** LoRA task updates represent minor task-specific residual corrections, allowing multi-layer activation blending without cumulative activation divergence.
- **Support:** In the 4-layer physical MLP (Table 5), layer-by-layer representational tracking of cosine similarities between SABLE blended activations and true oracle expert activations reveals exceptionally high values ($>0.83$) across all intermediate layers and logits, mathematically and empirically validating the drift control claims.
