# 4. Experimental Check and Empirical Support

### Evaluation Setup
The empirical evaluations are conducted on very small-scale, low-complexity benchmarks:
* A 3-layer MLP with hidden dimension $d = 256$ on Split-MNIST.
* A 3-layer MLP with hidden dimension $d = 256$ on Split-CIFAR-10.
* A custom micro-Vision Transformer (ViT) with sequence length 16 and embedding dimension $d=32$ on Split-MNIST.

While the authors provide a mathematical scale analysis showing that noise propagation worsens quadratically with dimension $d$, they do not provide any experiments on actual, modern-scale models (e.g., standard ViT-B, ResNets, or standard LLMs) where model merging is practically used.

### Support of Claims
Crucially, the experimental results in the paper **do not support** the claim that geometric manifold merging is practically superior to simple Euclidean methods. In fact, they demonstrate the opposite:

1. **Standard Euclidean Training (Table 1)**:
   - Simple Euclidean **Task Arithmetic (TA, $\lambda = 0.3$)** achieves **91.11%** average accuracy.
   - The authors' proposed **RIMO-Pruned** ($\text{keep}=0.2, \rho_{\text{scale}}=0.2$) achieves **90.47%**.
   - Simple Task Arithmetic is infinitely easier to implement and outperforms the proposed complex manifold method.

2. **Orthogonal Regularization Training (Table 2)**:
   - Simple Euclidean **Task Arithmetic (TA, $\lambda = 1.0$)** achieves **94.00%** average accuracy.
   - The proposed **RIMO-Pruned** ($\text{keep}=0.1, \rho_{\text{scale}}=1.0$) achieves **91.49%**.
   - Again, Task Arithmetic outperforms the proposed manifold method by a wide margin (**2.51%** absolute gap), despite the models being trained under the soft orthogonal constraint.

3. **Multi-Task Scaling ($N = 5$ Experts, Table 5)**:
   - The authors argue that Euclidean averaging suffers from representational magnitude decay for larger $N$.
   - However, in their own 5-task experiment, **Task Arithmetic ($\lambda = 0.4$)** achieves **91.48%** average accuracy.
   - The proposed **RIMO-Pruned ($\text{keep} = 0.4$)** achieves **91.46%**.
   - Even when $N=5$, simple Task Arithmetic (with a simple scale parameter) is still slightly better and vastly simpler.

The empirical evidence clearly indicates that the immense mathematical and computational complexity introduced by RIMO-Pruned is completely unjustified in practice. A single line of Euclidean addition (Task Arithmetic) consistently matches or outperforms this entire geometric apparatus.
