# Calibration-Free Zero-Shot Task Clustering with Online Centroid Refinement (CF-ZTC)

## 1. Persona Alignment
In real-world machine learning deployment, obtaining labeled calibration datasets ($|\mathcal{C}_k|=64$ samples per task) is a major operational bottleneck. Practitioners frequently deploy multi-task systems on edge devices under tight data-privacy regulations, or where streaming tasks are completely unlabeled and dynamic. 
Our proposed framework, **CF-ZTC (Calibration-Free Zero-Shot Task Clustering with Online Centroid Refinement)**, directly aligns with the traits and goals of **The Pragmatist**:
1. **Data-Free Deployment:** It eliminates any dependency on offline calibration data or task labels, enabling a pure plug-and-play dynamic ensembling experience.
2. **Zero-Backpropagation Latency:** All clustering, alignment, and tracking operations are performed in the forward-pass activation space of the model. It introduces zero backward-pass training, keeping inference latency extremely low and constant.
3. **Resilience to Domain Shift:** Real-world streams are notoriously noisy and undergo covariate shift. By tracking and updating centroids on-the-fly, CF-ZTC dynamically adapts to domain drift without requiring manual recalibration or retraining.

## 2. Core Techniques
CF-ZTC introduces three core geometrically-grounded techniques integrated with Single-Pass Activation-Space Dynamic Blending (SPS):
1. **Online Unsupervised Stream Clustering:** Instead of pre-computing task centroids offline, CF-ZTC dynamically discovers $K$ task centroids $\{\mu_c^{(3)}\}_{c=1}^K$ directly from the early representation space (Layer 3) of incoming stream activations using online, stream-wise K-Means clustering.
2. **Zero-Shot Cluster-to-Expert Alignment:** To solve the unsupervised "Cluster-to-Expert Alignment Problem", we feed the discovered centroids through the mid-to-late layers of each task expert in the registry. Because specialized experts exhibit maximum confidence (minimum prediction entropy) on their own task domain, we construct an entropy-based cost matrix and solve the bipartite matching problem via Hungarian matching. This requires zero calibration data or labels.
3. **Unsupervised Centroid Refinement (Continuous EMA Tracking):** To handle real-world covariate shifts and representational drift, we continuously refine the aligned task centroids on-the-fly using an online, confidence-weighted exponential moving average (EMA) tracker of incoming activations.

## 3. Mathematical Formulation

### A. Online Unsupervised Stream Clustering
For each incoming sample $x_b$ in batch step $t$, we execute the shared, adapter-free early-stage backbone (Layers 1--3) of the pre-trained base model $f_\theta$ to extract its early-stage representation $h_b^{(3)} = \text{Pool}(f_{\theta, \text{block3}}(x_b)) \in \mathbb{R}^D$.
We compute the cosine similarity between $h_b^{(3)}$ and the current $K$ cluster centroids $\{\mu_{c, t}^{(3)}\}_{c=1}^K$:
$$u_{c, b} = \text{cos\_sim}(h_b^{(3)}, \mu_{c, t}^{(3)}) = \frac{h_b^{(3)} \cdot \mu_{c, t}^{(3)}}{\|h_b^{(3)}\|_2 \|\mu_{c, t}^{(3)}\|_2}$$

For the first $T$ batch steps, we dynamically group and partition the stream. Each sample $h_b^{(3)}$ is assigned to its nearest cluster:
$$c^*(b) = \arg\max_{c \in \{1, \dots, K\}} u_{c, b}$$

We update the cluster centroids using an online running average with tracking rate $\beta \in [0, 1]$:
$$\mu_{c, t+1}^{(3)} \leftarrow \text{Normalize}\left( (1 - \beta) \mu_{c, t}^{(3)} + \beta \bar{h}_c^{(3)} \right)$$
where $\bar{h}_c^{(3)} = \frac{1}{|\mathcal{B}_c|} \sum_{b \in \mathcal{B}_c} h_b^{(3)}$ is the average representation of samples in the current batch assigned to cluster $c$.

### B. Bipartite Cluster-to-Expert Alignment
At batch step $T$ (once the discovered centroids have stabilized), we perform a one-time zero-shot alignment.
Let $E_k$ be the $k$-th task-specific expert in our registry. Passing a representation $h$ through the remaining layers (Layers 4 to $L$) of expert $E_k$ yields a prediction probability distribution:
$$p_k(h) = \text{Softmax}\left( E_k.\text{head}(E_{k, \text{layers } 4 \dots L}(h)) \right) \in \Delta^{Y_k - 1}$$
where $Y_k$ is the class dimensionality of expert $E_k$.
The prediction confidence is measured using the Shannon entropy:
$$H(p_k(h)) = - \sum_{y=1}^{Y_k} p_{k, y}(h) \log p_{k, y}(h)$$

We construct a bipartite matching cost matrix $C \in \mathbb{R}^{K \times K}$, where the cost of aligning cluster centroid $\mu_c^{(3)}$ with expert $E_k$ is the prediction entropy:
$$C_{c, k} = H(p_k(\mu_c^{(3)}))$$

The optimal bijective alignment mapping $\pi: \{1, \dots, K\} \rightarrow \{1, \dots, K\}$ minimizes the total matching cost:
$$\pi^* = \arg\min_{\pi} \sum_{c=1}^K C_{c, \pi(c)}$$
This is solved using the classical Hungarian (Kuhn-Munkres) algorithm. Once solved, we map the task centroids to the correct experts: $\mu_k^{(3)} = \mu_{\pi^{-1}(k)}^{(3)}$.

### C. Continuous EMA Centroid Refinement
During subsequent serving, we continuously refine the task centroids on-the-fly to follow streaming domain drift and covariate noise:
$$\mu_{k, t+1}^{(3)} \leftarrow \text{Normalize}\left( (1 - \eta) \mu_{k, t}^{(3)} + \eta \sum_{b=1}^B \alpha_{k, b} h_b^{(3)} \right)$$
where $\eta \in [0, 1]$ is the continuous online adaptation rate, and $\alpha_{k, b}$ are the soft routing coefficients:
$$\alpha_{k, b} = \frac{\exp(u_{\pi^{-1}(k), b} / \tau)}{\sum_{j=1}^K \exp(u_{\pi^{-1}(j), b} / \tau)}$$

## 4. Architecture Specifications
- **Base Model Backbone:** Frozen Vision Transformer (`vit_tiny_patch16_224`, $L=14$ total block groups including patch embedding and head, feature dimension $D=192$).
- **Early-stage Blocks (Layers 1--3):** Frozen and executed completely task-agnostically with zero LoRA adapters.
- **Mid-to-Late Blocks (Layers 4--12):** Low-Rank Adaptation (LoRA) adapters (rank $r=8$) inserted into the query, key, value, and output projection layers.
- **Centroid Vector Representation:** 192-dimensional vector extracted via global average pooling at the output of Layer 3.
- **Classification Heads:** Task-specific heads attached at the output of the final block layer (Layer 12).

## 5. Baselines
We evaluate CF-ZTC against four critical baselines:
1. **Expert Ceiling (0 params):** isolated task-specific execution with perfect routing.
2. **Static Uniform Merging (0 params):** weight-space uniform average of all expert adapters.
3. **SPS-ZCA (Offline SOTA, 0 params):** uses *offline labeled calibration sets* ($|\mathcal{C}_k|=64$) to pre-compute task centroids. This represents the empirical upper-bound for training-free dynamic ensembling.
4. **PFSR (0 params):** non-parametric classification-head-dependent projection routing.

## 6. Step-by-Step Interaction
1. **Initialization:** The edge device begins receiving a heterogeneous stream of unlabeled and mixed-task inputs $X$.
2. **Online Unsupervised Clustering Phase (Steps 1 to $T$):**
   - For each input $x_b$ in the batch, execute Layers 1--3 of the base model to extract pooled representation $h_b^{(3)}$.
   - Assign $h_b^{(3)}$ to its nearest cluster and update the centroids $\{\mu_c^{(3)}\}_{c=1}^K$ using the online K-Means update rule.
3. **One-Time Zero-Shot Alignment (At Step $T$):**
   - Pass each discovered centroid $\mu_c^{(3)}$ through the remaining layers (Layers 4--12) and head of all $K$ experts.
   - Compute the Shannon prediction entropy matrix $C \in \mathbb{R}^{K \times K}$.
   - Solve the bipartite matching problem to assign each centroid to its correct expert adapter, mapping $\mu_k^{(3)} = \mu_{\pi^{-1}(k)}^{(3)}$.
4. **Serving and Refinement Phase (Steps $> T$):**
   - For each input sample $x_b$, extract $h_b^{(3)}$ and compute soft routing coefficients $\alpha_{k, b}$ using the aligned centroids $\mu_k^{(3)}$.
   - Execute mid-to-late layers (Layers 4--12) using Single-Pass Activation-Space Dynamic Blending (SPS) based on $\alpha_{k, b}$.
   - Update the aligned task centroids $\mu_k^{(3)}$ on-the-fly via a confidence-weighted exponential moving average to track representation drift and covariate noise in real-time.
