# Idea Proposal: Winner-Take-All Sign Election (WTA-Sign)

## 1. Persona Alignment
This project is perfectly aligned with the traits and goals of **The Minimalist** persona. Modern model merging methods like TIES-Merging introduce significant complexity—requiring hyperparameter-dependent weight trimming (pruning the bottom $k\%$ of values), sign consensus voting across all models, filtering non-conforming parameters, and then rescaling the remaining parameters to compensate for lost energy. 

In contrast, our proposed **Winner-Take-All Sign Election (WTA-Sign)** method strips away these arbitrary and convoluted heuristics. Our key minimalist insights are:
1. **Confidence as Magnitude:** The task vector containing the largest absolute update at a given parameter index is the most confident about that specific weight.
2. **Occam's Razor Conflict Resolution:** Instead of computing a complex voting consensus across all models, we let the sign of this "winning" maximum-update model dictate the direction of the merge.
3. **Parameter-Free and Training-Free:** WTA-Sign requires zero hyperparameter tuning (no trim thresholds, no random seeds, no scaling multipliers) and completely avoids test-time optimization or backpropagation. It is a pure, elegant, closed-form mathematical operation.

---

## 2. Core Techniques
WTA-Sign introduces an elegant, closed-form, training-free mechanism for weight-space conflict resolution. It is built upon and references the following foundational methods:
- **Task Vectors & Task Arithmetic (Ilharco et al., 2022):** We represent task-specific knowledge as task vectors ($\tau_k = \Theta_k - \Theta_{\text{pre}}$) capturing the deviation from a pre-trained base model.
- **TIES-Merging (Yadav et al., 2023):** We target the sign conflict problem (where different models try to push a parameter in opposite directions), but replace TIES' multi-step voting, trimming, and rescaling with a direct winner-take-all election.

---

## 3. Mathematical Formulation
Let $\Theta_{\text{pre}} \in \mathbb{R}^D$ represent the parameters of a shared pre-trained base model, and let $\Theta_k \in \mathbb{R}^D$ represent the parameters of $K$ independently fine-tuned expert models for tasks $k \in \{1, \ldots, K\}$.

### Step 1: Compute Task Vectors
For each expert model, we compute the task vector $\tau_k \in \mathbb{R}^D$:
$$\tau_k = \Theta_k - \Theta_{\text{pre}}$$

### Step 2: Winner-Take-All Indexing
For each parameter index $j \in \{1, \ldots, D\}$, we identify the index $k^*(j)$ of the task vector that contains the maximum absolute update:
$$k^*(j) = \arg\max_{k \in \{1, \ldots, K\}} \left| \tau_{k, j} \right|$$

### Step 3: Sign Election
The elected sign $s_j \in \{-1, 0, 1\}$ for parameter $j$ is the sign of the winning update:
$$s_j = \text{sign}\left( \tau_{k^*(j), j} \right)$$

### Step 4: Conformity Masking
For each task $k$ and parameter $j$, we construct a binary mask $M_{k, j}$ indicating whether the expert's update direction conforms with the elected sign $s_j$:
$$M_{k, j} = \mathbb{I}\left( \text{sign}\left( \tau_{k, j} \right) == s_j \right)$$
This mask dynamically filters out any updates that oppose the elected winner's sign, neutralizing destructive interference.

### Step 5: Conformity Averaging
The merged task vector $\tau_{\text{merged}} \in \mathbb{R}^D$ is computed as the element-wise average of only the conforming updates:
$$\tau_{\text{merged}, j} = \frac{\sum_{k=1}^K M_{k, j} \cdot \tau_{k, j}}{\sum_{k=1}^K M_{k, j} + \epsilon}$$
where $\epsilon = 10^{-8}$ is a small numerical stabilizer. If all updates are zero, $\tau_{\text{merged}, j} = 0$.

### Step 6: Parameter Re-Integration
The final merged parameters $\Theta_{\text{merged}} \in \mathbb{R}^D$ are constructed by adding the merged task vector back to the pre-trained base model, scaled by a global coefficient $\lambda$:
$$\Theta_{\text{merged}} = \Theta_{\text{pre}} + \lambda \cdot \tau_{\text{merged}}$$

---

## 4. Architecture Specifications
WTA-Sign is architecturally agnostic and operates directly on standard neural network weight tensors. 
- **Input Representations:**
  - Base model weights $\Theta_{\text{pre}}$ and $K$ expert model task vectors $\{\tau_k\}_{k=1}^K$ loaded from PyTorch state dicts (including transformer multi-head attention weights, MLP layer weights, and layer norms).
- **Intermediate Representations:**
  - An absolute maximum index tensor $k^* \in \{1, \ldots, K\}^D$ identifying the winning model per parameter.
  - An elected sign tensor $s \in \{-1, 0, 1\}^D$ storing the direction of the winning model.
  - A conformity mask $M \in \{0, 1\}^{K \times D}$ representing the sign agreement of each expert.
- **Output Representation:**
  - The final merged model weight tensor $\Theta_{\text{merged}}$ of shape identical to $\Theta_{\text{pre}}$. WTA-Sign does not alter the model architecture, add parameter adapters, or introduce any inference latency.

---

## 5. Baselines
We will validate our proposed method against the following standard model merging baselines:
1. **Task Arithmetic (Ilharco et al., 2022):** A simple linear sum of task vectors ($\tau_{\text{merged}} = \sum_k \tau_k$). This is the standard baseline but suffers from high interference when updates oppose each other.
2. **TIES-Merging (Yadav et al., 2023):** The leading sign-conflict resolution baseline. It uses heuristic trimming (top $k\%$), sign voting, non-conforming zeroing, and scaling. WTA-Sign aims to match or exceed TIES' performance while completely removing its trimming and scaling hyperparameters.
3. **Model Soups (Wortsman et al., 2022):** Direct parameter averaging ($\Theta_{\text{merged}} = \frac{1}{K} \sum_k \Theta_k$), representing uniform weight combination.

---

## 6. Step-by-Step Interaction
1. **Load Checkpoints:** Load PyTorch weight dictionaries for the pre-trained base model $\Theta_{\text{pre}}$ and the $K$ fine-tuned task experts $\Theta_1, \ldots, \Theta_K$.
2. **Extract Updates:** For each layer, subtract $\Theta_{\text{pre}}$ from each expert to extract the raw task vectors $\tau_1, \ldots, \tau_K$.
3. **Find Maximum Magnitudes (Winner Election):** For each parameter index, compute absolute values and run `argmax` over the $K$ task vectors. This elects the winning expert $k^*(j)$ for every parameter.
4. **Determine Elected Signs:** Retrieve the sign of the winning expert's update at each index $s_j = \text{sign}(\tau_{k^*(j), j})$.
5. **Apply Conformity Filter:** Generate a boolean mask across the $K$ models. For each expert, check if its parameter update sign agrees with the elected sign $s_j$. Keep the value if it agrees; mask it to zero if it opposes.
6. **Average Active Updates:** For each parameter index, average the non-masked expert updates.
7. **Scale and Merge:** Multiply the resulting merged task vector by the global scaling parameter $\lambda$, and add it back to the base model weights to yield $\Theta_{\text{merged}}$.
8. **Inference:** Evaluate the merged model $\Theta_{\text{merged}}$ across all $K$ downstream tasks.
