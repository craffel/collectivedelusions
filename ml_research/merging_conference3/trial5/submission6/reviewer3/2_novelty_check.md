# 2. Novelty Check

## Assessment of Key Novel Aspects & Conceptual "Delta"
The paper presents **SLD-Merge** as a novel framework for "dynamic model merging." However, from a rigorous conceptual perspective, the novelty is highly questionable because SLD-Merge is not actually a model merging method in the established scientific sense.

In traditional weight-space model merging, the objective is to fuse multiple specialized expert models into a *single set of unified weights* $W_{merged}$ that can process all inputs. Because the weights are physically merged, a single forward pass $X W_{merged}$ handles all inputs, avoiding the need to store separate expert models in memory.

SLD-Merge completely departs from this paradigm:
1. It extracts specialized task vectors $V_k$ and compresses them offline using Singular Value Decomposition (SVD) to create a set of separate, low-rank adapters $(B_k, A_k)$ for each of the $K$ tasks.
2. During inference, instead of using a merged set of weights, it maintains $K$ separate sets of low-rank adapters in memory.
3. It uses a router to dynamically choose which of the $K$ separate low-rank adapters to execute for a given input sample.

This is **not model merging**; it is literally a **multi-LoRA Mixture-of-Experts (MoE)** or a multi-adapter routing framework (similar to existing works like *LoRA-Hub*, *LoRA-MoE*, or *LLaVA-MoE*). 
The "delta" from prior weight-merging work is that SLD-Merge simply bypasses the fundamental challenge of weight-space merging by keeping the parameters of the experts completely separate. Because the parameters are never merged, it trivially avoids "heterogeneity collapse" and "batch-dependency." Comparing SLD-Merge to actual weight-merging baselines (like QWS-Merge or AdaMerging) and claiming it "resolves" their bottlenecks is an unfair comparison, as SLD-Merge operates under a different paradigm (MoE) that retains separate task parameter pathways.

## SVD on Task Vectors
Performing SVD on task vectors or weight differences to obtain low-rank approximations is a well-established technique in model compression and Parameter-Efficient Fine-Tuning (PEFT). Standard post-hoc decomposition methods and recent merging methods (e.g., *Reversible Model Merging (RMM)*, *TSV-Merge*, or *LORE-Merging*) have extensively leveraged SVD to decompose or align task weight updates. The application of SVD here is a straightforward application of classic linear algebra and does not introduce new theoretical or methodological insights.

## Activation-Space Mean Initialization
The proposed **Activation-Space Mean Initialization** sets the routing basis vectors $\Phi_k^{(l)}$ to the empirical mean activation representing each task. While presented as a major novelty, this is conceptually identical to classical prototype-based learning, prototypical networks, and nearest-centroid classifiers. Aligning basis vectors with activation centroids is a standard and intuitive heuristic in clustering and classification, representing a very incremental engineering contribution.

## Characterization of Novelty
The overall novelty of this paper is **incremental**. The paper takes existing, well-known concepts—SVD-based low-rank compression, prototype-based activation centroid routing, and multi-LoRA Mixture-of-Experts—and re-packages them under the umbrella of "dynamic model merging." While the combination is pragmatic and delivers good engineering performance on the evaluated benchmarks, the conceptual novelty is limited and overclaimed.
