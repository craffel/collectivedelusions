# 2. Novelty Check

## Key Novel Aspects & "Delta" from Prior Work
The proposed SPS-ZCA framework introduces several elegant and lightweight innovations that distinguish it from prior work in weight-space and activation-space model merging:

1. **Delta from Static Weight Merging (Task Arithmetic, TIES-Merging, DARE):**
   * *Prior Work:* Merges model weights *offline* prior to deployment. This creates a single set of static weights, forcing a compromise that leads to "heterogeneity collapse" when processing mixed-task streams.
   * *SPS-ZCA:* Avoids weight merging altogether. Instead, it performs **on-the-fly activation-space blending** inside a single parallel forward pass. This preserves specialized, task-specific pathways while serving heterogeneous streams concurrently, with zero offline parameter compromise.

2. **Delta from Dynamic Weight Merging SOTA (PFSR + MBH):**
   * *Prior Work (MBH):* Bypasses parameter interference by partitioning mixed-task input batches into homogeneous micro-batches and routing each to its respective expert. This requires up to $K$ sequential backbone passes, introducing a linear latency penalty ($O(K)$) that scales with the number of active tasks, which is unacceptable for resource-constrained edge CPUs.
   * *SPS-ZCA:* Implements sample-wise activation-space dynamic blending (SPS), running the heavy base model backbone exactly *once* for the entire batch. Task-specific LoRA activations are scaled and added sample-wise, which scales compute at a flat $O(1)$ constant backbone latency.

3. **Delta from Traditional Dynamic Routers:**
   * *Prior Work (PFSR):* Routes inputs by projecting features against classification heads. This is highly specific to task-specific label spaces, making it noisy under domain shifts and out-of-distribution (OOD) tasks. More critically, traditional penultimate-layer routers suffer from the "Routing Paradox" (requiring late-stage representations to compute routing weights, forcing the system to run the backbone twice).
   * *SPS-ZCA:* Proposes **Zero-Shot Centroid Alignment (ZCA)**. It projects inputs onto robust task centroids pre-computed from a tiny, 64-sample calibration split in the pre-trained backbone's **early representation space (Layer 3)**.
   * *Resolving the Routing Paradox:* To resolve the temporal circular dependency of early routing, the authors propose a hardware-software co-designed execution layout. They restrict LoRA adapters strictly to layers 4 to $L$, leaving layers 1--3 frozen and shared. This allows the system to execute the first 3 layers task-agnostically with zero mismatch, extract Layer 3 CLS representations, compute routing weights, and blend the subsequent layers dynamically in a single forward pass.

4. **Delta from High-Resource Multi-Tenant PEFT Frameworks (S-LoRA, Punica):**
   * *Prior Work:* Designed for high-throughput multi-tenant LLM serving on massive GPU clusters. They rely on heavy systems scheduling, custom CUDA memory paging, and large server footprints.
   * *SPS-ZCA:* Operates on a fundamentally different paradigm—merging task-specific activations directly inside the shared neural layers in a single pass. It is training-free, parameter-free, compiler-friendly, and lightweight enough for resource-constrained edge CPUs.

5. **Delta from Complex OOD Detectors:**
   * *Prior Work:* Employs expensive parametric OOD classifiers.
   * *SPS-ZCA:* Fits a diagonal Gaussian Mixture Model (GMM) with only $M=2$ components directly over the **4D routing similarity coordinates**. This low-dimensional density estimation provides high-precision OOD detection (95.2% true rejection) at negligible computational and parameter cost.

## Characterization of Novelty
The novelty of SPS-ZCA is **significant, highly elegant, and conceptually clean**. 

Rather than chasing marginal performance gains by introducing complex learned routing layers, multi-stage fine-tuning schedules, or expensive scheduling layers, the paper solves a difficult, systems-heavy deployment bottleneck using **pure representation geometry** and **lightweight, training-free operations**:
- It exploits the existing visual/textual semantic representations of pre-trained models.
- It leverages the physical abstraction of deep neural networks—the fact that early layers capture generic visual/textual features, while mid-to-late layers specialize.
- Restricting LoRA to mid-to-late layers (blocks 4+) and using early representations (Layer 3) to route is a brilliantly simple architectural choice. It completely resolves the routing paradox while incurring a practically negligible joint accuracy degradation of only **-0.02%** absolute.
- The calibrations (UNC, IDC) and OOD density estimation (Coordinate GMM) are performed entirely in the low-dimensional routing coordinate space ($\mathbb{R}^K$), which is extremely data-efficient and avoids high-dimensional overfitting.

By prioritizing simplicity, training-free calibration, and hardware efficiency, the authors champion a minimalist systems-ML co-design that bridges the gap between theoretical FLOP savings and real-world edge execution.
