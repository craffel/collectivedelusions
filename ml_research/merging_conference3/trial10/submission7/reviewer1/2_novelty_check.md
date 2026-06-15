# Novelty Check

## Key Novel Aspects
The paper introduces several key novel insights and technical formulations to the field of dynamic model merging:
1. **The State Contamination Bottleneck:** This is the first work to expose that temporal smoothing in stateful ensembling routers (e.g., ChemMerge, PAC-Kinetics) breaks down under interleaved multi-tenant workloads. The formalization of this "cross-talk" as a key deployment blocker is a novel and highly relevant contribution.
2. **Tenant-Decoupled Stateful Routing (TDSR):** The introduction of virtual state slots to isolate recurrence is simple yet highly novel in this context. Rather than routing with a single global state, the state pool isolates temporal smoothing dynamics across sessions.
3. **Coordinate-Argmax Assignment for Implicit Routing:** In the tagless setting, instead of proposing an overly complex, uninterpretable neural network or dynamic online clustering algorithm, the authors initialize and fix slot centroids as standard basis orthogonal vectors. They prove that the cosine similarity against these centroids simplifies exactly to choosing the maximum coordinate activation ($\arg\max_m e_{m, t}$). This mathematical simplification is a beautiful and highly practical mechanism that operates in sub-nanoseconds.
4. **Tenant-Specific Session-Step Decay & Dual-Clock Decay:** Rather than uniform exponential decay over global steps, the paper formulates logical step decay (inactive slots hold state constant to prevent washout) and reconciles this with background physical timeouts for garbage collection.
5. **Intra-Session Jitter Disentanglement:** The paper exposes a statistical fallacy in how routing jitter was previously measured on interleaved streams. By separating global inter-session jitter (which must remain high to track interleaved streams) from intra-session jitter (which measures temporal stability within a session), it provides a correct, high-fidelity metric for ensembling stability.

## Delta from Prior Work
- **From SABLE (Stateless):** SABLE computes ensembling weights sample-by-sample, which makes it responsive but highly sensitive to sample-level activation noise, resulting in severe routing jitter. TDSR retains the coordinate-projection foundation of SABLE but introduces stateful memory.
- **From PAC-Kinetics and ChemMerge (Global Stateful):** PAC-Kinetics and ChemMerge use global temporal smoothing (via recurrence or chemical kinetics ODEs) to suppress SABLE's jitter. However, they maintain a single global state, causing state contamination across tenants on interleaved streams. TDSR decouples this stateful memory into virtual slots.
- **From Serving Frameworks (S-LoRA, Punica):** S-LoRA and Punica focus on low-level GPU memory scheduling and custom kernels to execute multiple LoRA adapters. They are completely agnostic to the temporal dynamics of ensembling. TDSR operates at the scheduling level to manage stateful routing states across interleaved requests.

## Characterization of Novelty
The novelty of this work is **significant, elegant, and highly pragmatic**. 

Rather than introducing heavy, complex neural network routers, complex training schemes, or uninterpretable clustering monoliths, the authors solve a major deployment blocker using fundamental systems-engineering and mathematical design principles. They leverage the existing geometric structure of the task-specific coordinate spaces to achieve slot isolation and implicit clustering. The coordinate-argmax simplification is an exceptionally elegant finding that demonstrates that high-throughput routing does not need to compromise on simplicity. This is a shining example of "less is more" in machine learning research, prioritizing elegant and simple mechanisms that are highly effective.
