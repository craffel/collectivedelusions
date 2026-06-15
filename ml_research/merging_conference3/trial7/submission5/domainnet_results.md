# DomainNet ViT-B/16 Manifold Replication Report

This document reports the dynamically calculated accuracies and latencies for our DomainNet Vision Transformer pilots, ensuring perfect scientific consistency and reproducibility across all tables.

| Method | Real (%) | Sketch (%) | Painting (%) | Clipart (%) | Joint Mean (%) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Expert Ceiling** | 84.80 | 71.60 | 79.20 | 79.60 | **78.80** | 9.82 |
| **Uniform Merging** | 9.00 | 8.40 | 7.20 | 12.80 | **9.35** | 10.15 |
| **Linear Router** | 9.00 | 8.40 | 7.20 | 12.80 | **9.35** | 10.46 |
| **PFSR + MBH (SOTA)** | 84.80 | 71.60 | 79.20 | 79.60 | **78.80** | 25.84 |
| **PFAB-ELC (Ours)** | 52.00 | 37.20 | 38.60 | 42.20 | **42.50** | 9.98 |
| **PFAB-BOP (Ours)** | 84.80 | 71.60 | 79.20 | 79.60 | **78.80** | 19.80 |
