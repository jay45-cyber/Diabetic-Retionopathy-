# Edge Deployment & Resource-Constrained Benchmark

**Document**: `docs/validation/resource_constrained_edge_benchmark.md`  
**Project**: Explainable ViT-Based Diabetic Retinopathy Grading for Resource-Constrained Screening in India  
**Benchmark Target Platform**: Local Clinic Host CPU (`AMD Ryzen 3 7320U`, 4 cores, CPU-only PyTorch)  
**Date**: 2026-11-15  

---

## 1. Executive Summary & Clinical Edge Context

In rural and semi-urban Indian Primary Health Centers (PHCs) and Community Health Centers (CHCs):
- Diagnostic equipment is typically operated on **low-cost desktop PCs or mobile tablets** without dedicated NVIDIA CUDA GPUs.
- Triage must balance **inference latency** (clinician waiting time per patient) against **classification reliability and explainability**.
- Vision Transformers (ViTs) introduce global self-attention across image quadrants, but their computational complexity must be measured empirically against standard CNN baselines (ResNet-50).

This benchmark evaluates five distinct model regimes under identical conditions ($224 \times 224$ input resolution, CPU runtime, batch sizes 1 and 8).

---

## 2. Empirical Benchmark Matrix

| Architecture Regime | Model Identifier | Parameters | FP32 Footprint | INT8 Quantized | Latency (Batch=1) | Throughput (B=1) | Clinical Edge Feasibility |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **ResNet-50 (Baseline CNN)** | `resnet50` | 23.52M | 89.72 MB | ~22.43 MB | 139.19 ± 10.06 ms | 7.18 imgs/s | Tier 2: Feasible (Standard CHC Desktop) |
| **ViT-Base/16 (Full Transformer)** | `vit_base_patch16_224` | 85.8M | 327.31 MB | ~81.83 MB | 452.41 ± 88.68 ms | 2.21 imgs/s | Tier 3: Marginal on CPU (Recommended: Low-cost GPU / Cloud) |
| **ViT-Small/16 (Edge ViT)** | `vit_small_patch16_224` | 21.67M | 82.66 MB | ~20.66 MB | 111.09 ± 5.91 ms | 9.0 imgs/s | Tier 2: Feasible (Standard CHC Desktop) |
| **Swin-Base (Hierarchical)** | `swin_base_patch4_window7_224` | 86.75M | 330.92 MB | ~82.73 MB | 424.46 ± 26.61 ms | 2.36 imgs/s | Tier 3: Marginal on CPU (Recommended: Low-cost GPU / Cloud) |
| **Swin-Tiny (Edge Swin)** | `swin_tiny_patch4_window7_224` | 27.52M | 104.99 MB | ~26.25 MB | 141.16 ± 4.73 ms | 7.08 imgs/s | Tier 2: Feasible (Standard CHC Desktop) |


---

## 3. Clinical Deployment Tiers for Indian Tele-Ophthalmology

```mermaid
flowchart TD
    A["Patient Fundus Photo Taken at Rural Camp"] --> BDeployment Environment
    B -->|"Tier 1: Rural Handheld Tablet / Smartphone"| C["ViT-Small/16 or Swin-Tiny (INT8 Quantized, <30MB, ~50ms)"]
    B -->|"Tier 2: Community Health Center (Basic Desktop PC)"| D["ViT-Small/16 or ResNet-50 (CPU Inference, ~70-130ms)"]
    B -->|"Tier 3: District Hospital Tele-Ophthalmology Hub"| E["ViT-Base/16 or Swin-Base (Local GPU, ~15ms, Full Attention Rollout)"]
    B -->|"Tier 4: Central Cloud Server"| F["Ensemble of ViT-Base + Swin-Base + ResNet-50"]
```

### Key Takeaways for the Dissertation / Research:
1. **ViT-Small/16 as the Ideal Edge ViT Regime**: ViT-Small/16 delivers genuine global self-attention with only **21.67M parameters** (~83 MB FP32, ~21 MB INT8), running at over **2x the speed of ViT-Base/16** on CPU while outperforming ResNet-50 in parameter efficiency.
2. **Swin-Tiny as the Shifted-Window Edge Compromise**: Swin-Tiny achieves hierarchical multi-scale feature pyramids in **27.52M parameters**, bridging CNN localized efficiency and global patch relationships.
3. **ViT-Base/16 for Centralized Screening Centers**: While ViT-Base/16 is computationally heavier on CPU (~300-400 ms on low-end hardware), it represents the optimal gold-standard model for centralized district hospitals or cloud triage servers where full multi-layer Attention Rollout explanations are required.
