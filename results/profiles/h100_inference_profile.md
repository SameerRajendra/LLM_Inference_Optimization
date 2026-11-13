# End-to-End Inference Decode Profile

- **Model**: `Qwen/Qwen3-8B`
- **Method**: `paged_fp8`
- **GPU**: NVIDIA H100 80GB HBM3
- **Profile steps**: `5` (warmup `3`, block size `64`)
- **Profiler source**: `cuda_events`
- **Generated (UTC)**: `2026-11-13T08:36:38Z`

## Summary Breakdown (`ms/step` and `% of step`)

| Stage | B=1, ctx=4K | B=16, ctx=4K | B=64, ctx=4K | B=1, ctx=64K |
|---|---:|---:|---:|---:|
| `Embedding` | 0.011 ms (0.0%) | 0.014 ms (0.0%) | 0.012 ms (0.0%) | 0.011 ms (0.0%) |
| `RMSNorm` | 9.492 ms (17.9%) | 9.670 ms (18.0%) | 7.159 ms (13.6%) | 9.111 ms (17.9%) |
| `QKV_Proj` | 2.533 ms (4.8%) | 2.415 ms (4.5%) | 2.000 ms (3.8%) | 2.383 ms (4.7%) |
| `RoPE_KV_Append` | 20.933 ms (39.5%) | 21.570 ms (40.0%) | 21.066 ms (39.9%) | 20.041 ms (39.3%) |
| `Paged_Attention` | 7.522 ms (14.2%) | 7.992 ms (14.8%) | 13.816 ms (26.2%) | 8.130 ms (15.9%) |
| `O_Proj` | 2.934 ms (5.5%) | 2.929 ms (5.4%) | 0.816 ms (1.5%) | 2.376 ms (4.7%) |
| `MLP_SwiGLU` | 5.413 ms (10.2%) | 5.506 ms (10.2%) | 5.188 ms (9.8%) | 5.356 ms (10.5%) |
| `LM_Head` | 0.458 ms (0.9%) | 0.459 ms (0.9%) | 0.465 ms (0.9%) | 0.451 ms (0.9%) |
| `Other` | 3.763 ms (7.1%) | 3.303 ms (6.1%) | 2.289 ms (4.3%) | 3.180 ms (6.2%) |
| **`Decode_Step` (Total)** | **53.060 ms (100.0%)** | **53.860 ms (100.0%)** | **52.810 ms (100.0%)** | **51.038 ms (100.0%)** |

## Per-Configuration Breakdown

### B=1, ctx=4K

| Stage | ms/step | % of step | Calls/step |
|---|---:|---:|---:|
| `Embedding` | 0.011 | 0.0% | 1.0 |
| `RMSNorm` | 9.492 | 17.9% | 109.0 |
| `QKV_Proj` | 2.533 | 4.8% | 36.0 |
| `RoPE_KV_Append` | 20.933 | 39.5% | 36.0 |
| `Paged_Attention` | 7.522 | 14.2% | 36.0 |
| `O_Proj` | 2.934 | 5.5% | 36.0 |
| `MLP_SwiGLU` | 5.413 | 10.2% | 36.0 |
| `LM_Head` | 0.458 | 0.9% | 1.0 |
| `Other` | 3.763 | 7.1% | - |
| **`Decode_Step` (Total)** | **53.060** | **100.0%** | **1.0** |

### B=16, ctx=4K

| Stage | ms/step | % of step | Calls/step |
|---|---:|---:|---:|
| `Embedding` | 0.014 | 0.0% | 1.0 |
| `RMSNorm` | 9.670 | 18.0% | 109.0 |
| `QKV_Proj` | 2.415 | 4.5% | 36.0 |
| `RoPE_KV_Append` | 21.570 | 40.0% | 36.0 |
| `Paged_Attention` | 7.992 | 14.8% | 36.0 |
| `O_Proj` | 2.929 | 5.4% | 36.0 |
| `MLP_SwiGLU` | 5.506 | 10.2% | 36.0 |
| `LM_Head` | 0.459 | 0.9% | 1.0 |
| `Other` | 3.303 | 6.1% | - |
| **`Decode_Step` (Total)** | **53.860** | **100.0%** | **1.0** |

### B=64, ctx=4K

| Stage | ms/step | % of step | Calls/step |
|---|---:|---:|---:|
| `Embedding` | 0.012 | 0.0% | 1.0 |
| `RMSNorm` | 7.159 | 13.6% | 109.0 |
| `QKV_Proj` | 2.000 | 3.8% | 36.0 |
| `RoPE_KV_Append` | 21.066 | 39.9% | 36.0 |
| `Paged_Attention` | 13.816 | 26.2% | 36.0 |
| `O_Proj` | 0.816 | 1.5% | 36.0 |
| `MLP_SwiGLU` | 5.188 | 9.8% | 36.0 |
| `LM_Head` | 0.465 | 0.9% | 1.0 |
| `Other` | 2.289 | 4.3% | - |
| **`Decode_Step` (Total)** | **52.810** | **100.0%** | **1.0** |

### B=1, ctx=64K

| Stage | ms/step | % of step | Calls/step |
|---|---:|---:|---:|
| `Embedding` | 0.011 | 0.0% | 1.0 |
| `RMSNorm` | 9.111 | 17.9% | 109.0 |
| `QKV_Proj` | 2.383 | 4.7% | 36.0 |
| `RoPE_KV_Append` | 20.041 | 39.3% | 36.0 |
| `Paged_Attention` | 8.130 | 15.9% | 36.0 |
| `O_Proj` | 2.376 | 4.7% | 36.0 |
| `MLP_SwiGLU` | 5.356 | 10.5% | 36.0 |
| `LM_Head` | 0.451 | 0.9% | 1.0 |
| `Other` | 3.180 | 6.2% | - |
| **`Decode_Step` (Total)** | **51.038** | **100.0%** | **1.0** |

### B=16, ctx=64K

_Skipped / failed: OOM_

### B=64, ctx=64K

_Skipped / failed: OOM_
