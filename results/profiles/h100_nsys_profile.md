# End-to-End Inference Decode Profile

- **Model**: `Qwen/Qwen3-8B`
- **Method**: `paged_fp8`
- **GPU**: NVIDIA H100 80GB HBM3
- **Profile steps**: `5` (warmup `3`, block size `64`)
- **Profiler source**: `nsys stats --report nvtx_sum (--sample=none --cpuctxsw=none)`
- **Generated (UTC)**: `2026-11-13T21:39:29Z`

## Summary Breakdown (`ms/step` and `% of step`)

| Stage | B=1, ctx=4K | B=16, ctx=4K | B=64, ctx=4K | B=1, ctx=64K |
|---|---:|---:|---:|---:|
| `Embedding` | 0.039 ms (0.1%) | 0.033 ms (0.1%) | 0.045 ms (0.1%) | 0.036 ms (0.1%) |
| `RMSNorm` | 14.546 ms (24.5%) | 13.651 ms (23.9%) | 14.770 ms (24.8%) | 14.618 ms (24.7%) |
| `QKV_Proj` | 3.086 ms (5.2%) | 2.868 ms (5.0%) | 2.911 ms (4.9%) | 3.064 ms (5.2%) |
| `RoPE_KV_Append` | 25.705 ms (43.3%) | 25.130 ms (44.0%) | 26.159 ms (43.9%) | 25.662 ms (43.3%) |
| `Paged_Attention` | 7.684 ms (13.0%) | 7.510 ms (13.2%) | 7.600 ms (12.7%) | 7.622 ms (12.9%) |
| `O_Proj` | 2.702 ms (4.6%) | 2.597 ms (4.5%) | 2.438 ms (4.1%) | 2.694 ms (4.5%) |
| `MLP_SwiGLU` | 4.064 ms (6.9%) | 3.891 ms (6.8%) | 4.182 ms (7.0%) | 4.068 ms (6.9%) |
| `LM_Head` | 0.036 ms (0.1%) | 0.034 ms (0.1%) | 0.038 ms (0.1%) | 0.037 ms (0.1%) |
| `Other` | 1.462 ms (2.5%) | 1.378 ms (2.4%) | 1.483 ms (2.5%) | 1.439 ms (2.4%) |
| **`Decode_Step` (Total)** | **59.325 ms (100.0%)** | **57.094 ms (100.0%)** | **59.625 ms (100.0%)** | **59.240 ms (100.0%)** |

## Per-Configuration Breakdown

### B=1, ctx=4K

| Stage | ms/step | % of step | Calls/step |
|---|---:|---:|---:|
| `Embedding` | 0.039 | 0.1% | 1.0 |
| `RMSNorm` | 14.546 | 24.5% | 109.0 |
| `QKV_Proj` | 3.086 | 5.2% | 36.0 |
| `RoPE_KV_Append` | 25.705 | 43.3% | 36.0 |
| `Paged_Attention` | 7.684 | 13.0% | 36.0 |
| `O_Proj` | 2.702 | 4.6% | 36.0 |
| `MLP_SwiGLU` | 4.064 | 6.9% | 36.0 |
| `LM_Head` | 0.036 | 0.1% | 1.0 |
| `Other` | 1.462 | 2.5% | - |
| **`Decode_Step` (Total)** | **59.325** | **100.0%** | **1.0** |

### B=16, ctx=4K

| Stage | ms/step | % of step | Calls/step |
|---|---:|---:|---:|
| `Embedding` | 0.033 | 0.1% | 1.0 |
| `RMSNorm` | 13.651 | 23.9% | 109.0 |
| `QKV_Proj` | 2.868 | 5.0% | 36.0 |
| `RoPE_KV_Append` | 25.130 | 44.0% | 36.0 |
| `Paged_Attention` | 7.510 | 13.2% | 36.0 |
| `O_Proj` | 2.597 | 4.5% | 36.0 |
| `MLP_SwiGLU` | 3.891 | 6.8% | 36.0 |
| `LM_Head` | 0.034 | 0.1% | 1.0 |
| `Other` | 1.378 | 2.4% | - |
| **`Decode_Step` (Total)** | **57.094** | **100.0%** | **1.0** |

### B=64, ctx=4K

| Stage | ms/step | % of step | Calls/step |
|---|---:|---:|---:|
| `Embedding` | 0.045 | 0.1% | 1.0 |
| `RMSNorm` | 14.770 | 24.8% | 109.0 |
| `QKV_Proj` | 2.911 | 4.9% | 36.0 |
| `RoPE_KV_Append` | 26.159 | 43.9% | 36.0 |
| `Paged_Attention` | 7.600 | 12.7% | 36.0 |
| `O_Proj` | 2.438 | 4.1% | 36.0 |
| `MLP_SwiGLU` | 4.182 | 7.0% | 36.0 |
| `LM_Head` | 0.038 | 0.1% | 1.0 |
| `Other` | 1.483 | 2.5% | - |
| **`Decode_Step` (Total)** | **59.625** | **100.0%** | **1.0** |

### B=1, ctx=64K

| Stage | ms/step | % of step | Calls/step |
|---|---:|---:|---:|
| `Embedding` | 0.036 | 0.1% | 1.0 |
| `RMSNorm` | 14.618 | 24.7% | 109.0 |
| `QKV_Proj` | 3.064 | 5.2% | 36.0 |
| `RoPE_KV_Append` | 25.662 | 43.3% | 36.0 |
| `Paged_Attention` | 7.622 | 12.9% | 36.0 |
| `O_Proj` | 2.694 | 4.5% | 36.0 |
| `MLP_SwiGLU` | 4.068 | 6.9% | 36.0 |
| `LM_Head` | 0.037 | 0.1% | 1.0 |
| `Other` | 1.439 | 2.4% | - |
| **`Decode_Step` (Total)** | **59.240** | **100.0%** | **1.0** |

### B=16, ctx=64K

_Skipped / failed: OOM_

### B=64, ctx=64K

_Skipped / failed: OOM_
