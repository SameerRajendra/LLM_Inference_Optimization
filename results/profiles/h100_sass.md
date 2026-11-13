# Decode Kernel SASS Instruction Summary (sm_90a)

| Kernel | HMMA | LDGSTS | LDG.E.128 | LDG | LDS | LDSM | FFMA/HFMA2 | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gqa_decode_v4_fp8_kernel | 0 | 0 | 2 | 9 | 232 | 0 | 262 | 1984 |
| gqa_decode_v4_kernel | 0 | 0 | 2 | 7 | 232 | 0 | 260 | 1896 |
| gqa_combine_kernel | 0 | 0 | 0 | 44 | 0 | 0 | 10 | 304 |
| gqa_decode_splitkv_kernel | 0 | 0 | 0 | 15 | 103 | 0 | 94 | 1144 |
| gqa_decode_tc_kernel<128,64,true> | 32 | 34 | 5 | 7 | 171 | 8 | 66 | 2784 |
| gqa_decode_tc_kernel<128,64,false> | 32 | 34 | 5 | 5 | 75 | 24 | 2 | 2520 |
| gqa_decode_tc_kernel<64,64,true> | 16 | 34 | 5 | 7 | 107 | 4 | 34 | 2432 |
| gqa_decode_tc_kernel<64,64,false> | 16 | 34 | 5 | 5 | 59 | 12 | 2 | 2304 |
| gqa_combine_tc_kernel | 0 | 0 | 0 | 44 | 0 | 0 | 10 | 336 |
