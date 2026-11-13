"""CPU-only unit & CLI smoke tests for WP-5 profiling and TP scripts.

Runs without CUDA, GPUs, ncu, nsys, or cuobjdump.
"""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmarks import bench_tp_decode, profile_e2e_nsys, profile_ncu  # noqa: E402


# --------------------------------------------------------------------------- CLI --help smoke tests


@pytest.mark.parametrize(
    "rel_path,expected_flags",
    [
        (
            os.path.join("benchmarks", "profile_ncu.py"),
            ["--ncu", "--ncu-csv", "--ncu-out", "--sass", "--sass-out", "--impl"],
        ),
        (
            os.path.join("benchmarks", "profile_e2e_nsys.py"),
            ["--nsys-sweep", "--cuda-events", "--out", "--method", "--batch", "--ctx"],
        ),
        (
            os.path.join("benchmarks", "bench_tp_decode.py"),
            ["--tp", "--shape", "--dry-run", "--dry-run-cpu", "--nsys-bracket"],
        ),
    ],
)
def test_wp5_scripts_help_cli(rel_path, expected_flags):
    script_path = os.path.join(ROOT, rel_path)
    proc = subprocess.run(
        [sys.executable, script_path, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    for flag in expected_flags:
        assert flag in proc.stdout


# --------------------------------------------------------------------------- benchmarks/profile_ncu.py


def test_profile_ncu_constants_and_metrics():
    expected_impls = {
        "v3",
        "v4",
        "v4_fp8",
        "tc",
        "tc_fp8",
        "triton",
        "triton_fp8",
        "paged_triton",
        "flashinfer",
        "sdpa",
    }
    assert expected_impls.issubset(set(profile_ncu.IMPL_CHOICES))
    for impl in profile_ncu.IMPL_CHOICES:
        assert impl in profile_ncu.KERNELS

    required_metrics = {
        "gpu__time_duration.sum",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__bytes_read.sum",
        "sm__inst_issued.avg.pct_of_peak_sustained_active",
        "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
        "smsp__warp_issue_stalled_long_scoreboard_per_instruction_executed.avg",
        "smsp__warp_issue_stalled_mio_throttle_per_instruction_executed.avg",
        "lts__t_sector_hit_rate.pct",
    }
    assert required_metrics.issubset(set(profile_ncu.METRICS))


def test_summarize_sass_counts_and_filters():
    sample_sass = """
Fatbin elf code:
================
arch = sm_90
code for sm_90
\tFunction : _Z20gqa_decode_tc_kernelILi128ELi64ELb0EEvPK6__half
\t/*0000*/ LDGSTS.E.BYPASS.LTC128B [R2], [R4];
\t/*0010*/ @P0 LDSM.16.M88.4 R8, [R6];
\t/*0020*/ HMMA.16816.F32 R12, R8, R10, R12;
\t/*0030*/ HMMA.16816.F32 R16, R8, R10, R16;
\t/*0040*/ LDG.E.128 R20, [R24];
\t/*0050*/ LDS.U.128 R24, [R28];
\t/*0060*/ FFMA R4, R4, R5, R6;
\t/*0070*/ EXIT;
\tFunction : _Z23unrelated_helper_kernelv
\t/*0000*/ LDG.E.128 R2, [R4];
\t/*0010*/ EXIT;
"""
    md = profile_ncu.summarize_sass(sample_sass)
    assert "sm_90" in md
    assert "gqa_decode_tc_kernel<128,64,false>" in md
    assert "unrelated_helper_kernel" not in md
    # Columns: Kernel | HMMA | LDGSTS | LDG.E.128 | LDG | LDS | LDSM | FFMA/HFMA2 | Total
    assert "| gqa_decode_tc_kernel<128,64,false> | 2 | 1 | 1 | 1 | 1 | 1 | 1 | 8 |" in md


def test_summarize_sass_empty():
    out = profile_ncu.summarize_sass("Function : non_decode_kernel\n/*0000*/ EXIT;\n")
    assert "No decode kernels" in out


def test_parse_ncu_csv_and_format_markdown():
    sample_csv = """==PROF== Connected to process 1234
"ID","Process Name","Host Name","Kernel Name","Metric Name","Metric Unit","Metric Value"
"0","python","localhost","void gqa_decode_tc_kernel<128, 64>(half const*)","gpu__time_duration.sum","ns","45,000.0"
"0","python","localhost","void gqa_decode_tc_kernel<128, 64>(half const*)","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","72.5"
"0","python","localhost","void gqa_decode_tc_kernel<128, 64>(half const*)","dram__bytes_read.sum","byte","536,870,912"
"0","python","localhost","void gqa_decode_tc_kernel<128, 64>(half const*)","sm__inst_issued.avg.pct_of_peak_sustained_active","%","31.2"
"0","python","localhost","void gqa_decode_tc_kernel<128, 64>(half const*)","sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active","%","18.4"
"0","python","localhost","void gqa_decode_tc_kernel<128, 64>(half const*)","smsp__warp_issue_stalled_long_scoreboard_per_instruction_executed.avg","inst","1.250"
"0","python","localhost","void gqa_decode_tc_kernel<128, 64>(half const*)","smsp__warp_issue_stalled_mio_throttle_per_instruction_executed.avg","inst","0.125"
"0","python","localhost","void gqa_decode_tc_kernel<128, 64>(half const*)","lts__t_sector_hit_rate.pct","%","64.0"
"0","python","localhost","void gqa_decode_tc_kernel<128, 64>(half const*)","sm__warps_active.avg.pct_of_peak_sustained_active","%","81.3"
"""
    kname, metrics = profile_ncu.parse_ncu_csv(sample_csv)
    assert "gqa_decode_tc_kernel" in kname
    assert metrics["gpu__time_duration.sum"] == (45000.0, "ns")
    assert metrics["dram__bytes_read.sum"] == (536870912.0, "byte")

    md = profile_ncu.format_ncu_markdown(
        [
            ("tc", kname, metrics, None),
            ("flashinfer", "", {}, "error (exit 1)"),
        ],
        batch=1,
        ctx=32768,
    )
    assert "batch=1, ctx=32768" in md
    assert "| tc | gqa_decode_tc_kernel<128, 64> | 45.00 | 72.50 | 536.87 | 31.20 | 18.40 | 1.250 | 0.125 | 64.00 | 81.30 |" in md
    assert "| flashinfer | error (exit 1) | - |" in md

    assert profile_ncu.parse_ncu_csv("") == ("", {})


# --------------------------------------------------------------------------- benchmarks/profile_e2e_nsys.py


def test_profile_e2e_nsys_constants():
    assert profile_e2e_nsys.OUTER_STAGE == "Decode_Step"
    for stage in (
        "RMSNorm",
        "QKV_Proj",
        "RoPE_KV_Append",
        "Paged_Attention",
        "O_Proj",
        "MLP_SwiGLU",
    ):
        assert stage in profile_e2e_nsys.STAGE_ORDER
    assert set(profile_e2e_nsys.METHODS) == {
        "paged_fp16",
        "paged_fp8",
        "quest_fp16",
        "quest_fp8",
    }


def test_parse_nvtx_csv_and_format_profile_markdown():
    sample_csv = """Generating '/tmp/report.sqlite'
Processing [nvtx_sum] report...
Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Style,Range
100.0,"10,000,000",5,"2,000,000","2,000,000","1,900,000","2,100,000","50,000",PushPop,:Decode_Step
50.0,"5,000,000",160,"31,250","31,000","30,000","33,000","500",PushPop,:Paged_Attention
30.0,"3,000,000",160,"18,750","18,700","18,000","19,500","200",PushPop,:MLP_SwiGLU
10.0,"1,000,000",320,"3,125","3,100","3,000","3,500","100",PushPop,:RMSNorm
"""
    parsed = profile_e2e_nsys.parse_nvtx_csv(sample_csv)
    by_name = {r["range"]: r for r in parsed}
    assert set(by_name.keys()) == {"Decode_Step", "Paged_Attention", "MLP_SwiGLU", "RMSNorm"}
    assert by_name["Decode_Step"]["total_ms"] == pytest.approx(10.0)
    assert by_name["Decode_Step"]["instances"] == 5
    assert by_name["Paged_Attention"]["avg_ms"] == pytest.approx(0.03125)

    assert profile_e2e_nsys.parse_nvtx_csv("") == []

    runs = [
        {
            "method": "quest_fp8",
            "batch": 1,
            "ctx": 4096,
            "profile_steps": 5,
            "stages": parsed,
        },
        {
            "method": "quest_fp8",
            "batch": 16,
            "ctx": 65536,
            "profile_steps": 5,
            "stages": [
                {"range": "Decode_Step", "total_ms": 20.0, "instances": 5},
                {"range": "Paged_Attention", "total_ms": 12.0, "instances": 160},
                {"range": "MLP_SwiGLU", "total_ms": 6.0, "instances": 160},
            ],
        },
    ]
    md = profile_e2e_nsys.format_profile_markdown(
        runs,
        metadata={
            "model": "Qwen/Qwen3-8B",
            "method": "quest_fp8",
            "gpu": "NVIDIA H100 80GB",
            "profile_steps": 5,
            "warmup": 3,
            "block_size": 64,
            "quest_budget": 0.25,
            "dense_prefix_layers": 2,
            "recent_pages": 1,
            "source": "nsys stats --report nvtx_sum",
        },
    )
    assert "## Summary Breakdown" in md
    assert "B=1, ctx=4K" in md
    assert "B=16, ctx=64K" in md
    assert "`Paged_Attention`" in md
    assert "`Other`" in md
    assert "Quest config" in md


def test_format_profile_markdown_handles_errors_only():
    md = profile_e2e_nsys.format_profile_markdown(
        [{"method": "paged_fp16", "batch": 64, "ctx": 65536, "error": "OOM"}]
    )
    assert "_No profiling records available._" in md
    assert "OOM" in md


# --------------------------------------------------------------------------- benchmarks/bench_tp_decode.py


def test_compute_tp_partition():
    p2 = bench_tp_decode.compute_tp_partition("qwen3-8b", 2)
    assert p2["hq_local"] == 16
    assert p2["hkv_local"] == 4
    assert p2["q_dim_local"] == 16 * 128
    assert p2["k_dim_local"] == 4 * 128
    assert p2["qkv_dim_local"] == (16 + 8) * 128
    assert p2["o_in_dim"] == 16 * 128
    assert p2["o_out_dim"] == 4096
    assert p2["intermediate_local"] == 6144
    assert p2["gate_up_dim_local"] == 12288

    p4 = bench_tp_decode.compute_tp_partition("qwen3-8b", 4)
    assert p4["hq_local"] == 8
    assert p4["hkv_local"] == 2
    assert p4["intermediate_local"] == 3072

    with pytest.raises(ValueError, match="Unknown shape"):
        bench_tp_decode.compute_tp_partition("nonexistent-model", 2)

    with pytest.raises(ValueError, match="not divisible"):
        bench_tp_decode.compute_tp_partition("tiny", 4)

    with pytest.raises(ValueError, match="positive integer"):
        bench_tp_decode.compute_tp_partition("qwen3-8b", 0)

    with pytest.raises(ValueError, match="positive integer"):
        bench_tp_decode.compute_tp_partition("qwen3-8b", True)


def test_compute_tp_metrics_and_find_free_port():
    m = bench_tp_decode.compute_tp_metrics(compute_ms=1.5, eager_tp_ms=2.0, graph_tp_ms=1.6)
    assert m["comm_ms"] == pytest.approx(0.5)
    assert m["comm_pct"] == pytest.approx(25.0)
    assert m["graph_speedup"] == pytest.approx(2.0 / 1.6)
    assert m["compute_to_comm_ratio"] == pytest.approx(3.0)

    m_no_graph = bench_tp_decode.compute_tp_metrics(compute_ms=1.5, eager_tp_ms=1.5, graph_tp_ms=None)
    assert m_no_graph["comm_ms"] == pytest.approx(0.0)
    assert m_no_graph["graph_speedup"] is None
    assert m_no_graph["compute_to_comm_ratio"] is None

    port = bench_tp_decode.find_free_port()
    assert isinstance(port, int)
    assert 1 <= port <= 65535


def test_bench_tp_decode_dry_run_and_dry_run_cpu():
    script_path = os.path.join(ROOT, "benchmarks", "bench_tp_decode.py")
    dry = subprocess.run(
        [
            sys.executable,
            script_path,
            "--dry-run",
            "--shape",
            "tiny",
            "--tp",
            "2",
            "--batches",
            "1",
            "--contexts",
            "16",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert dry.returncode == 0, dry.stderr
    assert "TP Partition | shape=tiny tp=2" in dry.stdout

    dry_cpu = subprocess.run(
        [
            sys.executable,
            script_path,
            "--dry-run-cpu",
            "--shape",
            "tiny",
            "--tp",
            "2",
            "--batches",
            "1",
            "--contexts",
            "16",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert dry_cpu.returncode == 0, dry_cpu.stderr
    assert "equivalence check passed" in dry_cpu.stdout
