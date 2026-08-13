import os
from setuptools import setup

ext_modules = []

if os.environ.get("SKIP_CUDA_BUILD") != "1":
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension

        NVCC_FLAGS = [
            "-O3",
            "-arch=sm_90a",
            "--use_fast_math",
            "-lineinfo",
            "-DUSE_HOPPER=1",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-Xcompiler", "-fPIC",
        ]

        CXX_FLAGS = ["-O3", "-std=c++17"]

        include_dirs = ["csrc/kernels", "csrc/pybind"]
        # Header-only CUTLASS/CuTe, cloned by cluster/setup_env.sh (git-ignored).
        cutlass_inc = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "third_party", "cutlass", "include",
        )
        if os.path.isdir(cutlass_inc):
            include_dirs.append(cutlass_inc)
            NVCC_FLAGS.append("-DHAS_CUTLASS=1")

        ext_modules = [
            CUDAExtension(
                name="sparse_kv._C",
                sources=[
                    "csrc/pybind/bindings.cpp",
                    "csrc/kernels/kv_evict_quant.cu",
                    "csrc/kernels/sparse_attention.cu",
                    "csrc/kernels/gqa_decode.cu",
                    "csrc/kernels/kv_fp8.cu",
                ],
                include_dirs=include_dirs,
                extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
                extra_link_args=["-lcuda"],
            )
        ]

        setup(
            ext_modules=ext_modules,
            cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
        )

    except ImportError as e:
        print(f"[sparse-kv] WARNING: torch not found, skipping CUDA build — {e}")
        setup()
else:
    setup()