"""Correctness tests: JAX reference vs Pallas vs PyTorch CUDA kernel.

Run with:
    pytest jax_ref/test_pallas.py -v
or standalone:
    python jax_ref/test_pallas.py
"""

import math
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from jax_ref import sparse_attention_pallas, sparse_attention_reference


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand(shape, dtype=jnp.float32, seed=0):
    key = jax.random.PRNGKey(seed)
    return jax.random.normal(key, shape, dtype=dtype)


# ---------------------------------------------------------------------------
# Reference correctness
# ---------------------------------------------------------------------------

class TestReference:

    def test_output_shape(self):
        Q = _rand((1, 4, 1, 64))
        K = _rand((1, 4, 256, 64))
        V = _rand((1, 4, 256, 64))
        out = sparse_attention_reference(Q, K, V, block_size=64, top_k_blocks=4)
        assert out.shape == (1, 4, 1, 64)

    def test_dtype_preserved_fp16(self):
        Q = _rand((1, 2, 1, 64), dtype=jnp.float16)
        K = _rand((1, 2, 512, 64), dtype=jnp.float16)
        V = _rand((1, 2, 512, 64), dtype=jnp.float16)
        out = sparse_attention_reference(Q, K, V, block_size=64, top_k_blocks=4)
        assert out.dtype == jnp.float16

    def test_output_finite(self):
        Q = _rand((1, 2, 1, 64))
        K = _rand((1, 2, 128, 64))
        V = _rand((1, 2, 128, 64))
        out = sparse_attention_reference(Q, K, V, block_size=64, top_k_blocks=2)
        assert jnp.all(jnp.isfinite(out))
        assert jnp.any(out != 0.0)

    def test_deterministic(self):
        """Same inputs → identical outputs on two calls."""
        Q = _rand((1, 1, 1, 32), seed=42)
        K = _rand((1, 1, 256, 32), seed=7)
        V = _rand((1, 1, 256, 32), seed=3)
        out1 = sparse_attention_reference(Q, K, V, block_size=64, top_k_blocks=1)
        out2 = sparse_attention_reference(Q, K, V, block_size=64, top_k_blocks=1)
        np.testing.assert_array_equal(np.array(out1), np.array(out2))

    def test_non_multiple_seqlen(self):
        """T=300 is not a multiple of block_size=64 — padding must not corrupt output."""
        Q = _rand((1, 2, 1, 64))
        K = _rand((1, 2, 300, 64))
        V = _rand((1, 2, 300, 64))
        out = sparse_attention_reference(Q, K, V, block_size=64, top_k_blocks=4)
        assert out.shape == (1, 2, 1, 64)
        assert jnp.all(jnp.isfinite(out))

    def test_top_k_clipped_to_num_blocks(self):
        """top_k_blocks > num_blocks should not crash — clips to num_blocks."""
        Q = _rand((1, 1, 1, 32))
        K = _rand((1, 1, 64, 32))   # exactly 1 block
        V = _rand((1, 1, 64, 32))
        out = sparse_attention_reference(Q, K, V, block_size=64, top_k_blocks=99)
        assert out.shape == (1, 1, 1, 32)
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# Pallas vs reference numerical agreement  (GPU only)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    jax.default_backend() != "gpu",
    reason="Pallas kernel requires a GPU backend"
)
class TestPallasVsReference:

    @pytest.mark.parametrize("T,block_size,top_k", [
        (256,  64,  4),
        (1024, 64,  8),
        (4096, 64, 16),
    ])
    def test_numerical_agreement(self, T, block_size, top_k):
        B, H, D = 1, 4, 64
        Q = _rand((B, H, 1, D))
        K = _rand((B, H, T, D))
        V = _rand((B, H, T, D))

        ref = sparse_attention_reference(Q, K, V, block_size, top_k)
        pal = sparse_attention_pallas(Q, K, V, block_size, top_k)

        np.testing.assert_allclose(
            np.array(ref), np.array(pal), rtol=1e-4, atol=1e-4,
            err_msg=f"Mismatch T={T}, block={block_size}, top_k={top_k}"
        )


# ---------------------------------------------------------------------------
# Optional cross-check against PyTorch reference
# ---------------------------------------------------------------------------

try:
    import torch
    from sparse_kv.attention import _reference_sparse_attention as _pt_ref
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch / sparse_kv not available")
class TestJaxVsPyTorchReference:
    """Both JAX and PyTorch references must produce finite outputs.

    Note: the JAX reference uses per-BLOCK top-k scoring;
    the PyTorch fallback uses per-TOKEN top-k.  They implement
    different sparsity strategies so exact numerical agreement is
    not expected — this test guards against trivially broken outputs.
    """

    def test_both_finite(self):
        B, H, T, D = 1, 2, 256, 64
        rng = np.random.default_rng(0)
        Q_np = rng.standard_normal((B, H, 1, D)).astype(np.float32)
        K_np = rng.standard_normal((B, H, T, D)).astype(np.float32)
        V_np = rng.standard_normal((B, H, T, D)).astype(np.float32)

        jax_out = sparse_attention_reference(
            jnp.array(Q_np), jnp.array(K_np), jnp.array(V_np),
            block_size=64, top_k_blocks=4,
        )
        pt_out = _pt_ref(
            torch.from_numpy(Q_np),
            torch.from_numpy(K_np),
            torch.from_numpy(V_np),
            top_k=4 * 64,
            sink_tokens=0,
        ).numpy()

        assert np.all(np.isfinite(np.array(jax_out))), "JAX output has non-finite values"
        assert np.all(np.isfinite(pt_out)), "PyTorch output has non-finite values"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running reference correctness tests...")
    t = TestReference()
    t.test_output_shape();           print("  output_shape         OK")
    t.test_dtype_preserved_fp16();   print("  dtype_fp16           OK")
    t.test_output_finite();          print("  output_finite        OK")
    t.test_deterministic();          print("  deterministic        OK")
    t.test_non_multiple_seqlen();    print("  non_multiple_seqlen  OK")
    t.test_top_k_clipped_to_num_blocks(); print("  top_k_clip           OK")
    print("All reference tests passed.\n")

    if jax.default_backend() == "gpu":
        print("GPU detected — running Pallas vs reference tests...")
        tc = TestPallasVsReference()
        for T, bs, tk in [(256, 64, 4), (1024, 64, 8), (4096, 64, 16)]:
            tc.test_numerical_agreement(T, bs, tk)
            print(f"  T={T:5d}, block={bs}, top_k={tk}  OK")
        print("All Pallas tests passed.")
    else:
        print("No GPU backend detected — skipping Pallas tests.")
        print("To run on GPU: XLA_FLAGS=--xla_gpu_cuda_data_dir=... python jax_ref/test_pallas.py")