# sparse_kv/__init__.py
__version__ = "0.1.0"

from .attention import sparse_attention

try:
    from .eviction import fused_kv_evict
    from sparse_kv._C import kv_evict_quant_forward
    CUDA_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    CUDA_AVAILABLE = False
    import warnings
    warnings.warn(f"[sparse_kv] CUDA extension not loaded: {e}. "
                  "Run: pip install -e . --no-build-isolation")

__all__ = ["sparse_attention", "fused_kv_evict", "kv_evict_quant_forward"]