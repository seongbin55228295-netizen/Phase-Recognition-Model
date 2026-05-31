from .co_attention import CoAttentionFusion
from .concat import ConcatFusion

__all__ = ["CoAttentionFusion", "ConcatFusion", "build_fusion"]


def build_fusion(name: str, **kwargs):
    """Factory for fusion modules. GMU is out of scope for the current variant set."""
    if name == "co_attention":
        return CoAttentionFusion(**kwargs)
    if name == "concat":
        return ConcatFusion(**kwargs)
    raise ValueError(f"Unknown fusion module: {name!r}")
