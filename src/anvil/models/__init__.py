"""Model wrappers used by Anvil's vision cascade.

Heavy implementations live behind the ``[full]`` extra. The bare install
exposes only the abstract base; ``load_embedder()`` returns ``None`` when no
backend can be loaded.
"""

from anvil.models.embedder import GlobalEmbedder, load_embedder

__all__ = ["GlobalEmbedder", "load_embedder"]
