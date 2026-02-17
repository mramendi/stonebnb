#!/usr/bin/env python3
"""
Expert Weight Storage System for StoneBnB

This module implements a unified virtual weight system with LRU caching for
expert weights, supporting both quantized and CPU-offloaded storage modes.

Key components:
- GlobalExpertCache: Singleton LRU cache shared across all layers
- ExpertWeightSource: Abstract base for different storage backends
- QuantizedExpertSource: For 4-bit quantized storage
- CPUOffloadedExpertSource: For full-precision CPU storage
- ExpertWeightRegistry: Maps (layer_idx, weight_type) to sources

This enables lazy per-expert dequantization with dramatic VRAM savings:
- Old: ~5GB (dequantize all 72 experts at once)
- New: ~140MB (dequantize 1-2 experts at a time)
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Optional, Tuple
import torch
import bitsandbytes.functional as bnb_F


# ============================================================================
# Component 1: Global Unified LRU Cache
# ============================================================================

class GlobalExpertCache:
    """
    Singleton LRU cache for all expert weights across all layers.

    Key insight: Training progresses layer-by-layer (Layer 0 → 1 → ... → 39),
    so at any moment we only need experts from 1-2 layers, not all 40 layers.

    Cache sizing:
    - max_entries=10: Minimal (~700MB), covers ~5 experts per current layer
    - max_entries=20: Balanced (~1.4GB), covers ~10 experts × 2 layers
    - max_entries=40-80: Aggressive (~2.8-5.6GB), caches multiple layers ahead

    Memory comparison:
    - Old (per-layer cache, size=2): 80 layers × 2 = 160 cached = ~11GB VRAM
    - New (global cache, size=20): 20 experts total = ~1.4GB VRAM
    """

    _instance = None
    _cache = None
    _max_entries = 20
    _stats = {'hits': 0, 'misses': 0}

    @classmethod
    def get_instance(cls, max_entries: int = 20):
        """
        Get or create the singleton instance.

        Args:
            max_entries: Maximum number of cached experts (total across all layers)

        Returns:
            GlobalExpertCache singleton instance
        """
        if cls._instance is None:
            cls._instance = cls()
            cls._cache = OrderedDict()
            cls._max_entries = max_entries
        return cls._instance

    @classmethod
    def get(cls, key: Tuple[int, str, int]) -> Optional[torch.Tensor]:
        """
        Get cached expert weight.

        Args:
            key: (layer_idx, weight_type, expert_idx)
                 e.g., (0, "input_linear", 5) for layer 0, input_linear, expert 5

        Returns:
            Cached tensor (BF16, 2D) or None if not cached
        """
        cache = cls._cache
        if cache is None:
            # Cache not initialized
            return None

        if key in cache:
            # Cache hit - move to end (most recently used)
            cache.move_to_end(key)
            cls._stats['hits'] += 1
            return cache[key]

        # Cache miss
        cls._stats['misses'] += 1
        return None

    @classmethod
    def put(cls, key: Tuple[int, str, int], tensor: torch.Tensor):
        """
        Add tensor to cache, evicting oldest if needed.

        Args:
            key: (layer_idx, weight_type, expert_idx)
            tensor: Expert weight tensor (BF16, 2D)
        """
        cache = cls._cache
        if cache is None:
            # Cache not initialized
            return

        # Add to cache and mark as most recently used
        cache[key] = tensor
        cache.move_to_end(key)

        # Evict oldest entries if over capacity
        while len(cache) > cls._max_entries:
            old_key, old_tensor = cache.popitem(last=False)
            del old_tensor  # Explicit VRAM free

    @classmethod
    def get_stats(cls):
        """
        Get cache statistics.

        Returns:
            Dict with hits, misses, hit_rate, cached_entries, max_entries
        """
        total = cls._stats['hits'] + cls._stats['misses']
        hit_rate = cls._stats['hits'] / total if total > 0 else 0.0

        return {
            'hits': cls._stats['hits'],
            'misses': cls._stats['misses'],
            'hit_rate': hit_rate,
            'cached_entries': len(cls._cache) if cls._cache else 0,
            'max_entries': cls._max_entries
        }

    @classmethod
    def clear(cls):
        """Clear the cache and reset statistics."""
        if cls._cache is not None:
            cls._cache.clear()
        cls._stats = {'hits': 0, 'misses': 0}

    @classmethod
    def reset_stats(cls):
        """Reset statistics without clearing cache."""
        cls._stats = {'hits': 0, 'misses': 0}


# ============================================================================
# Component 2: ExpertWeightSource (Abstract Base)
# ============================================================================

class ExpertWeightSource(ABC):
    """
    Base class for expert weight storage backends.

    Provides a unified interface for different weight storage modes:
    - QuantizedExpertSource: 4-bit quantized storage
    - CPUOffloadedExpertSource: Full-precision CPU storage
    - Future: PerExpertQuantizedSource (72 separate 2D tensors)
    """

    @abstractmethod
    def get_expert(self, expert_id: int) -> torch.Tensor:
        """
        Get dequantized/loaded expert weight (2D BF16).

        Args:
            expert_id: Expert index (0 to num_experts-1)

        Returns:
            Expert weight tensor (BF16, 2D: [out_features, in_features])
        """
        pass

    @abstractmethod
    def get_all_experts(self) -> torch.Tensor:
        """
        Get all experts as 3D tensor (for compatibility).

        Returns:
            All expert weights (BF16, 3D: [num_experts, out_features, in_features])
        """
        pass


# ============================================================================
# Component 3: QuantizedExpertSource
# ============================================================================

class QuantizedExpertSource(ExpertWeightSource):
    """
    Expert weight source for 4-bit quantized storage.

    CRITICAL: HuggingFace Compatibility via Reference (No Duplication)
    - Quantized data stays in model: model.layers.X.block_sparse_moe.input_linear.weight
      (which IS a Params4bit object)
    - This class holds a **Python reference** to that Params4bit object, NOT a copy
    - Single source of truth: model's Params4bit.data (no VRAM duplication)
    - When we access self.params4bit.data, we're accessing the same tensor in the model

    Memory savings:
    - Old: Dequantize all 72 experts at once (~5GB for Granite Small)
    - New: Dequantize only 1 expert at a time (~70MB), cache with LRU
    """

    def __init__(self, params4bit_ref, layer_idx: int, weight_type: str):
        """
        Create a quantized expert source.

        Args:
            params4bit_ref: Reference to model's Params4bit object (NOT a copy!)
            layer_idx: Layer index (0 to num_layers-1)
            weight_type: "input_linear" or "output_linear"
        """
        self.params4bit = params4bit_ref  # Reference to model's Params4bit object
        self.layer_idx = layer_idx
        self.weight_type = weight_type

        # Extract shape info from quant_state
        # quant_state.shape is the original 3D shape: (num_experts, out_features, in_features)
        self.num_experts = params4bit_ref.quant_state.shape[0]
        self.out_features = params4bit_ref.quant_state.shape[1]
        self.in_features = params4bit_ref.quant_state.shape[2]

    def get_expert(self, expert_id: int) -> torch.Tensor:
        """
        Get single expert weight with global caching.

        Args:
            expert_id: Expert index (0 to num_experts-1)

        Returns:
            Expert weight tensor (BF16, 2D: [out_features, in_features])
        """
        # Check global cache first
        cache_key = (self.layer_idx, self.weight_type, expert_id)
        cached = GlobalExpertCache.get(cache_key)
        if cached is not None:
            return cached

        # Cache miss - dequantize only this expert's slice
        expert_weight = self._dequantize_single_expert(expert_id)

        # Put in global cache
        GlobalExpertCache.put(cache_key, expert_weight)
        return expert_weight

    def _dequantize_single_expert(self, expert_id: int) -> torch.Tensor:
        """
        Dequantize only one expert's 2D slice.

        Args:
            expert_id: Expert index (0 to num_experts-1)

        Returns:
            Expert weight tensor (BF16, 2D: [out_features, in_features])
        """
        # Access quantized data from model's Params4bit (reference, not copy!)
        # Current 3D shape: (num_experts, out_features, in_features)
        # Stored as 2D: (num_experts * out_features, in_features)

        # Dequantize full tensor (temporary allocation)
        full_dequant = bnb_F.dequantize_4bit(
            self.params4bit.data,           # Model's quantized tensor
            self.params4bit.quant_state     # Model's quant_state
        )  # Returns (num_experts*out_features, in_features) in BF16

        # Reshape to 3D
        full_3d = full_dequant.reshape(self.num_experts, self.out_features, self.in_features)

        # Extract only this expert (clone to avoid holding reference to full_3d)
        expert_2d = full_3d[expert_id].clone()

        # full_dequant and full_3d are garbage collected here
        return expert_2d  # Shape: (out_features, in_features)

    def get_all_experts(self) -> torch.Tensor:
        """
        Get all experts as 3D tensor (for compatibility).

        Returns:
            All expert weights (BF16, 3D: [num_experts, out_features, in_features])
        """
        # Dequantize full tensor
        full_dequant = bnb_F.dequantize_4bit(
            self.params4bit.data,
            self.params4bit.quant_state
        )

        # Reshape to 3D
        full_3d = full_dequant.reshape(self.num_experts, self.out_features, self.in_features)
        return full_3d


# ============================================================================
# Component 4: CPUOffloadedExpertSource
# ============================================================================

class CPUOffloadedExpertSource(ExpertWeightSource):
    """
    Expert weight source for full-precision CPU storage.

    This is for future CPU offloading mode where expert weights are stored
    on CPU and loaded to GPU on demand.

    Memory savings:
    - GPU VRAM: Only cached experts (~1-2GB with LRU cache)
    - CPU RAM: Full expert weights (~10GB for Granite Small)
    """

    def __init__(self, expert_tensors_cpu, layer_idx: int, weight_type: str):
        """
        Create a CPU-offloaded expert source.

        Args:
            expert_tensors_cpu: List/dict of BF16 tensors on CPU
            layer_idx: Layer index (0 to num_layers-1)
            weight_type: "input_linear" or "output_linear"
        """
        self.expert_tensors = expert_tensors_cpu  # List/dict of BF16 tensors on CPU
        self.layer_idx = layer_idx
        self.weight_type = weight_type

    def get_expert(self, expert_id: int) -> torch.Tensor:
        """
        Get single expert weight with global caching.

        Args:
            expert_id: Expert index (0 to num_experts-1)

        Returns:
            Expert weight tensor (BF16, 2D on GPU)
        """
        # Check global cache first
        cache_key = (self.layer_idx, self.weight_type, expert_id)
        cached = GlobalExpertCache.get(cache_key)
        if cached is not None:
            return cached

        # Cache miss - copy from CPU to GPU
        expert_weight = self.expert_tensors[expert_id].to('cuda')

        # Put in global cache
        GlobalExpertCache.put(cache_key, expert_weight)
        return expert_weight

    def get_all_experts(self) -> torch.Tensor:
        """
        Get all experts as 3D tensor (for compatibility).

        Returns:
            All expert weights (BF16, 3D on GPU)
        """
        # Stack all experts and move to GPU
        if isinstance(self.expert_tensors, dict):
            expert_list = [self.expert_tensors[i] for i in sorted(self.expert_tensors.keys())]
        else:
            expert_list = self.expert_tensors

        return torch.stack(expert_list, dim=0).to('cuda')


# ============================================================================
# Component 5: ExpertWeightRegistry
# ============================================================================

class ExpertWeightRegistry:
    """
    Global singleton managing weight sources.

    Maps (layer_idx, weight_type) to ExpertWeightSource instances.
    This allows WeightReference to fetch sources by layer/type.
    """

    _sources = {}  # {(layer_idx, weight_type): ExpertWeightSource}

    @classmethod
    def register(cls, layer_idx: int, weight_type: str, source: ExpertWeightSource):
        """
        Register a weight source.

        Args:
            layer_idx: Layer index (0 to num_layers-1)
            weight_type: "input_linear" or "output_linear"
            source: ExpertWeightSource instance
        """
        cls._sources[(layer_idx, weight_type)] = source

    @classmethod
    def get(cls, layer_idx: int, weight_type: str) -> Optional[ExpertWeightSource]:
        """
        Get a registered weight source.

        Args:
            layer_idx: Layer index (0 to num_layers-1)
            weight_type: "input_linear" or "output_linear"

        Returns:
            ExpertWeightSource instance or None if not registered
        """
        return cls._sources.get((layer_idx, weight_type))

    @classmethod
    def clear(cls):
        """Clear all registered sources."""
        cls._sources.clear()

    @classmethod
    def count(cls) -> int:
        """Get number of registered sources."""
        return len(cls._sources)
