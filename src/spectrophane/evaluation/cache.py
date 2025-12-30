from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import hashlib

from spectrophane.core.dataclasses import StackData

class CacheBackend(ABC):

    @abstractmethod
    def get(self, key: str) -> Any:
        """Returns the value for the given key. Returns None if key is not present."""
        ...

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        ...

    @abstractmethod
    def contains(self, key: str) -> bool:
        ...


class DictCacheBackend(CacheBackend):
    """Simple in-memory cache backend"""

    def __init__(self):
        self._store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        """Returns the value for the given key. Returns None if key is not present."""
        return self._store.get(key)
    
    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def contains(self, key: str) -> bool:
        return key in self._store

#add e.g. LRUCacheBackend, DiskCacheBackend

class ForwardCache:
    """Caches results of stack->color calculations of the physics model for faster lookup"""
    def __init__(self, cache_backend: str, value_length: int = 3, value_dtype = np.float64):
        self.value_length = value_length
        self.value_dtype = value_dtype
        if cache_backend == "dict":
            self._backend = DictCacheBackend()
        else:
            raise ValueError(f"Unknown cache backend: {cache_backend}")

    def batch_set(self, stacks: StackData, values):
        """Store calculated values in the cache"""
        hashes = self._batch_hash(stacks)
        for hash, value in zip(hashes, values):
            self._backend.set(str(hash), value)

    def batch_get(self, stacks: StackData):
        """Retrieve values from the cache"""
        hashes = self._batch_hash(stacks)
        found = self._batch_contains_hashed(hashes)
        result = np.empty((len(hashes), self.value_length), dtype=self.value_dtype)
        for i in range(len(hashes)):
            result[i] = self._backend.get(hashes[i])
        return found, result

    def batch_contains(self, stacks: StackData):
        """Check if values are already in the cache. Returns a boolean mask indicating if each element is present"""
        hashes = self._batch_hash(stacks)
        return self._batch_contains_hashed(hashes)
    
    def _batch_contains_hashed(self, hashes: np.ndarray):
        mask = np.empty_like(hashes, dtype=np.bool)
        for i in range(len(mask)):
            mask[i] = self._backend.contains(str(hashes[i]))
        return mask

    def _batch_hash(self, stacks: StackData) -> np.ndarray:
        #TODO: Replace cryptographic hashing with VECTORIZED semantic numeric hashing, canonicalizing data (0-ing unused layers [stack_counts:], integerizing thicknesses) and rolling hash -> better performance
        m = np.ascontiguousarray(stacks.material_nums, dtype=np.int64)
        t = np.ascontiguousarray(stacks.thicknesses,   dtype=np.float64)
        c = np.ascontiguousarray(stacks.stack_counts,  dtype=np.int64)
        assert m.shape[0] == t.shape[0], "Forward Cache hashing: Arrays must have the same size along the specified axis"
        assert m.shape[0] == c.shape[0], "Forward Cache hashing: Arrays must have the same size along the specified axis"

        hashes = np.array(["a"*32]*len(m), dtype=np.str_)
        for i in range(len(c)):
            hashes[i] = hashlib.sha256(m[i].tobytes() + t[i].tobytes() + c[i].tobytes()).hexdigest()
        return hashes