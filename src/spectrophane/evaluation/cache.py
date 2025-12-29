from abc import ABC, abstractmethod
from typing import Any

class CacheBackend(ABC):

    @abstractmethod
    def get(self, key: str) -> Any:
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
        return self._store[key]

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def contains(self, key: str) -> bool:
        return key in self._store

#add e.g. LRUCacheBackend, DiskCacheBackend

class ForwardCache:
    """Caches results of stack->color calculations of the physics model for faster lookup"""
    def __init__(self, cache_backend: str):
        pass

    def set(self, stacks, values):
        pass

    def get(self, stacks):
        """"""
        pass

    def contains(self, stacks):
        pass

    def _hash(stack):
        pass
    