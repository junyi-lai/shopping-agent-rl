"""Thread-safe explicit lease tracking for ShopSimulator worker slots."""

from __future__ import annotations

import threading


class SlotLeasePool:
    """A slot remains leased until its owner explicitly releases it."""

    def __init__(self, size: int):
        self._lock = threading.Lock()
        self._size = 0
        self._free: set[int] = set()
        self.reset(size)

    def reset(self, size: int) -> None:
        size = int(size)
        if size < 0:
            raise ValueError("slot pool size must be non-negative")
        with self._lock:
            self._size = size
            self._free = set(range(size))

    def acquire(self) -> int | None:
        with self._lock:
            if not self._free:
                return None
            return self._free.pop()

    def release(self, slot: int) -> bool:
        slot = int(slot)
        with self._lock:
            if slot < 0 or slot >= self._size:
                raise ValueError(f"invalid environment index: {slot}")
            was_leased = slot not in self._free
            self._free.add(slot)
            return was_leased

    def free_slots(self) -> frozenset[int]:
        with self._lock:
            return frozenset(self._free)
