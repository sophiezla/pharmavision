"""Perceptual-hash near-duplicate grouping (64-bit pHash, union-find over Hamming distance)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

HEX_LENGTH = 16  # 64-bit hash


def phash_hex(path: Path, hash_size: int = 8) -> str:
    """Return the pHash of an image as 16 hex characters (Zauner, 2010; via ``imagehash``).

    Args:
        path: Image file (normalized RGB PNG).
        hash_size: Side of the DCT low-frequency block; 8 gives a 64-bit hash.

    Returns:
        Hex string of the hash.
    """
    with Image.open(path) as im:
        return str(imagehash.phash(im.convert("RGB"), hash_size=hash_size))


def to_uint64(hashes: Sequence[str]) -> np.ndarray:
    """Convert 16-hex-character hashes to a uint64 array."""
    if any(len(h) != HEX_LENGTH for h in hashes):
        raise ValueError("expected 64-bit hashes (16 hex characters)")
    return np.array([int(h, 16) for h in hashes], dtype=np.uint64)


def pairwise_hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the Hamming distance between every hash in ``a`` and every hash in ``b``.

    Args:
        a: uint64 array of shape (n,).
        b: uint64 array of shape (m,).

    Returns:
        int array of shape (n, m).
    """
    xor = np.bitwise_xor(a[:, None], b[None, :])
    bits = np.unpackbits(xor.view(np.uint8).reshape(*xor.shape, 8), axis=-1)
    return bits.sum(axis=-1).astype(np.int64)


def group_ids(image_ids: Sequence[str], hashes: Sequence[str], threshold: int) -> list[str]:
    """Union-find grouping: images whose pHash distance is <= ``threshold`` share a group.

    Grouping is transitive (a chain of close pairs forms one group). The group id is
    ``g_`` + the smallest image id in the group, so it is deterministic and order-independent.

    Args:
        image_ids: Image ids.
        hashes: pHash hex strings aligned with ``image_ids``.
        threshold: Maximum Hamming distance that links two images.

    Returns:
        Group id per image, aligned with ``image_ids``.
    """
    n = len(image_ids)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n:
        dist = pairwise_hamming(to_uint64(hashes), to_uint64(hashes))
        rows, cols = np.nonzero(np.triu(dist <= threshold, k=1))
        for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    members: dict[int, list[str]] = {}
    for i in range(n):
        members.setdefault(find(i), []).append(image_ids[i])
    smallest = {root: min(ids) for root, ids in members.items()}
    return [f"g_{smallest[find(i)]}" for i in range(n)]
