"""pHash Hamming distances and union-find grouping."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pharmachecked.data.dedup import group_ids, pairwise_hamming, phash_hex, to_uint64


def _h(bits: int) -> str:
    return f"{bits:016x}"


def test_pairwise_hamming_matches_bit_count() -> None:
    a = ["0000000000000000", "ffffffffffffffff", "00000000000000ff"]
    d = pairwise_hamming(to_uint64(a), to_uint64(a))
    assert d.tolist() == [[0, 64, 8], [64, 0, 56], [8, 56, 0]]


def test_grouping_is_transitive_and_respects_threshold() -> None:
    ids = ["c", "a", "b", "d"]
    hashes = [_h(0b0), _h(0b11), _h(0b1111), _h(0xFFFF_0000_0000_0000)]
    # c-a distance 2, a-b distance 2, c-b distance 4: chain links c, a, b at threshold 2.
    assert group_ids(ids, hashes, 2) == ["g_a", "g_a", "g_a", "g_d"]
    assert group_ids(ids, hashes, 1) == ["g_c", "g_a", "g_b", "g_d"]


def test_group_ids_independent_of_order() -> None:
    ids = ["c", "a", "b", "d"]
    hashes = [_h(0b0), _h(0b11), _h(0b1111), _h(0xFFFF_0000_0000_0000)]
    forward = dict(zip(ids, group_ids(ids, hashes, 2), strict=True))
    backward = dict(zip(ids[::-1], group_ids(ids[::-1], hashes[::-1], 2), strict=True))
    assert forward == backward


def test_bad_hash_length_rejected() -> None:
    with pytest.raises(ValueError):
        to_uint64(["abc"])


def test_phash_of_resized_copy_is_close(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    arr = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).resize((256, 256)).save(tmp_path / "big.png")
    Image.fromarray(arr).resize((128, 128)).save(tmp_path / "small.png")
    Image.fromarray(255 - arr).resize((256, 256)).save(tmp_path / "inv.png")
    hashes = [phash_hex(tmp_path / f"{n}.png") for n in ("big", "small", "inv")]
    assert all(len(h) == 16 for h in hashes)
    d = pairwise_hamming(to_uint64(hashes), to_uint64(hashes))
    assert d[0, 1] <= 4
    assert d[0, 2] > 16
