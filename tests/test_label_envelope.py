"""The envelope script's binning arithmetic.

`balanced_edges` is a dynamic program whose job is to stop the envelope table
reporting ceilings for partitions that cannot be trained on. If it is wrong,
the correction it exists to make is itself wrong, so it is tested against
hand-computable cases rather than only against the corpus.
"""

from __future__ import annotations

import collections
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / filename
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENV = _module("envelope", "18_label_envelope.py")


def counts_from(mass: dict[float, int]):
    """A `collect`-shaped mapping with `n` residues at each temperature.

    Residue identity is irrelevant to the partitioning, so every temperature
    gets the same single letter; the tests that care about information content
    use two.
    """
    return {
        (f"seq{i}", ogt): collections.Counter({"A": n})
        for i, (ogt, n) in enumerate(sorted(mass.items()))
    }


def test_two_classes_split_an_even_corpus_down_the_middle():
    counts = counts_from({10.0: 100, 20.0: 100, 30.0: 100, 40.0: 100})
    assert ENV.balanced_edges(counts, 2) == (30.0,)
    assert ENV.min_class_share(counts, (30.0,)) == pytest.approx(0.5)


def test_an_unsplittable_atom_caps_the_minimum_class():
    # 30 C holds 60% of the mass and cannot be divided, so it joins one side
    # whole and leaves the other with 200 of 1000 residues. The best minimum
    # is 20%, not the 40% an equal-mass scheme would aim for -- this is the
    # primary corpus's situation (30 C alone is 29.6% of its residues) and the
    # reason the quantile scheme degenerates as k grows.
    counts = counts_from({10.0: 200, 30.0: 600, 50.0: 200})
    assert ENV.min_class_share(counts, ENV.balanced_edges(counts, 2)) == pytest.approx(0.2)


def test_it_beats_the_quantile_scheme_on_a_lumpy_corpus():
    SWEEP = _module("sweep", "15_ceiling_sweep.py")
    mass = {10.0: 60, 20.0: 70, 37.0: 500, 50.0: 60, 60.0: 80, 70.0: 230}
    counts = counts_from(mass)
    for k in (3, 4, 5):
        balanced = ENV.min_class_share(counts, ENV.balanced_edges(counts, k))
        quantile = ENV.min_class_share(counts, SWEEP.quantile_edges(counts, k))
        assert balanced >= quantile, f"k={k}: {balanced} < {quantile}"


def test_it_is_optimal_against_brute_force():
    import itertools

    mass = {float(t): m for t, m in zip(
        range(10, 100, 7),
        [41, 9, 130, 17, 64, 5, 88, 23, 71, 12, 150, 34, 7],
        strict=True,
    )}
    counts = counts_from(mass)
    temps = sorted(mass)
    total = sum(mass.values())
    for k in (2, 3, 4, 5, 6):
        # Every contiguous partition into k classes, scored the same way.
        best = max(
            min(
                sum(mass[t] for t in temps[a:b])
                for a, b in zip((0,) + cuts, cuts + (len(temps),), strict=True)
            )
            for cuts in itertools.combinations(range(1, len(temps)), k - 1)
        )
        got = ENV.min_class_share(counts, ENV.balanced_edges(counts, k))
        assert got == pytest.approx(best / total), f"k={k}"


def test_edges_are_sorted_and_the_right_count():
    counts = counts_from({float(t): 100 + t for t in range(0, 100, 5)})
    for k in range(2, 9):
        edges = ENV.balanced_edges(counts, k)
        assert len(edges) == k - 1
        assert list(edges) == sorted(edges)
        assert len(set(edges)) == len(edges)


def test_asking_for_more_classes_than_temperatures_yields_nothing():
    counts = counts_from({10.0: 100, 20.0: 100})
    assert ENV.balanced_edges(counts, 3) == ()
    assert ENV.balanced_edges(counts, 1) == ()


def test_miller_madow_needs_two_classes_and_two_residues():
    one_class = {"30": collections.Counter({"A": 50, "C": 50})}
    assert ENV.miller_madow(one_class) == 0.0
    one_residue = {"30": collections.Counter({"A": 50}), "60": collections.Counter({"A": 50})}
    assert ENV.miller_madow(one_residue) == 0.0


def test_miller_madow_shrinks_as_the_corpus_grows():
    def at(scale: int) -> float:
        return ENV.miller_madow({
            "30": collections.Counter({"A": 10 * scale, "C": 30 * scale}),
            "60": collections.Counter({"A": 30 * scale, "C": 10 * scale}),
        })

    assert at(1000) == pytest.approx(at(1) / 1000)


def test_native_pooling_keeps_one_class_per_recorded_temperature():
    counts = counts_from({10.0: 100, 10.5: 100, 11.0: 100})
    native = ENV.by_native_ogt(counts)
    assert sorted(native) == ["10", "10.5", "11"]
    assert all(sum(c.values()) == 100 for c in native.values())
