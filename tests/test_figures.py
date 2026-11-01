"""Tests for the figure script.

`17_figures.py` reads the sweep tables and the per-run reports rather than
carrying transcribed numbers, because the first version of it did carry them
and put the second corpus's four-class bound on the primary corpus's panel.
What these tests protect is the part that parsing cannot: that the parsers
still match the files' layout, that each figure's points agree with the
analysis write-up they illustrate, and that the two statistics printed on
figure 1 are the ones in `measured_n8.md`.

Skipped where the result files are absent, as they are in a checkout that has
not fetched them.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "17_figures.py"
ANALYSIS = REPO_ROOT / "results" / "ceiling_sweep_20261030" / "measured_n8.md"


def load_script():
    spec = importlib.util.spec_from_file_location("figures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def figures():
    pytest.importorskip("matplotlib")
    return load_script()


@pytest.fixture(scope="module")
def data(figures):
    for path in figures.SWEEPS.values():
        if not path.is_file():
            pytest.skip(f"{path.name} not present")
    return figures.load()


class TestParsing:
    """The sweep tables and reports are machine-generated, so a parser that
    stops matching means their format changed and every figure is stale."""

    def test_every_sweep_yields_all_eleven_thresholds(self, data):
        for name, sweep in data["sweeps"].items():
            assert sorted(sweep) == [30, 35, 40, 45, 50, 55, 60, 65, 70, 75,
                                     80], name

    def test_the_rebalanced_table_is_not_read_as_thresholds(self, data):
        """Each sweep file holds three tables and only the first is the
        natural-balance threshold sweep. Reading on into the second would
        silently overwrite every bound with its rebalanced counterpart."""
        # The second corpus at 30 C reads +0.105% natural and +0.178% rebalanced.
        assert data["sweeps"]["external"][30][0] == pytest.approx(0.105)

    def test_the_four_class_bound_comes_from_the_primary_corpus(self, data):
        """The bug this file exists for: the external corpus's four-class row
        is +0.425% and the primary corpus's is +0.374%."""
        (corpus_wide, _), _ = data["c4"]
        assert corpus_wide == pytest.approx(0.374)

    def test_every_run_has_both_splits(self, data):
        for config, runs in data["gains"].items():
            for edge, splits in runs.items():
                assert set(splits) == {"val", "test"}, f"{config} {edge}"

    def test_error_bars_are_smaller_than_the_gains(self, data):
        """Gate 2 requires z >= 3, so one sigma is at most a third of the
        gain. An error bar larger than that means gain and z were paired from
        different runs."""
        for config, runs in data["gains"].items():
            for edge, splits in runs.items():
                for split, (gain, sigma) in splits.items():
                    assert 0 < sigma <= gain / 3 + 1e-9, \
                        f"{config} {edge} {split}"


class TestAgreementWithTheWriteUp:
    """Each figure illustrates one analysis file; if they disagree, the figure
    is the one a reader believes."""

    def test_the_printed_statistics_match_measured_n8(self, figures):
        if not ANALYSIS.is_file():
            pytest.skip("measured_n8.md not present")
            return
        text = ANALYSIS.read_text()
        for key, row in (("corpus", r"\| corpus-wide ceiling \|"),
                         ("within", r"\| \*\*within-cluster ceiling\*\* \|")):
            m = re.search(row + r"\s*\*{0,2}\+([\d.]+)\*{0,2}\s*\|"
                          r"\s*\*{0,2}([\d.]+)\*{0,2}\s*\|", text)
            assert m, key
            rho, p = figures.STATS[key]
            assert float(m.group(1)) == pytest.approx(rho)
            assert float(m.group(2)) == pytest.approx(p)

    @pytest.mark.parametrize(
        "config,edge,expected",
        [("primary", 50, 0.296), ("primary", 30, 0.037),
         ("external", 65, 0.449), ("external", 30, 0.110),
         ("small", 55, 0.200)],
    )
    def test_headline_gains(self, data, config, edge, expected):
        """The measured optima and cold ends, which the text quotes."""
        assert data["gains"][config][edge]["val"][0] == pytest.approx(expected)

    def test_progen2_small_reuses_the_primary_corpus_bounds(self, data):
        """Only the backbone changed, so a separate bound would be wrong."""
        assert data["bounds"]["small"] is data["bounds"]["primary"]


class TestRendering:
    def test_all_three_figures_render(self, figures, data, tmp_path):
        figures._style()
        written = [fn(data, tmp_path, "png")
                   for fn in (figures.figure1, figures.figure2,
                              figures.figure3)]
        assert len(written) == 3
        for path in written:
            assert path.is_file() and path.stat().st_size > 10_000
