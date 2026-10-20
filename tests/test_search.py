"""Search parameter handling, command construction and hit parsing.

The unit conversions here are the point. A coverage fraction passed where a
percentage is expected does not crash; it quietly admits nearly every fragment
in the database and changes what the whole project is trained on.
"""

from __future__ import annotations

import pytest

from thermocas.config import Config
from thermocas.data.search import (
    DIAMOND_FIELDS,
    HIT_COLUMNS,
    Hit,
    SearchParams,
    diamond_blastp_command,
    diamond_makedb_command,
    filter_hits,
    mmseqs_search_commands,
    parse_hits,
    unique_subjects,
)


def params(**overrides) -> SearchParams:
    base = dict(
        tool="diamond",
        database="uniref50",
        max_evalue=1e-5,
        min_query_coverage=0.70,
        max_target_seqs=5000,
        sensitivity="very-sensitive",
        threads=8,
    )
    base.update(overrides)
    return SearchParams(**base)


class TestParams:
    def test_loads_from_shipped_config(self):
        p = SearchParams.from_config(Config.load("data"))
        assert p.tool == "diamond"
        assert p.max_evalue == 1e-5
        assert p.min_query_coverage == 0.70

    def test_coverage_given_as_percent_is_rejected(self):
        """70 instead of 0.70 is the easy mistake; it must not be accepted."""
        cfg = Config(
            {
                "search": {
                    "tool": "diamond",
                    "database": "uniref50",
                    "max_evalue": 1e-5,
                    "min_query_coverage": 70,
                    "max_target_seqs": 100,
                    "sensitivity": "",
                    "threads": 1,
                }
            }
        )
        with pytest.raises(ValueError, match="fraction"):
            SearchParams.from_config(cfg)

    def test_unsupported_tool_rejected(self):
        with pytest.raises(ValueError, match="unsupported search tool"):
            params(tool="blastx")

        cfg = Config(
            {
                "search": {
                    "tool": "blastx",
                    "database": "uniref50",
                    "max_evalue": 1e-5,
                    "min_query_coverage": 0.7,
                    "max_target_seqs": 10,
                    "sensitivity": "",
                    "threads": 1,
                }
            }
        )
        with pytest.raises(ValueError, match="unsupported search tool"):
            SearchParams.from_config(cfg)

    def test_zero_threads_rejected(self):
        cfg = Config(
            {
                "search": {
                    "tool": "diamond",
                    "database": "uniref50",
                    "max_evalue": 1e-5,
                    "min_query_coverage": 0.7,
                    "max_target_seqs": 10,
                    "sensitivity": "",
                    "threads": 0,
                }
            }
        )
        with pytest.raises(ValueError, match="at least 1"):
            SearchParams.from_config(cfg)

    def test_percent_conversion(self):
        assert params(min_query_coverage=0.70).query_coverage_percent == 70.0
        assert params(min_query_coverage=1.0).query_coverage_percent == 100.0
        assert params(min_query_coverage=0.0).query_coverage_percent == 0.0


class TestDiamondCommand:
    def test_is_blastp_not_blastx(self, tmp_path):
        """Seeds and database are both protein; blastx would be wrong."""
        command = diamond_blastp_command(params(), tmp_path / "db", tmp_path / "q.fasta", tmp_path / "o.tsv")
        assert command[:2] == ["diamond", "blastp"]
        assert "blastx" not in command

    def test_query_cover_is_passed_as_a_percentage(self, tmp_path):
        command = diamond_blastp_command(
            params(min_query_coverage=0.70), tmp_path / "db", tmp_path / "q", tmp_path / "o"
        )
        assert command[command.index("--query-cover") + 1] == "70"

    def test_evalue_survives_as_scientific_notation(self, tmp_path):
        """1e-05 must not be rendered as 0.0 or 1.0000000000000001e-05."""
        command = diamond_blastp_command(params(), tmp_path / "db", tmp_path / "q", tmp_path / "o")
        value = command[command.index("--evalue") + 1]
        assert float(value) == 1e-5

    def test_outfmt_includes_coverage_and_subject_length(self, tmp_path):
        command = diamond_blastp_command(params(), tmp_path / "db", tmp_path / "q", tmp_path / "o")
        outfmt_at = command.index("--outfmt")
        assert command[outfmt_at + 1] == "6"
        emitted = command[outfmt_at + 2 : outfmt_at + 2 + len(DIAMOND_FIELDS)]
        assert emitted == list(DIAMOND_FIELDS)
        # Needed to re-verify coverage and to length-filter without a second pass.
        assert "qcovhsp" in emitted
        assert "slen" in emitted

    def test_sensitivity_becomes_a_flag(self, tmp_path):
        command = diamond_blastp_command(
            params(sensitivity="very-sensitive"), tmp_path / "db", tmp_path / "q", tmp_path / "o"
        )
        assert "--very-sensitive" in command

    def test_blank_sensitivity_adds_no_flag(self, tmp_path):
        command = diamond_blastp_command(
            params(sensitivity=""), tmp_path / "db", tmp_path / "q", tmp_path / "o"
        )
        assert not any(c.startswith("--sensitiv") for c in command)
        assert "--" not in command

    def test_makedb_command(self, tmp_path):
        command = diamond_makedb_command(tmp_path / "db.fasta", tmp_path / "idx", 4)
        assert command[:2] == ["diamond", "makedb"]
        assert command[command.index("--threads") + 1] == "4"


class TestMmseqsCommands:
    def test_coverage_is_a_fraction_not_a_percentage(self, tmp_path):
        """MMseqs2's -c differs in units from DIAMOND's --query-cover."""
        commands = mmseqs_search_commands(
            params(tool="mmseqs2", min_query_coverage=0.70),
            tmp_path / "q.fasta",
            tmp_path / "db.fasta",
            tmp_path / "o.tsv",
            tmp_path,
        )
        search = next(c for c in commands if c[1] == "search")
        assert search[search.index("-c") + 1] == "0.7"
        # cov-mode 1 makes -c mean query coverage specifically.
        assert search[search.index("--cov-mode") + 1] == "1"

    def test_pipeline_order(self, tmp_path):
        commands = mmseqs_search_commands(
            params(tool="mmseqs2"), tmp_path / "q", tmp_path / "db", tmp_path / "o", tmp_path
        )
        assert [c[1] for c in commands] == ["createdb", "createdb", "search", "convertalis"]


class TestHitParsing:
    def diamond_row(self, **overrides) -> str:
        row = {
            "qseqid": "Q99ZW2",
            "sseqid": "UniRef50_X",
            "pident": "41.2",
            "length": "1200",
            "qstart": "1",
            "qend": "1200",
            "sstart": "1",
            "send": "1190",
            "evalue": "1e-120",
            "bitscore": "800.5",
            "qcovhsp": "88.0",
            "scovhsp": "91.0",
            "slen": "1310",
        }
        row.update({k: str(v) for k, v in overrides.items()})
        return "\t".join(row[f] for f in DIAMOND_FIELDS)

    def test_parses_diamond_table(self, tmp_path):
        path = tmp_path / "hits.tsv"
        path.write_text(self.diamond_row() + "\n")
        (hit,) = list(parse_hits(path, "diamond"))
        assert hit.query_id == "Q99ZW2"
        assert hit.subject_id == "UniRef50_X"
        assert hit.percent_identity == 41.2
        assert hit.subject_length == 1310

    def test_diamond_percent_coverage_normalized_to_fraction(self, tmp_path):
        """Downstream compares against a fraction, so parsing must normalize."""
        path = tmp_path / "hits.tsv"
        path.write_text(self.diamond_row(qcovhsp="88.0") + "\n")
        (hit,) = list(parse_hits(path, "diamond"))
        assert hit.query_coverage == pytest.approx(0.88)

    def test_mmseqs_fractional_coverage_left_alone(self, tmp_path):
        path = tmp_path / "hits.tsv"
        path.write_text(
            "\t".join(["q", "t", "41.2", "1200", "1", "1200", "1", "1190", "1e-120", "800", "0.88", "0.91", "1310"])
            + "\n"
        )
        (hit,) = list(parse_hits(path, "mmseqs2"))
        assert hit.query_coverage == pytest.approx(0.88)

    def test_full_coverage_of_one_is_not_misread_as_one_percent(self, tmp_path):
        """A coverage of exactly 1.0 is 100%, and must stay 1.0."""
        path = tmp_path / "hits.tsv"
        path.write_text(
            "\t".join(["q", "t", "41", "10", "1", "10", "1", "10", "1e-9", "80", "1.0", "1.0", "100"]) + "\n"
        )
        (hit,) = list(parse_hits(path, "mmseqs2"))
        assert hit.query_coverage == 1.0

    def test_wrong_column_count_names_the_cause(self, tmp_path):
        path = tmp_path / "hits.tsv"
        path.write_text("a\tb\tc\n")
        with pytest.raises(ValueError, match="outfmt"):
            list(parse_hits(path, "diamond"))

    def test_comments_and_blank_lines_skipped(self, tmp_path):
        path = tmp_path / "hits.tsv"
        path.write_text(f"# header\n\n{self.diamond_row()}\n")
        assert len(list(parse_hits(path, "diamond"))) == 1

    def test_malformed_number_reports_line(self, tmp_path):
        path = tmp_path / "hits.tsv"
        path.write_text(self.diamond_row(evalue="not-a-number") + "\n")
        with pytest.raises(ValueError, match="hits.tsv:1"):
            list(parse_hits(path, "diamond"))


class TestFiltering:
    def hit(self, **overrides) -> Hit:
        base = dict(
            query_id="q",
            subject_id="s",
            percent_identity=40.0,
            alignment_length=1200,
            evalue=1e-50,
            bitscore=500.0,
            query_coverage=0.9,
            subject_coverage=0.9,
            subject_length=1300,
        )
        base.update(overrides)
        return Hit(**base)

    def test_thresholds_reapplied_with_full_accounting(self):
        hits = [
            self.hit(subject_id="keep"),
            self.hit(subject_id="bad_evalue", evalue=1e-2),
            self.hit(subject_id="bad_cover", query_coverage=0.10),
        ]
        kept, counts = filter_hits(iter(hits), params())
        assert [h.subject_id for h in kept] == ["keep"]
        assert counts == {"total": 3, "failed_evalue": 1, "failed_coverage": 1, "kept": 1}

    def test_threshold_boundaries_are_inclusive(self):
        p = params(max_evalue=1e-5, min_query_coverage=0.70)
        kept, _ = filter_hits(iter([self.hit(evalue=1e-5, query_coverage=0.70)]), p)
        assert len(kept) == 1

    def test_unique_subjects_preserves_first_seen_order(self):
        hits = [self.hit(subject_id=s) for s in ("b", "a", "b", "c", "a")]
        assert unique_subjects(hits) == ["b", "a", "c"]


class TestHitTableRoundTrip:
    """The hit table is the contract between stage 1b and stage 1c."""

    def hits(self) -> list[Hit]:
        return [
            Hit("Q99ZW2", "UniRef50_A", 41.2, 1200, 1e-120, 800.5, 0.88, 0.91, 1310),
            Hit("Q99ZW2", "UniRef50_B", 38.0, 1100, 2.5e-90, 640.0, 0.75, 0.80, 1290),
        ]

    def test_write_then_read_preserves_every_field(self, tmp_path):
        from thermocas.data.search import read_hit_table, write_hits

        path = tmp_path / "hits.tsv"
        assert write_hits(path, self.hits(), params()) == 2
        recovered = list(read_hit_table(path))
        assert recovered == self.hits()

    def test_table_records_the_thresholds_that_produced_it(self, tmp_path):
        from thermocas.data.search import write_hits

        path = tmp_path / "hits.tsv"
        write_hits(path, self.hits(), params())
        header = [ln for ln in path.read_text().splitlines() if ln.startswith("#")]
        joined = "\n".join(header)
        assert "tool=diamond" in joined
        assert "database=uniref50" in joined
        assert "max_evalue=1e-05" in joined
        assert "min_query_coverage=0.7" in joined

    def test_coverage_survives_round_trip_as_a_fraction(self, tmp_path):
        """A percentage creeping back in here would break the leakage math."""
        from thermocas.data.search import read_hit_table, write_hits

        path = tmp_path / "hits.tsv"
        write_hits(path, self.hits(), params())
        for hit in read_hit_table(path):
            assert 0.0 <= hit.query_coverage <= 1.0

    def test_header_line_is_not_parsed_as_a_hit(self, tmp_path):
        from thermocas.data.search import read_hit_table, write_hits

        path = tmp_path / "hits.tsv"
        write_hits(path, self.hits(), params())
        assert len(list(read_hit_table(path))) == 2

    def test_empty_result_writes_a_readable_table(self, tmp_path):
        from thermocas.data.search import read_hit_table, write_hits

        path = tmp_path / "hits.tsv"
        assert write_hits(path, [], params()) == 0
        assert list(read_hit_table(path)) == []

    def test_truncated_row_reports_the_line(self, tmp_path):
        from thermocas.data.search import read_hit_table

        path = tmp_path / "hits.tsv"
        path.write_text("# c\n" + "\t".join(HIT_COLUMNS) + "\nq\ts\t41\n")
        with pytest.raises(ValueError, match="hits.tsv:3"):
            list(read_hit_table(path))


def test_full_parse_filter_write_chain(tmp_path):
    """Simulates stage 1b: raw tool output in, normalized table out."""
    from thermocas.data.search import read_hit_table, write_hits

    raw = tmp_path / "raw.tsv"
    rows = [
        # keeper: strong e-value, 88% query coverage
        ["Q99ZW2", "good", "41.2", "1200", "1", "1200", "1", "1190", "1e-120", "800", "88.0", "91.0", "1310"],
        # rejected on coverage: a 200-residue fragment of a 1,368 aa query
        ["Q99ZW2", "fragment", "62.0", "200", "1", "200", "1", "200", "1e-40", "300", "14.6", "99.0", "205"],
        # rejected on e-value
        ["Q99ZW2", "weak", "22.0", "900", "1", "900", "1", "900", "0.004", "45", "80.0", "70.0", "1100"],
    ]
    raw.write_text("\n".join("\t".join(r) for r in rows) + "\n")

    p = params()
    kept, counts = filter_hits(parse_hits(raw, "diamond"), p)
    assert counts == {"total": 3, "failed_evalue": 1, "failed_coverage": 1, "kept": 1}

    out = tmp_path / "hits.tsv"
    write_hits(out, kept, p)
    (recovered,) = list(read_hit_table(out))
    assert recovered.subject_id == "good"
    assert recovered.query_coverage == pytest.approx(0.88)
