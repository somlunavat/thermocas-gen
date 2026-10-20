"""FASTA parsing and writing."""

from __future__ import annotations

import pytest

from thermocas.data.fasta import FastaRecord, read_fasta, write_fasta


def write(tmp_path, text, name="t.fasta"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_parses_multiline_and_strips_whitespace(tmp_path):
    path = write(tmp_path, ">a desc here\nMGK\n PIP \n\n>b\nAAA\n")
    records = list(read_fasta(path))
    assert len(records) == 2
    assert records[0].seq_id == "a"
    assert records[0].description == "desc here"
    assert records[0].sequence == "MGKPIP"
    assert records[1].seq_id == "b"
    assert records[1].description == ""


def test_lowercase_is_uppercased(tmp_path):
    (record,) = list(read_fasta(write(tmp_path, ">a\nmgkp\n")))
    assert record.sequence == "MGKP"


def test_duplicate_ids_raise(tmp_path):
    """Downstream joins are keyed by id; a silent overwrite would drop a sequence."""
    path = write(tmp_path, ">a\nMG\n>a\nKP\n")
    with pytest.raises(ValueError, match="duplicate"):
        list(read_fasta(path))


def test_sequence_before_header_raises(tmp_path):
    with pytest.raises(ValueError, match="before any header"):
        list(read_fasta(write(tmp_path, "MGKP\n>a\nMG\n")))


def test_empty_header_raises(tmp_path):
    with pytest.raises(ValueError, match="empty FASTA header"):
        list(read_fasta(write(tmp_path, ">\nMG\n")))


def test_empty_file_yields_nothing(tmp_path):
    assert list(read_fasta(write(tmp_path, ""))) == []


def test_final_record_is_emitted(tmp_path):
    """Regression: the last record has no following '>' to flush it."""
    records = list(read_fasta(write(tmp_path, ">a\nMG\n>b\nKP")))
    assert [r.seq_id for r in records] == ["a", "b"]
    assert records[-1].sequence == "KP"


def test_roundtrip_through_write(tmp_path):
    original = [
        FastaRecord("a", "first", "M" * 130),
        FastaRecord("b", "", "KP"),
    ]
    out = tmp_path / "out.fasta"
    assert write_fasta(out, original) == 2
    reparsed = list(read_fasta(out))
    assert [(r.seq_id, r.description, r.sequence) for r in reparsed] == [
        (r.seq_id, r.description, r.sequence) for r in original
    ]


def test_write_wraps_at_line_width(tmp_path):
    out = tmp_path / "out.fasta"
    write_fasta(out, [FastaRecord("a", "", "M" * 150)], line_width=60)
    lines = out.read_text().splitlines()
    assert lines[0] == ">a"
    assert [len(ln) for ln in lines[1:]] == [60, 60, 30]


def test_write_creates_parent_directories(tmp_path):
    out = tmp_path / "nested" / "deeper" / "out.fasta"
    assert write_fasta(out, [FastaRecord("a", "", "MG")]) == 1
    assert out.is_file()


def test_len_reports_sequence_length():
    assert len(FastaRecord("a", "", "MGKP")) == 4


class TestCanonicalSeqId:
    """MMseqs2 keys its cluster table on the bare accession.

    A corpus written with full sp|ACC|NAME identifiers produces a cluster
    table keyed on something else, so every downstream join finds nothing and
    the split comes out empty. This was a real bug, caught only because the
    Cas9 seeds happen to use plain accessions with no pipes.
    """

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("sp|P33247|SQHC_ALIAD", "P33247"),
            ("tr|A0A2M9CY37|A0A2M9CY37_9BACT", "A0A2M9CY37"),
            ("ref|NP_123456|something", "NP_123456"),
            ("gb|AAA12345|x", "AAA12345"),
        ],
    )
    def test_uniprot_style_headers_reduce_to_the_accession(self, raw, expected):
        from thermocas.data.fasta import canonical_seq_id

        assert canonical_seq_id(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        ["Q99ZW2", "WP_099233044.1", "UniRef50_Q99ZW2", "A0ABF7PG96", "G3ECR1"],
    )
    def test_pipe_free_identifiers_are_unchanged(self, raw):
        """MMseqs2 keeps these as-is, so canonicalizing must not alter them."""
        from thermocas.data.fasta import canonical_seq_id

        assert canonical_seq_id(raw) == raw

    def test_two_field_form_keeps_the_first_field(self):
        from thermocas.data.fasta import canonical_seq_id

        assert canonical_seq_id("ACC|NAME") == "ACC"

    def test_is_idempotent(self):
        """Applying it twice must not strip further."""
        from thermocas.data.fasta import canonical_seq_id

        once = canonical_seq_id("sp|P33247|SQHC_ALIAD")
        assert canonical_seq_id(once) == once

    def test_every_committed_seed_is_already_canonical(self):
        from thermocas.config import Config
        from thermocas.data.fasta import canonical_seq_id, read_fasta

        for record in read_fasta(Config.load("data").resolve_path("paths/seeds")):
            assert canonical_seq_id(record.seq_id) == record.seq_id
