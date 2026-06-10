"""
Regression tests for the onsitec merge join (bigbio/onsite#40).

The merge now operates on DataFrames, joining by spectrum_reference.
"""

import numpy as np
import pandas as pd

from onsite.onsitec import _join_psms_by_ref


def _df(rows):
    """Build a psms-like DataFrame from a list of dicts. Each dict has at least
    hit_index, spectrum_reference, sequence, and peptide_identification_index."""
    return pd.DataFrame(rows)


def test_join_aligns_by_reference_not_position():
    """AScore is missing the middle PSM (s2). Positional merge would fuse
    AScore's s3 with LucXor's s2; the key-join must pair s3 with s3."""
    ascore = _df([
        {"peptide_identification_index": 0, "hit_index": 0, "spectrum_reference": "s1", "sequence": "SAMPLER"},
        {"peptide_identification_index": 1, "hit_index": 0, "spectrum_reference": "s3", "sequence": "YEASTPEPK"},
    ])
    phosphors = _df([
        {"peptide_identification_index": 0, "hit_index": 0, "spectrum_reference": "s1", "sequence": "SAMPLER"},
        {"peptide_identification_index": 1, "hit_index": 0, "spectrum_reference": "s2", "sequence": "TESTPEPK"},
        {"peptide_identification_index": 2, "hit_index": 0, "spectrum_reference": "s3", "sequence": "YEASTPEPK"},
    ])
    lucxor = _df([
        {"peptide_identification_index": 0, "hit_index": 0, "spectrum_reference": "s1", "sequence": "SAMPLER"},
        {"peptide_identification_index": 1, "hit_index": 0, "spectrum_reference": "s2", "sequence": "TESTPEPK"},
        {"peptide_identification_index": 2, "hit_index": 0, "spectrum_reference": "s3", "sequence": "YEASTPEPK"},
    ])

    triples, stats = _join_psms_by_ref(ascore, phosphors, lucxor)

    assert stats["merged"] == 2

    # s3 triple: AScore's s3, LucXor's s3
    ai, _, li = triples[1]
    assert ai in ascore[ascore["spectrum_reference"] == "s3"].index
    assert li in lucxor[lucxor["spectrum_reference"] == "s3"].index

    assert stats["ascore_dropped"] == 0
    assert stats["phosphors_dropped"] == 1
    assert stats["lucxor_dropped"] == 1


def test_join_skips_backbone_mismatch():
    """If the backbones disagree for the same reference, the triple is skipped."""
    ascore = _df([
        {"peptide_identification_index": 0, "hit_index": 0, "spectrum_reference": "s1", "sequence": "PEPTIDEK"},
    ])
    phosphors = _df([
        {"peptide_identification_index": 0, "hit_index": 0, "spectrum_reference": "s1", "sequence": "PEPTIDEK"},
    ])
    lucxor = _df([
        {"peptide_identification_index": 0, "hit_index": 0, "spectrum_reference": "s1", "sequence": "DIFFERENTK"},
    ])

    triples, stats = _join_psms_by_ref(ascore, phosphors, lucxor)
    assert triples == []
    assert stats["seq_mismatch"] == 1
    assert stats["merged"] == 0


def test_join_identical_sets():
    """When all three tools share the same PSMs, every one is merged in order."""
    refs = ["a", "b", "c"]
    rows = [{"peptide_identification_index": i, "hit_index": 0, "spectrum_reference": r, "sequence": "PEPSK"}
            for i, r in enumerate(refs)]
    ascore = _df(rows)
    phosphors = _df(rows)
    lucxor = _df(rows)

    _, stats = _join_psms_by_ref(ascore, phosphors, lucxor)
    assert stats["merged"] == 3
    assert all(d == 0 for k, d in stats.items() if k.endswith("dropped"))
