"""
Regression tests for the onsitec merge join (bigbio/onsite#40).

The old merge paired the three tools' results by list position, so a single
dropped/missing PSM in one tool shifted every later index and fused scores from
different peptides. _join_psms_by_ref matches by spectrum_reference instead.
"""

from pyopenms import AASequence, PeptideHit, PeptideIdentification

from onsite.onsitec import _join_psms_by_ref


def _pid(spectrum_ref, sequence, score):
    """Build an in-memory PeptideIdentification with one hit (no file I/O)."""
    hit = PeptideHit()
    hit.setSequence(AASequence.fromString(sequence))
    hit.setScore(score)
    pid = PeptideIdentification()
    pid.setMetaValue("spectrum_reference", spectrum_ref)
    pid.setHits([hit])
    return pid


def test_join_aligns_by_reference_not_position():
    """AScore is missing the middle PSM (s2). Positional merge would fuse
    AScore's s3 with LucXor's s2; the key-join must pair s3 with s3."""
    ascore = [_pid("s1", "SAMPLER", 11.0), _pid("s3", "YEASTPEPK", 33.0)]   # no s2
    phosphors = [_pid("s1", "SAMPLER", 1.0), _pid("s2", "TESTPEPK", 2.0), _pid("s3", "YEASTPEPK", 3.0)]
    lucxor = [_pid("s1", "SAMPLER", 91.0), _pid("s2", "TESTPEPK", 92.0), _pid("s3", "YEASTPEPK", 93.0)]

    triples, stats = _join_psms_by_ref(ascore, phosphors, lucxor)

    # Only the spectra present in all three tools survive, in LucXor order.
    assert stats["merged"] == 2
    refs = [l.getMetaValue("spectrum_reference") for (_a, _p, l) in triples]
    assert refs == ["s1", "s3"]

    # The s3 triple pairs AScore-s3 with LucXor-s3 (the bug would pair s3 with s2).
    a3, p3, l3 = triples[1]
    assert a3.getHits()[0].getScore() == 33.0   # AScore's s3, not its s1
    assert l3.getHits()[0].getScore() == 93.0   # LucXor's s3, not s2 (92.0)
    assert (
        a3.getHits()[0].getSequence().toUnmodifiedString()
        == l3.getHits()[0].getSequence().toUnmodifiedString()
        == "YEASTPEPK"
    )

    # s2 is reported as excluded from AScore, not silently merged.
    assert stats["ascore_dropped"] == 0          # AScore has exactly the common set
    assert stats["phosphors_dropped"] == 1       # s2 only-in-2-tools
    assert stats["lucxor_dropped"] == 1


def test_join_skips_backbone_mismatch():
    """If the backbones disagree for the same reference, the triple is skipped
    (guards against a key collision fusing different peptides)."""
    ascore = [_pid("s1", "PEPTIDEK", 5.0)]
    phosphors = [_pid("s1", "PEPTIDEK", 6.0)]
    lucxor = [_pid("s1", "DIFFERENTK", 7.0)]   # same ref, different peptide

    triples, stats = _join_psms_by_ref(ascore, phosphors, lucxor)
    assert triples == []
    assert stats["seq_mismatch"] == 1
    assert stats["merged"] == 0


def test_join_identical_sets():
    """When all three tools share the same PSMs, every one is merged in order."""
    refs = ["a", "b", "c"]
    ascore = [_pid(r, "PEPS(Phospho)K", 1.0) for r in refs]
    phosphors = [_pid(r, "PEPS(Phospho)K", 2.0) for r in refs]
    lucxor = [_pid(r, "PEPS(Phospho)K", 3.0) for r in refs]

    triples, stats = _join_psms_by_ref(ascore, phosphors, lucxor)
    assert stats["merged"] == 3
    assert [l.getMetaValue("spectrum_reference") for (_a, _p, l) in triples] == refs
    assert all(d == 0 for k, d in stats.items() if k.endswith("dropped"))
