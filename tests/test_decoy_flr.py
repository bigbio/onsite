"""
Tests for the unified decoy-amino-acid FLR estimator (onsite/decoy_flr.py, #40).

The estimator and the parsing are tested independently of any idXML I/O.
"""

import numpy as np

from onsite.decoy_flr import (
    flr_curve,
    sites_at_flr,
    parse_localized_sites,
    is_unambiguous,
    _collapse_sites,
    parse_tool_idxml,
    compute_tool_flr,
    compute_decoy_flr,
    PSMRecord,
)


# ───────────────────────────── pure estimator ──────────────────────────────
def test_flr_curve_hand_computed():
    # ratio T_c/X_c = 1 -> factor 2; one decoy at the worst (5th) rank.
    curve = flr_curve([False, False, False, False, True], t_c=4, x_c=4)
    assert list(curve["cum_decoy"]) == [0, 0, 0, 0, 1]
    assert list(curve["cum_target"]) == [1, 2, 3, 4, 4]
    # flr_raw: 0,0,0,0, 2*(4/4)*1/5 = 0.4
    np.testing.assert_allclose(curve["flr_raw"], [0, 0, 0, 0, 0.4])
    # qval == flr_raw here (already monotone non-decreasing)
    np.testing.assert_allclose(curve["qval"], [0, 0, 0, 0, 0.4])


def test_flr_capped_at_one_when_decoys_dominate():
    # All decoys -> raw 2*1=2 at every rank, must cap at 1.0.
    curve = flr_curve([True, True, True], t_c=10, x_c=10)
    assert np.all(curve["flr_raw"] == 1.0)
    assert np.all(curve["qval"] == 1.0)
    assert sites_at_flr(curve, 0.05) == (0, 0, 0)


def test_qvalue_monotonization():
    # A single decoy at the TOP poisons the whole list (q-value style).
    curve = flr_curve([True] + [False] * 5, t_c=10, x_c=10)
    # flr_raw decreases (1,1,0.667,...0.333) but qval is the reverse cummin.
    assert np.all(np.diff(curve["qval"]) >= -1e-12)  # non-decreasing
    np.testing.assert_allclose(curve["qval"], np.full(6, curve["flr_raw"][-1]))
    # Cannot reach 5% FLR; at 40% everything is included.
    assert sites_at_flr(curve, 0.05) == (0, 0, 0)
    assert sites_at_flr(curve, 0.4) == (6, 5, 1)


def test_sites_at_flr_picks_deepest_rank():
    curve = flr_curve([False, False, False, False, True], t_c=4, x_c=4)
    # qval = [0,0,0,0,0.4]; at 5% the deepest admissible rank is 4.
    assert sites_at_flr(curve, 0.05) == (4, 4, 0)
    assert sites_at_flr(curve, 0.5) == (5, 4, 1)


def test_flr_curve_empty():
    curve = flr_curve([], t_c=0, x_c=0)
    assert curve["rank"].size == 0
    assert sites_at_flr(curve, 0.05) == (0, 0, 0)


def test_flr_curve_zero_decoys_no_normalization_crash():
    # X_c == 0 must not divide by zero; ratio falls back to 0.
    curve = flr_curve([False, False], t_c=5, x_c=0)
    assert np.all(curve["flr_raw"] == 0.0)


# ───────────────────────────── sequence parsing ────────────────────────────
def test_parse_localized_sites_phospho():
    unmod, sites = parse_localized_sites("FLLEPTDVVTSRS(Phospho)K")
    assert unmod == "FLLEPTDVVTSRSK"
    assert sites == [(12, "S", "Phospho")]  # 0-based index of the modified S


def test_parse_localized_sites_decoy():
    unmod, sites = parse_localized_sites("ALLSSSA(PhosphoDecoy)VLYK")
    assert unmod == "ALLSSSAVLYK"
    assert sites == [(6, "A", "PhosphoDecoy")]


def test_parse_localized_sites_mixed():
    unmod, sites = parse_localized_sites("LALT(Phospho)LA(PhosphoDecoy)VRK")
    assert unmod == "LALTLAVRK"
    assert sites == [(3, "T", "Phospho"), (5, "A", "PhosphoDecoy")]


def test_is_unambiguous():
    # one candidate (single S), one phospho -> no localization choice
    assert is_unambiguous("PEPSK", 1) is True
    # three candidates (S,T,Y), one phospho -> ambiguous
    assert is_unambiguous("PEPSTYK", 1) is False
    # decoy A counts as a candidate site
    assert is_unambiguous("PEPSAK", 2) is True   # S + A both occupied
    assert is_unambiguous("PEPSAK", 1) is False  # S or A -> a choice


# ───────────────────────────── site collapsing ─────────────────────────────
def test_collapse_takes_max_and_counts_psms():
    raw = [
        ("PEPSTYK", 3, "S", 5.0, False),
        ("PEPSTYK", 3, "S", 9.0, False),  # same site, higher score, 2 PSMs
        ("PEPSTYK", 5, "A", 1.0, True),
    ]
    collapsed = sorted(_collapse_sites(raw), key=lambda t: t[2], reverse=True)
    # site (PEPSTYK,3): max score 9.0, not decoy, 2 supporting PSMs
    top = [c for c in collapsed if c[2] == 2][0]
    assert top == (9.0, False, 2)
    # site (PEPSTYK,5): decoy, 1 PSM
    assert (1.0, True, 1) in _collapse_sites(raw)


# ───────────────────────────── idXML parsing ───────────────────────────────
_PHOSPHORS_IDXML = """<?xml version="1.0" encoding="UTF-8"?>
<IdXML version="1.5">
 <PeptideIdentification score_type="PhosphoRSScore" spectrum_reference="scan=42">
  <PeptideHit score="0.0" sequence="FLLEPTDVVTSRS(Phospho)K" charge="2">
   <UserParam type="string" name="target_decoy" value="target"/>
   <UserParam type="float" name="q-value" value="0.002"/>
   <UserParam type="string" name="PhosphoRS_site_probs" value="{5: 0.01, 10: 42.3, 12: 57.6}"/>
   <UserParam type="string" name="PhosphoRS_site_delta" value="{5: -30.0, 10: -5.0, 12: 12.5}"/>
  </PeptideHit>
 </PeptideIdentification>
</IdXML>
"""


def test_parse_tool_idxml_phosphors(tmp_path):
    """The PhosphoRS leg ranks on the peptide-score delta (not the saturated
    probability); parse_tool_idxml picks up PhosphoRS_site_delta."""
    p = tmp_path / "phosphors.idXML"
    p.write_text(_PHOSPHORS_IDXML)
    recs = parse_tool_idxml(str(p), "phosphors")
    assert len(recs) == 1
    r = recs[0]
    assert r.spectrum_ref == "scan=42"
    assert r.unmod_seq == "FLLEPTDVVTSRSK"
    assert r.sites == [(12, "S", "Phospho")]
    # ranked on the peptide-score delta; winning S@12 has the largest delta
    assert r.site_scores == {5: -30.0, 10: -5.0, 12: 12.5}
    assert max(r.site_scores, key=r.site_scores.get) == 12
    assert r.is_ident_decoy is False
    assert r.q_value == 0.002


def test_compute_tool_flr_filters_intersect_and_normalizes():
    """Orchestration: ident-decoy drop, q-filter, intersection, unambiguous
    exclusion, T_c/X_c over the analyzed set, and the Eq. 2 wiring."""
    recs = [
        # ambiguous, target win (kept)
        PSMRecord("s1", "PEPSTYK", [(3, "S", "Phospho")], {3: 10.0, 5: 2.0, 6: 1.0}, False, 0.001),
        # ambiguous, decoy A win (kept)
        PSMRecord("s2", "PEPSAK", [(4, "A", "PhosphoDecoy")], {3: 1.0, 4: 5.0}, False, 0.001),
        # identification decoy -> dropped
        PSMRecord("s3", "PEPSTYK", [(3, "S", "Phospho")], {3: 9.0}, True, 0.001),
        # q-value above threshold -> dropped
        PSMRecord("s4", "PEPSTYK", [(3, "S", "Phospho")], {3: 8.0}, False, 0.5),
        # unambiguous (one candidate, one phospho) -> excluded from analysis
        PSMRecord("s5", "PEPSK", [(3, "S", "Phospho")], {3: 1000.0}, False, 0.001),
    ]
    keep = {"s1", "s2", "s4", "s5"}  # s3 also fails the ident filter regardless
    res = compute_tool_flr(recs, "ascore", keep, q_threshold=0.01, flr_threshold=0.05)

    assert res.n_after_ident_filter == 3   # s1, s2, s5 (s3 decoy, s4 q>0.01)
    assert res.n_in_intersection == 3      # all three are in keep
    assert res.n_analyzed_psms == 2        # s5 unambiguous excluded
    # T_c/X_c over analyzed peptides: PEPSTYK -> S,T,Y=3,A=0 ; PEPSAK -> S=1,A=1
    assert res.t_c == 4 and res.x_c == 1

    # Ranked best-first: s1 target (10.0) then s2 decoy (5.0); factor 2*(4/1)=8.
    # At 5% FLR only the top target site qualifies.
    total, target, decoy = sites_at_flr(res.curve, 0.05)
    assert (total, target, decoy) == (1, 1, 0)


# ───────────────────── extensive 3-tool file-based integration ─────────────
def _make_idxml(score_meta, psms):
    """Minimal idXML for one tool. psms: list of (spectrum_ref, sequence, {pos: score})."""
    rows = []
    for sref, seq, scores in psms:
        rows.append(
            f'  <PeptideIdentification score_type="X" spectrum_reference="{sref}">\n'
            f'   <PeptideHit score="0.0" sequence="{seq}" charge="2">\n'
            f'    <UserParam type="string" name="target_decoy" value="target"/>\n'
            f'    <UserParam type="float" name="q-value" value="0.001"/>\n'
            f'    <UserParam type="string" name="{score_meta}" value="{scores}"/>\n'
            f"   </PeptideHit>\n"
            f"  </PeptideIdentification>"
        )
    return '<?xml version="1.0"?>\n<IdXML>\n' + "\n".join(rows) + "\n</IdXML>\n"


def test_compute_decoy_flr_three_tool_integration(tmp_path):
    """End-to-end: parse three idXMLs -> shared filtered intersection -> collapse
    -> Eq. 2 -> threshold, for all three tools and their respective score-meta
    keys (AScore_site_scores / PhosphoRS_site_delta / Luciphor_site_scores).

    Peptide AGSYK has candidates S2, Y3 (target) and A0 (decoy). Three PSMs
    localize to S2, Y3, and A0 respectively, with the decoy scored lowest, so
    at 5% FLR the two target sites are recovered and the decoy is excluded.
    """
    psms = [
        ("scan=1", "AGS(Phospho)YK", {2: 50.0, 0: 10.0, 3: 5.0}),       # target S2
        ("scan=2", "AGSY(Phospho)K", {3: 40.0, 0: 8.0, 2: 6.0}),        # target Y3
        ("scan=3", "A(PhosphoDecoy)GSYK", {0: 30.0, 2: 5.0, 3: 4.0}),   # decoy A0 (lowest)
    ]
    tool_meta = {
        "ascore": "AScore_site_scores",
        "phosphors": "PhosphoRS_site_delta",
        "lucxor": "Luciphor_site_scores",
    }
    paths = {}
    for tool, meta in tool_meta.items():
        p = tmp_path / f"{tool}.idXML"
        p.write_text(_make_idxml(meta, psms))
        paths[tool] = str(p)

    res = compute_decoy_flr(paths, q_threshold=0.01, flr_threshold=0.05)
    assert set(res) == {"ascore", "phosphors", "lucxor"}
    for tool, r in res.items():
        assert r.n_in_intersection == 3, tool       # all spectra shared
        assert r.n_analyzed_psms == 3, tool          # all ambiguous, all scored
        assert (r.t_c, r.x_c) == (6, 3), tool        # STY=2, A=1 per PSM x 3
        # Decoy (A0, score lowest) ranks last; factor 2*(6/3)=4 -> FLR hits 1.0
        # at rank 3, so 5% FLR recovers the two target sites only.
        assert sites_at_flr(r.curve, 0.05) == (2, 2, 0), tool
