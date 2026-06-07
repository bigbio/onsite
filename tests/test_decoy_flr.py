"""
Tests for the unified decoy-amino-acid FLR estimator (onsite/decoy_flr.py, #40).
"""

import os
import tempfile

import numpy as np
import pandas as pd

from onsite.decoy_flr import (
    flr_curve,
    sites_at_flr,
    parse_localized_sites,
    is_unambiguous,
    _collapse_sites,
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
    assert sites == [(13, "S", "Phospho")]  # 0-based index of the modified S


def test_parse_localized_sites_decoy():
    unmod, sites = parse_localized_sites("ALLSSSA(PhosphoDecoy)VLYK")
    assert unmod == "ALLSSSAVLYK"
    assert sites == [(7, "A", "PhosphoDecoy")]


def test_parse_localized_sites_mixed():
    unmod, sites = parse_localized_sites("LALT(Phospho)LA(PhosphoDecoy)VRK")
    assert unmod == "LALTLAVRK"
    assert sites == [(4, "T", "Phospho"), (6, "A", "PhosphoDecoy")]


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
        ("PEPSTYK", 4, "S", 5.0, False),
        ("PEPSTYK", 4, "S", 9.0, False),  # same site, higher score, 2 PSMs
        ("PEPSTYK", 6, "A", 1.0, True),
    ]
    collapsed = sorted(_collapse_sites(raw), key=lambda t: t[2], reverse=True)
    # site (PEPSTYK,3): max score 9.0, not decoy, 2 supporting PSMs
    top = [c for c in collapsed if c[2] == 2][0]
    assert top == (9.0, False, 2)
    # site (PEPSTYK,5): decoy, 1 PSM
    assert (1.0, True, 1) in _collapse_sites(raw)


# ─────────────────────────── idParquet parsing ───────────────────────────────


def _make_parquet_dir(tmp_path: str, tool: str, psm_rows: list) -> str:
    """Create a minimal idparquet directory with a psms.parquet for one tool.

    psm_rows: list of dicts with keys matching psms.parquet schema.
    """
    out_dir = os.path.join(tmp_path, tool)
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(psm_rows)
    df.to_parquet(os.path.join(out_dir, "psms.parquet"), index=False)
    return out_dir


def test_parse_tool_idparquet_phosphors(tmp_path):
    """PhosphoRS parquet parsing: site scores picked up from psm_metavalues."""
    rows = [
        {
            "peptide_identification_index": 0,
            "hit_index": 0,
            "spectrum_reference": "scan=42",
            "peptidoform": "FLLEPTDVVTSRS[UNIMOD:21]K",
            "sequence": "FLLEPTDVVTSRSK",
            "is_decoy": False,
            "psm_metavalues": np.array([
                {"name": "target_decoy", "value": "target", "value_type": "string"},
                {"name": "percolator_q_value", "value": "0.002", "value_type": "double"},
                {"name": "PhosphoRS_site_probs", "value": "{6: 0.01, 11: 42.3, 13: 57.6}", "value_type": "string"},
                {"name": "PhosphoRS_site_delta", "value": "{6: -30.0, 11: -5.0, 13: 12.5}", "value_type": "string"},
            ], dtype=object),
            "score": 0.0,
            "score_type": "PhosphoRSScore",
            "higher_score_better": False,
        }
    ]
    d = _make_parquet_dir(tmp_path, "phosphors", rows)
    from onsite.decoy_flr import parse_tool_idparquet
    recs = parse_tool_idparquet(d, "phosphors")
    assert len(recs) == 1
    r = recs[0]
    assert r.spectrum_ref == "scan=42"
    assert r.unmod_seq == "FLLEPTDVVTSRSK"
    assert r.sites == [(13, "S", "Phospho")]
    assert r.site_scores == {6: -30.0, 11: -5.0, 13: 12.5}
    assert max(r.site_scores, key=r.site_scores.get) == 13
    assert r.is_ident_decoy is False
    assert r.q_value == 0.002


def test_compute_tool_flr_filters_intersect_and_normalizes():
    """Orchestration: ident-decoy drop, q-filter, intersection, unambiguous
    exclusion, T_c/X_c over the analyzed set, and the Eq. 2 wiring."""
    recs = [
        # ambiguous, target win (kept)
        PSMRecord("s1", "PEPSTYK", [(4, "S", "Phospho")], {4: 10.0, 6: 2.0, 7: 1.0}, False, 0.001),
        # ambiguous, decoy A win (kept)
        PSMRecord("s2", "PEPSAK", [(5, "A", "PhosphoDecoy")], {4: 1.0, 5: 5.0}, False, 0.001),
        # identification decoy -> dropped
        PSMRecord("s3", "PEPSTYK", [(4, "S", "Phospho")], {4: 9.0}, True, 0.001),
        # q-value above threshold -> dropped
        PSMRecord("s4", "PEPSTYK", [(4, "S", "Phospho")], {4: 8.0}, False, 0.5),
        # unambiguous (one candidate, one phospho) -> excluded from analysis
        PSMRecord("s5", "PEPSK", [(4, "S", "Phospho")], {4: 1000.0}, False, 0.001),
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


def _make_psm_row(pep_idx, hit_idx, sref, seq, peptidoform, score_meta, scores, is_decoy=False):
    """Build a single psms.parquet row dict."""
    return {
        "peptide_identification_index": pep_idx,
        "hit_index": hit_idx,
        "spectrum_reference": sref,
        "sequence": seq,
        "peptidoform": peptidoform,
        "is_decoy": is_decoy,
        "score": 0.0,
        "score_type": "X",
        "higher_score_better": True,
        "psm_metavalues": np.array([
            {"name": "target_decoy", "value": "decoy" if is_decoy else "target", "value_type": "string"},
            {"name": "percolator_q_value", "value": "0.001", "value_type": "double"},
            {"name": score_meta, "value": str(scores), "value_type": "string"},
        ], dtype=object),
    }


def test_compute_decoy_flr_three_tool_integration(tmp_path):
    """End-to-end: parse three idparquet dirs -> shared filtered intersection
    -> collapse -> Eq. 2 -> threshold, for all three tools.

    Peptide AGSYK has candidates S2, Y3 (target) and A0 (decoy). Three PSMs
    localize to S2, Y3, and A0 respectively, with the decoy scored lowest, so
    at 5% FLR the two target sites are recovered and the decoy is excluded.
    """
    tool_meta = {
        "ascore": "AScore_site_scores",
        "phosphors": "PhosphoRS_site_delta",
        "lucxor": "Luciphor_site_scores",
    }
    # Common PSMs across tools: same spectra, same sites, same scores.
    psms_data = [
        (0, 0, "scan=1", "AGS(Phospho)YK", "AGS[UNIMOD:21]YK", {3: 50.0, 1: 10.0, 4: 5.0}, False),
        (1, 0, "scan=2", "AGSY(Phospho)K", "AGSY[UNIMOD:21]K", {4: 40.0, 1: 8.0, 3: 6.0}, False),
        (2, 0, "scan=3", "A(PhosphoDecoy)GSYK", "A[UNIMOD:21]GSYK", {1: 30.0, 3: 5.0, 4: 4.0}, False),
    ]
    paths = {}
    for tool, meta in tool_meta.items():
        rows = [_make_psm_row(p[0], p[1], p[2], p[3], p[4], meta, p[5], p[6])
                for p in psms_data]
        d = os.path.join(tmp_path, tool)
        os.makedirs(d, exist_ok=True)
        pd.DataFrame(rows).to_parquet(os.path.join(d, "psms.parquet"), index=False)
        paths[tool] = d

    res = compute_decoy_flr(paths, q_threshold=0.01, flr_threshold=0.05)
    assert set(res) == {"ascore", "phosphors", "lucxor"}
    for tool, r in res.items():
        assert r.n_in_intersection == 3, tool
        assert r.n_analyzed_psms == 3, tool
        assert (r.t_c, r.x_c) == (6, 3), tool
        assert sites_at_flr(r.curve, 0.05) == (2, 2, 0), tool
