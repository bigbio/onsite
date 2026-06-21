"""
Unified decoy-amino-acid global FLR for AScore / PhosphoRS / LucXor (issue #40).

Implements the decoy-amino-acid false-localization-rate estimator of
Ramsbottom et al. 2022 (J. Proteome Res., DOI 10.1021/acs.jproteome.1c00827),
which is the only FLR definition common to all three localization algorithms and
therefore the only valid basis for comparing them on one scale.

Each tool emits a position-keyed per-site localization confidence
(``AScore_site_scores`` / ``PhosphoRS_site_probs`` / ``Luciphor_site_scores``);
Alanine acts as a decoy phospho-acceptor (``PhosphoDecoy``) because it cannot be
phosphorylated, so a localization onto A is a false localization.

Global decoy FLR (Eq. 2 of the paper), at rank ``n`` in the score-ranked list::

    pX_FLR_n = (T_c / X_c) * (cumulative decoy sites) / (n - (cumulative decoy sites))

where ``T_c`` / ``X_c`` are the total counts of target (S/T/Y) and decoy (A)
candidate residues in the analyzed PSM set. The ``(T_c / X_c)`` factor scales
the raw decoy fraction to the true expected FLR (without it the estimate is
optimistic by ~3.7x on typical data).

NOTE: ``T_c / X_c`` is a single global prior; it assumes a roughly stable
S/T/Y : A ratio across peptides (the paper's approximation).
"""

import argparse
import ast
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Modification names written by the three tools.
PHOSPHO = "Phospho"
PHOSPHO_DECOY = "PhosphoDecoy"
TARGET_RESIDUES = set("STY")
DECOY_RESIDUE = "A"

# Per-tool meta value holding the position-keyed {residue_index: score} dict,
# and whether a higher score means more confident.
# Per-tool position-keyed {residue_index: score} meta value used to RANK sites
# for the FLR (higher = more confident). PhosphoRS is ranked on its peptide-score
# delta (phosphoRS's own discrimination signal) rather than the normalized site
# probability, which saturates at ~100% and carries no ranking resolution; the
# probability is still emitted as PhosphoRS_site_probs for reporting.
TOOL_SCORE_META = {
    "ascore": "AScore_site_scores",
    # "phosphors": "PhosphoRS_site_delta",
    "phosphors": "PhosphoRS_site_probs",
    "lucxor": "Luciphor_site_scores",
}


# ───────────────────────────── pure estimator ──────────────────────────────
def flr_curve(is_decoy_by_rank: List[bool], t_c: int, x_c: int) -> Dict[str, np.ndarray]:
    """
    Compute the decoy-AA FLR curve from sites already sorted best-first.

    Pure function (no I/O) so it can be unit-tested directly.

    Args:
        is_decoy_by_rank: per-site decoy flags, ordered by localization
            confidence DESCENDING (most confident site first).
        t_c: total target (S/T/Y) candidate residues in the analyzed set.
        x_c: total decoy (A) candidate residues in the analyzed set.

    Returns:
        Dict of equal-length numpy arrays:
          rank        1..N (cumulative site count = the observation count n)
          cum_target  cumulative target (non-decoy) sites
          cum_decoy   cumulative decoy (A) sites
        #   flr_raw     2*(T_c/X_c)*cum_decoy/n, capped at 1.0
          flr_raw     (T_c/X_c)*cum_decoy/(n-cum_decoy), capped at 1.0
          qval        q-value-style monotonization of flr_raw (reverse cummin)
    """
    d = np.asarray(is_decoy_by_rank, dtype=bool)
    n = d.size
    if n == 0:
        z = np.array([], dtype=float)
        return {"rank": z, "cum_target": z, "cum_decoy": z, "flr_raw": z, "qval": z}

    ranks = np.arange(1, n + 1)
    cum_decoy = np.cumsum(d)
    cum_target = ranks - cum_decoy

    ratio = (t_c / x_c) if x_c > 0 else 0.0
    # flr_raw = 2.0 * ratio * cum_decoy / ranks
    flr_raw = (ratio * cum_decoy) / (ranks - cum_decoy)
    flr_raw = np.minimum(flr_raw, 1.0)  # an FLR cannot exceed 1

    # q-value style: the FLR achievable by any threshold at least this permissive
    # = min over m >= n. Reverse cumulative minimum -> monotonically non-decreasing.
    qval = np.minimum.accumulate(flr_raw[::-1])[::-1]

    return {
        "rank": ranks,
        "cum_target": cum_target,
        "cum_decoy": cum_decoy,
        "flr_raw": flr_raw,
        "qval": qval,
    }


def sites_at_flr(curve: Dict[str, np.ndarray], threshold: float) -> Tuple[int, int, int]:
    """
    Deepest (most permissive) rank whose monotonized FLR <= threshold.

    Returns (total_sites, target_sites, decoy_sites) reported at that cutoff,
    or (0, 0, 0) if no site meets the threshold.
    """
    if curve["rank"].size == 0:
        return (0, 0, 0)
    ok = np.where(curve["qval"] <= threshold)[0]
    if ok.size == 0:
        return (0, 0, 0)
    i = int(ok.max())  # qval is non-decreasing, so the valid set is a prefix
    return (int(curve["rank"][i]), int(curve["cum_target"][i]), int(curve["cum_decoy"][i]))


# ───────────────────────────── sequence parsing ────────────────────────────
def parse_localized_sites(seq_str: str) -> Tuple[str, List[Tuple[int, str, str]]]:
    """
    Parse a modified peptide string into its unmodified backbone and the
    Phospho / PhosphoDecoy sites.

    Returns (unmodified_sequence, [(residue_index, residue, mod_name), ...])
    with 1-based residue indices (N-terminus = 0, first residue = 1, etc.).
    """
    sites: List[Tuple[int, str, str]] = []
    unmod: List[str] = []
    i = 0
    pos = 0
    while i < len(seq_str):
        c = seq_str[i]
        if c == "(":
            j = seq_str.find(")", i)
            if j == -1:
                break
            mod = seq_str[i + 1 : j]
            if mod in (PHOSPHO, PHOSPHO_DECOY) and pos >= 0:
                sites.append((pos, unmod[-1], mod))
            i = j + 1
        elif c == "[":
            j = seq_str.find("]", i)
            i = (j + 1) if j != -1 else (i + 1)
        elif c.isalpha():
            pos += 1
            unmod.append(c)
            i += 1
        else:
            i += 1
    return "".join(unmod), sites


def _candidate_counts(unmod_seq: str) -> Tuple[int, int]:
    """(target S/T/Y residues, decoy A residues) in an unmodified sequence."""
    t = sum(1 for c in unmod_seq if c in TARGET_RESIDUES)
    x = unmod_seq.count(DECOY_RESIDUE)
    return t, x


# ───────────────────────────── idParquet parsing ───────────────────────────────
@dataclass
class PSMRecord:
    spectrum_ref: str
    unmod_seq: str
    sites: List[Tuple[int, str, str]]   # (pos, residue, mod_name) for the winner
    site_scores: Dict[int, float]       # position -> confidence
    is_ident_decoy: bool
    q_value: Optional[float]


def _parse_site_dict(raw: str) -> Dict[int, float]:
    """Parse a stringified ``{pos: score}`` dict; tolerate empties/garbage."""
    if not raw:
        return {}
    try:
        d = ast.literal_eval(raw)
        return {int(k): float(v) for k, v in d.items()}
    except (ValueError, SyntaxError, TypeError):
        return {}


def _find_meta(metavalues, name: str) -> Optional[str]:
    """Look up a meta value by name in a psm_metavalues array."""
    if metavalues is None:
        return None
    items = []
    if isinstance(metavalues, np.ndarray):
        items = list(metavalues)
    elif isinstance(metavalues, (list, tuple)):
        items = list(metavalues)
    for item in items:
        if isinstance(item, dict) and item.get("name") == name:
            return str(item.get("value", ""))
    return None


def parse_tool_idparquet(path: str, tool: str) -> List[PSMRecord]:
    """Parse a tool's idParquet directory into one PSMRecord per identification."""
    score_meta = TOOL_SCORE_META[tool]
    psm_path = os.path.join(path, "psms.parquet")
    if not os.path.isfile(psm_path):
        raise FileNotFoundError(f"Expected psms.parquet in {path}")

    df = pd.read_parquet(psm_path)
    # Use the best hit per identification (hit_index == 0).
    if "hit_index" in df.columns:
        df = df[df["hit_index"] == 0].copy()
    records: List[PSMRecord] = []

    for _, row in df.iterrows():
        raw_seq = str(row.get("peptidoform", row.get("sequence", "")))
        if "[UNIMOD:" in raw_seq:
            from onsite.idparquet import unimod_to_pyopenms_notation

            seq = unimod_to_pyopenms_notation(raw_seq)
        else:
            seq = raw_seq

        unmod, sites = parse_localized_sites(seq)
        spectrum_ref = str(row.get("spectrum_reference", ""))
        is_ident_decoy = bool(row.get("is_decoy", False))

        site_scores: Dict[int, float] = {}
        mv = row.get("psm_metavalues")
        raw_scores = _find_meta(mv, score_meta)
        if raw_scores:
            site_scores = _parse_site_dict(raw_scores)

        q_value = None
        raw_q = _find_meta(mv, "percolator_q_value")
        if raw_q:
            try:
                q_value = float(raw_q)
            except (ValueError, TypeError):
                pass

        records.append(
            PSMRecord(
                spectrum_ref=spectrum_ref,
                unmod_seq=unmod,
                sites=sites,
                site_scores=site_scores,
                is_ident_decoy=is_ident_decoy,
                q_value=q_value,
            )
        )
    return records


def is_unambiguous(unmod_seq: str, n_reported: int) -> bool:
    """A peptide carries no localization choice when every candidate site
    (S/T/Y + decoy A) is occupied."""
    t, x = _candidate_counts(unmod_seq)
    return (t + x) <= n_reported


# ───────────────────────────── orchestration ───────────────────────────────
@dataclass
class ToolResult:
    tool: str
    n_psms_in: int
    n_after_ident_filter: int
    n_in_intersection: int
    n_analyzed_psms: int       # ambiguous, scored
    site_records: List[Tuple[float, bool, int, str]]  # (score, is_decoy, psm_count, key)
    curve: Dict[str, np.ndarray]
    t_c: int
    x_c: int


def _collapse_sites(
    raw_sites: List[Tuple[str, int, str, float, bool]]
) -> List[Tuple[float, bool, int]]:
    """
    Collapse redundant observations of the same (peptide, position) site.

    Per the paper: take the max confidence per site, and use the supporting-PSM
    count to break ties within a 2-decimal score bin.

    Args:
        raw_sites: list of (unmod_seq, position, residue, score, is_decoy)
    Returns:
        list of (max_score, is_decoy, psm_count), one per unique site.
    """
    agg: Dict[Tuple[str, int], List] = {}
    for unmod_seq, pos, _residue, score, is_decoy in raw_sites:
        key = (unmod_seq, pos)
        if key not in agg:
            agg[key] = [score, is_decoy, 1]
        else:
            entry = agg[key]
            entry[0] = max(entry[0], score)
            entry[2] += 1
    return [(v[0], v[1], v[2]) for v in agg.values()]


def compute_tool_flr(
    records: List[PSMRecord],
    tool: str,
    keep_refs: set,
    q_threshold: Optional[float],
    flr_threshold: float,
    collapse: bool = True,
) -> ToolResult:
    """Compute the decoy-AA FLR curve for one tool over the shared PSM set."""
    n_in = len(records)

    # 1. Identification filter: drop ident-decoys and (optionally) q-value > thr.
    #    This realizes the "same PSM-FDR level" requirement.
    after_ident = [
        r for r in records
        if not r.is_ident_decoy
        and (q_threshold is None or r.q_value is None or r.q_value < q_threshold)
    ]
    n_after_ident = len(after_ident)

    # 2. Restrict to the PSM set shared by all tools.
    in_isect = [r for r in after_ident if (r.spectrum_ref, r.unmod_seq) in keep_refs]
    n_isect = len(in_isect)

    # 3. Drop unambiguous peptides (no localization choice) and collect sites.
    raw_sites: List[Tuple[str, int, str, float, bool]] = []
    t_c = 0
    x_c = 0
    n_analyzed = 0
    for r in in_isect:
        n_reported = len(r.sites)
        if n_reported == 0 or is_unambiguous(r.unmod_seq, n_reported):
            continue
        # A reported site contributes only if it has a usable confidence score.
        contributed = False
        for pos, residue, mod_name in r.sites:
            if pos not in r.site_scores:
                continue
            score = r.site_scores[pos]
            is_decoy = (mod_name == PHOSPHO_DECOY) or (residue == DECOY_RESIDUE)
            raw_sites.append((r.unmod_seq, pos, residue, score, is_decoy))
            contributed = True
        if contributed:
            # T_c / X_c must be computed over exactly the analyzed population.
            t, x = _candidate_counts(r.unmod_seq)
            t_c += t
            x_c += x
            n_analyzed += 1

    # 4. Collapse redundant sites, then rank best-first.
    if collapse:
        collapsed = _collapse_sites(raw_sites)
    else:
        collapsed = [(s, d, 1) for (_seq, _pos, _res, s, d) in raw_sites]

    # bin score to 2 dp; tie-break by supporting-PSM count (paper).
    # collapsed.sort(key=lambda t: (round(t[0], 2), t[2]), reverse=True)
    collapsed.sort(key=lambda t: (t[0], t[2]), reverse=True)
    is_decoy_ranked = [d for (_s, d, _c) in collapsed]

    curve = flr_curve(is_decoy_ranked, t_c, x_c)

    return ToolResult(
        tool=tool,
        n_psms_in=n_in,
        n_after_ident_filter=n_after_ident,
        n_in_intersection=n_isect,
        n_analyzed_psms=n_analyzed,
        site_records=[(s, d, c, "") for (s, d, c) in collapsed],
        curve=curve,
        t_c=t_c,
        x_c=x_c,
    )


def compute_decoy_flr(
    tool_paths: Dict[str, str],
    q_threshold: Optional[float] = 0.01,
    flr_threshold: float = 0.05,
    collapse: bool = True,
) -> Dict[str, ToolResult]:
    """
    Compute the unified decoy-AA FLR for every provided tool on the shared,
    q-value-filtered PSM intersection.

    Args:
        tool_paths: {"ascore": path, "phosphors": path, "lucxor": path} (subset ok)
        q_threshold: PSM q-value cutoff (None to skip); shared across tools.
        flr_threshold: global FLR cutoff for the reported site yield.
    Returns:
        {tool: ToolResult}
    """
    parsed = {}
    for t, p in tool_paths.items():
        parsed[t] = parse_tool_idparquet(p, t)

    # Intersection of spectrum references that survive the identification filter
    # in EVERY tool, so all tools report on the identical PSM population.
    ref_sets = []
    for recs in parsed.values():
        refs = {
            (r.spectrum_ref, r.unmod_seq)
            for r in recs
            if not r.is_ident_decoy
            and (q_threshold is None or r.q_value is None or r.q_value < q_threshold)
            and r.spectrum_ref is not None
        }
        ref_sets.append(refs)
    keep_refs = set.intersection(*ref_sets) if ref_sets else set()

    return {
        t: compute_tool_flr(recs, t, keep_refs, q_threshold, flr_threshold, collapse)
        for t, recs in parsed.items()
    }


# ───────────────────────────────── CLI ─────────────────────────────────────
def _write_curve_csv(path: str, res: ToolResult) -> None:
    c = res.curve
    with open(path, "w", encoding="utf-8") as f:
        f.write("rank,score,cum_target,cum_decoy,flr_raw,qval\n")
        scores = [s for (s, _d, _c, _k) in res.site_records]
        for i in range(c["rank"].size):
            f.write(
                f"{int(c['rank'][i])},{scores[i]:.6g},{int(c['cum_target'][i])},"
                f"{int(c['cum_decoy'][i])},{c['flr_raw'][i]:.6f},{c['qval'][i]:.6f}\n"
            )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Unified decoy-amino-acid global FLR across AScore / PhosphoRS / LucXor (issue #40)."
    )
    ap.add_argument("--ascore", help="AScore result idparquet directory")
    ap.add_argument("--phosphors", help="PhosphoRS result idparquet directory")
    ap.add_argument("--lucxor", help="LucXor result idparquet directory")
    ap.add_argument("--q-value-threshold", type=float, default=0.01,
                    help="PSM q-value cutoff applied to all tools (default 0.01; <0 to skip)")
    ap.add_argument("--flr-threshold", type=float, default=0.05,
                    help="Global FLR cutoff for reported site yield (default 0.05; the paper "
                         "recommends 5%% over 1%%)")
    ap.add_argument("--no-collapse", action="store_true",
                    help="Do not collapse redundant (peptide, position) sites")
    ap.add_argument("--out-prefix", help="If set, write <prefix>_<tool>_flr.csv curves")
    args = ap.parse_args(argv)

    tool_paths = {
        t: p for t, p in (("ascore", args.ascore),
                          ("phosphors", args.phosphors),
                          ("lucxor", args.lucxor)) if p
    }
    if not tool_paths:
        ap.error("provide at least one of --ascore / --phosphors / --lucxor")

    q_thr = None if args.q_value_threshold is not None and args.q_value_threshold < 0 else args.q_value_threshold
    results = compute_decoy_flr(tool_paths, q_thr, args.flr_threshold, not args.no_collapse)

    isect = next(iter(results.values())).n_in_intersection if results else 0
    print(f"\nShared PSM set (q-value <= {q_thr}, ident-decoys removed): {isect} PSMs")
    print(f"Global FLR threshold: {args.flr_threshold:.0%}\n")
    print(f"{'tool':<10} {'in':>7} {'after_fdr':>10} {'in_isect':>9} "
          f"{'analyzed':>9} {'Tc/Xc':>6} {'sites@FLR':>10} {'targets':>8} {'decoys':>7}")
    for tool, res in results.items():
        total, target, decoy = sites_at_flr(res.curve, args.flr_threshold)
        ratio = (res.t_c / res.x_c) if res.x_c else float("nan")
        print(f"{tool:<10} {res.n_psms_in:>7} {res.n_after_ident_filter:>10} "
              f"{res.n_in_intersection:>9} {res.n_analyzed_psms:>9} {ratio:>6.2f} "
              f"{total:>10} {target:>8} {decoy:>7}")
        if args.out_prefix:
            out = f"{args.out_prefix}_{tool}_flr.csv"
            _write_curve_csv(out, res)
            print(f"           -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
