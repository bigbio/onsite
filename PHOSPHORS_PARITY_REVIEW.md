# PhosphoRS parity review: `onsite` vs. compomics-utilities `PhosphoRS.java`

Reference: https://github.com/CompOmics/compomics-utilities/blob/master/src/main/java/com/compomics/util/experiment/identification/modification/scores/PhosphoRS.java

Local implementation: `onsite/phosphors/phosphors.py` (function `calculate_phospho_localization_compomics_style`).

> **Status:** static/structural comparison. The Java reference is not run here, so this is
> not empirical parity. Part 2 (below) designs side-by-side tests to make it empirical.

---

## Verdict

**The Python is not a faithful port — it is a reimplementation that deliberately rewrites
the depth-optimization core and redefines several structural quantities (the binomial `n`,
the ion matcher, the ion model, the `w` range).** It will **not** produce numerically
equivalent scores to compomics. Several divergences are author-flagged "corrections" of
behavior judged buggy in the reference; others appear to be unintended drift. The
high-level scaffolding (spectrum filtering, profile enumeration, the `getp`/binomial
formulas, the `1/P` normalization) does match.

---

## What matches (genuine parity)

| Component | Python | Java | Status |
|---|---|---|---|
| `filterSpectrum` (window = 10·tol or 100; maxPeaks = 10 or w/tol; keep most intense per window) | `filter_spectrum_for_phosphors` (338–428) | `filterSpectrum` (1248–1369) | OK faithful |
| `getp` = `d·n/w`, clamp >1→1, `floorDouble(p, nDecimals)` | `getp_style` (74–96) | `getp` (828–857) | OK faithful |
| Binomial score: `k==0 → 1`, else cumulative tail | `binomial_tail_probability` (115–226) | `getPhosphoRsScoreP` (734–773) | **NOT equivalent — see Part 2 / D13.** onsite=`P(X≥k)`, compomics=`P(X>k)` (verified on live JVM). Structure matches; tail convention differs. |
| Profile enumeration (sorted combinations of sites choose nMods) | `itertools.combinations` (1068) | `getPossibleModificationProfiles` (1060) | OK |
| Trivial case (sites == mods → 100%) and final `pInv/total·100` normalization | 1016–1038, 1197–1214 | 677–681, 626–673 | OK |
| Constants `WINDOW_SIZE=100`, `MAX_DEPTH=8`, 100-m/z window sweep | 35–36, 853–875 | 55–60, 306–584 | OK |

---

## Material divergences (change the scores)

### 1. Depth-optimization criterion — the heart of PhosphoRS (intentional rewrite)
- **Reference** (365–510, SDI branch): per depth, score the distinct profiles → ascending
  p-value array; `delta[j] = bigP[j]/bigP[j+1]` (≤1); pick the depth maximizing
  **`delta[0]` = best/second-best ratio**, i.e. where the top two p-values are *closest*.
  The depth binomial uses **n = nPeaks (experimental peaks above threshold)**; `k` is
  counted from matches against the *filtered* spectrum. Floored to **MIN_DEPTH=2**, capped
  at MAX_DEPTH=8.
- **Python** (`_choose_window_depth`, 774–815): per depth (top-`depth` peaks by intensity),
  compute each isoform's peptide score `−10·log10 P` with **n = unique theoretical ions in
  window**; if the window has site-determining ions, maximize the **score *gap* `(r1−r2, …)`**
  (separation), else maximize the best absolute score. Depth range `1..min(8, #peaks)`,
  **no MIN_DEPTH floor**.

These optimize **opposite directions**: the reference maximizes `p_j/p_{j+1}` (closest),
Python maximizes `r1−r2` (most separated). The reference criterion also looks inverted vs.
Taus et al. 2011 (which maximizes the score *difference*) — the author's comment at 720–726
says exactly this ("whose depth-selection ratio was inverted"). For *parity* the
load-bearing fact is: **Python does not reproduce the reference here.** Sub-differences
stack: binomial `n` (exp peaks vs. theoretical ions), depth = distinct-intensity thresholds
with ties (Java) vs. exactly top-N peaks (Python), no MIN_DEPTH=2 in Python, and profile
dedup-by-SDI-set (Java) vs. scoring all isoforms (Python).

### 2. No-SDI window handling (Python diverges from a reference bug)
Reference no-SDI branch (524–566): `double bestP = 0.0; … if (bigP < bestP)`. Since
`bigP ∈ [0,1]` (enforced by `checkProbabilityRange`), `bigP < 0.0` is **never true** →
`bestI` stays 0 → keeps only the most-intense peak(s) in windows with no site-determining
ions. Genuine reference bug (init was almost certainly meant to be `1.0`). Python instead
maximizes the best absolute score, keeping more peaks.

### 3. Binomial `n` in final scoring (intentional)
- Reference (633, 650): `n = profileToN` = total count of **all expected theoretical
  fragment ions** across selected charges.
- Python (`_count_matched_ions`, 1174): `n` = **unique theoretical m/z after merging ions
  within one tolerance window**. Different `n` → different `bigP` for every profile.

### 4. Ion matcher / `k` definition (intentional)
- Reference: `PeptideSpectrumAnnotator` (one match per ion, standard annotator dedup).
- Python: bespoke **consume-once matcher** that also merges indistinguishable theoretical
  ions (229–271). Asymmetrically affects decoy isomers carrying extra phospho-loss ions.

### 5. `w` (m/z range) for the final `p` (unflagged drift)
- Reference (608): `w = filteredSpectrum.maxMz − minMz` (**filtered** range).
- Python (1133–1137): `w` from `spectrum.get_peaks()` = **original, unfiltered** range. The
  Python comment asserts "FULL m/z range" — but the reference uses the filtered span.

### 6. Ion model (approximation)
- Reference: configured `PEPTIDE_FRAGMENT_ION` subtypes + configured selected charges;
  neutral losses gated by `|lossMass − modMass| > tol`.
- Python: **hardcoded b/y only**, charges `1..precursor_charge`, all losses generated then
  name-filtered for `-HPO3`/`-PO3H`.

### 7. Internal inconsistency in Python's own ion set (accidental)
The depth-reduction generator sets `add_metainfo=false` (840), so `_isoform_theo_mz`
**cannot** apply the `-HPO3/-PO3H` filter → **depth selection sees phospho-loss ions**,
while final scoring (`add_metainfo=true`, 1104) strips them.

---

## Minor divergences
- **`nDecimals`**: both derive it from `−log10(d/w)+1`, but the differing `w` (item 5) makes
  the final-stage value differ slightly.
- **Per-window k m/z bound**: reference uses `peakMz < maxMz` (the *global* max, not
  `tempMax`) at 403/440/548 — *appears* to be a reference bug; Python correctly restricts to
  the window.
- **Div-by-zero guard**: if `getp` floors `p` to 0, Python returns `p_inv=0` (graceful, 1181);
  the reference would compute `1/0 → Inf` and `Inf/Inf → NaN`.

---

## Dead code / hygiene (not parity findings)
- `calculate_phosphors_score` (611) and `get_occurrence_probability` (552): **unused**; a
  different intensity-/mass-error-weighted scheme that is *not* PhosphoRS.
- The binomial distribution **cache is ineffective**: `binomial_tail_probability` checks the
  cache then `pass`/recomputes (132–136); stored scalar keyed by `(p,n)` is never reused
  across `k`.
- `add_ion_types` / `max_ion_charge` parameters accepted but **ignored** (only consumer
  `_expected_fragment_mzs` is dead code).

## onsite-specific extension (not in the reference)
`site_deltas_from_isomers` (297–334) — per-site peptide-score gap used for FLR thresholding.

---

## Bottom line
For **numerical parity** with compomics this is not achieved, primarily because of the
depth-optimization rewrite (#1), the redefined binomial `n` (#3), and the custom matcher
(#4) — all intentional — plus unflagged drift in `w` (#5) and the metainfo inconsistency
(#7). For a **paper-aligned scorer**, several deviations are defensible (and #2 is one place
where copying the reference would be *worse*).

<!-- PART2_MARKER: deep-dive (bug classification + side-by-side tests) appended below -->

---
---

# Part 2 — Deep dive: which divergences are true bugs, and side-by-side tests

This part **empirically adjudicates** each divergence. We ran the **real compomics JVM**
(`utilities-5.1.17.jar` + `commons-math-2.2`) against the **real onsite functions**
(pyopenms 3.5.0), in three tiers:

- **Tier A — pure functions.** A Java driver calling `Util.floorDouble` and
  `BinomialDistribution.getDescendingCumulativeProbabilityAt` vs onsite `getp_style` /
  `binomial_tail_probability` over a numeric grid. *(ran, 558 cells)*
- **Tier B — depth selection.** A faithful Java replica of the compomics delta-loop
  (`PhosphoRS.java` 461–510 + the no-SDI branch 524–566) vs the onsite rule and the real
  `_choose_window_depth`. *(ran)*
- **Tier C — end-to-end.** The **real** `PhosphoRS.getSequenceProbabilities` vs onsite
  `calculate_phospho_localization_compomics_style` on a byte-identical synthetic spectrum.
  *(ran; the losses-OFF result was re-run in this session from `tierC/Driver.java`)*

Artifacts live under `/tmp/parity/{tierA,tierB,tierC}` (drivers + outputs).

> **Sourcing caveat.** The Taus et al. 2011 paper is paywalled (every fetch returned 403),
> so "paper-correct" calls rest on two independent secondary descriptions + internal
> model-consistency, not verbatim quotes. The *code-level* facts (and all Tier A/B/C numbers)
> are primary and reproduced here. Where a verdict depends only on code (dead loops, tail
> arithmetic, NaN), confidence is HIGH regardless of the paper.

## Headline empirical results

1. **End-to-end, clean b/y spectrum (neutral losses OFF): the two implementations AGREE.**
   On `PEPS(Phospho)TIDE`, tol 0.5 Da, both call **S4 dominant**: compomics **99.094 %**
   vs onsite **98.958 %** — a **0.14 percentage-point** gap. Despite all the structural
   differences, on a clean backbone spectrum they converge. The residual is consistent with
   the always-on tail-convention difference (D13), partly cancelled by 2-isoform normalization.

2. **End-to-end with each stack's native neutral-loss handling ON: the calls FLIP.**
   compomics → S4 (88 %), onsite → **T5 (92 %)**. The flip is *consistent with* the
   neutral-loss + fragment-charge divergence (D9), **not** the dead `-HPO3` name filter (D10,
   which matched **0 of 66** ions — pyopenms names the phospho loss `-H3O4P1`, never `-HPO3`).
   **Caveat:** this losses-ON comparison is *uncontrolled* — the peaks were crafted loss-free,
   then each stack flooded its own loss model at its own charges, so the flip *magnitude*
   conflates the real D9 effect with a peak-crafting artifact. Treat it as suggestive evidence
   that D9 can change the call on loss-rich spectra, not as a quantified result.

3. **The binomial tail is off by one term.** Against the live JVM,
   `getDescendingCumulativeProbabilityAt(k) = regularizedBeta(p, k+1, n−k) = P(X>k)` — the
   **strict** tail — while onsite returns `P(X≥k)`. Every cell's difference equals exactly
   `P(X=k)` (e.g. `n=2,k=1,p=0.5`: Java 0.25 vs onsite 0.75). **onsite is paper-correct here**
   ("at least k"); compomics computes one term too few. Separately, compomics returns **NaN**
   at `k==n` (regularizedBeta with `b=0`) — a real reference bug onsite does not have.
   **But note the scope:** this difference is large *per isoform* (a ~10-point peptide-score
   gap on some cells), yet **practically negligible end-to-end** — it largely cancels in the
   `1/P` site-probability normalization (it is the 0.14 pp residual in result 1, not a call
   change). **D9, not the tail, is the divergence that moves real localization calls.**

4. **Depth selection is inverted in the reference.** Tier B, real javac vs real onsite, on a
   crafted per-depth p-value matrix: compomics picks the depth of **least** isoform
   discrimination (max `p₀/p₁`), onsite picks **maximum separation** (max `r₁−r₂`). For the
   no-SDI window, the compomics branch is a **dead loop** (`bestP=0.0; if(bigP<bestP)` is never
   true) that always keeps depth 1; onsite keeps the best-scoring depth.

## Verdict table (all 13, after adversarial verification)

Legend: **bug_java** = reference is wrong, onsite correct · **bug_python** = onsite wrong,
reference correct · **both_ok** = defensible-but-different · **cosmetic** = no/negligible effect.

| ID | Divergence | Verdict | Who is correct | Conf. |
|----|------------|---------|----------------|-------|
| **D1** | Depth-selection direction: max `p₀/p₁` (closest) vs max `r₁−r₂` (separation) | **bug_java** | **onsite** (maximize discrimination) | HIGH |
| **D2** | Binomial `n` during depth selection: exp peaks (`nPeaks`) vs theoretical ions | **bug_java** | **onsite** (`n`=theoretical ions) | HIGH |
| D3 | Depth granularity: distinct-intensity thresholds (keep all ties) vs exact top-N | cosmetic | — (rare; needs intensity ties) | high |
| D4 | `MIN_DEPTH=2` floor present (Java) / absent (Python) | both_ok | unsettled (paper silent) | med |
| **D5** | No-SDI window: Java dead loop → always depth 1 vs onsite max-score | **bug_java** | **onsite** | HIGH |
| D6 | Final `n`: sum over (ion×valid-charge), no merge vs tolerance-merged unique m/z | both_ok | both self-consistent; merge is rare | HIGH |
| D7 | Matcher `k`: annotator (no-consume, no-merge) vs consume-once + theo-merge | both_ok | paper-underdetermined | HIGH |
| **D8** | `w` for final `p`: filtered-spectrum range vs original-spectrum range | bug_java | **onsite** ("full mass range") | med |
| **D9** | Fragment-charge ladder + neutral-loss gate | **bug_python** | **compomics** (charge-validated; mass gate) | HIGH |
| D10 | `add_metainfo=false` in depth reduction → phospho-loss name filter is dead | cosmetic | — (filter is a no-op in pyopenms) | high |
| **D11** | Per-window depth `k` upper bound: global `maxMz` (Java) vs window bound | bug_java | **onsite** | high |
| **D12** | `1/bigP` when `getp` floors `p→0`: Java `+Inf→NaN` vs onsite guard `→0` | bug_java | **onsite** | high |
| **D13** | Pure fns: binomial tail `P(X>k)` vs `P(X≥k)`; `floorDouble` decimal vs binary | **bug_both** | tail→onsite; floor→compomics | high |

**Tally:** reference (compomics) bugs onsite avoids: **D1, D2, D5, D8, D11, D12, + D13-tail, + k==n NaN** (8).
onsite bugs: **D9 charge ladder, D13 `_floor_double`** (2). Defensible-different: D4, D6, D7. Cosmetic/dead: D3, D10.

## True bugs, grouped

### A. Genuine bugs in the compomics reference (onsite is right to diverge)
- **D1 — inverted depth criterion.** `delta = bigP[j]/bigP[j+1]` on an *ascending* p-value
  set, then `argmax delta[0]`. Since the peptide-score gap is `−10·log10(delta₀)` (strictly
  *decreasing* in `delta₀`), maximizing `delta₀` is *minimizing* discrimination — the opposite
  of the paper's "largest score difference between rank-1 and rank-2." *Tier B proved it:
  Java→depth 3 (sep 0.67 dB), onsite→depth 2 (sep 16.99 dB).*
- **D2 — wrong binomial `n` in depth selection.** Reference passes `nPeaks` (experimental
  peaks above threshold) as the binomial trial count. The trials must be the *theoretical*
  ions; the experimental peak count is already consumed inside `p = N·d/w`, so it cannot
  re-enter as `n`. onsite uses theoretical ions.
- **D5 — no-SDI dead loop.** `double bestP = 0.0; … if (bigP < bestP)` — a probability is
  never `< 0`, so `bestI` is pinned at 0 and no-SDI windows keep only the most intense peak.
  *Tier B replicated it: Java always returns depth 1.* (Intended init was surely `1.0`.)
- **D8 — `w` is the filtered span, not the full spectrum range.** Secondary sources say `w` =
  "full mass range of the MS/MS spectrum"; the reference uses the *filtered* spectrum's span.
- **D11 — global `maxMz` upper bound in the per-window `k` count** (`peakMz < maxMz`, not
  `< tempMax`) at 403/440/548 — counts matches above the window start up to the spectrum end.
- **D12 — `pInv = 1/bigP` with no zero guard.** If `getp` floors `p→0`, `bigP→0`, `pInv→+Inf`,
  and normalization → NaN. onsite guards (`→0`).
- **D13a / `k==n` NaN.** `regularizedBeta(p, k+1, 0)` is out of domain → NaN for fully-matched
  isoforms; onsite returns the correct `p^n`. (Verified at the `betainc` level; Apache's exact
  throw/NaN behavior at `b=0` not independently run, but the malfunction is real.)
- **D13b — binomial tail off-by-one.** `getDescendingCumulativeProbabilityAt` returns
  `P(X>k)`, not the `P(X≥k)` its name and the algorithm imply. onsite's `P(X≥k)` is
  paper-correct. *(The cache at `BinomialDistribution.java:134` is also a latent no-op — it
  reads `pCache`, which this path never populates — but harmless.)*

### B. Genuine bugs in onsite (the reference is right)
- **D9 — fragment-charge ladder over-generates ions.** onsite calls
  `getSpectrum(seq, 1, precursor_charge)` → fragment charges `1..precursor` *inclusive*.
  compomics' `chargeValidated` requires `charge < precursorCharge` **and** `charge ≤ ionNumber`,
  so it never emits a charge-`p` fragment of a `p+` precursor nor a `y1`/`b1` at 2+. Measured:
  for `PEPS(Phospho)TIDE` at precursor 3, onsite `n=99` vs compomics-equivalent `n=64` — **35 %
  inflation with physically impossible ions.** (onsite's own `_expected_fragment_mzs` documents
  the intended `1..min(precursor−1, 2)` cap, but that function is **dead code** — never called.)
  The **n-inflation is primary-source-certain** (`chargeValidated` at `dl_SpectrumAnnotator`
  504–506; the extra ions are physically impossible). The **Tier-C losses-ON flip is only
  suggestive** of this — it came from an uncontrolled comparison (see headline result 2), so it
  shows D9 *can* change a call, not by how much.
- **D13c — `_floor_double` is a binary floor, not a decimal floor.** Docstring says "Mimic
  `Util.floorDouble`", but `math.floor(value*10**n)/10**n` floors the IEEE product, whereas
  Java floors the decimal string (`BigDecimal(String.valueOf(d)).setScale(n, FLOOR)`). E.g.
  `_floor_double(0.29,2)=0.28` vs Java `0.29`; this propagates a ~17 % relative shift into
  `p` for some inputs (`getp_style(3,100,0.02)=0.0005` vs Java `0.0006`). A failed port.

### C. Defensible-but-different (no "bug", but blocks numeric parity)
- **D4** (`MIN_DEPTH=2`): the paper is silent; the reference floors depth to 2, onsite allows 1.
- **D6** (final `n`: sum-over-charges vs tolerance-merge): each side is *internally* consistent
  (Java multi-counts both `n` and `k`; onsite merges both). Merging rarely fires in practice
  (1+/2+ of one fragment differ by ~half the m/z, never within tolerance); the real-world `n`
  gap is driven by **D9's charge ladder**, not by merging.
- **D7** (matcher: no-consume vs consume-once): onsite's consume-once + theo-merge is a
  deliberate anti-inflation / anti-decoy-bias choice; the paper does not pin the sub-tolerance
  rule. On a shared crafted input the two give e.g. `(n=3,k=3)` vs `(n=2,k=2)` → ~20× `bigP`.

### D. Cosmetic / dead code
- **D3** intensity-tie handling (rare without intensity quantization).
- **D10** the `-HPO3`/`-PO3H` filter is dead: pyopenms emits `-H3O4P1`/`-H2O1`, so it removes 0
  ions — but because the loss it *means* to remove (HPO3 ≈ phospho mass) is also never emitted,
  the scored loss sets coincide with the reference. Internally-wrong dead code, nil numeric impact.

## Side-by-side test harness (runnable here)

All three tiers ran in this environment (JDK 21, pyopenms 3.5.0). To reproduce / extend:

**Tier A — pure-function grid** (`/tmp/parity/tierA/`):
`GetpBinom.java` (compomics) + `getp_binom_py.py` (onsite) emit `TAG\tinputs\tvalue`; `diff_side_by_side.py` joins them.
```bash
cd /tmp/parity/tierA && javac -cp /tmp/parity/jars/'*' GetpBinom.java && java -cp .:/tmp/parity/jars/'*' GetpBinom > java_out.tsv
PYTHONPATH=/home/sachsenb/Development/onsite python3 getp_binom_py.py > py_out.tsv && python3 diff_side_by_side.py
```
Result: 265 EXACT / 293 DIFF; every BINOM diff `= P(X=k)`; `getp` diverges on ~4.2 % of a 32 680-cell sweep via `floorDouble`.

**Tier B — depth selection** (`/tmp/parity/tierB/`): `JavaDepthRule.java` (faithful compomics
delta-loop + no-SDI branch) vs `py_depth_rule.py` (onsite rule + the real `_choose_window_depth`).
```bash
cd /tmp/parity/tierB && javac JavaDepthRule.java && java JavaDepthRule
PYTHONPATH=/home/sachsenb/Development/onsite python3 py_depth_rule.py
```
Output (see `RESULTS.txt`): D1 Java→depth 3 / onsite→depth 2; D5 Java→depth 1 always / onsite→depth 3.

**Tier C — end-to-end** (`/tmp/parity/tierC/`): `Driver.java` calls the real
`PhosphoRS.getSequenceProbabilities`; `onsite_e2e.py` feeds the byte-identical peak list to
`calculate_phospho_localization_compomics_style`.
```bash
cd /tmp/parity/tierC && javac -cp /tmp/parity/jars/'*' Driver.java && java -cp .:/tmp/parity/jars/'*' Driver
PYTHONPATH=/home/sachsenb/Development/onsite python3 onsite_e2e.py
```
Result: losses-OFF agree (S4 99.09 % vs 98.96 %); losses-ON flip (S4 vs T5) via D9.

**The single highest-value regression test** to add to `tests/`: a frozen Tier-C-style case
(small hardcoded spectrum + peptide, losses off) asserting onsite's site probabilities stay
within tolerance of the compomics numbers captured above (S4 ≈ 99 %). This is a **drift anchor**,
not a parity test — onsite is deliberately non-parity (and arguably more correct) elsewhere;
this just catches onsite *regressions* on the one config where the two stacks happen to agree.

## Recommended onsite fixes (priority order)

1. **D9 charge ladder (real bug, flips calls):** cap fragment charge at `min(precursor−1, …)`
   and enforce `charge ≤ ion number` — i.e. apply compomics' `chargeValidated` in
   `_isoform_theo_mz` (735) and the final scorer (1144). This is the change most likely to move
   real localization results.
2. **D13c `_floor_double`:** if parity matters, port `Util.floorDouble` faithfully
   (`Decimal(repr(p)).quantize(…, ROUND_FLOOR)`); otherwise document that it is an approximation.
3. **Decide the tail convention (D13b) explicitly.** onsite's `P(X≥k)` is paper-correct;
   compomics' `P(X>k)` is the parity target. Pick one *on purpose* and write it down — you
   cannot match the paper and compomics simultaneously here.
4. **D10 / dead code:** delete the `-HPO3`/`-PO3H` filter (or fix it to a mass gate vs the
   actual modification mass, per D9's reference behavior), and remove `calculate_phosphors_score`,
   `get_occurrence_probability`, `_expected_fragment_mzs`, and the no-op binomial cache.
5. **D8 `w`:** choose filtered vs original range deliberately (paper favors original/full).

## Fixes applied (this session)

Two real onsite bugs were fixed in `onsite/phosphors/phosphors.py`; all 178 tests pass.

- **D9 (fragment-charge over-generation) — FIXED.** New `_theo_mz_charge_valid()` generates
  b/y ions at charge `1..max(1, precursor−1)` and drops any ion with `charge > ion_number`
  (compomics `chargeValidated`), plus the phospho-loss name filter. Both live theoretical-ion
  paths (final scoring + `_isoform_theo_mz` for depth selection) route through it. Verified to
  reproduce the reference ion count exactly (`PEPS(Phospho)TIDE` @3+, losses on: 99 → **64**,
  matching the live compomics JVM). As a side effect this also closes **D10**: the
  depth-reduction generator now uses `add_metainfo='true'`, so depth and final scoring share
  one charge-validated, loss-filtered ion set.
- **D13c (`_floor_double`) — FIXED.** Confirmed *live* (called by `getp_style`, which feeds
  both depth selection and final scoring). It claimed to "Mimic `Util.floorDouble`" but did a
  binary floor; replaced with a decimal-string floor (`Decimal(repr(value)).quantize(…,
  ROUND_FLOOR)`). Now matches Java exactly (`0.29→0.29`, `getp_style(3,100,0.02)=0.0006`).

Not changed: the binomial tail convention (D13b — a deliberate paper-vs-compomics choice), the
now-superseded dead `_expected_fragment_mzs`, and the depth-selection criterion (D1, where
onsite is already paper-correct). Note the Tier-C losses-off number moved 98.96 % → **99.92 %**
after the D9 fix (compomics 99.09 %): for a 2+ precursor the fix correctly restricts onsite to
charge-1 fragments (matching compomics' ion set), so the residual now reflects the *other*
compomics-side bugs onsite avoids — the prior 0.14 pp agreement was partly coincidental
cancellation from the impossible ions.

## Net assessment

onsite is **not** a byte-faithful port — and on the depth-selection core, the binomial tail,
the no-SDI branch, and several edge cases, that is a *good* thing: it sidesteps **eight** real
defects in the compomics reference. It carries **two** real bugs of its own — the fragment-charge
over-generation (D9, which can flip a localization call on loss-rich spectra) and the binary
`_floor_double` — plus a handful of deliberate, defensible modeling differences. On a clean
b/y spectrum the two agree to ~0.14 pp; the divergences bite on noisy, loss-rich, or
high-charge spectra.

