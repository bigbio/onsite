# Decoy-Amino-Acid FLR Documentation

## Overview

`onsite.decoy_flr` estimates a global false localization rate (FLR) for AScore, PhosphoRS and LucXor results on one common scale, using the decoy-amino-acid method of Ramsbottom et al. 2022 ([DOI 10.1021/acs.jproteome.1c00827](https://doi.org/10.1021/acs.jproteome.1c00827)).

A residue that cannot carry the modification (Alanine for phosphorylation) is offered to each tool as a decoy site (`PhosphoDecoy`). Any site a tool localizes onto a decoy residue is a known false localization, so the decoy rate in the score-ranked site list estimates the FLR.

## Method

1. **Shared PSM set.** For each tool, keep the best hit per spectrum, drop identification decoys, and apply the PSM q-value cutoff (`percolator_q_value`). Then keep only the PSMs present in *every* tool, so all tools are compared on the same population.
2. **Ambiguous peptides only.** Skip peptides where every candidate residue (target + decoy) is occupied, because they have no localization choice.
3. **Rank sites** by each tool's confidence:

   | Tool | Meta value used for ranking |
   |------|-----------------------------|
   | AScore | `AScore_site_scores` |
   | PhosphoRS | `PhosphoRS_site_delta` (the site probability saturates near 100% and cannot rank) |
   | LucXor | `Luciphor_site_scores` |

   By default, repeated observations of the same (peptide, position) site are collapsed to their maximum score.
4. **FLR at rank n**:
   ```
   FLR_n = (T_c / X_c) * D_n / (n - D_n)
   ```
   Where:
   - `D_n` = cumulative decoy sites in the top `n`
   - `T_c` / `X_c` = total target / decoy candidate residues in the analyzed peptides

   The `T_c / X_c` factor corrects for target residues outnumbering decoy residues. The curve is capped at 1 and made monotone like a q-value (reverse cumulative minimum).
5. **Report** the deepest rank whose FLR is at or below the threshold, as total / target / decoy site counts.

## Usage

Run the tools with decoys enabled (`--add-decoys` for AScore and PhosphoRS; LucXor generates decoys itself), then:

```bash
python -m onsite.decoy_flr \
  --ascore ascore.idparquet \
  --phosphors phosphors.idparquet \
  --lucxor lucxor.idparquet \
  --out-prefix results/flr
```

Any subset of the three tools works. The output is a summary table on stdout, plus one `<prefix>_<tool>_flr.csv` curve per tool with `--out-prefix` (columns `rank,score,cum_target,cum_decoy,flr_raw,qval`).

### Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `--ascore` / `--phosphors` / `--lucxor` | — | Result idParquet directory per tool (at least one) |
| `--q-value-threshold` | `0.01` | PSM q-value cutoff shared by all tools; negative to skip |
| `--flr-threshold` | `0.05` | FLR cutoff for the reported site yield (the paper recommends 5% over 1%) |
| `--no-collapse` | off | Count each PSM's site separately instead of collapsing per (peptide, position) |
| `--out-prefix` | — | Write per-tool FLR curves to `<prefix>_<tool>_flr.csv` |
| `--modification` | `Phospho` | Modification being localized (see below) |
| `--target-residues` | preset | Override target residues: `STY`, `S,T,Y` or `S T Y` |
| `--decoy-residues` | `A` | Override decoy residues; must not overlap the targets |

## Other modifications

> **Note:** AScore, PhosphoRS and LucXor still localize **phosphorylation only**. `--modification` currently changes only how `decoy_flr` reads their results. Generalizing the scoring tools is tracked in [#57](https://github.com/bigbio/onsite/issues/57). Until then, a non-phospho `--modification` on phospho results simply finds no sites.

`--modification` accepts a preset or any modification name in the pyOpenMS `ModificationsDB` together with `--target-residues`:

| Preset | Target residues | Decoy |
|--------|-----------------|-------|
| `Phospho` | S, T, Y | A |
| `Acetyl` | K | A |
| `Methyl` | K, R | A |
| `Dimethyl` | K, R | A |
| `Trimethyl` | K | A |
| `GG` (aliases `GlyGly`, `ub`) | K | A |

The decoy is a modification named `<Modification>Decoy` with the same mass as the target, registered on every residue. Alanine is the default decoy for every preset. It has been validated for phosphorylation only, so treat FLRs for other modifications as provisional until they are checked on data with known sites.

Protein N-terminal modifications are ignored: they have no residue choice to localize.
