from pyopenms import *
import math
import logging


class AScore:
    """
    AScore algorithm for phosphorylation site localization.

    Identifies the most probable phosphorylation site(s) for a given peptide
    sequence and MS/MS spectrum, calculating probability scores for each site.
    """

    def __init__(self):
        """Initialize AScore with default parameters."""
        self.fragment_mass_tolerance_ = 0.05
        self.fragment_tolerance_ppm_ = False
        self.max_peptide_length_ = 40
        self.max_permutations_ = 16384
        self.add_decoys_ = False
        self.base_match_probability_ = 0.0
        self.unambiguous_score_ = 1000.0

        # Initialize TheoreticalSpectrumGenerator
        self.spectrum_generator_ = TheoreticalSpectrumGenerator()
        p = self.spectrum_generator_.getParameters()
        p.setValue("isotope_model", "none")
        p.setValue("add_first_prefix_ion", "true")
        p.setValue("add_b_ions", "true")
        p.setValue("add_y_ions", "true")
        p.setValue("add_a_ions", "false")
        p.setValue("add_c_ions", "false")
        p.setValue("add_x_ions", "false")
        p.setValue("add_z_ions", "false")
        p.setValue("add_losses", "false")
        p.setValue("add_metainfo", "false")
        p.setValue("add_precursor_peaks", "false")
        p.setValue("add_abundant_immonium_ions", "false")
        self.spectrum_generator_.setParameters(p)

        self.updateMembers_()

    def _get_logger(self):
        """Get debug logger"""
        return logging.getLogger("debug_logger")

    def updateMembers_(self):
        """Update member variables from parameter object."""
        pass

    def setAddDecoys(self, add_decoys):
        """Set whether to include A (PhosphoDecoy) sites."""
        self.add_decoys_ = add_decoys

    @staticmethod
    def isPhosphoDecoySite(residue):
        """Check if residue is a PhosphoDecoy site (A)."""
        return residue == "A"

    @staticmethod
    def isPhosphoSite(residue):
        """Check if residue is a phosphorylation site (S, T, Y)."""
        return residue in ["S", "T", "Y"]

    def compute(self, hit, real_spectrum):
        """
        Compute AScore and return modified PeptideHit with phosphorylation site scores.
        """
        phospho = PeptideHit(hit)

        phospho.setScore(-1.0)
        phospho.setMetaValue("search_engine_sequence", hit.getSequence().toString())

        if real_spectrum.size() == 0:
            return phospho

        sequence_str = phospho.getSequence().toString()
        unmodified_sequence_str = phospho.getSequence().toUnmodifiedString()

        number_of_phosphorylation_events = self.numberOfPhosphoEvents_(sequence_str)
        seq_without_phospho = self.removePhosphositesFromSequence_(sequence_str)

        regular_phospho_count = sequence_str.count("(Phospho)")
        decoy_phospho_count = sequence_str.count("(PhosphoDecoy)")

        if (not self.add_decoys_) and decoy_phospho_count > 0:
            self._get_logger().info(
                "Warning: PhosphoDecoy sites found but add_decoys is false. Returning original hit."
            )
            return phospho

        if (self.max_peptide_length_ > 0) and (
            len(unmodified_sequence_str) > self.max_peptide_length_
        ):
            return phospho

        sites = self.getSites_(unmodified_sequence_str)
        number_of_sites = len(sites)

        if number_of_phosphorylation_events == 0 or number_of_sites == 0:
            return phospho

        if self.add_decoys_:
            phospho.setMetaValue("regular_phospho_count", regular_phospho_count)
            phospho.setMetaValue("phospho_decoy_count", decoy_phospho_count)
        phospho.setMetaValue("phospho_sites", number_of_phosphorylation_events)

        # If all possible sites are phosphorylated, localization is unambiguous
        if number_of_sites == number_of_phosphorylation_events:
            phospho.setScore(self.unambiguous_score_)

            site_scores = {}
            for site in sites:
                site_scores[site] = self.unambiguous_score_

            proforma = self.generateProFormaString_(phospho.getSequence(), site_scores)
            phospho.setMetaValue("ProForma", proforma)
            phospho.setMetaValue("AScore_site_scores", str(site_scores))
            phospho.setMetaValue("AScore_pep_score", self.unambiguous_score_)

            for i, site in enumerate(sites):
                phospho.setMetaValue(f"AScore_{i+1}", self.unambiguous_score_)

            return phospho

        permutations = self.computePermutations_(
            sites, number_of_phosphorylation_events
        )

        if not permutations or (
            (self.max_permutations_ > 0)
            and (len(permutations) > self.max_permutations_)
        ):
            return phospho

        th_spectra = self.createTheoreticalSpectra_(permutations, seq_without_phospho)

        if not real_spectrum.isSorted():
            real_spectrum.sortByPosition()
        windows_top10 = self.peakPickingPerWindowsInSpectrum_(real_spectrum)
        # Precompute, once per PSM, for each 100-Da window the m/z arrays of its
        # top-i peaks (i=1..10) sorted ascending by m/z. This replaces the
        # per-call MSSpectrum copy+truncate+sort that numberOfMatchedIons_ used
        # to do (10x per window). See windowDepthMZArrays_.
        windows_depth_mz = [self.windowDepthMZArrays_(w) for w in windows_top10]

        self.base_match_probability_ = self.computeBaseProbability_(
            real_spectrum[real_spectrum.size() - 1].getMZ()
        )

        peptide_site_scores = self.calculatePermutationPeptideScores_(
            th_spectra, windows_depth_mz
        )

        ranking = self.rankWeightedPermutationPeptideScores_(peptide_site_scores)

        if len(ranking) < 2:
            return phospho

        best_score = ranking[-1][0]
        best_permutation_idx = ranking[-1][1]
        seq1 = th_spectra[best_permutation_idx].getName()
        phospho.setSequence(AASequence.fromString(seq1))

        peptide1_score = best_score
        phospho.setMetaValue("AScore_pep_score", peptide1_score)

        second_best_score = ranking[-2][0]
        peptide2_score = second_best_score

        phospho_sites = []
        self.determineHighestScoringPermutations_(
            peptide_site_scores, phospho_sites, permutations, ranking
        )

        rank = 1
        best_Ascore = float("inf")
        site2score = {}

        for phospho_site in phospho_sites:
            Ascore = 0.0
            if peptide1_score == peptide2_score:
                self._get_logger().info(
                    f"Debug: Best and second best peptide scores are equal ({peptide1_score})"
                )
                Ascore = 0.0
            else:
                site_determining_ions = []
                self.computeSiteDeterminingIons_(
                    th_spectra, phospho_site, site_determining_ions
                )

                if (
                    not site_determining_ions
                    or len(site_determining_ions) < 2
                    or site_determining_ions[0].size() == 0
                    or site_determining_ions[1].size() == 0
                ):
                    self._get_logger().info(
                        f"Debug: No site-determining ions found for site {phospho_site['first']}"
                    )
                    Ascore = 0.0
                else:
                    N = site_determining_ions[0].size()
                    p = float(phospho_site["peak_depth"]) * self.base_match_probability_

                    depth = phospho_site["peak_depth"]
                    th_first_mz = self.spectrumMZArray_(site_determining_ions[0])
                    th_second_mz = self.spectrumMZArray_(site_determining_ions[1])

                    n_first = 0
                    for window_depth_mz in windows_depth_mz:
                        n_first += self.countMatchedMZ_(
                            th_first_mz, window_depth_mz, depth
                        )

                    P_first = self.computeCumulativeScore_(N, n_first, p)

                    n_second = 0
                    for window_depth_mz in windows_depth_mz:
                        n_second += self.countMatchedMZ_(
                            th_second_mz, window_depth_mz, depth
                        )

                    N2 = site_determining_ions[1].size()
                    P_second = self.computeCumulativeScore_(N2, n_second, p)

                    if P_first <= 0:
                        P_first = 1e-100
                    if P_second <= 0:
                        P_second = 1e-100

                    score_first = abs(-10.0 * math.log10(P_first))
                    score_second = abs(-10.0 * math.log10(P_second))

                    Ascore = score_first - score_second

            if Ascore < best_Ascore:
                best_Ascore = Ascore

            phospho.setMetaValue(f"AScore_{rank}", Ascore)
            site2score[phospho_site["first"]] = Ascore
            rank += 1

        proforma = self.generateProFormaString_(phospho.getSequence(), site2score)
        phospho.setMetaValue("ProForma", proforma)
        # Position-keyed per-site AScores for a site-level decoy-AA FLR
        # (mirrors PhosphoRS_site_probs / Luciphor_site_scores; see #40).
        phospho.setMetaValue("AScore_site_scores", str(site2score))
        phospho.setScore(best_Ascore)

        return phospho

    def computeBaseProbability_(self, ppm_reference_mz):
        """Compute base probability of random peak match."""
        base_match_probability = 2.0 * self.fragment_mass_tolerance_ / 100.0
        if self.fragment_tolerance_ppm_:
            base_match_probability *= ppm_reference_mz * 1e-6
        return base_match_probability

    def computeCumulativeScore_(self, N, n, p):
        """Compute cumulative binomial probability P(X ≥ n)."""
        assert n <= N, "n cannot exceed N"
        assert 0 <= p <= 1.0, "p must be a probability [0,1]"

        if n == 0:
            return 1.0

        score = 0.0
        for k in range(n, N + 1):
            try:
                from math import comb

                coeff = comb(N, k)
            except (ImportError, OverflowError):
                log_coeff = 0.0
                for i in range(1, k + 1):
                    log_coeff += math.log((N - k + i) / i)
                coeff = math.exp(log_coeff)

            pow1 = p**k
            pow2 = (1 - p) ** (N - k)
            score += coeff * pow1 * pow2

        return score

    def determineHighestScoringPermutations_(
        self, peptide_site_scores, sites, permutations, ranking
    ):
        """
        Find highest scoring permutations for each phosphorylation site.
        """
        sites.clear()

        if not ranking or len(ranking) == 1:
            return

        best_peptide_sites = permutations[ranking[-1][1]]

        for i in range(len(best_peptide_sites)):
            site_info = {"first": best_peptide_sites[i], "seq_1": ranking[-1][1]}

            rank_index = len(ranking) - 2
            peptide_not_found = True

            while peptide_not_found and rank_index >= 0:
                current_permutation = permutations[ranking[rank_index][1]]

                for j in range(len(best_peptide_sites)):
                    if j == i:
                        if best_peptide_sites[j] in current_permutation:
                            peptide_not_found = True
                            break
                        else:
                            peptide_not_found = False
                    else:
                        if best_peptide_sites[j] not in current_permutation:
                            peptide_not_found = True
                            break
                        else:
                            peptide_not_found = False

                if peptide_not_found:
                    rank_index -= 1

            if rank_index >= 0:
                site_info["seq_2"] = ranking[rank_index][1]

                for position in permutations[site_info["seq_2"]]:
                    if position not in best_peptide_sites:
                        site_info["second"] = position
                        break

                maximum_score_difference = 0.0
                site_info["peak_depth"] = 1

                max_depth = min(
                    len(peptide_site_scores[site_info["seq_1"]]),
                    len(peptide_site_scores[site_info["seq_2"]]),
                )

                for depth in range(1, max_depth + 1):
                    phospho_at_site_score = peptide_site_scores[site_info["seq_1"]][
                        depth - 1
                    ]
                    no_phospho_at_site_score = peptide_site_scores[site_info["seq_2"]][
                        depth - 1
                    ]
                    score_difference = phospho_at_site_score - no_phospho_at_site_score

                    if score_difference > maximum_score_difference:
                        maximum_score_difference = score_difference
                        site_info["peak_depth"] = depth

                sites.append(site_info)

    def computeSiteDeterminingIons_(
        self, th_spectra, candidates, site_determining_ions
    ):
        """Find site-determining ions for phosphorylation site candidates."""
        site_determining_ions.clear()
        site_determining_ions.extend([MSSpectrum(), MSSpectrum()])

        spectrum_first = th_spectra[candidates["seq_1"]]
        spectrum_second = th_spectra[candidates["seq_2"]]

        spectrum_first_diff = MSSpectrum()
        self.getSpectrumDifference_(
            spectrum_first, spectrum_second, spectrum_first_diff
        )

        spectrum_second_diff = MSSpectrum()
        self.getSpectrumDifference_(
            spectrum_second, spectrum_first, spectrum_second_diff
        )

        site_determining_ions[0] = spectrum_first_diff
        site_determining_ions[1] = spectrum_second_diff

        site_determining_ions[0].sortByPosition()
        site_determining_ions[1].sortByPosition()

    def getSpectrumDifference_(self, spectrum1, spectrum2, result):
        """Find peaks in spectrum1 but not in spectrum2."""
        result.clear(True)

        i, j = 0, 0
        while i < spectrum1.size() and j < spectrum2.size():
            mz1 = spectrum1[i].getMZ()
            mz2 = spectrum2[j].getMZ()
            val = self.compareMZ_(mz1, mz2)

            if val == -1:
                result.push_back(spectrum1[i])
                i += 1
            elif val == 1:
                j += 1
            else:
                j += 1
                while (
                    j < spectrum2.size()
                    and self.compareMZ_(mz1, spectrum2[j].getMZ()) == 0
                ):
                    j += 1

                i += 1
                while (
                    i < spectrum1.size()
                    and self.compareMZ_(spectrum1[i].getMZ(), mz2) == 0
                ):
                    i += 1

        while i < spectrum1.size():
            result.push_back(spectrum1[i])
            i += 1

    def spectrumMZArray_(self, spectrum):
        """Extract a spectrum's peak m/z values as a plain float64 numpy array.

        The spectrum is assumed already sorted ascending by m/z (theoretical
        spectra from the generator and the site-determining ion spectra are
        sorted by position before use). One get_peaks() call replaces all the
        per-peak getMZ() binding calls that numberOfMatchedIons_ used to do.
        """
        mz, _ = spectrum.get_peaks()
        return np.asarray(mz, dtype=np.float64)

    def windowDepthMZArrays_(self, window):
        """Precompute the top-i peak m/z arrays (i=1..10) for one window.

        `window` is an MSSpectrum already ordered by intensity descending
        (peakPickingPerWindowsInSpectrum_ calls sortByIntensity(True)). For each
        depth i, the reduced window is the top-i peaks; we sort those by m/z
        ascending once. Returns a list indexed [i-1] -> sorted float64 m/z array
        of the top-i peaks. This reproduces, exactly, the per-call
        copy/truncate-to-depth/sortByPosition that numberOfMatchedIons_ did 10x.
        """
        mz, _ = window.get_peaks()
        mz = np.asarray(mz, dtype=np.float64)
        size = mz.shape[0]
        depth_arrays = []
        for i in range(1, 11):
            k = i if size > i else size
            top = np.sort(mz[:k])
            depth_arrays.append(top)
        return depth_arrays

    def countMatchedMZ_(self, th_mz, window_depth_mz, depth):
        """Count matched ions between theoretical and reduced-window m/z arrays.

        `th_mz` is the theoretical spectrum's ascending m/z array.
        `window_depth_mz` is the precomputed list (windowDepthMZArrays_) of
        top-i m/z arrays; the reduced window for this call is window_depth_mz at
        index min(depth, 10)-1 (depth is always 1..10 here). Reproduces the
        exact two-pointer semantics of the former numberOfMatchedIons_: a match
        (|th - window| <= tolerance) consumes both pointers; otherwise the
        smaller m/z advances.
        """
        idx = depth - 1
        if idx >= len(window_depth_mz):
            idx = len(window_depth_mz) - 1
        window_mz = window_depth_mz[idx]

        n_th = th_mz.shape[0]
        n_win = window_mz.shape[0]

        matched_peaks = 0
        th_idx = 0
        window_idx = 0

        ppm = self.fragment_tolerance_ppm_
        base_tol = self.fragment_mass_tolerance_

        while th_idx < n_th and window_idx < n_win:
            t = th_mz[th_idx]
            w = window_mz[window_idx]

            tolerance = base_tol
            if ppm:
                tolerance = base_tol * ((t + w) / 2.0) / 1.0e6

            error = abs(t - w)

            if error <= tolerance:
                matched_peaks += 1
                th_idx += 1
                window_idx += 1
            elif t < w:
                th_idx += 1
            else:
                window_idx += 1

        return matched_peaks

    def compareMZ_(self, mz1, mz2):
        """Compare two m/z values using fragment mass tolerance."""
        tolerance = self.fragment_mass_tolerance_
        error = mz1 - mz2

        if self.fragment_tolerance_ppm_:
            avg_mass = (mz1 + mz2) / 2.0
            tolerance = tolerance * avg_mass / 1.0e6

        if error < -tolerance:
            return -1
        elif error > tolerance:
            return 1
        else:
            return 0

    def peptideScore_(self, scores):
        """Compute weighted peptide score."""
        assert len(scores) == 10, "Scores vector must contain 10 scores"

        return (
            scores[0] * 0.5
            + scores[1] * 0.75
            + scores[2]
            + scores[3]
            + scores[4]
            + scores[5]
            + scores[6] * 0.75
            + scores[7] * 0.5
            + scores[8] * 0.25
            + scores[9] * 0.25
        ) / 7.0

    def getSites_(self, unmodified):
        """Get potential phosphorylation sites in peptide sequence."""
        phospho_sites = self.getPhosphoSites_(unmodified)

        if self.add_decoys_:
            decoy_sites = self.getPhosphoDecoySites_(unmodified)
            phospho_sites.extend(decoy_sites)

        # Decoy (A) sites are appended after the S/T/Y sites, so the list is no
        # longer position-ordered. computePermutations_ -> combinations() would
        # then emit descending index combos (e.g. [4, 1]), and
        # createTheoreticalSpectra_ assumes ascending order and would silently
        # drop the out-of-order site. Sort to keep positions ascending.
        return sorted(phospho_sites)

    def getPhosphoSites_(self, unmodified):
        """Find phosphorylation sites (S, T, Y) in sequence."""
        ret = []
        for i in range(len(unmodified)):
            if self.isPhosphoSite(unmodified[i]):
                ret.append(i)
        return ret

    def getPhosphoDecoySites_(self, unmodified):
        """Find PhosphoDecoy sites (A) in sequence."""
        ret = []
        for i in range(len(unmodified)):
            if self.isPhosphoDecoySite(unmodified[i]):
                ret.append(i)
        return ret

    def numberOfPhosphoEvents_(self, sequence):
        """Count phosphorylation events in sequence."""
        cnt_phospho_events = 0

        pos = 0
        while True:
            pos = sequence.find("(Phospho)", pos)
            if pos == -1:
                break
            cnt_phospho_events += 1
            pos += 9

        if self.add_decoys_:
            pos = 0
            while True:
                pos = sequence.find("(PhosphoDecoy)", pos)
                if pos == -1:
                    break
                cnt_phospho_events += 1
                pos += 14

        self._get_logger().info(
            f"Debug: Found {cnt_phospho_events} phosphorylation events in sequence: {sequence}"
        )

        return cnt_phospho_events

    def computePermutations_(self, sites, n_phosphorylation_events):
        """Generate all possible phosphorylation site combinations."""
        permutations = []

        if self.max_permutations_ > 0 and n_phosphorylation_events >= 1:
            if n_phosphorylation_events > len(sites):
                self._get_logger().info(
                    f"Debug: More phosphorylation events ({n_phosphorylation_events}) than available sites ({len(sites)})"
                )
                return permutations
            else:
                try:
                    from math import comb

                    estimated_permutations = comb(len(sites), n_phosphorylation_events)
                except (ImportError, OverflowError):
                    try:
                        estimated_permutations = 1
                        for i in range(1, n_phosphorylation_events + 1):
                            estimated_permutations *= (len(sites) - i + 1) / i
                    except OverflowError:
                        estimated_permutations = float("inf")

                if estimated_permutations > self.max_permutations_:
                    self._get_logger().info(
                        f"Debug: Estimated permutations ({estimated_permutations}) exceeds maximum ({self.max_permutations_})"
                    )
                    return permutations

        if n_phosphorylation_events == 0:
            return permutations
        elif n_phosphorylation_events == 1:
            for i in range(len(sites)):
                permutations.append([sites[i]])
            return permutations
        elif len(sites) == n_phosphorylation_events:
            permutations.append(list(sites))
            return permutations
        else:
            try:
                from itertools import combinations

                for combo in combinations(sites, n_phosphorylation_events):
                    permutations.append(list(combo))
                    if (
                        self.max_permutations_ > 0
                        and len(permutations) > self.max_permutations_
                    ):
                        self._get_logger().info(
                            f"Debug: Early termination during iteration: current permutations ({len(permutations)}) exceeds maximum ({self.max_permutations_})"
                        )
                        return []
                return permutations
            except ImportError:

                def backtrack(start, current_combo):
                    if len(current_combo) == n_phosphorylation_events:
                        permutations.append(list(current_combo))
                        return

                    for i in range(start, len(sites)):
                        current_combo.append(sites[i])
                        backtrack(i + 1, current_combo)
                        current_combo.pop()

                        if (
                            self.max_permutations_ > 0
                            and len(permutations) > self.max_permutations_
                        ):
                            return

                backtrack(0, [])

                if (
                    self.max_permutations_ > 0
                    and len(permutations) > self.max_permutations_
                ):
                    return []

                return permutations

    def removePhosphositesFromSequence_(self, sequence):
        """Remove phosphorylation markers from sequence."""
        seq = sequence
        seq = seq.replace("(Phospho)", "")

        if self.add_decoys_:
            seq = seq.replace("(PhosphoDecoy)", "")

        without_phospho = AASequence.fromString(seq)

        return without_phospho

    def createTheoreticalSpectra_(self, permutations, seq_without_phospho):
        """Create theoretical spectra for phosphorylation site permutations."""
        th_spectra = []
        for _ in range(len(permutations)):
            th_spectra.append(MSSpectrum())

        seq_string = seq_without_phospho.toUnmodifiedString()

        for i in range(len(permutations)):
            seq = AASequence(seq_without_phospho)
            permu = 0

            # This loop walks positions ascending and advances through the
            # permutation in order, so the modification indices must be sorted;
            # otherwise an out-of-order index is skipped and silently dropped.
            perm = sorted(permutations[i])

            for as_pos in range(seq.size()):
                if permu < len(perm) and as_pos == perm[permu]:
                    residue = seq_string[as_pos]

                    if residue in ["S", "T", "Y"]:
                        seq.setModification(as_pos, "Phospho")
                        self._get_logger().info(
                            f"Debug: Set Phospho modification at position {as_pos} (residue {residue})"
                        )
                    elif self.add_decoys_ and residue == "A":
                        seq.setModification(as_pos, "PhosphoDecoy")
                        self._get_logger().info(
                            f"Debug: Set PhosphoDecoy modification at position {as_pos} (residue {residue})"
                        )

                    permu += 1

                if permu == len(perm):
                    break

            self.spectrum_generator_.getSpectrum(th_spectra[i], seq, 1, 1)
            th_spectra[i].setName(seq.toString())

        return th_spectra

    def peakPickingPerWindowsInSpectrum_(self, real_spectrum):
        """Pick top 10 intensity peaks for each 100 Da window."""
        windows_top10 = []

        spect_lower_bound = math.floor(real_spectrum[0].getMZ() / 100) * 100
        spect_upper_bound = (
            math.ceil(real_spectrum[real_spectrum.size() - 1].getMZ() / 100) * 100
        )

        number_of_windows = int(
            math.ceil((spect_upper_bound - spect_lower_bound) / 100)
        )
        for _ in range(number_of_windows):
            windows_top10.append(MSSpectrum())

        it_current_peak = 0
        window_upper_bound = spect_lower_bound + 100

        for current_window in range(number_of_windows):
            real_window = MSSpectrum()

            while (
                it_current_peak < real_spectrum.size()
                and real_spectrum[it_current_peak].getMZ() <= window_upper_bound
            ):
                real_window.push_back(real_spectrum[it_current_peak])
                it_current_peak += 1

            real_window.sortByIntensity(True)

            for i in range(min(10, real_window.size())):
                windows_top10[current_window].push_back(real_window[i])

            window_upper_bound += 100

        return windows_top10

    def calculatePermutationPeptideScores_(self, th_spectra, windows_depth_mz):
        """Calculate scores for each permutation at different peak depths.

        windows_depth_mz is a list (one per 100-Da window) of precomputed
        top-i m/z arrays (see windowDepthMZArrays_), so the inner matching no
        longer rebuilds/sorts MSSpectrum objects per depth.
        """
        permutation_peptide_scores = []
        for _ in range(len(th_spectra)):
            permutation_peptide_scores.append([0.0] * 10)

        for idx, spectrum in enumerate(th_spectra):
            N = spectrum.size()
            th_mz = self.spectrumMZArray_(spectrum)

            for i in range(1, 11):
                n = 0
                for window_depth_mz in windows_depth_mz:
                    n += self.countMatchedMZ_(th_mz, window_depth_mz, i)

                p = float(i) * self.base_match_probability_
                cumulative_score = self.computeCumulativeScore_(N, n, p)

                permutation_peptide_scores[idx][i - 1] = abs(
                    (-10.0 * math.log10(cumulative_score))
                )

        return permutation_peptide_scores

    def rankWeightedPermutationPeptideScores_(self, peptide_site_scores):
        """Rank permutations by weighted peptide scores."""
        ranking = []

        for i in range(len(peptide_site_scores)):
            weighted_score = self.peptideScore_(peptide_site_scores[i])
            ranking.append((weighted_score, i))

        ranking.sort(key=lambda x: x[0])

        return ranking

    def generateProFormaString_(self, peptide, ascores):
        """Generate ProForma string with phosphorylation site scores."""
        unmodified_str = peptide.toUnmodifiedString()

        position_scores = {}

        for position, ascore in ascores.items():
            probability = 1.0 - (10.0 ** (-ascore / 10.0))
            probability = max(0.0, min(1.0, probability))
            position_scores[position] = probability

        result = ""
        for i in range(len(unmodified_str)):
            result += unmodified_str[i]

            if i in position_scores:
                residue = unmodified_str[i]
                mod_name = ""

                if self.isPhosphoSite(residue):
                    mod_name = "Phospho"
                elif self.add_decoys_ and self.isPhosphoDecoySite(residue):
                    mod_name = "PhosphoDecoy"
                else:
                    continue

                score_str = "{:.4f}".format(position_scores[i])
                result += "[" + mod_name + "|score=" + score_str + "]"

        return result
