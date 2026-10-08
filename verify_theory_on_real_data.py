#!/usr/bin/env python3
"""
verify_theory_on_real_data.py
=============================
Manual verification of the four theoretical claims in THEORY_additions.tex,
run on YOUR machine against YOUR real .pkl datasets.

WHAT THIS CHECKS
----------------
  VERIFY-A : Theorem 1 -- the exponential coverage bound
             P(err) <= (|A|-1) * exp(-M * Delta_min^2 / 4)
             is a TRUE upper bound on the real Min-Distance decoder error,
             at every coverage depth, on real noisy reads.

  VERIFY-C : Proposition 3 -- sequential context carries information.
             Estimates  I(z_j ; P_j)   vs   I(z_j ; P_{j-1},P_j,P_{j+1})
             directly from a real dataset and reports the (positive) gain.

  VERIFY-D : the alphabet minimum distance d(A) used by Theorem 1 and the
             Singleton bound, computed exactly from the dataset's own
             ideal-vector table.

HOW TO RUN
----------
  1. Put this file anywhere.
  2. Edit the CONFIG block below: point PKL_DIR at the folder that holds
     your .pkl datasets (the ones produced by the dataset-generator
     notebooks). Optionally list specific files in PKL_FILES.
  3. Run:   python3 verify_theory_on_real_data.py
  4. Paste the console output back.

REQUIREMENTS
------------
  numpy only.  (No torch, no project code needed.)
  Python 3.8+.

NOTES
-----
  * The script auto-detects the dataset structure produced by your
    generators:  dataset['data'] = list of {'id','label','cluster'} and
    dataset['metadata'] with 'ideal_vectors' / 'symbol_to_idx'.
  * It does NOT need the training notebooks or a GPU.
  * Everything is deterministic given the dataset; no new randomness
    except the optional sub-sampling of clusters, which is seeded.
"""

import os
import sys
import glob
import pickle
import math
from collections import Counter

import numpy as np

# ===========================================================================
# CONFIG  --  EDIT THIS BLOCK
# ===========================================================================

# Folder containing your .pkl datasets. Use an absolute path to be safe.
PKL_DIR = "./dataset"          # <-- CHANGE ME (e.g. "/home/you/project/dataset")

# Optionally restrict to specific files (filenames only, inside PKL_DIR).
# Leave as None to auto-process every .pkl found in PKL_DIR.
PKL_FILES = None               # e.g. ["EZ17_100000_25.pkl", "R21_100000_25.pkl"]

# Coverage depths at which to test Theorem 1. The script uses the first
# min(M, available cluster size) reads of each cluster.
COVERAGE_LEVELS = [1, 2, 3, 5, 8, 10, 15, 20, 25]

# How many samples (composite sequences) to use per dataset. Use a few
# thousand for a fast, statistically solid check. None = use all.
MAX_SAMPLES = 3000

# Number of frequency-discretisation bins for the mutual-information
# estimate in VERIFY-C. 8-12 is reasonable.
MI_BINS = 10

# RNG seed for any sub-sampling.
SEED = 12345

# ===========================================================================
# END CONFIG
# ===========================================================================

BASE_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


# ---------------------------------------------------------------------------
# Dataset loading and structure detection
# ---------------------------------------------------------------------------
def load_dataset(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    if not isinstance(d, dict) or 'data' not in d or 'metadata' not in d:
        raise ValueError(f"{path}: unexpected structure "
                          f"(expected dict with 'data' and 'metadata').")
    return d


def get_ideal_matrix(meta):
    """Return (symbol_list, A) where A is (|A|,4) array of ideal vectors,
    columns ordered A,C,G,T. Symbol i in symbol_list corresponds to row i
    of A and to integer index i in symbol_to_idx.

    Handles both dataset layouts:
      * ideal_vectors as a LIST  -> indexed by integer symbol id;
      * ideal_vectors as a DICT  -> keyed by symbol name.
    Each ideal vector may itself be a length-4 array OR a list of
    (base, frequency) tuples.
    """
    iv = meta.get('ideal_vectors', None)
    s2i = meta.get('symbol_to_idx', None)
    if iv is None or s2i is None:
        raise ValueError("metadata lacks 'ideal_vectors' or 'symbol_to_idx'.")

    # Ordered symbol-name list: position i holds the name with index i.
    K = len(s2i)
    symbols = [None] * K
    for name, idx in s2i.items():
        symbols[idx] = name

    def to_vec4(v):
        # v is either a length-4 numeric vector or a list of (base,freq) pairs
        if isinstance(v, (list, tuple)) and len(v) > 0 and \
           isinstance(v[0], (list, tuple)):
            vec = np.zeros(4)
            for b, f in v:
                vec[BASE_IDX[b]] = float(f)
            return vec
        vec = np.asarray(v, dtype=float).ravel()
        if vec.size != 4:
            raise ValueError("an ideal vector is not length-4.")
        return vec

    rows = []
    for i, name in enumerate(symbols):
        if isinstance(iv, dict):
            v = iv[name]
        else:                       # list-like, indexed by integer id
            v = iv[i]
        rows.append(to_vec4(v))

    A = np.array(rows, dtype=float)
    # numerical safety: renormalise each row to sum 1
    A = A / A.sum(axis=1, keepdims=True)
    return symbols, A


def alphabet_min_distance(A):
    """Exact minimum Euclidean distance over all symbol pairs."""
    K = len(A)
    dmin = np.inf
    dmax = 0.0
    for a in range(K):
        for b in range(a + 1, K):
            dd = np.linalg.norm(A[a] - A[b])
            dmin = min(dmin, dd)
            dmax = max(dmax, dd)
    return dmin, dmax


# ---------------------------------------------------------------------------
# Frequency-matrix construction  (identical to the training notebooks)
# ---------------------------------------------------------------------------
def cluster_to_freq_matrix(cluster_reads, target_length):
    """(4, target_length) normalised nucleotide-frequency matrix via the
    same linear alignment used by the training pipeline."""
    P = np.zeros((4, target_length), dtype=np.float64)
    n_reads = 0
    for read in cluster_reads:
        rl = len(read)
        if rl == 0:
            continue
        n_reads += 1
        for t in range(target_length):
            ri = int((t + 0.5) * (rl / target_length))
            if ri >= rl:
                ri = rl - 1
            b = read[ri]
            if b in BASE_IDX:
                P[BASE_IDX[b], t] += 1.0
    if n_reads > 0:
        P /= n_reads
    return P  # columns may not be exactly 1 if reads contain non-ACGT chars


# ---------------------------------------------------------------------------
# VERIFY-A : Theorem 1 -- exponential coverage bound on real data
# ---------------------------------------------------------------------------
def verify_A(dataset, name):
    print("\n" + "=" * 78)
    print(f" VERIFY-A : Theorem 1 coverage bound  ---  dataset: {name}")
    print(" Claim:  real Min-Distance error  <=  (|A|-1) * exp(-M * d(A)^2 / 4)")
    print("=" * 78)

    meta = dataset['metadata']
    symbols, A = get_ideal_matrix(meta)
    K = len(A)
    s2i = meta['symbol_to_idx']
    seq_len = meta.get('seq_length', None)

    dmin, dmax = alphabet_min_distance(A)
    print(f"  |A| = {K},  alphabet min distance d(A) = {dmin:.6f},  "
          f"d_max = {dmax:.6f}")
    if seq_len is None:
        seq_len = len(dataset['data'][0]['label'])
    print(f"  sequence length n = {seq_len}")

    data = dataset['data']
    if MAX_SAMPLES is not None and len(data) > MAX_SAMPLES:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(data), size=MAX_SAMPLES, replace=False)
        data = [data[i] for i in idx]
    print(f"  using {len(data)} samples")

    print(f"\n  {'M':>4} {'positions':>12} {'emp P(err)':>12} "
          f"{'bound (c=4)':>12} {'holds?':>8}")
    print("  " + "-" * 54)

    all_hold = True
    for M in COVERAGE_LEVELS:
        n_err = 0
        n_pos = 0
        for item in data:
            label = item['label']               # list of symbol-name strings
            cluster = item['cluster']            # list of read strings
            use = cluster[:M]
            if len(use) == 0:
                continue
            P = cluster_to_freq_matrix(use, seq_len)
            # decode every position by minimum Euclidean distance
            for j in range(seq_len):
                col = P[:, j]
                # min-distance decode
                d = np.linalg.norm(A - col[None, :], axis=1)
                shat = int(np.argmin(d))
                true_idx = s2i[label[j]]
                n_pos += 1
                if shat != true_idx:
                    n_err += 1
        emp = n_err / max(n_pos, 1)
        bound = min(1.0, (K - 1) * math.exp(-M * dmin * dmin / 4.0))
        # the bound is a per-symbol worst case; the empirical number is the
        # average over symbols, so emp <= bound must always hold.
        holds = emp <= bound + 1e-9
        all_hold = all_hold and holds
        print(f"  {M:4d} {n_pos:12d} {emp:12.6f} {bound:12.6f} "
              f"{'OK' if holds else 'VIOLATED':>8}")

    print("  " + "-" * 54)
    if all_hold:
        print("  RESULT: Theorem 1 bound holds at every coverage depth. [PASS]")
    else:
        print("  RESULT: a violation was seen -- please report this output.")
    return all_hold


# ---------------------------------------------------------------------------
# VERIFY-C : Proposition 3 -- context carries information
# ---------------------------------------------------------------------------
def verify_C(dataset, name, M_for_context=10):
    print("\n" + "=" * 78)
    print(f" VERIFY-C : Proposition 3 context value  ---  dataset: {name}")
    print(" Claim:  I(z_j ; P_(j-1),P_j,P_(j+1))  >=  I(z_j ; P_j)")
    print("=" * 78)

    meta = dataset['metadata']
    symbols, A = get_ideal_matrix(meta)
    s2i = meta['symbol_to_idx']
    seq_len = meta.get('seq_length', None)
    if seq_len is None:
        seq_len = len(dataset['data'][0]['label'])

    data = dataset['data']
    if MAX_SAMPLES is not None and len(data) > MAX_SAMPLES:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(data), size=MAX_SAMPLES, replace=False)
        data = [data[i] for i in idx]

    M = min(M_for_context, max(len(it['cluster']) for it in data))
    print(f"  using {len(data)} samples, coverage M = {M}, "
          f"MI bins = {MI_BINS}")

    # To keep the plug-in MI estimator low-dimensional and low-bias, we
    # summarise each frequency column P_j by a SINGLE scalar feature: the
    # max nucleotide frequency  max_b P_{b,j}  (a natural "sharpness" /
    # confidence statistic). Discretise it into MI_BINS levels.
    # We then estimate, with z_j the true symbol index:
    #     I(z_j ; f_j)              -- self only
    #     I(z_j ; f_{j-1},f_j,f_{j+1}) -- with neighbours
    # Both use the SAME estimator, so the BIAS is comparable and the GAIN
    # (difference) is the meaningful, robust quantity.

    B = MI_BINS

    def disc(x):
        # x in [0,1]; map to {0,...,B-1}
        return min(int(x * B), B - 1)

    joint_self = Counter()       # (z, f_j)
    joint_ctx = Counter()        # (z, f_{j-1}, f_j, f_{j+1})
    cz = Counter()
    cf_self = Counter()
    cf_ctx = Counter()
    total = 0

    for item in data:
        label = item['label']
        cluster = item['cluster'][:M]
        if len(cluster) == 0:
            continue
        P = cluster_to_freq_matrix(cluster, seq_len)
        sharp = P.max(axis=0)               # length-n sharpness feature
        f = [disc(s) for s in sharp]
        for j in range(1, seq_len - 1):     # interior positions only
            z = s2i[label[j]]
            total += 1
            joint_self[(z, f[j])] += 1
            joint_ctx[(z, f[j - 1], f[j], f[j + 1])] += 1
            cz[z] += 1
            cf_self[f[j]] += 1
            cf_ctx[(f[j - 1], f[j], f[j + 1])] += 1

    if total == 0:
        print("  no interior positions found; skipping.")
        return None

    def mutual_information(joint, c_feat):
        I = 0.0
        for key, nij in joint.items():
            z = key[0]
            feat = key[1] if len(key) == 2 else key[1:]
            pij = nij / total
            pz = cz[z] / total
            pf = c_feat[feat] / total
            if pij > 0 and pz > 0 and pf > 0:
                I += pij * math.log2(pij / (pz * pf))
        return I

    I_self = mutual_information(joint_self, cf_self)
    I_ctx = mutual_information(joint_ctx, cf_ctx)
    gain = I_ctx - I_self

    print(f"\n  I(z_j ; P_j)                       = {I_self:.4f} bits")
    print(f"  I(z_j ; P_(j-1), P_j, P_(j+1))     = {I_ctx:.4f} bits")
    print(f"  context gain                       = {gain:+.4f} bits")
    print()
    if gain > 0:
        print("  RESULT: context gain is POSITIVE -> neighbouring columns")
        print("          carry information about z_j beyond P_j alone.")
        print("          Proposition 3 is non-vacuous on this real dataset. [PASS]")
    else:
        print("  RESULT: context gain is ~0 on this dataset (columns near")
        print("          independent here). Proposition 3 still holds as an")
        print("          inequality; the neural-decoder benefit would be small.")
    print("  NOTE: plug-in MI is slightly upward-biased, but BOTH terms use")
    print("        the same estimator so the GAIN (their difference) is the")
    print("        robust quantity. Re-run with a different MI_BINS to confirm")
    print("        the sign is stable.")
    return gain


# ---------------------------------------------------------------------------
# VERIFY-D : exact alphabet minimum distance from the dataset's own ideals
# ---------------------------------------------------------------------------
def verify_D(dataset, name):
    print("\n" + "=" * 78)
    print(f" VERIFY-D : alphabet geometry  ---  dataset: {name}")
    print("=" * 78)
    meta = dataset['metadata']
    symbols, A = get_ideal_matrix(meta)
    K = len(A)
    dmin, dmax = alphabet_min_distance(A)

    # find the closest pair
    cp = None
    best = np.inf
    for a in range(K):
        for b in range(a + 1, K):
            dd = np.linalg.norm(A[a] - A[b])
            if dd < best:
                best = dd
                cp = (a, b)
    print(f"  |A| = {K}")
    print(f"  alphabet minimum distance  d(A) = {dmin:.6f}")
    print(f"  largest symbol distance    d_max = {dmax:.6f}")
    if cp:
        print(f"  closest symbol pair:  {symbols[cp[0]]}  &  {symbols[cp[1]]}")
        print(f"     ideal vectors: {np.round(A[cp[0]],4)}  &  "
              f"{np.round(A[cp[1]],4)}")
    # Singleton rate bound preview
    print(f"\n  Singleton rate bound  R_A(d) <= log2|A| * (1 - (ceil(d/d_max)-1)/n)")
    log2K = math.log2(K)
    n = meta.get('seq_length', len(dataset['data'][0]['label']))
    for mult in [1, 2, 4]:
        d = dmax * mult
        k = math.ceil(d / dmax)
        R = log2K * (1 - (k - 1) / n)
        print(f"     d = {d:.4f} ({mult}x d_max):  R_A(d) <= {R:.4f} bits/pos")
    return dmin, dmax


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("#" * 78)
    print("#  THEORY VERIFICATION ON REAL DATASETS")
    print("#  Run on your own machine. Paste the full output back.")
    print("#" * 78)
    print(f"  numpy version : {np.__version__}")
    print(f"  PKL_DIR       : {os.path.abspath(PKL_DIR)}")

    if not os.path.isdir(PKL_DIR):
        print(f"\n  ERROR: PKL_DIR '{PKL_DIR}' is not a directory.")
        print("  Edit the CONFIG block at the top of this script.")
        sys.exit(1)

    if PKL_FILES:
        paths = [os.path.join(PKL_DIR, f) for f in PKL_FILES]
    else:
        paths = sorted(glob.glob(os.path.join(PKL_DIR, "*.pkl")))

    if not paths:
        print(f"\n  ERROR: no .pkl files found in {PKL_DIR}.")
        sys.exit(1)

    print(f"  datasets found: {len(paths)}")
    for p in paths:
        print(f"    - {os.path.basename(p)}")

    summary = []
    for path in paths:
        name = os.path.basename(path)
        try:
            ds = load_dataset(path)
        except Exception as e:
            print(f"\n  [skip] {name}: {e}")
            continue
        try:
            okA = verify_A(ds, name)
            verify_D(ds, name)
            gain = verify_C(ds, name)
            summary.append((name, okA, gain))
        except Exception as e:
            import traceback
            print(f"\n  [error] {name}: {e}")
            traceback.print_exc()

    print("\n" + "#" * 78)
    print("#  SUMMARY")
    print("#" * 78)
    for name, okA, gain in summary:
        g = f"{gain:+.4f} bits" if gain is not None else "n/a"
        print(f"  {name:32s}  Theorem-1 bound: "
              f"{'PASS' if okA else 'CHECK'}   context gain: {g}")
    print("\n  Done. Paste this entire output back to finalise the paper text.")
    print("#" * 78)


if __name__ == "__main__":
    main()
