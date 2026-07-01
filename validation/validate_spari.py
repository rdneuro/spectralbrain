#!/usr/bin/env python
"""Validation harness for SpectralBrain's spatially-aware (adjusted) Rand index.

The published spARI (Yan, Feng & Luo 2025, Biometrics 81(3):ujaf127) ships an
R-only reference. SpectralBrain's :func:`spectralbrain.statistics.spatial_rand_index`
is a structural reimplementation in the weighted-Rand / chance-corrected family
(distance-weighted disagreement pairs + permutation chance baseline), flagged
``validated_against_R=False`` until confirmed cell-for-cell against that package.

This script provides three layers of validation:

1. **Analytical properties** (run anywhere, no R): every correct spatial ARI must
   satisfy these, and they are exact or near-exact checks —
   - identity:            spARI(P, P) == 1
   - label invariance:    relabelling either partition does not change spARI
   - symmetry:            spARI(A, B) == spARI(B, A)
   - **reduction to ARI**: as the spatial length-scale -> 0 (weights -> indicator),
     spRI -> classical Rand index and spARI -> classical ARI. This is the
     strongest internal check: in the no-spatial limit the index must collapse
     onto the exact, scikit-learn-validated ARI.
   - chance level:        spARI ~ 0 for independent random partitions
   - locality monotonicity: moving a fixed disagreement closer (smaller distance)
     does not decrease spARI (near disagreements are penalised less).

2. **Library vs independent reference**: an independent reimplementation
   (``reference_spari`` below) is compared against the library on random cases.
   Agreement rules out transcription/indexing bugs (it does NOT prove identity
   with the R paper, since both follow the same documented structure).

3. **R bridge** (``--with-r``, run on a machine with R + the ``spARI`` package):
   generates random (a, b, coords) cases, computes spARI in Python and in R,
   writes a CSV of paired values, and reports max |Δ| and correlation. This is
   the cell-for-cell validation that justifies flipping ``validated_against_R``.

Usage
-----
    python validate_spari.py                 # properties + reference cross-check
    python validate_spari.py --with-r        # also run the R cell-for-cell bridge
    python validate_spari.py --n-cases 200   # more random cases
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")  # silence the not-yet-validated notice during testing

from spectralbrain.statistics import spatial_rand_index
from sklearn.metrics import adjusted_rand_score


# ----------------------------------------------------------------------
# Independent reference implementation (structural, NOT the R package)
# ----------------------------------------------------------------------
def reference_spari(a, b, coords=None, distance=None, length_scale=None,
                    n_perm=400, seed=0):
    """Independent weighted-Rand reference, derived from the documented structure.

    Concordant pairs score 1; disagreement pairs score w_ij = exp(-d_ij / ell).
    spRI = mean pair score; spARI = (spRI - E)/(1 - E) with E from label
    permutation. Written from scratch (different code path / vectorisation than
    the library) so agreement is a genuine cross-check.
    """
    a = np.asarray(a); b = np.asarray(b); n = a.shape[0]
    if distance is not None:
        D = np.asarray(distance, float)
    else:
        from scipy.spatial.distance import cdist
        coords = np.asarray(coords, float)
        D = cdist(coords, coords)
    if length_scale is None:
        iu0 = np.triu_indices(n, 1)
        length_scale = float(np.median(D[iu0]))
    W = np.exp(-D / length_scale)

    iu = np.triu_indices(n, 1)
    sa = (a[:, None] == a[None, :])[iu]
    sb = (b[:, None] == b[None, :])[iu]
    concord = (sa == sb)
    w = W[iu]
    score = np.where(concord, 1.0, w)
    spri = float(score.mean())

    rng = np.random.default_rng(seed)
    base = np.empty(n_perm)
    for p in range(n_perm):
        bp = b[rng.permutation(n)]
        sbp = (bp[:, None] == bp[None, :])[iu]
        base[p] = np.where(sa == sbp, 1.0, w).mean()
    e = float(base.mean())
    spari = (spri - e) / (1 - e) if abs(1 - e) > 1e-12 else float("nan")
    return {"spRI": spri, "spARI": spari, "length_scale": length_scale}


def _lib(a, b, **kw):
    return spatial_rand_index(a, b, **kw)


# ----------------------------------------------------------------------
# Property tests
# ----------------------------------------------------------------------
def _random_case(n, k_a, k_b, rng, dim=2):
    a = rng.integers(0, k_a, size=n)
    b = rng.integers(0, k_b, size=n)
    coords = rng.normal(size=(n, dim))
    return a, b, coords


def test_identity(rng):
    a, _, coords = _random_case(80, 4, 4, rng)
    r = _lib(a, a, coords=coords)
    ok = abs(r["spARI"] - 1.0) < 1e-9 and abs(r["spRI"] - 1.0) < 1e-9
    return ok, f"spARI(P,P)={r['spARI']:.6f} spRI={r['spRI']:.6f} (want 1)"


def test_label_invariance(rng):
    a, b, coords = _random_case(80, 4, 3, rng)
    r1 = _lib(a, b, coords=coords)["spARI"]
    # relabel a and b arbitrarily
    pa = rng.permutation(a.max() + 1); pb = rng.permutation(b.max() + 1)
    r2 = _lib(pa[a], pb[b], coords=coords)["spARI"]
    ok = abs(r1 - r2) < 1e-9
    return ok, f"relabel Δ={abs(r1 - r2):.2e}"


def test_symmetry(rng):
    a, b, coords = _random_case(80, 5, 4, rng)
    # symmetry holds for a fixed distance matrix (avoid permutation-seed asymmetry
    # by supplying a precomputed distance and many permutations is not needed for spRI)
    from scipy.spatial.distance import cdist
    D = cdist(coords, coords)
    r_ab = _lib(a, b, distance=D, adjusted=False)["spRI"]
    r_ba = _lib(b, a, distance=D, adjusted=False)["spRI"]
    ok = abs(r_ab - r_ba) < 1e-9
    return ok, f"spRI sym Δ={abs(r_ab - r_ba):.2e}"


def test_reduction_to_ari(rng):
    """As length_scale -> 0, disagreement weights -> 0, so spARI -> classical ARI."""
    deltas = []
    for _ in range(8):
        a, b, coords = _random_case(120, 5, 4, rng)
        from scipy.spatial.distance import cdist
        D = cdist(coords, coords)
        # tiny length scale relative to the smallest nonzero distance
        dmin = np.min(D[D > 0])
        r = _lib(a, b, distance=D, length_scale=dmin * 1e-3)
        ari = adjusted_rand_score(a, b)
        deltas.append(abs(r["spARI"] - ari))
    md = float(np.max(deltas))
    # permutation baseline (200 draws) introduces small MC error; tolerance 0.03
    ok = md < 0.03
    return ok, f"max|spARI(ell->0) - ARI| = {md:.4f} (tol 0.03)"


def test_chance_level(rng):
    """Independent random partitions -> spARI near 0 on average."""
    vals = []
    for _ in range(20):
        a, b, coords = _random_case(100, 4, 4, rng)
        vals.append(_lib(a, b, coords=coords)["spARI"])
    m = float(np.mean(vals))
    ok = abs(m) < 0.10
    return ok, f"mean spARI over random pairs = {m:+.4f} (want ~0)"


def test_locality_monotonicity(rng):
    """A disagreement between two NEAR points is penalised less than between FAR
    points, so making a fixed disagreement closer must not decrease spRI."""
    n = 60
    a = np.zeros(n, int); a[: n // 2] = 1            # two equal blocks
    b = a.copy()
    # introduce one disagreement: move point 0 to b's other cluster
    b[0] = 1 - b[0]
    # near geometry: place point 0 close to the cluster it disagrees about
    coords_near = rng.normal(size=(n, 2)); coords_near[0] = coords_near[1] + 1e-3
    coords_far = coords_near.copy(); coords_far[0] = coords_near[0] + 50.0
    from scipy.spatial.distance import cdist
    s_near = _lib(a, b, distance=cdist(coords_near, coords_near), adjusted=False)["spRI"]
    s_far = _lib(a, b, distance=cdist(coords_far, coords_far), adjusted=False)["spRI"]
    ok = s_near >= s_far - 1e-9
    return ok, f"spRI near={s_near:.5f} >= far={s_far:.5f}"


def test_reference_agreement(rng, n_cases):
    """Library vs the independent reference on random cases (fixed seeds)."""
    deltas = []
    for i in range(n_cases):
        a, b, coords = _random_case(70, 5, 4, np.random.default_rng(1000 + i))
        from scipy.spatial.distance import cdist
        D = cdist(coords, coords)
        # match permutation seed/handling: compare spRI (deterministic) exactly,
        # and spARI within MC tolerance.
        lib = _lib(a, b, distance=D)
        ref = reference_spari(a, b, distance=D)
        deltas.append((abs(lib["spRI"] - ref["spRI"]), abs(lib["spARI"] - ref["spARI"])))
    deltas = np.array(deltas)
    ok = float(deltas[:, 0].max()) < 1e-9 and float(deltas[:, 1].max()) < 0.05
    return ok, (f"max|ΔspRI|={deltas[:,0].max():.2e} (exact)  "
                f"max|ΔspARI|={deltas[:,1].max():.4f} (MC tol 0.05)")


# ----------------------------------------------------------------------
# R bridge (cell-for-cell against the spARI package)
# ----------------------------------------------------------------------
R_TEMPLATE = r"""
# Requires: install.packages("spARI")  (Yan, Feng & Luo 2025)
suppressMessages(library(spARI))
args <- commandArgs(trailingOnly = TRUE)
a <- as.integer(read.csv(args[1], header=FALSE)[,1])
b <- as.integer(read.csv(args[2], header=FALSE)[,1])
xy <- as.matrix(read.csv(args[3], header=FALSE))
# spARI(labels1, labels2, coordinates) -> list/vector with spRI and spARI.
res <- spARI(a, b, xy)
cat(sprintf("%.10f,%.10f\n", as.numeric(res$spRI), as.numeric(res$spARI)))
"""


def run_r_bridge(n_cases=50, seed=0, tol=1e-3):
    """If Rscript + the spARI package are available, compare cell-for-cell."""
    if not _have_rscript():
        print("  [skip] Rscript not found on PATH. Install R + the 'spARI' "
              "package, then re-run with --with-r on your workstation.")
        return None
    tmp = Path(tempfile.mkdtemp())
    rfile = tmp / "spari_ref.R"
    rfile.write_text(R_TEMPLATE)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_cases):
        n = int(rng.integers(40, 120))
        a = rng.integers(0, rng.integers(2, 6), size=n)
        b = rng.integers(0, rng.integers(2, 6), size=n)
        xy = rng.normal(size=(n, 2))
        fa, fb, fxy = tmp / "a.csv", tmp / "b.csv", tmp / "xy.csv"
        np.savetxt(fa, a, fmt="%d"); np.savetxt(fb, b, fmt="%d")
        np.savetxt(fxy, xy, delimiter=",")
        try:
            out = subprocess.run(["Rscript", str(rfile), str(fa), str(fb), str(fxy)],
                                 capture_output=True, text=True, timeout=120)
            if out.returncode != 0:
                print("  [R error]", out.stderr.strip().splitlines()[-1:] or out.stderr)
                return None
            r_spri, r_spari = map(float, out.stdout.strip().split(","))
        except Exception as exc:  # pragma: no cover
            print("  [R bridge failed]", exc)
            return None
        py = spatial_rand_index(a, b, coords=xy)
        rows.append((i, r_spri, py["spRI"], r_spari, py["spARI"]))
    rows = np.array(rows, float)
    csv = tmp / "spari_py_vs_r.csv"
    np.savetxt(csv, rows, delimiter=",",
               header="case,R_spRI,py_spRI,R_spARI,py_spARI", comments="")
    d_spri = np.abs(rows[:, 1] - rows[:, 2]).max()
    d_spari = np.abs(rows[:, 3] - rows[:, 4]).max()
    corr = float(np.corrcoef(rows[:, 3], rows[:, 4])[0, 1])
    print(f"  R bridge: {len(rows)} cases | max|ΔspRI|={d_spri:.2e} "
          f"max|ΔspARI|={d_spari:.2e} corr(spARI)={corr:.5f}")
    print(f"  paired values written to {csv}")
    agree = d_spari < tol
    print(f"  -> {'MATCH' if agree else 'MISMATCH'} at tol={tol}. "
          f"{'You may set validated_against_R=True.' if agree else 'Port the R weighting before publication.'}")
    return agree


def _have_rscript():
    try:
        subprocess.run(["Rscript", "--version"], capture_output=True, timeout=15)
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Validate spARI.")
    ap.add_argument("--with-r", action="store_true", help="run the R cell-for-cell bridge")
    ap.add_argument("--n-cases", type=int, default=60, help="random cases for the reference cross-check")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("=" * 70)
    print("spARI validation — analytical properties (no R required)")
    print("=" * 70)
    checks = [
        ("identity  spARI(P,P)=1", test_identity(rng)),
        ("label invariance", test_label_invariance(rng)),
        ("symmetry", test_symmetry(rng)),
        ("REDUCTION to ARI (ell->0)", test_reduction_to_ari(rng)),
        ("chance level ~0", test_chance_level(rng)),
        ("locality monotonicity", test_locality_monotonicity(rng)),
        (f"library vs reference (n={args.n_cases})", test_reference_agreement(rng, args.n_cases)),
    ]
    n_pass = 0
    for name, (ok, msg) in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:32s} | {msg}")
        n_pass += int(ok)
    print(f"\n  {n_pass}/{len(checks)} property checks passed.")

    print("\n" + "=" * 70)
    print("Cell-for-cell vs the R 'spARI' package")
    print("=" * 70)
    if args.with_r:
        run_r_bridge()
    else:
        print("  [not run] pass --with-r on a machine with R + spARI installed.")
        print("  This is the check that justifies flipping validated_against_R=True.")

    return 0 if n_pass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
