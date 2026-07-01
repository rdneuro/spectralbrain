"""Classic morphometric spectra: EFD, eigenshape, principal warps, SPHARM.

These are the long-established spectral/decomposition methods of
morphometrics, complementing the operator spectra elsewhere in
:mod:`spectralbrain.expanded`:

- **Elliptic Fourier Descriptors (EFD)** — the Fourier series of a closed
  2-D outline (Kuhl & Giardina 1982); the workhorse of outline-based shape
  analysis, with optional rotation/scale/phase normalisation.
- **Eigenshape analysis** — PCA of aligned shape coordinates (Lohmann
  1983): the principal modes of shape variation in a sample.
- **Principal warps** — the eigen-decomposition of the thin-plate-spline
  bending-energy matrix (Bookstein 1989): an orthogonal basis of
  increasingly local landmark deformations; the affine subspace is its
  3-dimensional null space.
- **SPHARM** — spherical-harmonic expansion of a genus-0 surface
  (Brechbühler et al. 1995); the gold-standard parametric descriptor for
  subcortical structures, with a rotation-invariant power spectrum.

All four are implemented in pure NumPy/SciPy.

References
----------
Kuhl FP, Giardina CR. "Elliptic Fourier features of a closed contour."
*Computer Graphics and Image Processing* 18(3):236–258, 1982.
Lohmann GP. "Eigenshape analysis of microfossils." *Mathematical Geology*
15:659–672, 1983.
Bookstein FL. "Principal warps: thin-plate splines and the decomposition
of deformations." *IEEE TPAMI* 11(6):567–585, 1989.
Brechbühler C, Gerig G, Kübler O. "Parametrization of closed surfaces for
3-D shape description." *CVGIP* 61(2):154–170, 1995.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from spectralbrain.runtime import (
    DescriptorMatrix,
    GlobalDescriptor,
    get_logger,
)

logger = get_logger(__name__)

_EPS = 1e-12


# ======================================================================
# §1  ELLIPTIC FOURIER DESCRIPTORS  (Kuhl & Giardina 1982)
# ======================================================================


def elliptic_fourier_descriptors(
    contour: np.ndarray,
    *,
    order: int = 10,
    normalize: bool = True,
) -> DescriptorMatrix:
    """Elliptic Fourier descriptors of a closed 2-D outline.

    Parameters
    ----------
    contour : ndarray, shape (P, 2)
        Ordered ``(x, y)`` vertices of a **closed** contour (the closing
        segment from the last point back to the first is added implicitly).
    order : int
        Number of harmonics.
    normalize : bool
        If ``True``, apply the Kuhl–Giardina normalisation (invariant to
        translation, scale, rotation and starting point): the first
        harmonic is rotated to its semi-major axis and the coefficients are
        scaled by its magnitude.

    Returns
    -------
    ndarray, shape (order, 4)
        Rows ``[a_n, b_n, c_n, d_n]`` per harmonic.

    Notes
    -----
    A circle yields a dominant first harmonic with negligible higher
    harmonics; after normalisation ``a₁ = 1`` and ``b₁ = c₁ = 0``.
    """
    c = np.asarray(contour, dtype=np.float64)
    if c.ndim != 2 or c.shape[1] != 2:
        raise ValueError(f"contour must be (P, 2), got {c.shape}.")

    dxy = np.diff(c, axis=0, append=c[:1])  # closing segment
    dt = np.sqrt((dxy**2).sum(axis=1))
    dt = np.clip(dt, _EPS, None)
    t = np.concatenate([[0.0], np.cumsum(dt)])
    T = t[-1]

    coeffs = np.zeros((order, 4), dtype=np.float64)
    two_pi = 2.0 * np.pi
    for n in range(1, order + 1):
        const = T / (2.0 * np.pi**2 * n**2)
        phi = two_pi * n * t / T  # (P+1,)
        cos_d = np.cos(phi[1:]) - np.cos(phi[:-1])
        sin_d = np.sin(phi[1:]) - np.sin(phi[:-1])
        a = const * np.sum((dxy[:, 0] / dt) * cos_d)
        b = const * np.sum((dxy[:, 0] / dt) * sin_d)
        cc = const * np.sum((dxy[:, 1] / dt) * cos_d)
        d = const * np.sum((dxy[:, 1] / dt) * sin_d)
        coeffs[n - 1] = [a, b, cc, d]

    if normalize:
        coeffs = _normalize_efd(coeffs)
    return coeffs


def _normalize_efd(coeffs: np.ndarray) -> np.ndarray:
    """Kuhl–Giardina normalisation (rotation/scale/start-point invariant)."""
    a1, b1, c1, d1 = coeffs[0]
    # starting-phase rotation theta1
    theta1 = 0.5 * np.arctan2(2.0 * (a1 * b1 + c1 * d1), a1**2 + c1**2 - b1**2 - d1**2)
    n = np.arange(1, coeffs.shape[0] + 1)
    out = np.zeros_like(coeffs)
    for i, k in enumerate(n):
        ct, st = np.cos(k * theta1), np.sin(k * theta1)
        mat = np.array([[coeffs[i, 0], coeffs[i, 1]], [coeffs[i, 2], coeffs[i, 3]]])
        rot = mat @ np.array([[ct, -st], [st, ct]])
        out[i] = rot.ravel()
    # rotation psi1 to align semi-major axis, and scale by magnitude E
    psi1 = np.arctan2(out[0, 2], out[0, 0])
    cps, sps = np.cos(psi1), np.sin(psi1)
    rot_psi = np.array([[cps, sps], [-sps, cps]])
    e = np.hypot(out[0, 0], out[0, 2])
    e = e if e > _EPS else 1.0
    for i in range(coeffs.shape[0]):
        mat = np.array([[out[i, 0], out[i, 1]], [out[i, 2], out[i, 3]]])
        res = (rot_psi @ mat) / e
        out[i] = res.ravel()
    return out


# ======================================================================
# §2  EIGENSHAPE ANALYSIS  (Lohmann 1983)
# ======================================================================


def eigenshape_analysis(
    shapes: np.ndarray,
    *,
    n_components: int | None = None,
) -> dict[str, Any]:
    """Principal modes of shape variation (PCA of aligned coordinates).

    Parameters
    ----------
    shapes : ndarray, shape (S, D)
        ``S`` shapes, each a flattened, pre-aligned coordinate vector of
        length ``D`` (e.g. ``2·n_landmarks``).
    n_components : int, optional
        Number of eigenshapes to keep (all if ``None``).

    Returns
    -------
    dict
        ``mean`` (D,), ``eigenshapes`` (C, D), ``explained_variance`` (C,),
        ``explained_variance_ratio`` (C,) and ``scores`` (S, C) — the shape
        coordinates in the eigenshape basis.
    """
    x = np.asarray(shapes, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"shapes must be 2-D (S, D), got {x.shape}.")
    s = x.shape[0]
    mean = x.mean(axis=0)
    xc = x - mean
    # SVD of the centred data
    u, sv, vt = np.linalg.svd(xc, full_matrices=False)
    var = (sv**2) / max(s - 1, 1)
    total = var.sum()
    if n_components is not None:
        vt = vt[:n_components]
        sv = sv[:n_components]
        var = var[:n_components]
        u = u[:, :n_components]
    scores = u * sv[None, :]
    return {
        "mean": mean,
        "eigenshapes": vt,
        "explained_variance": var,
        "explained_variance_ratio": var / max(total, _EPS),
        "scores": scores,
    }


# ======================================================================
# §3  PRINCIPAL WARPS  (Bookstein 1989, TPS bending energy)
# ======================================================================


def _tps_kernel(landmarks: np.ndarray) -> np.ndarray:
    """Thin-plate-spline kernel ``U(r) = r² ln r`` over landmark pairs."""
    p = landmarks.shape[0]
    diff = landmarks[:, None, :] - landmarks[None, :, :]
    r2 = np.sum(diff**2, axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = 0.5 * r2 * np.log(np.clip(r2, _EPS, None))
    np.fill_diagonal(k, 0.0)
    return k


def principal_warps(landmarks: np.ndarray) -> dict[str, np.ndarray]:
    """Principal warps and their bending energies (Bookstein 1989).

    Builds the TPS interpolation system ``L = [[K, P], [Pᵀ, 0]]`` for the
    reference landmarks and eigendecomposes the **bending-energy matrix**
    (the ``p × p`` upper-left block of ``L⁻¹``).  The eigenvectors are the
    *principal warps* — an orthogonal basis of deformations ordered from
    global/affine to increasingly local — and the eigenvalues are their
    bending energies.

    Parameters
    ----------
    landmarks : ndarray, shape (P, 2)
        Reference landmark configuration (2-D).

    Returns
    -------
    dict
        ``bending_energies`` (P,) ascending, ``principal_warps`` (P, P)
        (columns are warps), and ``affine_nullity`` — the count of
        zero-energy modes (3 for non-degenerate 2-D configurations: the
        affine subspace).
    """
    lm = np.asarray(landmarks, dtype=np.float64)
    if lm.ndim != 2 or lm.shape[1] != 2:
        raise ValueError(f"landmarks must be (P, 2), got {lm.shape}.")
    p = lm.shape[0]

    k = _tps_kernel(lm)
    pmat = np.column_stack([np.ones(p), lm])  # (P, 3)
    top = np.hstack([k, pmat])
    bot = np.hstack([pmat.T, np.zeros((3, 3))])
    big = np.vstack([top, bot])
    big_inv = np.linalg.pinv(big)
    be = big_inv[:p, :p]  # bending-energy matrix
    be = 0.5 * (be + be.T)

    w, v = np.linalg.eigh(be)
    order = np.argsort(w)
    w, v = w[order], v[:, order]
    scale = max(np.abs(w).max(), _EPS)
    nullity = int(np.sum(np.abs(w) < 1e-9 * scale))
    return {
        "bending_energies": w,
        "principal_warps": v,
        "affine_nullity": nullity,
    }


# ======================================================================
# §4  SPHERICAL HARMONICS  (SPHARM, Brechbühler et al. 1995)
# ======================================================================


def _sph_harm(
    m: int, l: int, azimuth: np.ndarray, polar: np.ndarray
) -> np.ndarray:
    """Spherical harmonic ``Y_l^m`` with azimuth∈[0,2π], polar∈[0,π].

    Compatibility shim across SciPy versions: the legacy
    ``scipy.special.sph_harm(m, n, theta_azimuth, phi_polar)`` and the newer
    ``scipy.special.sph_harm_y(n, m, theta_polar, phi_azimuth)`` swap both
    the argument order and the angle convention.
    """
    from scipy import special

    if hasattr(special, "sph_harm_y"):
        return np.asarray(special.sph_harm_y(l, m, polar, azimuth))
    return np.asarray(special.sph_harm(m, l, azimuth, polar))


def _real_sph_design(
    theta: np.ndarray, phi: np.ndarray, l_max: int
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Complex spherical-harmonic design matrix and its (l, m) index list.

    ``theta`` is the azimuth (0–2π), ``phi`` the polar angle (0–π).
    """
    cols: list[np.ndarray] = []
    index: list[tuple[int, int]] = []
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            cols.append(_sph_harm(m, l, theta, phi))
            index.append((l, m))
    return np.column_stack(cols), index


def spharm_coefficients(
    theta: np.ndarray,
    phi: np.ndarray,
    values: np.ndarray,
    *,
    l_max: int = 12,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Least-squares spherical-harmonic coefficients of a function on S².

    Parameters
    ----------
    theta : ndarray, shape (N,)
        Azimuthal angle of each sample (0–2π).
    phi : ndarray, shape (N,)
        Polar angle of each sample (0–π).
    values : ndarray, shape (N,) or (N, C)
        Function value(s) at each sample (e.g. radius, or x/y/z).
    l_max : int
        Maximum degree.

    Returns
    -------
    coeffs : ndarray, shape ((l_max+1)², ) or ((l_max+1)², C)
        Complex SH coefficients.
    index : list of (l, m)
        Degree/order for each coefficient row.
    """
    y, index = _real_sph_design(
        np.asarray(theta, float), np.asarray(phi, float), l_max
    )
    vals = np.asarray(values, dtype=np.complex128)
    coeffs, *_ = np.linalg.lstsq(y, vals, rcond=None)
    return coeffs, index


def spharm_power_spectrum(
    coeffs: np.ndarray,
    index: list[tuple[int, int]],
) -> GlobalDescriptor:
    """Rotation-invariant SPHARM power spectrum ``P_l = Σ_m |c_{lm}|²``.

    Parameters
    ----------
    coeffs : ndarray
        Coefficients from :func:`spharm_coefficients`.
    index : list of (l, m)

    Returns
    -------
    ndarray, shape (l_max+1,)
        Per-degree power — invariant to rotations of the surface.
    """
    l_max = max(l for l, _ in index)
    power = np.zeros(l_max + 1, dtype=np.float64)
    mag = np.abs(coeffs) ** 2
    if mag.ndim > 1:
        mag = mag.sum(axis=1)
    for (l, _m), val in zip(index, mag):
        power[l] += float(val)
    return power


def spharm_surface(
    vertices: np.ndarray,
    *,
    l_max: int = 12,
    center: np.ndarray | None = None,
) -> dict[str, Any]:
    """SPHARM expansion of a star-shaped surface's radial function.

    Parameterises each vertex by the spherical angles of its direction from
    the centroid and expands the radius ``r(θ, φ)`` in spherical harmonics.
    Suitable for star-shaped subcortical structures (a practical SPHARM
    descriptor without a full conformal spherical parameterisation).

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
    l_max : int
        Maximum SH degree.
    center : ndarray, shape (3,), optional
        Expansion centre (centroid if ``None``).

    Returns
    -------
    dict
        ``coeffs`` (complex SH coefficients of the radius), ``index``,
        and ``power_spectrum`` (rotation-invariant ``P_l``).

    Notes
    -----
    A sphere yields all power in ``P₀`` (constant radius); departures into
    higher ``P_l`` quantify lobation/asymmetry.
    """
    v = np.asarray(vertices, dtype=np.float64)
    if center is None:
        center = v.mean(axis=0)
    d = v - center
    r = np.linalg.norm(d, axis=1)
    rn = np.clip(r, _EPS, None)
    phi = np.arccos(np.clip(d[:, 2] / rn, -1.0, 1.0))  # polar 0..π
    theta = np.arctan2(d[:, 1], d[:, 0]) % (2.0 * np.pi)  # azimuth 0..2π
    coeffs, index = spharm_coefficients(theta, phi, r, l_max=l_max)
    return {
        "coeffs": coeffs,
        "index": index,
        "power_spectrum": spharm_power_spectrum(coeffs, index),
    }


__all__ = [
    "eigenshape_analysis",
    "elliptic_fourier_descriptors",
    "principal_warps",
    "spharm_coefficients",
    "spharm_power_spectrum",
    "spharm_surface",
]
