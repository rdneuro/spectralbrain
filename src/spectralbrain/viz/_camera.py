"""Shared anatomical camera conventions for all 3D renderers.

Every 3D renderer in :mod:`spectralbrain.viz` (vedo, PyVista, FURY) uses the
same **RAS** convention, so a view name always means the same thing:

* ``x`` = right (+) / left (−), ``y`` = anterior (+) / posterior (−),
  ``z`` = superior (+) / inferior (−).
* ``azimuth`` is measured in the axial plane from +x towards +y and
  ``elevation`` towards +z; the camera sits at
  ``center + d · (cos el · cos az, cos el · sin az, sin el)``.

So the *left lateral* camera sits at −x, *anterior* at +y and *superior* at +z.
Medial views assume a single hemisphere (``left_medial`` looks from +x).

Cameras are always built as explicit ``position``/``focal_point``/``viewup``
dicts — never as relative ``azimuth``/``elevation`` rotations, which several
backends silently override (vedo's ``viewup="z"`` resets the camera).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

#: (azimuth, elevation) in degrees, RAS convention (see module docstring).
VIEW_ANGLES: dict[str, tuple[float, float]] = {
    "anterior": (90.0, 0.0),
    "posterior": (-90.0, 0.0),
    "left_lateral": (180.0, 0.0),
    "right_lateral": (0.0, 0.0),
    "superior": (0.0, 90.0),
    "inferior": (0.0, -90.0),
    "left_medial": (0.0, 0.0),
    "right_medial": (180.0, 0.0),
    "oblique_left": (135.0, 30.0),
    "oblique_right": (45.0, 30.0),
    # Short aliases used by the tractography renderers.
    "left": (180.0, 0.0),
    "right": (0.0, 0.0),
    "oblique": (135.0, 25.0),
}


def presets(names: Iterable[str]) -> dict[str, dict[str, Any]]:
    """``{name: {"azimuth", "elevation"}}`` presets for the given view names."""
    return {n: {"azimuth": VIEW_ANGLES[n][0], "elevation": VIEW_ANGLES[n][1]} for n in names}


def validate_views(views: Iterable[str], allowed: Iterable[str] | None = None) -> list[str]:
    """Return ``views`` as a list, raising ``ValueError`` on unknown names."""
    allowed_set = set(VIEW_ANGLES if allowed is None else allowed)
    views = list(views)
    bad = [v for v in views if v not in allowed_set]
    if bad:
        raise ValueError(f"Unknown view name(s) {bad}. Valid views: {sorted(allowed_set)}")
    return views


def view_direction(view: str, angles: dict[str, Any] | None = None) -> np.ndarray:
    """Unit vector from the focal point towards the camera for ``view``."""
    if angles is not None and view in angles:
        a = angles[view]
        az, el = (a["azimuth"], a["elevation"]) if isinstance(a, dict) else a
    elif view in VIEW_ANGLES:
        az, el = VIEW_ANGLES[view]
    else:
        raise ValueError(f"Unknown view name {view!r}. Valid views: {sorted(VIEW_ANGLES)}")
    az, el = np.deg2rad(az), np.deg2rad(el)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def view_up(direction: np.ndarray) -> tuple[float, float, float]:
    """+z is up, except for (near) top/bottom views where +y (anterior) is up."""
    return (0.0, 1.0, 0.0) if abs(direction[2]) > 0.98 else (0.0, 0.0, 1.0)


def camera_for_view(
    view: str,
    points: np.ndarray,
    *,
    angles: dict[str, Any] | None = None,
    view_angle: float = 30.0,
    margin: float = 1.15,
) -> dict[str, Any]:
    """Explicit camera dict framing ``points`` from the anatomical ``view``.

    The dict uses the keys understood by ``vedo.utils.camera_from_dict``
    (``position``, ``focal_point``, ``viewup``, ``clipping_range``,
    ``view_angle``).
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.size == 0:
        center, radius = np.zeros(3), 1.0
    else:
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        center = 0.5 * (lo + hi)
        radius = float(np.linalg.norm(hi - lo)) / 2.0 or 1.0
    d = view_direction(view, angles)
    dist = radius * margin / np.tan(np.deg2rad(view_angle) / 2.0)
    return {
        "position": tuple(center + dist * d),
        "focal_point": tuple(center),
        "viewup": view_up(d),
        "view_angle": view_angle,
        "clipping_range": (max(dist - 3.0 * radius, 1e-3 * dist), dist + 3.0 * radius),
    }


__all__ = ["VIEW_ANGLES", "camera_for_view", "presets", "validate_views", "view_direction"]
