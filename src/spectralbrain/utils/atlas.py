"""Brain atlas label registries.

Maps atlas label IDs to human-readable region names, hemispheres,
network assignments, and canonical colours.  Covers the 19 atlases
in :class:`~spectralbrain.runtime.AtlasScheme`.

The two primary use cases are:

1. **Point-cloud extraction**: look up label IDs for a structure
   (``get_label_id("aseg", "Left-Hippocampus") → 17``).
2. **Geometric connectome**: map Schaefer parcels to Yeo networks
   for block-level aggregation.

Label tables for subcortical atlases (aseg, thalamic nuclei,
hippocampal subfields, amygdala nuclei) are embedded.  Cortical
atlases (Schaefer, DKT, Destrieux) load from FreeSurfer annotation
files when available.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

from spectralbrain.runtime import get_logger

logger = get_logger(__name__)


# ======================================================================
# §1  FREESURFER ASEG
# ======================================================================

ASEG_LABELS: dict[int, str] = {
    2: "Left-Cerebral-White-Matter",
    3: "Left-Cerebral-Cortex",
    4: "Left-Lateral-Ventricle",
    5: "Left-Inf-Lat-Vent",
    7: "Left-Cerebellum-White-Matter",
    8: "Left-Cerebellum-Cortex",
    10: "Left-Thalamus",
    11: "Left-Caudate",
    12: "Left-Putamen",
    13: "Left-Pallidum",
    14: "3rd-Ventricle",
    15: "4th-Ventricle",
    16: "Brain-Stem",
    17: "Left-Hippocampus",
    18: "Left-Amygdala",
    24: "CSF",
    26: "Left-Accumbens-area",
    28: "Left-VentralDC",
    30: "Left-vessel",
    31: "Left-choroid-plexus",
    41: "Right-Cerebral-White-Matter",
    42: "Right-Cerebral-Cortex",
    43: "Right-Lateral-Ventricle",
    44: "Right-Inf-Lat-Vent",
    46: "Right-Cerebellum-White-Matter",
    47: "Right-Cerebellum-Cortex",
    49: "Right-Thalamus",
    50: "Right-Caudate",
    51: "Right-Putamen",
    52: "Right-Pallidum",
    53: "Right-Hippocampus",
    54: "Right-Amygdala",
    58: "Right-Accumbens-area",
    60: "Right-VentralDC",
    62: "Right-vessel",
    63: "Right-choroid-plexus",
    77: "WM-hypointensities",
    85: "Optic-Chiasm",
    251: "CC_Posterior",
    252: "CC_Mid_Posterior",
    253: "CC_Central",
    254: "CC_Mid_Anterior",
    255: "CC_Anterior",
}


# ======================================================================
# §2  HIPPOCAMPAL SUBFIELDS (FreeSurfer v7.x, T1-based)
# ======================================================================

HIPPOCAMPAL_SUBFIELDS: dict[int, str] = {
    203: "parasubiculum",
    204: "presubiculum-head",
    205: "presubiculum-body",
    206: "subiculum-head",
    207: "subiculum-body",
    208: "CA1-head",
    209: "CA1-body",
    210: "CA2/3-head",
    211: "CA2/3-body",
    212: "CA4-head",
    213: "CA4-body",
    214: "GC-ML-DG-head",
    215: "GC-ML-DG-body",
    226: "molecular_layer_HP-head",
    227: "molecular_layer_HP-body",
    228: "hippocampal-fissure",
    229: "HATA",
    230: "fimbria",
    231: "hippocampal_tail",
    232: "whole_hippocampal_head",
    233: "whole_hippocampal_body",
}

# FreeSurfer writes one hippocampal-subfield volume per hemisphere
# (``lh.hippoAmygLabels*.mgz`` / ``rh.hippoAmygLabels*.mgz``) and both use
# the *same* label IDs — there is no "+1000" right-hemisphere code.  The
# right-hemisphere table is therefore identical to the left one.
HIPPOCAMPAL_SUBFIELDS_RIGHT: dict[int, str] = dict(HIPPOCAMPAL_SUBFIELDS)

# Atlases whose label IDs carry no hemisphere (one file per hemisphere).
_HEMISPHERE_AGNOSTIC: frozenset[str] = frozenset({"hippocampal_subfields"})


# ======================================================================
# §3  THALAMIC NUCLEI (FreeSurfer v7.x)
# ======================================================================

THALAMIC_NUCLEI: dict[int, str] = {
    8103: "Left-AV",
    8104: "Left-CeM",
    8105: "Left-CL",
    8106: "Left-CM",
    8108: "Left-LD",
    8109: "Left-LGN",
    8110: "Left-LP",
    8111: "Left-L-Sg",
    8112: "Left-MDl",
    8113: "Left-MDm",
    8115: "Left-MGN",
    8116: "Left-MV(Re)",
    8117: "Left-Pc",
    8118: "Left-Pf",
    8119: "Left-Pt",
    8120: "Left-PuA",
    8121: "Left-PuI",
    8122: "Left-PuL",
    8123: "Left-PuM",
    8126: "Left-VA",
    8127: "Left-VAmc",
    8128: "Left-VLa",
    8129: "Left-VLp",
    8130: "Left-VM",
    8131: "Left-VPL",
    8133: "Left-Whole_thalamus",
    # Right = Left + 100
}

THALAMIC_NUCLEI_RIGHT: dict[int, str] = {
    k + 100: v.replace("Left", "Right") for k, v in THALAMIC_NUCLEI.items()
}


# ======================================================================
# §4  AMYGDALA NUCLEI (FreeSurfer v7.x)
# ======================================================================

AMYGDALA_NUCLEI: dict[int, str] = {
    7001: "Left-Lateral-nucleus",
    7002: "Left-Basal-nucleus",
    7003: "Left-Accessory-Basal-nucleus",
    7004: "Left-Anterior-amygdaloid-area",
    7005: "Left-Central-nucleus",
    7006: "Left-Medial-nucleus",
    7007: "Left-Cortical-nucleus",
    7008: "Left-Corticoamygdaloid-transition",
    7009: "Left-Paralaminar-nucleus",
    7010: "Left-Whole-amygdala",
}

AMYGDALA_NUCLEI_RIGHT: dict[int, str] = {
    k + 1000: v.replace("Left", "Right") for k, v in AMYGDALA_NUCLEI.items()
}


# ======================================================================
# §5  YEO NETWORK ASSIGNMENTS
# ======================================================================

YEO_7_NETWORKS: dict[int, str] = {
    1: "Visual",
    2: "Somatomotor",
    3: "DorsalAttention",
    4: "VentralAttention",
    5: "Limbic",
    6: "Frontoparietal",
    7: "Default",
}

YEO_17_NETWORKS: dict[int, str] = {
    1: "VisCent",
    2: "VisPeri",
    3: "SomMotA",
    4: "SomMotB",
    5: "DorsAttnA",
    6: "DorsAttnB",
    7: "SalVentAttnA",
    8: "SalVentAttnB",
    9: "LimbicA",
    10: "LimbicB",
    11: "ContA",
    12: "ContB",
    13: "ContC",
    14: "DefaultA",
    15: "DefaultB",
    16: "DefaultC",
    17: "TempPar",
}


# ======================================================================
# §6  UNIFIED LOOKUP
# ======================================================================

_REGISTRIES: dict[str, dict[int, str]] = {
    "aseg": ASEG_LABELS,
    "hippocampal_subfields": dict(HIPPOCAMPAL_SUBFIELDS),
    "thalamic_nuclei": {
        **THALAMIC_NUCLEI,
        **THALAMIC_NUCLEI_RIGHT,
    },
    "amygdala_nuclei": {
        **AMYGDALA_NUCLEI,
        **AMYGDALA_NUCLEI_RIGHT,
    },
}


def get_label_name(atlas: str, label_id: int) -> str:
    """Look up the region name for a label ID.

    Parameters
    ----------
    atlas : str
        Atlas name (e.g. ``"aseg"``, ``"thalamic_nuclei"``).
    label_id : int

    Returns
    -------
    str
        Region name, or ``"Unknown-{label_id}"``.
    """
    registry = _REGISTRIES.get(atlas, {})
    return registry.get(label_id, f"Unknown-{label_id}")


def get_label_id(atlas: str, name: str) -> int | None:
    """Reverse lookup: region name → label ID.

    An exact (case-insensitive) name match always wins.  Otherwise a
    case-insensitive substring match is used; if several regions match
    (e.g. ``"Hippocampus"`` → Left *and* Right) the first in table order is
    returned and a :class:`UserWarning` lists the alternatives — pass the
    full name (``"Left-Hippocampus"``) to disambiguate.

    Parameters
    ----------
    atlas : str
    name : str
        Region name (exact, or case-insensitive substring).

    Returns
    -------
    int or None
    """
    registry = _REGISTRIES.get(atlas, {})
    name_lower = name.lower()
    for lid, lname in registry.items():
        if lname.lower() == name_lower:
            return lid
    matches = [(lid, lname) for lid, lname in registry.items() if name_lower in lname.lower()]
    if not matches:
        return None
    if len(matches) > 1:
        alts = ", ".join(f"{n} ({i})" for i, n in matches)
        msg = (
            f"get_label_id({atlas!r}, {name!r}) is ambiguous; returning "
            f"{matches[0][1]} ({matches[0][0]}). Candidates: {alts}"
        )
        logger.warning(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
    return matches[0][0]


def list_labels(atlas: str) -> dict[int, str]:
    """Return all label ID → name mappings for an atlas.

    Parameters
    ----------
    atlas : str

    Returns
    -------
    dict
    """
    return dict(_REGISTRIES.get(atlas, {}))


def get_structure_ids(
    atlas: str,
    hemisphere: Literal["left", "right", "both"] = "both",
) -> list[int]:
    """Get all label IDs for a hemisphere.

    Parameters
    ----------
    atlas : str
    hemisphere : str

    Returns
    -------
    list of int
    """
    registry = _REGISTRIES.get(atlas, {})
    if hemisphere == "both" or atlas in _HEMISPHERE_AGNOSTIC:
        # Hemisphere-agnostic atlases use the same IDs in the lh./rh. files.
        return sorted(registry.keys())

    ids = []
    for lid, name in registry.items():
        name_l = name.lower()
        if hemisphere == "left" and ("left" in name_l or "lh" in name_l):
            ids.append(lid)
        elif hemisphere == "right" and ("right" in name_l or "rh" in name_l):
            ids.append(lid)
    return sorted(ids)


_SCHAEFER_7_TOKENS: dict[str, str] = {
    "Vis": "Visual",
    "SomMot": "Somatomotor",
    "DorsAttn": "DorsalAttention",
    "SalVentAttn": "VentralAttention",
    "Limbic": "Limbic",
    "Cont": "Frontoparietal",
    "Default": "Default",
}

# Schaefer 17-network label tokens → names used in :data:`YEO_17_NETWORKS`.
_SCHAEFER_17_TOKENS: dict[str, str] = {
    "VisCent": "VisCent",
    "VisPeri": "VisPeri",
    "SomMotA": "SomMotA",
    "SomMotB": "SomMotB",
    "DorsAttnA": "DorsAttnA",
    "DorsAttnB": "DorsAttnB",
    "SalVentAttnA": "SalVentAttnA",
    "SalVentAttnB": "SalVentAttnB",
    "LimbicA": "LimbicA",
    "LimbicB": "LimbicB",
    "ContA": "ContA",
    "ContB": "ContB",
    "ContC": "ContC",
    "DefaultA": "DefaultA",
    "DefaultB": "DefaultB",
    "DefaultC": "DefaultC",
    "TempPar": "TempPar",
}


def schaefer_name_to_yeo(name: str, n_networks: int | None = None) -> str:
    """Parse the Yeo network from a Schaefer parcel name.

    Parameters
    ----------
    name : str
        A Schaefer label such as ``"7Networks_LH_Vis_1"`` or
        ``"17Networks_RH_DefaultA_PFCm_2"``.
    n_networks : int, optional
        7 or 17.  Inferred from the ``"<n>Networks_"`` prefix if omitted.

    Returns
    -------
    str
        Network name (values of :data:`YEO_7_NETWORKS` / :data:`YEO_17_NETWORKS`).

    Raises
    ------
    ValueError
        If the name does not follow the Schaefer convention.
    """
    parts = str(name).split("_")
    if len(parts) < 3 or not parts[0].endswith("Networks"):
        raise ValueError(f"Not a Schaefer parcel name: {name!r}")
    if n_networks is None:
        try:
            n_networks = int(parts[0][: -len("Networks")])
        except ValueError as exc:
            raise ValueError(f"Not a Schaefer parcel name: {name!r}") from exc
    token = parts[2]
    table = _SCHAEFER_7_TOKENS if n_networks == 7 else _SCHAEFER_17_TOKENS
    if token not in table:
        raise ValueError(f"Unknown {n_networks}-network token {token!r} in {name!r}")
    return table[token]


def schaefer_to_yeo(
    parcel_id: int,
    n_parcels: int = 200,
    n_networks: int = 7,
    *,
    parcel_names: Sequence[str] | None = None,
) -> str:
    """Map a Schaefer parcel ID to its Yeo network name.

    Schaefer parcels encode the network in their naming convention:
    ``7Networks_LH_Vis_1`` → "Visual".

    Parameters
    ----------
    parcel_id : int
        1-indexed Schaefer parcel ID.
    n_parcels : int
        Total parcels (100, 200, 400, etc.).
    n_networks : int
        7 or 17.
    parcel_names : sequence of str, optional
        The atlas label names in parcel order (index ``parcel_id - 1``),
        e.g. from the Schaefer LUT/annotation or
        ``nilearn.datasets.fetch_atlas_schaefer_2018()["labels"]``.  When
        given, the mapping is **exact**.

    Returns
    -------
    str
        Network name.

    Notes
    -----
    Without *parcel_names* the mapping is a heuristic that assumes an
    equal number of parcels per network, which is **not** true for the
    Schaefer atlases (e.g. Schaefer-100/7: LH Vis = parcels 1–9, but the
    heuristic assigns 8–9 to Somatomotor).  A :class:`UserWarning` is
    emitted in that case.
    """
    if parcel_names is not None:
        if not 1 <= parcel_id <= len(parcel_names):
            raise ValueError(
                f"parcel_id {parcel_id} out of range for {len(parcel_names)} parcel names"
            )
        name = parcel_names[parcel_id - 1]
        if isinstance(name, bytes):
            name = name.decode()
        return schaefer_name_to_yeo(name, n_networks)

    warnings.warn(
        "schaefer_to_yeo() without parcel_names uses an equal-size-network "
        "heuristic that mislabels many Schaefer parcels; pass the atlas label "
        "names via parcel_names= for an exact mapping.",
        UserWarning,
        stacklevel=2,
    )
    networks = YEO_7_NETWORKS if n_networks == 7 else YEO_17_NETWORKS
    parcels_per_hemi = n_parcels // 2
    parcels_per_net = max(1, parcels_per_hemi // n_networks)

    # Determine which network this parcel belongs to.
    hemi_id = (parcel_id - 1) % parcels_per_hemi
    net_idx = min(hemi_id // parcels_per_net, n_networks - 1) + 1
    return networks.get(net_idx, f"Network-{net_idx}")


__all__ = [
    "AMYGDALA_NUCLEI",
    "ASEG_LABELS",
    "HIPPOCAMPAL_SUBFIELDS",
    "THALAMIC_NUCLEI",
    "YEO_7_NETWORKS",
    "YEO_17_NETWORKS",
    "get_label_id",
    "get_label_name",
    "get_structure_ids",
    "list_labels",
    "schaefer_name_to_yeo",
    "schaefer_to_yeo",
]
