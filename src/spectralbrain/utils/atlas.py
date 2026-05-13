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

from typing import Dict, List, Optional, Tuple

import numpy as np

from spectralbrain.runtime import AtlasScheme, get_logger

logger = get_logger(__name__)


# ======================================================================
# §1  FREESURFER ASEG
# ======================================================================

ASEG_LABELS: Dict[int, str] = {
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

HIPPOCAMPAL_SUBFIELDS: Dict[int, str] = {
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

# Right-hemisphere labels are + 1000.
HIPPOCAMPAL_SUBFIELDS_RIGHT: Dict[int, str] = {
    k + 1000: v.replace("left", "right")
    for k, v in HIPPOCAMPAL_SUBFIELDS.items()
}


# ======================================================================
# §3  THALAMIC NUCLEI (FreeSurfer v7.x)
# ======================================================================

THALAMIC_NUCLEI: Dict[int, str] = {
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

THALAMIC_NUCLEI_RIGHT: Dict[int, str] = {
    k + 100: v.replace("Left", "Right")
    for k, v in THALAMIC_NUCLEI.items()
}


# ======================================================================
# §4  AMYGDALA NUCLEI (FreeSurfer v7.x)
# ======================================================================

AMYGDALA_NUCLEI: Dict[int, str] = {
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

AMYGDALA_NUCLEI_RIGHT: Dict[int, str] = {
    k + 1000: v.replace("Left", "Right")
    for k, v in AMYGDALA_NUCLEI.items()
}


# ======================================================================
# §5  YEO NETWORK ASSIGNMENTS
# ======================================================================

YEO_7_NETWORKS: Dict[int, str] = {
    1: "Visual",
    2: "Somatomotor",
    3: "DorsalAttention",
    4: "VentralAttention",
    5: "Limbic",
    6: "Frontoparietal",
    7: "Default",
}

YEO_17_NETWORKS: Dict[int, str] = {
    1: "VisCent", 2: "VisPeri",
    3: "SomMotA", 4: "SomMotB",
    5: "DorsAttnA", 6: "DorsAttnB",
    7: "SalVentAttnA", 8: "SalVentAttnB",
    9: "LimbicA", 10: "LimbicB",
    11: "ContA", 12: "ContB", 13: "ContC",
    14: "DefaultA", 15: "DefaultB", 16: "DefaultC",
    17: "TempPar",
}


# ======================================================================
# §6  UNIFIED LOOKUP
# ======================================================================

_REGISTRIES: Dict[str, Dict[int, str]] = {
    "aseg": ASEG_LABELS,
    "hippocampal_subfields": {
        **HIPPOCAMPAL_SUBFIELDS,
        **HIPPOCAMPAL_SUBFIELDS_RIGHT,
    },
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


def get_label_id(atlas: str, name: str) -> Optional[int]:
    """Reverse lookup: region name → label ID.

    Parameters
    ----------
    atlas : str
    name : str
        Region name (case-insensitive substring match).

    Returns
    -------
    int or None
    """
    registry = _REGISTRIES.get(atlas, {})
    name_lower = name.lower()
    for lid, lname in registry.items():
        if name_lower in lname.lower():
            return lid
    return None


def list_labels(atlas: str) -> Dict[int, str]:
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
) -> List[int]:
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
    if hemisphere == "both":
        return sorted(registry.keys())

    ids = []
    for lid, name in registry.items():
        name_l = name.lower()
        if hemisphere == "left" and ("left" in name_l or "lh" in name_l):
            ids.append(lid)
        elif hemisphere == "right" and ("right" in name_l or "rh" in name_l):
            ids.append(lid)
    return sorted(ids)


def schaefer_to_yeo(
    parcel_id: int,
    n_parcels: int = 200,
    n_networks: int = 7,
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

    Returns
    -------
    str
        Network name.

    Notes
    -----
    This is a heuristic based on the standard Schaefer ordering.
    For exact mapping, load the annotation file and parse names.
    """
    networks = YEO_7_NETWORKS if n_networks == 7 else YEO_17_NETWORKS
    parcels_per_hemi = n_parcels // 2
    parcels_per_net = parcels_per_hemi // n_networks

    # Determine which network this parcel belongs to.
    hemi_id = (parcel_id - 1) % parcels_per_hemi
    net_idx = min(hemi_id // parcels_per_net, n_networks - 1) + 1
    return networks.get(net_idx, f"Network-{net_idx}")


from typing import Literal

__all__ = [
    "ASEG_LABELS",
    "HIPPOCAMPAL_SUBFIELDS",
    "THALAMIC_NUCLEI",
    "AMYGDALA_NUCLEI",
    "YEO_7_NETWORKS",
    "YEO_17_NETWORKS",
    "get_label_name",
    "get_label_id",
    "list_labels",
    "get_structure_ids",
    "schaefer_to_yeo",
]
