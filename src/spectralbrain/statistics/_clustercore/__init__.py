"""Vendored, self-contained clustering + cluster-statistics cores.

These modules are ported verbatim (only relative imports adjusted) from the
``brainmosaic`` library so that SpectralBrain gains its distance-dependent
Chinese-Restaurant-Process clustering (ddCRP with NIW collapsed marginal
likelihood), consensus clustering, hyperparameter autotuning, partition-overlap
statistics (ARI/AMI/NMI/VI, spARI), cluster-vs-atlas concordance with
size-matched nulls, and eigenstrapping/BrainSMASH spatial nulls.

User-facing, SpectralBrain-harmonised entry points live in
``spectralbrain.statistics.ddcrp`` and ``spectralbrain.statistics.atlas_stats``;
import those, not this private subpackage.
"""

from ._meshgeom import (  # noqa: F401
    adjacency_list_from_faces, adjacency_list_from_sparse, edge_distances,
)
from .ddcrp import DDCRP, DDCRPResult, NIWPrior, cluster_ddcrp  # noqa: F401
from .ddcrp_functional import cluster_ddcrp_functional  # noqa: F401
from .consensus import (  # noqa: F401
    co_association_matrix, consensus_partition, stability_per_vertex,
)
from .partition import (  # noqa: F401
    adjusted_rand_index, normalized_mutual_info, spatial_rand_index,
    variation_of_information,
)
from .atlas import (  # noqa: F401
    aggregate_across_subjects, cluster_atlas_concordance, compare_partitions,
    homogeneity_vs_null, parcel_overlap, random_parcellation,
    spectral_homogeneity,
)
from .nulls import (  # noqa: F401
    brainsmash_surrogates, eigenstrapping_surrogates, paired_label_permutation,
)
from .optimize import autotune_ddcrp, autotune_ddcrp_functional  # noqa: F401
from .cluster_stats import intra_inter_homogeneity, spatial_silhouette  # noqa: F401
