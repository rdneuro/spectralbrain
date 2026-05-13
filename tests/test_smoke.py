"""Smoke tests — verify that all subpackages import without errors.

Run with: pytest tests/test_smoke.py -v
"""

import pytest


def test_runtime_imports():
    from spectralbrain.runtime import get_logger, GeometryFormat, BackendName
    logger = get_logger("test")
    assert logger is not None
    assert hasattr(GeometryFormat, "FREESURFER")


def test_core_imports():
    from spectralbrain.core import SpectralDecomposition, BrainMesh, BrainPointCloud
    assert SpectralDecomposition is not None


def test_io_imports():
    from spectralbrain.io import detect_format, load, save_hdf5, save_mesh


def test_spectral_imports():
    from spectralbrain.spectral import (
        compute_shapedna, compute_hks, compute_wks, compute_gps,
        wesd, shapedna_distance, sgw_transform,
    )


def test_statistics_imports():
    from spectralbrain.statistics import (
        SpectralQCReport, spectral_qc,
        VertexWiseResult, vertexwise_ttest, tfce,
        BayesianModel, HorseshoeRegression,
        NormativeModel, z_score_map,
        bootstrap_ci, null_spin_permutation,
        SyntheticMesh, SyntheticPointCloud,
    )


def test_backends_imports():
    from spectralbrain.backends import NumpyBackend, ram_status


def test_utils_imports():
    from spectralbrain.utils import (
        ASEG_LABELS, HIPPOCAMPAL_SUBFIELDS, THALAMIC_NUCLEI,
        get_label_name, seed_everything, timer,
    )


def test_viz_graphics_imports():
    from spectralbrain.viz.graphics import (
        PALETTE, set_style, figure, savefig, distplot,
    )


def test_viz_brainplots_imports():
    from spectralbrain.viz.brainplots import (
        VIEWS_CORTEX, BrainPlotSpec, plot_brain,
    )


def test_viz_geometry_imports():
    from spectralbrain.viz.geometry import (
        plot_point_cloud, plot_mesh, plot_curvature,
        plot_clusters, plot_wireframe,
    )


def test_viz_hipp_imports():
    from spectralbrain.viz.hipp import plot_hippocampus


def test_viz_bayes_imports():
    from spectralbrain.viz.bayes import (
        plot_posterior, plot_forest, plot_gp_trajectory,
    )


def test_toplevel_import():
    """The top-level import should expose the core API."""
    import spectralbrain as sb
    assert hasattr(sb, "__version__")
    assert hasattr(sb, "BrainMesh")
    assert hasattr(sb, "compute_hks")
    assert hasattr(sb, "load_freesurfer_surface")
    assert hasattr(sb, "seed_everything")
