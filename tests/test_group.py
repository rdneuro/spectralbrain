"""Tests for group loading (:mod:`spectralbrain.io.group`).

Exercises the three discovery modes (BIDS glob, FreeSurfer ``SUBJECTS_DIR``,
explicit list), both load modes (vertex-corresponded ``maps`` and the full
``pipeline``), parallel/sequential invariance, and the glue into the
vertex-wise group statistics.
"""

import numpy as np
import pytest

import spectralbrain as sb


def _icosphere():
    """Small unit icosphere (42 vertices, 80 faces)."""
    phi = (1.0 + 5.0**0.5) / 2.0
    v = np.array(
        [
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
        ],
        dtype=np.float64,
    )  # fmt: skip
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    f = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )  # fmt: skip
    return v, f


def _make_bids_overlays(root, n=6):
    """Write per-vertex GIFTI overlays in a BIDS derivatives layout."""
    nib = pytest.importorskip("nibabel")
    v, _ = _icosphere()
    n_vert = len(v)
    rng = np.random.default_rng(0)
    for i in range(n):
        sub = f"{i + 1:02d}"
        grp = "patient" if i % 2 else "control"
        d = root / f"sub-{sub}" / "anat"
        d.mkdir(parents=True)
        scal = rng.normal(2.0, 0.3, n_vert).astype(np.float32)
        if grp == "patient":
            scal[:10] += 1.0  # localised group effect
        da = nib.gifti.GiftiDataArray(scal, intent="NIFTI_INTENT_NONE")
        img = nib.gifti.GiftiImage(darrays=[da])
        nib.save(img, str(d / f"sub-{sub}_group-{grp}_hemi-L_thickness.shape.gii"))
    return n_vert


def test_discover_bids_and_load_maps(tmp_path):
    """BIDS discovery → maps load → stacked array with parsed covariates."""
    pytest.importorskip("nibabel")
    n_vert = _make_bids_overlays(tmp_path, n=6)

    files = sb.discover_bids(tmp_path, "sub-{sub}/anat/sub-{sub}_*_hemi-L_thickness.shape.gii")
    assert len(files) == 6

    group = sb.load_group(files, mode="maps", n_jobs=1)
    assert group.is_stacked
    assert group.data.shape == (6, n_vert)
    assert group.n_subjects == 6
    assert list(group.covariate("group")) == [
        "control", "patient", "control", "patient", "control", "patient",
    ]  # fmt: skip


def test_load_group_njobs_invariant(tmp_path):
    """Loading is identical sequentially and in parallel."""
    pytest.importorskip("nibabel")
    _make_bids_overlays(tmp_path, n=6)
    files = sb.discover_bids(tmp_path, "sub-{sub}/anat/sub-{sub}_*_hemi-L_thickness.shape.gii")
    seq = sb.load_group(files, mode="maps", n_jobs=1)
    par = sb.load_group(files, mode="maps", n_jobs=2)
    assert np.allclose(seq.data, par.data)
    assert seq.subject_ids == par.subject_ids


def test_group_comparison_detects_effect(tmp_path):
    """The analysis glue runs a vertex-wise test and finds the seeded effect."""
    pytest.importorskip("nibabel")
    _make_bids_overlays(tmp_path, n=8)
    files = sb.discover_bids(tmp_path, "sub-{sub}/anat/sub-{sub}_*_hemi-L_thickness.shape.gii")
    group = sb.load_group(files, mode="maps")
    res = sb.group_comparison(group, group.covariate("group"), test="ttest", correction="none")
    # The effect was seeded into the first 10 vertices.
    assert res.significant[:10].sum() >= 5


def test_explicit_paths_parse_subject_ids(tmp_path):
    """A plain list of paths yields subject IDs parsed from filenames."""
    pytest.importorskip("nibabel")
    _make_bids_overlays(tmp_path, n=4)
    files = sb.discover_bids(tmp_path, "sub-{sub}/anat/sub-{sub}_*_hemi-L_thickness.shape.gii")
    group = sb.load_group(list(files.values()), mode="maps")
    assert sorted(group.subject_ids) == ["sub-01", "sub-02", "sub-03", "sub-04"]


def test_freesurfer_discovery_and_pipeline(tmp_path):
    """FreeSurfer surfaces → pipeline mode → stacked HKS descriptor fields."""
    nib = pytest.importorskip("nibabel")
    v, f = _icosphere()
    rng = np.random.default_rng(1)
    for i in range(3):
        sd = tmp_path / f"sub-{i + 1:02d}" / "surf"
        sd.mkdir(parents=True)
        vi = v + rng.normal(0, 0.01, v.shape)
        nib.freesurfer.write_geometry(
            str(sd / "lh.white"), vi.astype(np.float64), f.astype(np.int32)
        )

    files = sb.discover_freesurfer(tmp_path, hemi="lh", surface="white")
    assert len(files) == 3

    group = sb.load_group(
        files,
        mode="pipeline",
        descriptor="hks",
        k=8,
        descriptor_kwargs={"t_values": [1.0, 10.0, 100.0]},
    )
    assert group.is_stacked
    assert group.data.shape == (3, len(v), 3)
    assert np.isfinite(group.data).all()


def test_group_data_split():
    """GroupData.split partitions a stacked array by a 2-level label."""
    from spectralbrain.io.group import GroupData

    data = np.arange(40).reshape(4, 10).astype(float)
    g = GroupData(
        data=data,
        subject_ids=[f"sub-0{i}" for i in range(4)],
        entities=[{}, {}, {}, {}],
        paths=[],
    )
    a, b = g.split(np.array(["x", "y", "x", "y"]))
    assert a.shape == (2, 10) and b.shape == (2, 10)
    assert np.array_equal(a, data[[0, 2]])


def test_freesurfer_discovery_requires_one_of_surface_or_measure(tmp_path):
    """Specifying neither or both surface and measure is an error."""
    with pytest.raises(ValueError, match="exactly one"):
        sb.discover_freesurfer(tmp_path, surface="white", measure="thickness")
    with pytest.raises(ValueError, match="exactly one"):
        sb.discover_freesurfer(tmp_path)


# ----------------------------------------------------------------------
# Template resampling (FreeSurfer)
# ----------------------------------------------------------------------
def _write_fs_sphere(subjects_dir, name, v, f, *, thickness=None):
    """Write a FreeSurfer sphere.reg (and optional thickness) for a subject."""
    import nibabel as nib

    sd = subjects_dir / name / "surf"
    sd.mkdir(parents=True)
    nib.freesurfer.write_geometry(
        str(sd / "lh.sphere.reg"), v.astype(np.float64), f.astype(np.int32)
    )
    if thickness is not None:
        nib.freesurfer.write_morph_data(str(sd / "lh.thickness"), thickness.astype(np.float32))


def test_resample_to_template_identity(tmp_path):
    """Resampling a subject onto an identical template sphere is the identity."""
    pytest.importorskip("nibabel")
    v, f = _icosphere()
    _write_fs_sphere(tmp_path, "fsaverage", v, f)
    rng = np.random.default_rng(0)
    thick = rng.normal(2.5, 0.3, len(v))
    _write_fs_sphere(tmp_path, "sub-01", v, f, thickness=thick)

    out = sb.resample_to_template(thick, tmp_path, "sub-01", "lh", template="fsaverage")
    assert out.shape == (len(v),)
    assert np.allclose(out, thick)


def test_load_group_freesurfer_resamples_to_common_size(tmp_path):
    """Subjects with different vertex counts stack after resampling to template."""
    pytest.importorskip("nibabel")
    v_t, f_t = _icosphere()  # 12-vertex template
    _write_fs_sphere(tmp_path, "fsaverage", v_t, f_t)

    rng = np.random.default_rng(1)
    _write_fs_sphere(tmp_path, "sub-01", v_t, f_t, thickness=rng.normal(2.5, 0.3, len(v_t)))
    # A second subject with a *different* vertex count.
    v2 = np.vstack([v_t, -v_t])  # 24 vertices
    f2 = np.vstack([f_t, f_t + len(v_t)])
    _write_fs_sphere(tmp_path, "sub-02", v2, f2, thickness=rng.normal(2.5, 0.3, len(v2)))

    group = sb.load_group_freesurfer(
        tmp_path, measure="thickness", template="fsaverage",
        subjects=["sub-01", "sub-02"],
    )  # fmt: skip
    assert group.is_stacked
    assert group.data.shape == (2, len(v_t))
