"""Regression tests for the spectralbrain.io bug-fix pass.

Each test pins one previously confirmed bug (orientation, label mapping,
silent drops, broken fallbacks, …). External tools (FreeSurfer, HD-BET,
SynthSeg, FastSurfer) are never invoked: subprocess entry points are mocked.
"""

from __future__ import annotations

import importlib
import shlex
from pathlib import Path

import numpy as np
import pytest

# ``spectralbrain.io`` re-exports a *function* named ``parcellate`` that shadows
# the submodule attribute, so import the modules explicitly.
export = importlib.import_module("spectralbrain.io.export")
gpu_preprocess = importlib.import_module("spectralbrain.io.gpu_preprocess")
group = importlib.import_module("spectralbrain.io.group")
loaders = importlib.import_module("spectralbrain.io.loaders")
meshing = importlib.import_module("spectralbrain.io.meshing")
parcellate = importlib.import_module("spectralbrain.io.parcellate")
tractseg = importlib.import_module("spectralbrain.io.tractseg")

LIA = np.array(
    [[-1.0, 0, 0, 128], [0, 0, 1, -128], [0, -1, 0, 128], [0, 0, 0, 1]]
)  # FreeSurfer conformed orientation, det < 0


def _signed_volume(v: np.ndarray, f: np.ndarray) -> float:
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def _ball(n: int = 22, r: float = 6.0) -> np.ndarray:
    x, y, z = np.mgrid[:n, :n, :n]
    c = (n - 1) / 2
    return ((x - c) ** 2 + (y - c) ** 2 + (z - c) ** 2 < r**2).astype(np.uint8)


# ----------------------------------------------------------------------
# meshing (#7, #18)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("affine", [np.eye(4), LIA, np.diag([0.5, 0.5, 2.0, 1.0])])
@pytest.mark.parametrize("mode", ["sdf", "gaussian"])
def test_marching_cubes_field_outward_for_any_affine(affine, mode):
    field, level, pad = meshing._mask_to_field(_ball() > 0, mode=mode)
    v, f = meshing._marching_cubes_field(field, level, affine, pad, inside_low=(mode == "sdf"))
    assert _signed_volume(v, f) > 0


@pytest.mark.parametrize("affine", [np.eye(4), LIA])
@pytest.mark.parametrize("closed", [True, False])
def test_volume_to_mesh_improved_outward(affine, closed):
    pytest.importorskip("trimesh")
    v, f = meshing.volume_to_mesh(
        _ball(), affine, closed=closed, remesh=False, taubin_iterations=0
    )
    assert _signed_volume(v, f) > 0


@pytest.mark.parametrize("affine", [np.eye(4), LIA])
def test_volume_to_mesh_raw_outward(affine):
    v, f = meshing.volume_to_mesh(_ball(), affine, raw=True, label=1)
    assert _signed_volume(v, f) > 0


def test_mask_to_field_voxel_size_aware():
    m = np.zeros((9, 9, 9), bool)
    m[4, 4, 4] = True
    iso, _, _ = meshing._mask_to_field(m, presmooth_vox=0, sigma_vox=0, pad=0)
    same, _, _ = meshing._mask_to_field(
        m, presmooth_vox=0, sigma_vox=0, pad=0, voxel_size=(1, 1, 1)
    )
    np.testing.assert_array_equal(iso, same)  # isotropic: unchanged behaviour
    aniso, _, _ = meshing._mask_to_field(
        m, presmooth_vox=0, sigma_vox=0, pad=0, voxel_size=(1, 1, 2)
    )
    assert aniso[4, 4, 5] == pytest.approx(2.0)  # one coarse voxel = 2 fine units
    assert aniso[4, 5, 4] == pytest.approx(1.0)


# ----------------------------------------------------------------------
# loaders (#9, #16, #23, #24)
# ----------------------------------------------------------------------
def test_gifti_label_names_indexed_by_key(tmp_path):
    nib = pytest.importorskip("nibabel")
    from nibabel.gifti import GiftiDataArray, GiftiImage, GiftiLabel, GiftiLabelTable

    lt = GiftiLabelTable()
    for k, n in [(0, "???"), (181, "L_V1"), (182, "L_MST")]:
        lab = GiftiLabel(key=k)
        lab.label = n
        lt.labels.append(lab)
    img = GiftiImage(
        labeltable=lt,
        darrays=[
            GiftiDataArray(
                np.array([181, 182, 0], np.int32),
                intent="NIFTI_INTENT_LABEL",
                datatype="NIFTI_TYPE_INT32",
            )
        ],
    )
    path = tmp_path / "x.label.gii"
    nib.save(img, str(path))
    labels, names = loaders.load_gifti_label(path)
    assert names[181] == "L_V1" and names[182] == "L_MST" and names[0] == "???"
    new, new_names = loaders.remap_parcellation(labels, names, {"L_V1": "vis"})
    assert new_names[int(new[0])] == "vis"
    assert new_names[int(new[1])] == "unmapped"


def test_gifti_surface_uses_intents(tmp_path):
    nib = pytest.importorskip("nibabel")
    from nibabel.gifti import GiftiDataArray, GiftiImage

    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    f = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.int32)
    img = GiftiImage(
        darrays=[  # triangles stored FIRST
            GiftiDataArray(f, intent="NIFTI_INTENT_TRIANGLE", datatype="NIFTI_TYPE_INT32"),
            GiftiDataArray(v, intent="NIFTI_INTENT_POINTSET", datatype="NIFTI_TYPE_FLOAT32"),
        ]
    )
    path = tmp_path / "s.surf.gii"
    nib.save(img, str(path))
    vv, ff = loaders.load_gifti_surface(path)
    np.testing.assert_allclose(vv, v)
    np.testing.assert_array_equal(ff, f)


def test_apply_parcellation_ignores_unassigned():
    v = np.random.default_rng(0).normal(size=(6, 3))
    f = np.array([[0, 1, 2], [3, 4, 5]])
    labels = np.array([-1, -1, -1, 2, 2, 2])
    parcels = loaders.apply_parcellation(v, f, labels, ignore_labels=[0])
    assert list(parcels) == [2]
    legacy = loaders.apply_parcellation(v, f, labels, ignore_unassigned=False)
    assert -1 in legacy


def test_labels_to_pointcloud_float_labels():
    vol = np.zeros((4, 4, 4), np.float32)
    vol[1, 1, 1] = 16.9999997
    pts = loaders.labels_to_pointcloud(vol, np.eye(4), 17)
    assert pts.shape == (1, 3)


def test_aggregate_accepts_plain_callable():
    pytest.importorskip("pandas")
    data = np.arange(6, dtype=float).reshape(3, 2)
    df = loaders.aggregate_by_parcellation(data, np.array([1, 1, 2]), stat=lambda x: x.max())
    np.testing.assert_allclose(df.loc[1].to_numpy(), [2.0, 3.0])


def test_npz_loader(tmp_path):
    p = tmp_path / "a.npz"
    np.savez(p, x=np.arange(3))
    out = loaders.load(p)
    np.testing.assert_array_equal(out["x"], np.arange(3))


# ----------------------------------------------------------------------
# export (#10, #11)
# ----------------------------------------------------------------------
def test_save_connectome_header_and_precision(tmp_path):
    pd = pytest.importorskip("pandas")
    m = np.array([[1e-8, 2.0], [3.0, 4.0]])
    p = export.save_connectome(tmp_path / "c.tsv", m)
    back = pd.read_csv(p, sep="\t", index_col=0)
    assert back.shape == (2, 2)
    np.testing.assert_allclose(back.to_numpy(), m, rtol=1e-9)
    with pytest.raises(ValueError):
        export.save_connectome(tmp_path / "d.tsv", m, labels=["a"])


def test_save_mesh_stl_guard_and_vtp_order(tmp_path):
    pytest.importorskip("pyvista")
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]]) + 100.0
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    with pytest.raises(ValueError, match="STL"):
        export.save_mesh(tmp_path / "m.stl", v, f, allow_vertex_reorder=False)
    p = export.save_mesh(tmp_path / "sub" / "m.vtp", v, f)
    vv, ff = loaders.load_mesh(p)
    np.testing.assert_allclose(vv, v)
    np.testing.assert_array_equal(ff, f)


# ----------------------------------------------------------------------
# group (#8, #25)
# ----------------------------------------------------------------------
def _fs_thickness(root: Path, subs: dict[str, np.ndarray]) -> list[Path]:
    nib = pytest.importorskip("nibabel")
    paths = []
    for sid, vals in subs.items():
        d = root / sid / "surf"
        d.mkdir(parents=True)
        p = d / "lh.thickness"
        nib.freesurfer.write_morph_data(str(p), vals.astype(np.float32))
        paths.append(p)
    return paths


def test_load_group_list_ids_from_freesurfer_layout(tmp_path):
    paths = _fs_thickness(tmp_path, {"s1": np.ones(5), "s2": np.zeros(5)})
    g = group.load_group(paths)
    assert g.subject_ids == ["s1", "s2"]
    np.testing.assert_array_equal(g.data[0], np.ones(5))


def test_load_group_duplicate_ids_raise(tmp_path):
    a = tmp_path / "s1" / "surf" / "lh.thickness"
    b = tmp_path / "other" / "s1" / "surf" / "lh.thickness"
    with pytest.raises(ValueError, match="unique subject IDs"):
        group.load_group([a, b], loader=lambda p: np.zeros(2))


def test_load_group_records_failures_and_reraises_bugs(tmp_path):
    files = {"sub-01": tmp_path / "ok", "sub-02": tmp_path / "bad"}

    def loader(p):
        if Path(p).name == "bad":
            raise OSError("corrupt")
        return np.ones(3)

    g = group.load_group(files, loader=loader)
    assert g.subject_ids == ["sub-01"]
    assert g.metadata["failed_subjects"] == ["sub-02"]

    def buggy(p):
        raise TypeError("bug")

    with pytest.raises(TypeError):
        group.load_group(files, loader=buggy)


# ----------------------------------------------------------------------
# tractseg (#25)
# ----------------------------------------------------------------------
def test_tractseg_reraises_programming_errors_and_reports_failed(tmp_path):
    nib = pytest.importorskip("nibabel")
    seg = tmp_path / "ts" / "bundle_segmentations"
    seg.mkdir(parents=True)
    ok = np.zeros((12, 12, 12), np.uint8)
    ok[4:8, 4:8, 4:8] = 1
    nib.save(nib.Nifti1Image(ok, np.eye(4)), str(seg / "CST_left.nii.gz"))
    nib.save(nib.Nifti1Image(np.zeros_like(ok), np.eye(4)), str(seg / "EMPTY.nii.gz"))
    out, failed = tractseg.load_tractseg(tmp_path / "ts", return_failed=True)
    assert set(out) == {"CST_left"} and set(failed) == {"EMPTY"}
    with pytest.raises(TypeError):
        tractseg.load_tractseg(tmp_path / "ts", not_a_kwarg=1)


# ----------------------------------------------------------------------
# parcellate (#5, #6, #14, #15, #17, #25)
# ----------------------------------------------------------------------
def test_mni_atlas_requires_registration(tmp_path):
    spec = parcellate._resolve_atlas("harvard_oxford")
    with pytest.raises(ValueError, match="atlas_reg"):
        parcellate._parcellate_mni_volume(tmp_path, "s", spec, "lh", "white")


def test_brainnetome_fetch_raises_clearly():
    with pytest.raises(ValueError, match="atlas_volume_path"):
        parcellate._fetch_atlas_volume(parcellate._resolve_atlas("brainnetome"))


def _fake_subject(root: Path, c_ras=(10.0, -20.0, 5.0)):
    nib = pytest.importorskip("nibabel")
    sub = root / "s"
    (sub / "surf").mkdir(parents=True)
    (sub / "mri").mkdir()
    # Conformed 32^3 1 mm volume with a c_ras offset.
    aff = LIA.copy()
    aff[:3, 3] = 0
    aff[:3, 3] = -aff[:3, :3] @ np.array([16, 16, 16]) + np.array(c_ras)
    nib.save(nib.MGHImage(np.zeros((32, 32, 32), np.uint8), aff), str(sub / "mri" / "orig.mgz"))
    tkr = np.array([[-5.0, 0, 0], [5.0, 0, 0], [0, 0, 0]])  # surface coords in tkr-RAS
    faces = np.array([[0, 1, 2]], np.int32)
    for s in ("white", "pial"):
        nib.freesurfer.write_geometry(str(sub / "surf" / f"lh.{s}"), tkr, faces)
    return sub, tkr, np.array(c_ras)


def test_python_vol2surf_applies_c_ras(tmp_path, monkeypatch):
    nib = pytest.importorskip("nibabel")
    _sub, tkr, c_ras = _fake_subject(tmp_path)
    # Native-space label volume: 1 mm RAS grid; label = 1 for x<0, 2 for x>0
    # at the *scanner* position of the vertices (tkr + c_ras).
    aff = np.eye(4)
    aff[:3, 3] = c_ras - 20
    data = np.zeros((41, 41, 41), np.int16)
    data[:20] = 1
    data[21:] = 2
    atlas = tmp_path / "atlas.nii.gz"
    nib.save(nib.Nifti1Image(data, aff), str(atlas))
    monkeypatch.setattr(parcellate, "_has_freesurfer_cmd", lambda cmd: False)
    labels = parcellate._vol2surf_project(atlas, tmp_path, "s", "lh", atlas_space="native")
    np.testing.assert_array_equal(labels, [1, 2, 0])
    del tkr


def test_surf2surf_uses_temp_sd_when_fsaverage_elsewhere(tmp_path, monkeypatch):
    fs_home = tmp_path / "fshome" / "subjects" / "fsaverage"
    (fs_home / "label").mkdir(parents=True)
    src = fs_home / "label" / "lh.x.annot"
    src.write_bytes(b"x")
    sd = tmp_path / "sd"
    (sd / "subj").mkdir(parents=True)
    seen = {}

    def fake_run(cmd, **kw):
        sd_arg = Path(cmd[cmd.index("--sd") + 1])
        seen["fsa"] = (sd_arg / "fsaverage").resolve()
        Path(cmd[cmd.index("--tval") + 1]).write_bytes(b"annot")

        class R:
            returncode = 0
            stdout = stderr = ""

        return R()

    monkeypatch.setattr(parcellate.subprocess, "run", fake_run)
    target = sd / "subj" / "label" / "lh.x.annot"
    parcellate._surf2surf_freesurfer(sd, "subj", "lh", src, target)
    assert seen["fsa"] == fs_home.resolve()
    assert target.read_bytes() == b"annot"


def test_fastsurfer_seg_only_raises_clearly(tmp_path):
    (tmp_path / "sub-01" / "mri").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="seg_only"):
        parcellate._locate_fastsurfer_subject(
            tmp_path, Path("sub-01_T1w.nii.gz"), "lh", "white"
        )


def test_parcellate_batch_failures(monkeypatch, tmp_path):
    def fake(**kw):
        if kw["subject_id"] == "bad":
            raise FileNotFoundError("missing")
        if kw["subject_id"] == "bug":
            raise AttributeError("bug")
        return parcellate.ParcellationResult(
            atlas=parcellate._resolve_atlas("dkt"), hemi="lh", surface="white",
            vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), int), labels=np.zeros(0),
            label_names=[], parcels={}, strategy_used="x",
        )

    monkeypatch.setattr(parcellate, "parcellate", fake)
    res, failed = parcellate.parcellate_batch(tmp_path, ["ok", "bad"], return_failed=True)
    assert list(res) == ["ok"] and list(failed) == ["bad"]
    with pytest.raises(AttributeError):
        parcellate.parcellate_batch(tmp_path, ["bug"])


# ----------------------------------------------------------------------
# gpu_preprocess (#1, #2, #3, #4, #12, #13, #20, #21)
# ----------------------------------------------------------------------
def _touch_outputs_run_cmd(calls):
    def fake(cmd, label, **kw):
        calls.append(cmd)
        toks = shlex.split(cmd)
        for flag in ("-o", "--o"):
            if flag in toks:
                Path(toks[toks.index(flag) + 1]).write_bytes(b"x")
        return None

    return fake


def test_synthstrip_accepts_device_and_quotes_paths(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(gpu_preprocess, "_run_cmd", _touch_outputs_run_cmd(calls))
    d = tmp_path / "with space"
    d.mkdir()
    out, _mask = gpu_preprocess.skull_strip(
        d / "in.nii.gz", d / "brain.nii.gz", method="synthstrip", device="cuda:0"
    )
    assert out.exists()
    assert f"'{d / 'in.nii.gz'}'" in calls[0]


def test_skull_strip_auto_falls_back(tmp_path, monkeypatch):
    def hdbet_fail(*a, **k):
        raise RuntimeError("no hd-bet")

    monkeypatch.setattr(gpu_preprocess, "skull_strip_hdbet", hdbet_fail)
    calls: list[str] = []
    monkeypatch.setattr(gpu_preprocess, "_run_cmd", _touch_outputs_run_cmd(calls))
    _b, _m, method = gpu_preprocess._skull_strip_impl(
        tmp_path / "in.nii.gz", tmp_path / "b.nii.gz", method="auto", device="cuda:0",
        save_mask=True,
    )
    assert method == "synthstrip"


def test_segment_fastsurfer_writes_requested_file(tmp_path, monkeypatch):
    nib = pytest.importorskip("nibabel")

    def fake_fs(inp, out, subject_id, *, device="cuda:0"):
        mri = Path(out) / subject_id / "mri"
        mri.mkdir(parents=True)
        nib.save(
            nib.MGHImage(np.full((4, 4, 4), 17, np.int32), np.eye(4)),
            str(mri / "aparc.DKTatlas+aseg.deep.mgz"),
        )
        return Path(out) / subject_id

    monkeypatch.setattr(gpu_preprocess, "segment_fastsurfer", fake_fs)
    out = gpu_preprocess.segment(
        tmp_path / "sub-01_T1w.nii.gz", tmp_path / "seg.nii.gz", method="fastsurfer",
        parc=True, vol_path=None, qc_path=None,
    )
    assert out == tmp_path / "seg.nii.gz"
    assert np.all(np.asarray(nib.load(str(out)).dataobj) == 17)


def test_preprocess_gpu_segments_native_and_respects_skip_existing(tmp_path, monkeypatch):
    inp = tmp_path / "sub-01_T1w.nii.gz"
    inp.write_bytes(b"x")
    seen = {}

    def fake_strip(cur, out, **kw):
        seen["strip_overwrite"] = kw["overwrite"]
        return Path(out), None, "hdbet"

    def fake_register(cur, tmpl, out_dir, stem, **kw):
        return {"affine": None, "affine_xfm": None,
                "warped": Path(out_dir) / f"{stem}_MNI.nii.gz", "deformable_xfm": None}

    def fake_segment(cur, out, **kw):
        seen["seg_input"] = Path(cur)
        seen["seg_overwrite"] = kw["overwrite"]
        return Path(out), "synthseg"

    monkeypatch.setattr(gpu_preprocess, "_skull_strip_impl", fake_strip)
    monkeypatch.setattr(gpu_preprocess, "register", fake_register)
    monkeypatch.setattr(gpu_preprocess, "_segment_impl", fake_segment)
    monkeypatch.setattr(gpu_preprocess, "ensure_template", lambda *a, **k: tmp_path / "t.nii.gz")
    monkeypatch.setattr(gpu_preprocess, "vram_info", dict)
    res = gpu_preprocess.preprocess_gpu(
        inp, tmp_path / "out", steps=["skull_strip", "register", "segment"],
        skip_existing=False,
    )
    assert seen["seg_input"].name == "sub-01_T1w_brain.nii.gz"
    assert seen["strip_overwrite"] is True and seen["seg_overwrite"] is True
    assert res.methods["skull_strip"] == "hdbet"
    assert res.methods["segment"].startswith("synthseg")


def test_overwrite_flag_reruns_step(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(gpu_preprocess, "_run_cmd", _touch_outputs_run_cmd(calls))
    out = tmp_path / "seg.nii.gz"
    out.write_bytes(b"stale")
    gpu_preprocess.segment_synthseg(tmp_path / "in.nii.gz", out)
    assert calls == []  # cached
    gpu_preprocess.segment_synthseg(tmp_path / "in.nii.gz", out, overwrite=True)
    assert len(calls) == 1


def test_batch_keys_unique_for_same_filename(tmp_path):
    a = tmp_path / "sub-01" / "T1w.nii.gz"
    b = tmp_path / "sub-02" / "T1w.nii.gz"
    keys = gpu_preprocess._unique_subject_keys([a, b])
    assert len(set(keys)) == 2 and keys[0].startswith("sub-01")
    assert gpu_preprocess._unique_subject_keys([tmp_path / "x.nii.gz"]) == ["x"]


def test_pick_new_output_ignores_stale_and_ambiguous(tmp_path):
    stale = tmp_path / "old_parcellation.nii.gz"
    stale.write_bytes(b"x")
    before = gpu_preprocess._snapshot(tmp_path)
    new = tmp_path / "sub_parcellation.nii.gz"
    new.write_bytes(b"y")
    assert gpu_preprocess._pick_new_output(tmp_path, before, ("*parcellation*.nii*",), "t") == new
    (tmp_path / "a.nii.gz").write_bytes(b"1")
    (tmp_path / "b.nii.gz").write_bytes(b"2")
    with pytest.raises(RuntimeError):
        gpu_preprocess._pick_new_output(tmp_path, before, ("*.nii*",), "t")
