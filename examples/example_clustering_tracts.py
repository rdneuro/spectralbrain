# %% [markdown]
# # SpectralBrain — end-to-end: HippUnfold hippocampus -> ddCRP -> atlas stats -> 3D figures
#
# A complete, runnable walkthrough that chains the methods ported from
# brainmosaic with SpectralBrain's spectral core and the new 3D visualization:
#
#   1. load a HippUnfold hippocampal surface (real data if available, else synthetic)
#   2. LBO eigenpairs -> HKS + WKS descriptors
#   3. fuse descriptors -> contiguous ddCRP clustering (NIW marginal likelihood)
#   4. data-driven autotuning + consensus over an ensemble
#   5. compare the parcellation against a subfield atlas with a size-matched null
#      (ARI / spARI) and an eigenstrapping spatial null on a descriptor map
#   6. a parcellation-vs-clustering 3D grid (views x labelings)
#   7. an advanced multi-POV tract figure (bundle surface + HKS overlay)
#
# Set the paths in cell [1] to your own HippUnfold / TractSeg outputs; with the
# paths left as ``None`` the script synthesises a hippocampus-like mesh and a
# bundle mask so it runs anywhere. Replies/comments PT-BR; code English.

# %% [1] Configuração e dados (reais se os caminhos existirem; senão, sintéticos)
from __future__ import annotations

import os
import numpy as np

import spectralbrain as sb
import spectralbrain.statistics as sbstats
import spectralbrain.viz as sbviz

# --- aponte para seus dados reais (ou deixe None p/ fallback sintético) ---
HIPP_SURF = os.environ.get("SB_HIPP_SURF")        # ex.: ".../sub-XX_hemi-L_den-8k_label-hipp_midthickness.surf.gii"
HIPP_SUBFIELDS = os.environ.get("SB_HIPP_LABELS")  # ex.: subfields .label.gii (Iglesias/HippUnfold)
TRACT_MASK = os.environ.get("SB_TRACT_MASK")       # ex.: TractSeg "CST_left.nii.gz"
OUT = os.environ.get("SB_OUT", "./sb_example_out")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0)


def _synthetic_hippocampus(n_u=60, n_v=22):
    """A folded C-shaped sheet standing in for a HippUnfold midthickness surface,
    with an anterior->posterior graded 5-subfield 'atlas' for comparison."""
    u = np.linspace(0.15 * np.pi, 1.15 * np.pi, n_u)   # long axis (A->P), curved
    v = np.linspace(-1.0, 1.0, n_v)                    # proximal-distal
    U, V = np.meshgrid(u, v, indexing="ij")
    R = 6.0 + 1.4 * V
    X = R * np.cos(U)
    Y = R * np.sin(U)
    Z = 2.6 * V + 0.8 * np.sin(3 * U)
    verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float64)
    vid = lambda i, j: i * n_v + j
    faces = []
    for i in range(n_u - 1):
        for j in range(n_v - 1):
            faces.append([vid(i, j), vid(i + 1, j), vid(i, j + 1)])
            faces.append([vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)])
    faces = np.asarray(faces, dtype=np.int64)
    subfields = np.clip((np.arange(n_u) * 5 // n_u), 0, 4)         # A->P bands
    atlas = np.repeat(subfields, n_v)                              # (V,)
    return verts, faces, atlas


def _load_surface(path):
    d = sb.io.load(path)
    return np.asarray(d["vertices"], float), np.asarray(d["faces"], np.int64)


if HIPP_SURF and os.path.exists(HIPP_SURF):
    vertices, faces = _load_surface(HIPP_SURF)
    if HIPP_SUBFIELDS and os.path.exists(HIPP_SUBFIELDS):
        atlas = np.asarray(sb.io.load(HIPP_SUBFIELDS)["labels"], np.int64)
    else:
        atlas = None
    print(f"Hipocampo REAL: V={vertices.shape[0]} F={faces.shape[0]}")
else:
    vertices, faces, atlas = _synthetic_hippocampus()
    print(f"Hipocampo SINTÉTICO: V={vertices.shape[0]} F={faces.shape[0]} (defina SB_HIPP_SURF p/ dados reais)")


# %% [2] LBO -> HKS + WKS
mesh = sb.BrainMesh(vertices, faces)
decomp = mesh.decompose(k=100, laplacian_method="cotangent")  # 'robust' p/ malhas não-manifold
hks = np.asarray(sb.compute_hks(decomp, n_times=20))          # (V, 20)
wks = np.asarray(sb.compute_wks(decomp, n_energies=20))       # (V, 20)
print(f"HKS {hks.shape}  WKS {wks.shape}  λ1..λ3={decomp.eigenvalues[1:4].round(4)}")


# %% [3] Fusão de descritores -> matriz de features H (V, d)
# log + z-score por coluna, depois concatena (equivalente didático ao fuse_concatenate).
def _zlog(M):
    M = np.log(np.abs(M) + 1e-12)
    return (M - M.mean(0)) / (M.std(0) + 1e-12)

H = np.column_stack([_zlog(hks), _zlog(wks)])                 # (V, 40)
print(f"H fundido: {H.shape}")


# %% [4] ddCRP contíguo (verossimilhança marginal NIW, decay por distância de aresta real)
# PERF: o sampler usa estatísticas suficientes incrementais por componente (nunca
# recomputa o scatter do cluster grande) -> ~1-3 s/draw em ~7k vértices. Para um
# ganho extra e melhor condicionamento, comprima os descritores com n_components
# (HKS/WKS são redundantes entre escalas); use None para o espaço completo.
res_ddcrp = sbstats.cluster_ddcrp(
    H, faces=faces, vertices=vertices,        # vertices -> decay usa distâncias de aresta reais
    decay_kind="exponential", alpha=1.0, n_components=8,
    n_draws=150, burn_in=80, thin=2, chains=3, random_state=0, progress=True,
)
print(res_ddcrp, "| R-hat(K) =", round(res_ddcrp.quality["rhat_n_clusters"], 3))


# %% [5] Autotune dos hiperparâmetros (optuna se instalado; senão random search)
# NOTA DE CUSTO: cada trial é uma amostragem ddCRP completa. Em ~7k vértices,
# 30 trials levam minutos. Para um passe rápido use backend="random" e poucos
# trials; aumente eval_draws/n_trials para a busca definitiva. O guarda
# anti-degenerescência rejeita partições que colapsam em K=1 (score muito baixo).
tuned = sbstats.autotune_ddcrp(
    H, faces=faces, vertices=vertices,
    backend=None, n_trials=30, objective="silhouette",
    eval_draws=25, eval_burn_in=15, eval_chains=1, random_state=0, progress=True,
)
res_tuned = tuned.cluster_result or res_ddcrp
print("melhores params:", tuned.best_params, "| score:", round(tuned.best_score, 3))
print("ddCRP (autotuned):", res_tuned)


# %% [6] Consenso sobre um ensemble de execuções (estabilidade por vértice)
ensemble = [
    sbstats.cluster_ddcrp(H, faces=faces, vertices=vertices, decay_kind="exponential",
                          n_draws=60, burn_in=40, chains=1, random_state=s, progress=False).labels
    for s in range(5)
]
res_consensus = sbstats.cluster_consensus(ensemble)
print(res_consensus, "| estabilidade média =", round(res_consensus.quality["mean_stability"], 3))


# %% [7] Comparação cluster-vs-atlas (ARI/spARI vs nulo casado em tamanho)
if atlas is not None:
    concord = sbstats.cluster_atlas_concordance(
        res_tuned.labels, atlas, faces=faces, coords=vertices, n_null=1000, seed=0)
    m = concord["metrics"]
    print(f"ARI={m['ari']:.3f} AMI={m['ami']:.3f} VI={m['vi_bits']:.3f}")
    print(f"nulo casado: ARI z={concord['ari_null']['z']:.2f} p={concord['ari_null']['p']:.4f}")
    print(f"spARI={concord['spARI']['spARI']:.3f}  (espacialmente consciente)")
    print("Dice médio (melhor match):", round(concord.get("mean_best_dice", float('nan')), 3))

# Nulo espacial por eigenstrapping num mapa descritor (recomendado p/ superfícies limitadas)
hks_map = hks[:, 5]
evals, evecs = decomp.eigenvalues, decomp.eigenvectors
mass = np.asarray(decomp.mass.diagonal() if hasattr(decomp.mass, "diagonal") else decomp.mass)
surr = sbstats.null_eigenstrapping(hks_map, evals, evecs, mass, n_surrogates=200, seed=0)
obs_autocorr = float(np.corrcoef(hks_map[faces[:, 0]], hks_map[faces[:, 1]])[0, 1])
print(f"eigenstrapping: {len(surr)} surrogates | autocorr observada (arestas) ≈ {obs_autocorr:.3f}")


# %% [8] FIGURA — grade parcelamento-vs-clusterização (colunas=views, linhas=labelings)
labelings = {}
if atlas is not None:
    labelings["Subfields"] = atlas
labelings["ddCRP"] = res_tuned.labels
labelings["Consensus"] = res_consensus.labels

fig_grid, meta = sbviz.plot_parcellation_cluster_grid(
    vertices, faces, labelings,
    views=["left_lateral", "anterior", "superior"],  # colunas
    engine="vedo",                                   # 'pyvista' como alternativa
    title="Hippocampus — parcellation vs. ddCRP clustering",
    save=os.path.join(OUT, "hipp_parcellation_vs_clusters.png"),
)
print("grade salva:", meta["shape"], "->", os.path.join(OUT, "hipp_parcellation_vs_clusters.png"))


# %% [9] FIGURA — trato 3D multi-POV (superfície de bundle + overlay HKS)
if TRACT_MASK and os.path.exists(TRACT_MASK):
    import nibabel as nib
    vol = nib.load(TRACT_MASK)
    Vb, Fb = sbviz.mask_to_mesh(vol.get_fdata(), affine=vol.affine, smooth_sigma=1.0)
    bundle_name = os.path.basename(TRACT_MASK).split(".")[0]
else:
    # bundle sintético (elipsoide alongado) só para o exemplo rodar sem dados
    zz, yy, xx = np.mgrid[0:30, 0:30, 0:80]
    mask = (((xx - 40) / 36) ** 2 + ((yy - 15) / 9) ** 2 + ((zz - 15) / 9) ** 2 <= 1).astype(float)
    Vb, Fb = sbviz.mask_to_mesh(mask, affine=np.eye(4), smooth_sigma=1.0)
    bundle_name = "synthetic_bundle"

hks_bundle = sbviz.spectral_overlay(Vb, Fb, kind="hks", n_eigen=80, t_index=20)

# 9a — montagem multi-POV da superfície do bundle com overlay HKS
from spectralbrain.viz.tracts3d import compose_tract_panel, render_bundle_surface

pov_pngs = []
for view in ("left", "anterior", "superior", "oblique"):
    p = os.path.join(OUT, f"{bundle_name}_{view}.png")
    render_bundle_surface(Vb, Fb, scalars=hks_bundle, view=view,
                          out_path=p, symmetric=False)
    pov_pngs.append(p)
clim = sbviz.robust_clim(hks_bundle)
fig_tract = compose_tract_panel(
    pov_pngs, titles=["Left", "Anterior", "Superior", "Oblique"],
    colorbar={"kind": "sequential", "clim": clim, "label": "HKS(t=20)"},
    out_path=os.path.join(OUT, f"{bundle_name}_hks_multiview.pdf"),
)
print("painel de trato salvo:", os.path.join(OUT, f"{bundle_name}_hks_multiview.pdf"))

# 9b — (com streamlines reais .trk/.tck, use:)
#   sl, aff = sbviz.load_streamlines("CST_left.trk", to_space="world")   # requer dipy
#   sbviz.streamlines_multiview(sl, views=("left","anterior","superior","oblique"),
#                               color_by="orientation", out_path=os.path.join(OUT,"cst_dec.pdf"))

print("\n=== pipeline end-to-end concluído ===")
