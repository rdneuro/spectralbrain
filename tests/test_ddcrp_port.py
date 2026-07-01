"""Tests for the ddCRP / consensus / cluster-vs-atlas methods ported from
brainmosaic and harmonised into spectralbrain.statistics.
"""

import numpy as np
import pytest

import spectralbrain.statistics as st
from spectralbrain.statistics.clustering import ClusterResult


def _grid(side=12):
    V = np.array([[r, c, 0.0] for r in range(side) for c in range(side)], float)
    vid = lambda r, c: r * side + c
    F = []
    for r in range(side - 1):
        for c in range(side - 1):
            F += [[vid(r, c), vid(r + 1, c), vid(r, c + 1)],
                  [vid(r + 1, c), vid(r + 1, c + 1), vid(r, c + 1)]]
    return V, np.array(F)


def _two_blob_features(side=12, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([rng.normal(-3 if c < side // 2 else 3, 0.4, size=3)
                     for r in range(side) for c in range(side)])


def test_cluster_ddcrp_returns_clusterresult():
    side = 12
    V, F = _grid(side)
    X = _two_blob_features(side)
    res = st.cluster_ddcrp(X, faces=F, vertices=V, decay_kind="exponential",
                           n_draws=30, burn_in=20, thin=2, chains=2,
                           random_state=0, progress=False)
    assert isinstance(res, ClusterResult)
    assert res.method == "ddcrp"
    assert res.n_clusters >= 2
    assert res.metadata["co_association"].shape == (side * side, side * side)


def test_cluster_ddcrp_functional_two_blocks():
    side = 10
    V, F = _grid(side)
    X = _two_blob_features(side)
    hks, wks = np.cumsum(np.abs(X), axis=1), X ** 2
    res = st.cluster_ddcrp_functional([hks, wks], faces=F, vertices=V, n_fpca=2,
                                      decay_kind="exponential", n_draws=20,
                                      burn_in=15, thin=2, chains=2,
                                      random_state=0, progress=False)
    assert res.method == "ddcrp_functional"
    assert res.n_clusters >= 1


def test_cluster_consensus_threshold_default():
    side = 10
    V, F = _grid(side)
    X = _two_blob_features(side)
    parts = [st.cluster_ddcrp(X, faces=F, vertices=V, n_draws=15, burn_in=10,
                              chains=1, random_state=s, progress=False).labels
             for s in range(3)]
    con = st.cluster_consensus(parts)
    assert con.method == "consensus"
    assert "co_association" in con.metadata
    assert 0.0 <= con.quality["mean_stability"] <= 1.0


def test_ddcrp_requires_contiguity():
    X = _two_blob_features(8)
    with pytest.raises(ValueError):
        st.cluster_ddcrp(X, progress=False)  # no faces / adjacency


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cluster_atlas_concordance_with_null():
    side = 12
    V, F = _grid(side)
    atlas = np.array([(r >= side // 2) * 2 + (c >= side // 2)
                      for r in range(side) for c in range(side)])
    rep = st.cluster_atlas_concordance(atlas, atlas, faces=F, coords=V,
                                       n_null=30, seed=0)
    # identical partitions -> ARI == 1 and a large positive null z-score
    assert rep["metrics"]["ari"] == pytest.approx(1.0, abs=1e-6)
    assert rep["ari_null"]["z"] > 2.0


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_compare_partitions_and_spari():
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([0, 0, 1, 1, 2, 2])
    m = st.compare_partitions(a, b)
    assert m["ari"] == pytest.approx(1.0)
    coords = np.arange(6, dtype=float)[:, None] * np.ones(3)
    s = st.spatial_rand_index(a, b, coords=coords)
    assert 0.0 <= s["spARI"] <= 1.0
