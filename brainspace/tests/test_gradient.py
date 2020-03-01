""" Test gradient maps """

import pytest

import numpy as np
from scipy.sparse import coo_matrix

from brainspace.gradient import compute_affinity
from brainspace.gradient.alignment import (procrustes_alignment,
                                           ProcrustesAlignment)
from brainspace.gradient import embedding as emb
from brainspace.gradient import GradientMaps


def test_kernels():
    rs = np.random.RandomState(0)
    x = rs.randn(10, 15)
    a = np.corrcoef(x)

    # with pytest.warns(UserWarning):
    a2 = compute_affinity(a, sparsity=None)
    assert np.count_nonzero(a2 < 0) == 0

    a2 = compute_affinity(a, sparsity=.7)
    assert np.count_nonzero(a2 < 0) == 0
    assert np.all(np.count_nonzero(a2, axis=1) == 3)

    a2 = compute_affinity(a, kernel='cosine', sparsity=.7)
    assert np.count_nonzero(a2 < 0) == 0


def test_alignment():
    rs = np.random.RandomState(0)
    list_data = [rs.randn(10, 5) for _ in range(3)]
    ref = rs.randn(10, 5)

    # test without reference
    pa = ProcrustesAlignment()
    pa.fit(list_data)
    assert len(pa.aligned_) == len(list_data)
    assert pa.mean_.shape == list_data[0].shape

    aligned2, ref2 = procrustes_alignment(list_data, return_reference=True)
    assert np.allclose(ref2, pa.mean_)
    for i in range(3):
        assert np.allclose(aligned2[i], pa.aligned_[i])

    # test with reference
    pa2 = ProcrustesAlignment()
    pa2.fit(list_data, reference=ref)
    assert len(pa2.aligned_) == len(list_data)
    assert pa2.mean_.shape == list_data[0].shape

    aligned2, ref2 = procrustes_alignment(list_data, reference=ref,
                                          return_reference=True)
    assert np.allclose(ref2, pa2.mean_)
    for i in range(3):
        assert np.allclose(aligned2[i], pa2.aligned_[i])


def test_embedding_gradient():
    rs = np.random.RandomState(0)
    x = rs.randn(100, 50)
    x2 = rs.randn(100, 50)
    a = compute_affinity(x, kernel='gaussian', sparsity=0.7)
    a_sparse = coo_matrix(a)

    dop = {'dm': emb.DiffusionMaps, 'le': emb.LaplacianEigenmaps,
           'pca': emb.PCAMaps}

    for app_name, app in dop.items():
        # test dense
        m = app(random_state=0)
        m.fit(a)

        assert m.lambdas_.shape == (10,)
        assert m.maps_.shape == (100, 10)
        if app_name == 'le':
            assert np.allclose(m.lambdas_, np.sort(m.lambdas_))
        else:
            assert np.allclose(m.lambdas_, np.sort(m.lambdas_)[::-1])

        # test sparse
        m2 = app(random_state=0)
        if app_name == 'pca':
            with pytest.raises(Exception):
                m2.fit(a_sparse)
        else:
            m2.fit(a_sparse)
            assert np.allclose(m.lambdas_, m2.lambdas_)
            assert np.allclose(m.maps_, m2.maps_)

        # test with gradientmaps
        gm = GradientMaps(approach=app_name, kernel='gaussian', random_state=0)
        gm.fit(x, sparsity=0.7)

        assert np.allclose(gm.lambdas_, m.lambdas_)
        assert np.allclose(gm.gradients_, m.maps_)
        assert gm.aligned_ is None

    # test alignment
    for align in [None, 'procrustes', 'joint']:
        gm_dm = GradientMaps(approach='dm', kernel='gaussian', alignment=align,
                             random_state=0)
        gm_dm.fit([x, x2], sparsity=0.7)

        assert len(gm_dm.gradients_) == 2
        assert len(gm_dm.lambdas_) == 2

        if align is None:
            assert gm_dm.aligned_ is None

        elif align == 'procrutes':
            for i in range(2):
                assert not np.all(gm_dm.aligned_[i] == gm_dm.gradients_[i])

        elif align == 'joint':
            for i in range(2):
                assert np.all(gm_dm.aligned_[i] == gm_dm.gradients_[i])

    # test alignment with single matrix
    gm_ref = GradientMaps(approach='dm', kernel='gaussian',
                          alignment='procrustes', random_state=0)
    ref = gm_ref.fit(x, sparsity=0.7).gradients_

    gm_single = GradientMaps(approach='dm', kernel='gaussian',
                             alignment='procrustes', random_state=0)
    gm_single.fit(x2, sparsity=0.7, reference=ref)

    gm_list = GradientMaps(approach='dm', kernel='gaussian',
                           alignment='procrustes', random_state=0)
    gm_list.fit([x2], sparsity=0.7, reference=ref)

    assert gm_single.aligned_ is not None
    assert np.allclose(gm_single.aligned_, gm_list.aligned_[0])
