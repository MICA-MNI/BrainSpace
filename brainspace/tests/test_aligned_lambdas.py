"""Tests for proxy lambda correspondence after Procrustes alignment (#94)."""

import numpy as np

from brainspace.gradient import GradientMaps
from brainspace.gradient.alignment import (
    procrustes, procrustes_alignment, aligned_lambdas, ProcrustesAlignment,
)


def _psd(n, seed):
    rs = np.random.RandomState(seed)
    a = rs.randn(n, n + 5)
    return a @ a.T


def test_aligned_lambdas_identity_rotation_is_passthrough():
    lambdas = np.array([1.0, 0.5, 0.25, 0.1])
    proxy = aligned_lambdas(lambdas, np.eye(4))
    np.testing.assert_array_equal(proxy, lambdas)


def test_aligned_lambdas_permutation_rotation():
    lambdas = np.array([10.0, 5.0, 1.0])
    # Rotation that swaps gradients 0 and 1.
    perm = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    proxy = aligned_lambdas(lambdas, perm)
    np.testing.assert_array_equal(proxy, [5.0, 10.0, 1.0])


def test_procrustes_return_transform_consistency():
    rs = np.random.RandomState(0)
    src = rs.randn(40, 5)
    tgt = rs.randn(40, 5)
    aligned, t = procrustes(src, tgt, return_transform=True)
    np.testing.assert_allclose(aligned, src @ t, atol=1e-12)


def test_procrustes_alignment_returns_transforms():
    rs = np.random.RandomState(1)
    data = [rs.randn(30, 4) for _ in range(3)]
    aligned, mean, transforms = procrustes_alignment(
        data, return_reference=True, return_transforms=True)
    assert len(transforms) == len(data)
    for d, a, t in zip(data, aligned, transforms):
        np.testing.assert_allclose(a, d @ t, atol=1e-10)


def test_procrustes_alignment_class_exposes_transforms_attr():
    rs = np.random.RandomState(2)
    data = [rs.randn(20, 3) for _ in range(4)]
    pa = ProcrustesAlignment().fit(data)
    assert hasattr(pa, 'transforms_')
    assert len(pa.transforms_) == len(data)
    for d, a, t in zip(data, pa.aligned_, pa.transforms_):
        np.testing.assert_allclose(a, d @ t, atol=1e-10)


def test_gradientmaps_exposes_aligned_lambdas():
    mats = [_psd(15, seed=s) for s in (10, 11, 12)]
    gm = GradientMaps(n_components=3, alignment='procrustes',
                      random_state=0).fit(mats)
    assert gm.aligned_lambdas_ is not None
    assert len(gm.aligned_lambdas_) == 3
    for la, al in zip(gm.lambdas_, gm.aligned_lambdas_):
        # Each proxy entry must be drawn from the original lambda set
        # (possibly with repeats): every value of al lives in la.
        for v in al:
            assert np.any(np.isclose(la, v))


def test_no_alignment_means_no_aligned_lambdas():
    mats = [_psd(15, seed=s) for s in (20, 21)]
    gm = GradientMaps(n_components=3, alignment=None,
                      random_state=0).fit(mats)
    assert gm.aligned_ is None
    assert gm.aligned_lambdas_ is None


def test_single_dataset_with_reference_returns_1d_aligned_lambdas():
    a = _psd(20, seed=30)
    ref_gm = GradientMaps(n_components=3, random_state=0).fit(a)
    ref = ref_gm.gradients_
    other = _psd(20, seed=31)
    gm = GradientMaps(n_components=3, alignment='procrustes',
                      random_state=0).fit(other, reference=ref)
    # Single-dataset path collapses lists to single arrays.
    assert gm.aligned_lambdas_ is not None
    assert gm.aligned_lambdas_.shape == (3,)
