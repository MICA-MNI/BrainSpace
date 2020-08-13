""" Test datasest """


import pytest

import numpy as np

from brainspace.datasets import *
from brainspace.vtk_interface.wrappers import BSPolyData


parametrize = pytest.mark.parametrize


testdata_surface = [
    ({}),
    ({'as_sphere': True}),
    ({'as_sphere': False, 'with_normals': False}),
    ({'as_sphere': True, 'with_normals': False}),
]


@parametrize('kwds', testdata_surface)
def test_load_surface(kwds):

    surf_lh, surf_rh = load_conte69(**kwds)
    assert isinstance(surf_lh, BSPolyData)
    assert isinstance(surf_rh, BSPolyData)
    assert surf_lh.n_points == surf_rh.n_points
    if 'with_normals' not in kwds or kwds['with_normals']:
        assert 'Normals' in surf_lh.PointData.keys()
        assert 'Normals' in surf_rh.PointData.keys()
    else:
        assert 'Normals' not in surf_lh.PointData.keys()
        assert 'Normals' not in surf_rh.PointData.keys()


def test_load_mask():
    surf_lh, surf_rh = load_conte69()
    total_n_pts = surf_lh.n_points + surf_rh.n_points

    mask = load_mask(join=True)
    assert mask.shape == (total_n_pts,)
    assert mask.dtype == np.bool


@parametrize('name', ['vosdewael', 'schaefer'])
@parametrize('kwds', [{}] + [{'n_parcels': k} for k in [100, 200, 300, 400]])
def test_load_parcellation(name, kwds):
    mask = load_mask(join=True)
    total_n_pts = mask.size

    if 'n_parcels' in kwds:
        n = kwds['n_parcels']
        parc = load_parcellation(name, scale=n, join=True)
    else:
        n = 400
        parc = load_parcellation(name, join=True)

    assert parc.shape == (total_n_pts,)
    assert np.unique(parc).size == n + 1
    assert np.count_nonzero(np.isnan(parc[mask])) == 0


def test_load_thickness():
    mask = load_mask(join=True)
    total_n_pts = mask.size

    thick = load_marker('thickness', join=True)
    assert thick.shape == (total_n_pts,)
    assert thick.dtype == np.float
    assert not np.isnan(thick[mask]).any()
    assert np.isnan(thick[~mask]).all()


def test_load_myelin():
    mask = load_mask(join=True)
    total_n_pts = mask.size

    myelin = load_marker('t1wt2w', join=True)
    assert myelin.shape == (total_n_pts,)
    assert myelin.dtype == np.float
    assert not np.isnan(myelin[mask]).any()
    assert np.isnan(myelin[~mask]).all()


@parametrize('name', ['fc', 'mpc'])
@parametrize('kwds', [{}] + [{'idx': k} for k in [0, 1]])
def test_load_gradient(name, kwds):
    mask = load_mask(join=True)
    total_n_pts = mask.size

    if 'idx' in kwds:
        idx = kwds['idx']
        grad = load_gradient(name, idx=idx, join=True)
    else:
        grad = load_gradient(name, join=True)

    assert grad.shape == (total_n_pts,)
    assert grad.dtype == np.float
    assert not np.isnan(grad[mask]).any()
    assert np.isnan(grad[~mask]).all()


@parametrize('name', ['vosdewael', 'schaefer'])
@parametrize('kwds', [{}] + [{'n_parcels': k} for k in [100, 200, 300, 400]])
def test_load_group(name, kwds):

    if 'n_parcels' in kwds:
        n = kwds['n_parcels']
        cm = load_group_fc(name, scale=n)
    else:
        n = 400
        cm = load_group_fc(name)

    assert cm.shape == (n, n)
    assert cm.dtype == np.float
    assert np.allclose(cm, cm.T)


@parametrize('name', ['vosdewael', 'schaefer'])
@parametrize('kwds', [{}] + [{'n_parcels': k} for k in [100, 200, 300, 400]])
def test_load_holdout(name, kwds):

    if 'n_parcels' in kwds:
        n = kwds['n_parcels']
        cm = load_group_fc(name, scale=n, group='holdout')
    else:
        n = 400
        cm = load_group_fc(name, group='holdout')

    assert cm.shape == (n, n)
    assert cm.dtype == np.float
    assert np.allclose(cm, cm.T)
