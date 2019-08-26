""" Test datasest """


import pytest

import numpy as np

from brainspace.datasets import *
from brainspace.vtk_interface.wrappers import BSPolyData


def test_load_conte69():

    surf_lh, surf_rh = load_conte69()
    assert isinstance(surf_lh, BSPolyData)
    assert isinstance(surf_rh, BSPolyData)
    assert surf_lh.n_points == surf_rh.n_points
    assert 'Normals' in surf_lh.PointData.keys()
    assert 'Normals' in surf_rh.PointData.keys()

    surf_lh2, surf_rh2 = load_conte69(as_sphere=False)
    assert np.all(surf_lh2.Points == surf_lh.Points)
    assert np.all(surf_rh2.Points == surf_rh.Points)

    surf_lh3, surf_rh3 = load_conte69(with_normals=False)
    assert np.all(surf_lh3.Points == surf_lh.Points)
    assert np.all(surf_rh3.Points == surf_rh.Points)
    assert 'Normals' not in surf_lh3.PointData.keys()
    assert 'Normals' not in surf_rh3.PointData.keys()

    sphere_lh, sphere_rh = load_conte69(as_sphere=True, with_normals=True)
    assert sphere_lh.n_points == surf_lh.n_points
    assert sphere_rh.n_points == surf_rh.n_points
    assert np.any(sphere_lh.Points != surf_lh.Points)
    assert np.any(sphere_rh.Points != surf_rh.Points)
    assert 'Normals' in sphere_lh.PointData.keys()
    assert 'Normals' in sphere_rh.PointData.keys()


def test_load_mask():
    surf_lh, surf_rh = load_conte69()
    total_n_pts = surf_lh.n_points + surf_rh.n_points

    mask = load_mask()
    assert mask.shape == (total_n_pts,)
    assert mask.dtype == np.bool


def test_load_parcellation():
    mask = load_mask()
    total_n_pts = mask.size

    for name in ['vosdewael', 'schaefer']:
        for n in [100, 200, 300, 400]:
            parc = load_parcellation(name, n_parcels=n)

            assert parc.shape == (total_n_pts,)
            assert np.unique(parc).size == n + 1
            assert np.count_nonzero(np.isnan(parc[mask])) == 0


def test_load_thickness():
    mask = load_mask()
    total_n_pts = mask.size

    thick = load_thickness(parcellation=None, mask=None)
    assert thick.shape == (total_n_pts,)
    assert thick.dtype == np.float
    assert not np.isnan(thick[mask]).any()
    assert np.isnan(thick[~mask]).all()


def test_load_myelin():
    mask = load_mask()
    total_n_pts = mask.size

    myelin = load_t1t2(parcellation=None, mask=None)
    assert myelin.shape == (total_n_pts,)
    assert myelin.dtype == np.float
    assert not np.isnan(myelin[mask]).any()
    assert np.isnan(myelin[~mask]).all()


def test_load_gradient():
    mask = load_mask()
    total_n_pts = mask.size

    grad = load_gradient('fc')
    assert grad.shape == (total_n_pts,)
    assert grad.dtype == np.float
    assert not np.isnan(grad[mask]).any()
    assert np.isnan(grad[~mask]).all()

    grad0 = load_gradient('fc', idx=0)
    assert ((grad0 == grad) | (np.isnan(grad0) & np.isnan(grad))).all()

    grad1 = load_gradient('fc', idx=1)
    assert grad1.shape == (total_n_pts,)

    grad = load_gradient('mpc')
    assert grad.shape == (total_n_pts,)
    assert grad.dtype == np.float
    assert not np.isnan(grad[mask]).any()
    assert np.isnan(grad[~mask]).all()

    grad0 = load_gradient('mpc', idx=0)
    assert ((grad0 == grad) | (np.isnan(grad0) & np.isnan(grad))).all()

    grad1 = load_gradient('mpc', idx=1)
    assert grad1.shape == (total_n_pts,)
