""" Test null models """

import pytest

import numpy as np
from scipy.spatial.distance import pdist, squareform
from tempfile import gettempdir
from os.path import join, exists
import vtk

from brainspace.vtk_interface import wrap_vtk
from brainspace.vtk_interface.pipeline import to_data
from brainspace.mesh import mesh_elements as me
from brainspace.null_models.moran import (compute_mem, moran_randomization,
                                          MoranRandomization)
from brainspace.null_models.spin import (_generate_spins, spin_permutations,
                                         SpinPermutations)
# from brainspace.null_models.variogram import Base, Sampled,
from brainspace.null_models.variogram import (txt2memmap, SurrogateMaps,
                                              SampledSurrogateMaps)


def test_moran():
    # Sphere with points as locations to build spatial matrix
    sphere = wrap_vtk(vtk.vtkSphereSource, radius=20, thetaResolution=10,
                      phiResolution=5)
    sphere = to_data(sphere)
    n_pts = sphere.n_points

    # Features to randomize
    rs = np.random.RandomState(0)
    feats = rs.randn(n_pts, 2)

    # build spatial weight matrix
    a = me.get_immediate_distance(sphere)
    a.data **= -1

    # test default
    v, w = compute_mem(a, tol=1e-7)
    assert w.shape[0] <= (n_pts - 1)
    assert v.shape == (n_pts, w.shape[0])

    r1 = moran_randomization(feats[:, 0], v, n_rep=10, random_state=0)
    assert r1.shape == (10, n_pts)

    r2 = moran_randomization(feats, v, n_rep=10, random_state=0)
    assert r2.shape == (10, n_pts, 2)

    # test default dense
    mem, ev = compute_mem(a.toarray(), tol=1e-7)
    assert np.allclose(w, ev)
    assert np.allclose(v, mem)

    r1 = moran_randomization(feats[:, 0], mem, n_rep=10, random_state=0)
    assert r1.shape == (10, n_pts)

    r2 = moran_randomization(feats, mem, n_rep=10, random_state=0)
    assert r2.shape == (10, n_pts, 2)

    # test object api
    msr = MoranRandomization(n_rep=10, random_state=0, tol=1e-7)
    msr.fit(a)
    assert np.allclose(msr.mev_, ev)
    assert np.allclose(msr.mem_, mem)
    assert np.allclose(r1, msr.randomize(feats[:, 0]))
    assert np.allclose(r2, msr.randomize(feats))

    # test object api with PolyData
    msr = MoranRandomization(n_rep=10, random_state=0, tol=1e-7)
    msr.fit(sphere)
    assert np.allclose(msr.mev_, ev)
    assert np.allclose(msr.mem_, mem)
    assert np.allclose(r1, msr.randomize(feats[:, 0]))
    assert np.allclose(r2, msr.randomize(feats))


def test_spin():
    # Create dummy spheres or left and right hemispheres
    sphere_lh = wrap_vtk(vtk.vtkSphereSource, radius=20, thetaResolution=10,
                         phiResolution=5)
    sphere_lh = to_data(sphere_lh)
    pts_lh = sphere_lh.Points
    n_pts_lh = sphere_lh.n_points

    # Right with more points
    sphere_rh = wrap_vtk(vtk.vtkSphereSource, radius=20, thetaResolution=10,
                         phiResolution=6)
    sphere_rh = to_data(sphere_rh)
    pts_rh = sphere_rh.Points
    n_pts_rh = sphere_rh.n_points

    # Features to randomize
    rs = np.random.RandomState(0)
    feats_lh = rs.randn(n_pts_lh, 2)
    feats_rh = rs.randn(n_pts_rh, 2)

    # generate spin indices
    ridx1 = _generate_spins(pts_lh, n_rep=10, random_state=0)
    assert ridx1['lh'].shape == (10, n_pts_lh)
    assert 'rh' not in ridx1

    ridx2 = _generate_spins(pts_lh, points_rh=pts_rh, n_rep=10,
                            random_state=0)
    assert ridx2['lh'].shape == (10, n_pts_lh)
    assert ridx2['rh'].shape == (10, n_pts_rh)

    # test api lh
    sp = SpinPermutations(n_rep=10, random_state=0)
    sp.fit(pts_lh)
    assert np.all(sp.spin_lh_ == ridx1['lh'])
    assert sp.spin_rh_ is None
    assert sp.randomize(feats_lh[:, 0]).shape == (10, n_pts_lh)
    assert sp.randomize(feats_lh).shape == (10, n_pts_lh, 2)

    # test api lh and rh
    sp = SpinPermutations(n_rep=10, random_state=0)
    sp.fit(pts_lh, points_rh=pts_rh)
    assert np.all(sp.spin_lh_ == ridx2['lh'])
    assert np.all(sp.spin_rh_ == ridx2['rh'])

    r1 = sp.randomize(feats_lh[:, 0])
    assert r1[0].shape == (10, n_pts_lh)
    assert r1[1] is None

    r1bis = spin_permutations(pts_lh, feats_lh[:, 0], n_rep=10, random_state=0)
    assert np.all(r1[0] == r1bis)

    r2 = sp.randomize(feats_lh)
    assert r2[0].shape == (10, n_pts_lh, 2)
    assert r2[1] is None

    r2bis = spin_permutations(pts_lh, feats_lh, n_rep=10, random_state=0)
    assert np.all(r2[0] == r2bis)

    r1 = sp.randomize(feats_lh[:, 0], x_rh=feats_rh[:, 0])
    assert r1[0].shape == (10, n_pts_lh)
    assert r1[1].shape == (10, n_pts_rh)

    r1bis = spin_permutations({'lh': pts_lh, 'rh': pts_rh},
                              {'lh': feats_lh[:, 0], 'rh': feats_rh[:, 0]},
                              n_rep=10, random_state=0)
    assert np.all(r1[0] == r1bis[0])
    assert np.all(r1[1] == r1bis[1])

    r2 = sp.randomize(feats_lh, x_rh=feats_rh)
    assert r2[0].shape == (10, n_pts_lh, 2)
    assert r2[1].shape == (10, n_pts_rh, 2)

    r2bis = spin_permutations({'lh': pts_lh, 'rh': pts_rh},
                              {'lh': feats_lh, 'rh': feats_rh},
                              n_rep=10, random_state=0)
    assert np.all(r2[0] == r2bis[0])
    assert np.all(r2[1] == r2bis[1])


def test_variogram_base():
    # Base class
    # Sphere with points as locations to build distance matrix
    sphere = wrap_vtk(vtk.vtkSphereSource, radius=20, thetaResolution=20,
                      phiResolution=10)
    sphere = to_data(sphere)
    points = sphere.GetPoints()
    npoints = sphere.n_points
    distmat = squareform(pdist(points))  # pairwise distance matrix

    rs = np.random.RandomState(0)
    brainmap = rs.randn(npoints, 1).flatten()
    brainmap[0] = np.nan

    # Generate surrogates
    vsur = SurrogateMaps(random_state=672436)
    vsur.fit(distmat)
    surrs = vsur.randomize(brainmap, n_rep=10)

    # np.random.seed(672436)
    # gen = Base(brainmap, distmat)
    # surrs2 = gen(n=10)
    #
    # assert np.allclose(surrs, surrs2, equal_nan=True)
    assert surrs.shape == (10, npoints)
    assert np.allclose(vsur._dist, distmat)
    # assert np.allclose(gen.x.data[1:], brainmap[1:])
    # assert np.isnan(gen.x.data[0])
    assert not np.isnan(surrs).any()


def test_variogram_sampled():
    # Sampled class
    # Sphere with points as locations to build distance matrix
    sphere = wrap_vtk(vtk.vtkSphereSource, radius=20, thetaResolution=50,
                      phiResolution=25)
    sphere = to_data(sphere)
    points = sphere.GetPoints()
    npoints = sphere.n_points
    distmat = squareform(pdist(points))  # pairwise distance matrix

    rs = np.random.RandomState(0)
    brainmap = rs.randn(npoints, 1).flatten()
    brainmap[0] = np.nan

    # Create a mask file
    mask = np.zeros(npoints, dtype=int)
    mask[[3, 4, 5]] = 1

    # Save distmat and mask file
    temp = gettempdir()
    dist_file = join(temp, 'distmat.txt')
    mask_file = join(temp, 'mask.txt')
    np.savetxt(dist_file, distmat)
    np.savetxt(mask_file, mask)
    assert exists(dist_file) and exists(mask_file)

    # Convert to memmap
    files = txt2memmap(dist_file=dist_file, output_dir=temp, maskfile=mask_file)
    assert exists(files['index'])
    assert exists(files['distmat'])

    masked_map = brainmap[np.where(mask == 0)]

    # Generate surrogates
    vsur = SampledSurrogateMaps(knn=100, ns=100, random_state=43)
    vsur.fit(files['distmat'], files['index'])
    surrs = vsur.randomize(masked_map, n_rep=5)

    # np.random.seed(43)
    # gen = Sampled(masked_map, files['distmat'], files['index'],
    #               knn=100, ns=100)
    # surrs2 = gen(n=5)
    #
    # assert np.allclose(surrs, surrs2, equal_nan=True)
    assert vsur._dist.shape == (npoints-3, 100)
    assert surrs.shape == (5, npoints-3)
    # assert np.allclose(gen.x.data[1:], masked_map[1:])
    # assert np.isnan(gen.x.data[0])
    assert not np.isnan(surrs).any()
