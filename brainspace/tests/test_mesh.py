""" Test mesh operations """

import pytest

import os
import numpy as np

import vtk
from vtk.util.vtkConstants import VTK_TRIANGLE, VTK_LINE, VTK_VERTEX

from brainspace.vtk_interface import wrap_vtk
from brainspace.vtk_interface.wrappers import BSPolyData
from brainspace.mesh import mesh_io as mio
from brainspace.mesh import mesh_elements as me
from brainspace.mesh import mesh_creation as mc
from brainspace.mesh import mesh_operations as mop
from brainspace.mesh import mesh_cluster as mcluster
from brainspace.mesh import array_operations as aop


parametrize = pytest.mark.parametrize


try:
    import nibabel as nb
except ImportError:
    nb = None


def _generate_sphere():
    s = vtk.vtkSphereSource()
    s.Update()
    return wrap_vtk(s.GetOutput())


@parametrize('ext', ['fs', 'asc', 'ply', 'vtp', 'vtk'])
def test_io(ext):
    s = _generate_sphere()

    root_pth = os.path.dirname(__file__)
    io_pth = os.path.join(root_pth, 'test_sphere_io.{ext}').format(ext=ext)

    mio.write_surface(s, io_pth)
    s2 = mio.read_surface(io_pth)

    assert np.allclose(s.Points, s2.Points)
    assert np.all(s.GetCells2D() == s2.GetCells2D())

    os.remove(io_pth)


@pytest.mark.skipif(nb is None, reason="Requires nibabel")
def test_io_nb():
    s = _generate_sphere()

    root_pth = os.path.dirname(__file__)
    io_pth = os.path.join(root_pth, 'test_sphere_io.gii')
    mio.write_surface(s, io_pth)
    s2 = mio.read_surface(io_pth)

    assert np.allclose(s.Points, s2.Points)
    assert np.all(s.GetCells2D() == s2.GetCells2D())

    os.remove(io_pth)


def test_mesh_creation():
    st = _generate_sphere()
    sl = mc.to_lines(st)
    sv = mc.to_vertex(st)

    # build polydata with points and triangle cells
    pd = mc.build_polydata(st.Points, cells=st.GetCells2D())
    assert pd.n_points == st.n_points
    assert pd.n_cells == st.n_cells
    assert np.all(pd.cell_types == np.array([VTK_TRIANGLE]))
    assert isinstance(pd, BSPolyData)

    # build polydata with points vertices by default
    pd = mc.build_polydata(st.Points)
    assert pd.n_points == st.n_points
    assert pd.n_cells == 0
    assert np.all(pd.cell_types == np.array([VTK_VERTEX]))
    assert isinstance(pd, BSPolyData)

    # build polydata with points vertices
    pd = mc.build_polydata(st.Points, cells=sv.GetCells2D())
    assert pd.n_points == st.n_points
    assert pd.n_cells == st.n_points
    assert np.all(pd.cell_types == np.array([VTK_VERTEX]))
    assert isinstance(pd, BSPolyData)

    # build polydata with lines
    pd = mc.build_polydata(st.Points, cells=sl.GetCells2D())
    assert pd.n_points == sl.n_points
    assert pd.n_cells == sl.n_cells
    assert np.all(pd.cell_types == np.array([VTK_LINE]))
    assert isinstance(pd, BSPolyData)


@pytest.mark.xfail
def test_drop_cells():
    s = _generate_sphere()

    rs = np.random.RandomState(0)

    label_cells = rs.randint(0, 10, s.n_cells)
    cell_name = s.append_array(label_cells, at='c')

    n_cells = mop.drop_cells(s, cell_name, upp=3).n_cells
    assert n_cells == np.count_nonzero(label_cells > 3)


def test_select_cells():
    s = _generate_sphere()

    rs = np.random.RandomState(0)

    label_cells = rs.randint(0, 10, s.n_cells)
    cell_name = s.append_array(label_cells, at='c')

    n_cells = mop.select_cells(s, cell_name, low=0, upp=3).n_cells
    assert n_cells == np.count_nonzero(label_cells <= 3)


def test_mask_cells():
    s = _generate_sphere()

    rs = np.random.RandomState(0)

    label_cells = rs.randint(0, 10, s.n_cells)

    # Warns when array is boolean
    with pytest.warns(UserWarning):
        mask_cell_name = s.append_array(label_cells > 3, at='c')

    n_cells = mop.mask_cells(s, mask_cell_name).n_cells
    assert n_cells == np.count_nonzero(label_cells > 3)


@pytest.mark.xfail
def test_drop_points():
    s = _generate_sphere()

    rs = np.random.RandomState(0)

    label_points = rs.randint(0, 10, s.n_points)
    point_name = s.append_array(label_points, at='p')

    # Warns cause number of selected points may not coincide with
    # selected points
    with pytest.warns(UserWarning):
        n_pts = mop.drop_points(s, point_name, low=0, upp=3).n_points
        assert n_pts <= s.n_points


def test_select_points():
    s = _generate_sphere()

    rs = np.random.RandomState(0)

    label_points = rs.randint(0, 10, s.n_points)
    point_name = s.append_array(label_points, at='p')

    with pytest.warns(UserWarning):
        n_pts = mop.select_points(s, point_name, low=0, upp=3).n_points
        assert n_pts <= s.n_points


def test_mask_points():
    s = _generate_sphere()

    rs = np.random.RandomState(0)

    label_points = rs.randint(0, 10, s.n_points)
    with pytest.warns(UserWarning):
        mask_point_name = s.append_array(label_points > 3, at='p')

    with pytest.warns(UserWarning):
        n_pts = mop.mask_points(s, mask_point_name).n_points
        assert n_pts <= s.n_points


def test_mesh_elements():
    s = _generate_sphere()

    ee = vtk.vtkExtractEdges()
    ee.SetInputData(s.VTKObject)
    ee.Update()
    ee = wrap_vtk(ee.GetOutput())
    n_edges = ee.n_cells

    assert np.all(me.get_points(s) == s.Points)
    assert np.all(me.get_cells(s) == s.GetCells2D())
    assert me.get_extent(s).shape == (3,)

    pc = me.get_point2cell_connectivity(s)
    assert pc.shape == (s.n_points, s.n_cells)
    assert pc.dtype == np.uint8
    assert np.all(pc.sum(axis=0) == 3)

    cp = me.get_cell2point_connectivity(s)
    assert pc.dtype == np.uint8
    assert (pc - cp.T).nnz == 0

    adj = me.get_immediate_adjacency(s)
    assert adj.shape == (s.n_points, s.n_points)
    assert adj.dtype == np.uint8
    assert adj.nnz == (2*n_edges + s.n_points)

    adj2 = me.get_immediate_adjacency(s, include_self=False)
    assert adj2.shape == (s.n_points, s.n_points)
    assert adj2.dtype == np.uint8
    assert adj2.nnz == (2 * n_edges)

    radj = me.get_ring_adjacency(s)
    assert radj.dtype == np.uint8
    assert (adj - radj).nnz == 0

    radj2 = me.get_ring_adjacency(s, include_self=False)
    assert radj2.dtype == np.uint8
    assert (adj2 - radj2).nnz == 0

    radj3 = me.get_ring_adjacency(s, n_ring=2, include_self=False)
    assert radj3.dtype == np.uint8
    assert (radj3 - adj2).nnz > 0

    d = me.get_immediate_distance(s)
    assert d.shape == (s.n_points, s.n_points)
    assert d.dtype == np.float
    assert d.nnz == adj2.nnz

    d2 = me.get_immediate_distance(s, metric='sqeuclidean')
    d_sq = d.copy()
    d_sq.data **= 2
    assert np.allclose(d_sq.A, d2.A)

    rd = me.get_ring_distance(s)
    assert rd.dtype == np.float
    assert np.allclose(d.A, rd.A)

    rd2 = me.get_ring_distance(s, n_ring=2)
    assert (rd2 - d).nnz > 0

    assert me.get_cell_neighbors(s).shape == (s.n_cells, s.n_cells)
    assert me.get_edges(s).shape == (n_edges, 2)
    assert me.get_edge_length(s).shape == (n_edges,)

    assert me.get_boundary_points(s).size == 0
    assert me.get_boundary_edges(s).size == 0
    assert me.get_boundary_cells(s).size == 0


def test_mesh_cluster():
    s = _generate_sphere()

    cl_size = 10
    nc = s.n_points // cl_size

    cl, cc = mcluster.cluster_points(s, n_clusters=nc, random_state=0)
    assert np.all(cl > 0)
    assert np.unique(cl).size == nc
    assert np.unique(cl).size == np.unique(cc).size - 1

    cl2 = mcluster.cluster_points(s, n_clusters=nc, with_centers=False,
                                  random_state=0)
    assert np.all(cl == cl2)

    cl3, _ = mcluster.cluster_points(s, n_clusters=cl_size, is_size=True,
                                     random_state=0)
    assert np.all(cl == cl3)

    cl4, cc4 = mcluster.cluster_points(s, n_clusters=nc, approach='ward',
                                       random_state=0)
    assert np.all(cl4 > 0)
    assert np.unique(cl4).size == nc
    assert np.unique(cl4).size == np.unique(cc4).size - 1

    sp = mcluster.sample_points_clustering(s, random_state=0)
    assert np.count_nonzero(sp) == int(s.n_points * 0.1)

    sp2 = mcluster.sample_points_clustering(s, keep=0.2, approach='ward',
                                            random_state=0)
    assert np.count_nonzero(sp2) == int(s.n_points * 0.2)


def test_array_operations():
    s = _generate_sphere()

    # Cell area
    area = aop.compute_cell_area(s)
    assert isinstance(area, np.ndarray)
    assert area.shape == (s.n_cells, )

    s2 = aop.compute_cell_area(s, append=True, key='CellArea')
    assert s is s2
    assert np.allclose(s2.CellData['CellArea'], area)

    # Cell centers
    centers = aop.compute_cell_center(s)
    assert isinstance(centers, np.ndarray)
    assert centers.shape == (s.n_cells, 3)

    s2 = aop.compute_cell_center(s, append=True, key='CellCenter')
    assert s is s2
    assert np.allclose(s2.CellData['CellCenter'], centers)

    # Adjacent cells
    n_adj = aop.get_n_adjacent_cells(s)
    assert isinstance(n_adj, np.ndarray)
    assert n_adj.shape == (s.n_points,)

    s2 = aop.get_n_adjacent_cells(s, append=True, key='NAdjCells')
    assert s is s2
    assert np.all(s2.PointData['NAdjCells'] == n_adj)

    # map cell data to point data
    area2 = aop.map_celldata_to_pointdata(s, area)
    area3 = aop.map_celldata_to_pointdata(s, 'CellArea', red_func='mean')
    assert area.dtype == area2.dtype
    assert area.dtype == area3.dtype
    assert np.allclose(area2, area3)

    area4 = aop.map_celldata_to_pointdata(s, 'CellArea', red_func='mean',
                                          dtype=np.float32)
    assert area4.dtype == np.float32

    for op in ['sum', 'mean', 'mode', 'one_third', 'min', 'max']:
        ap = aop.map_celldata_to_pointdata(s, 'CellArea', red_func=op)
        assert ap.shape == (s.n_points,)

        name = 'CellArea_{}'.format(op)
        s2 = aop.map_celldata_to_pointdata(s, 'CellArea', red_func=op,
                                           append=True, key=name)
        assert np.allclose(s2.PointData[name], ap)

    # map point data to cell  data
    fc = aop.map_pointdata_to_celldata(s, n_adj)
    fc2 = aop.map_pointdata_to_celldata(s, 'NAdjCells', red_func='mean')
    assert fc.dtype == fc2.dtype
    assert fc.dtype == fc2.dtype
    assert np.allclose(fc, fc2)

    fc3 = aop.map_pointdata_to_celldata(s, 'NAdjCells', red_func='mean',
                                        dtype=np.float32)
    assert fc3.dtype == np.float32

    for op in ['sum', 'mean', 'mode', 'one_third', 'min', 'max']:
        ac = aop.map_pointdata_to_celldata(s, 'NAdjCells', red_func=op)
        assert ac.shape == (s.n_cells,)

        name = 'NAdjCells_{}'.format(op)
        s2 = aop.map_pointdata_to_celldata(s, 'NAdjCells', red_func=op,
                                           append=True, key=name)
        assert np.allclose(s2.CellData[name], ac)

    # Point area
    area = aop.compute_point_area(s)
    assert isinstance(area, np.ndarray)
    assert area.shape == (s.n_points, )

    s2 = aop.compute_point_area(s, append=True, key='PointArea')
    assert s is s2
    assert np.allclose(s2.PointData['PointArea'], area)

    s2 = aop.compute_point_area(s, cell_area='CellArea', append=True,
                                key='PointArea2')
    assert s is s2
    assert np.allclose(s2.PointData['PointArea2'], area)

    # Connected components
    cc = mop.get_connected_components(s)
    assert cc.shape == (s.n_points, )
    assert np.unique(cc).size == 1

    s2 = mop.get_connected_components(s, append=True, key='components')
    assert s is s2
    assert np.all(cc == s2.PointData['components'])

    # labeling border
    labeling = (s.Points[:, 0] > s.Points[:, 0].mean()).astype(int)
    s.append_array(labeling, name='parc', at='p')

    border = aop.get_labeling_border(s, labeling)
    assert border.shape == (s.n_points, )
    assert np.unique(border).size == 2

    border2 = aop.get_labeling_border(s, 'parc')
    assert np.all(border == border2)

    # parcellation centroids
    cent = aop.get_parcellation_centroids(s, labeling, non_centroid=2)
    assert cent.shape == (s.n_points,)
    assert np.unique(cent).size == 3
    assert np.count_nonzero(cent == 0) == 1
    assert np.count_nonzero(cent == 1) == 1
    assert np.count_nonzero(cent == 2) == s.n_points - 2

    cent2 = aop.get_parcellation_centroids(s, 'parc', non_centroid=2)
    assert np.all(cent == cent2)

    # propagate labeling
    labeling2 = labeling.astype(np.float32)
    labeling2[:10] = np.nan
    s.append_array(labeling2, name='parc2', at='p')

    pl1 = aop.propagate_labeling(s, labeling2)
    assert pl1.shape == (s.n_points,)
    assert np.count_nonzero(np.isnan(pl1)) == 0
    assert np.all(np.unique(pl1) == np.array([0, 1]))

    pl2 = aop.propagate_labeling(s, 'parc2')
    assert np.all(pl1 == pl2)

    # smooth array
    for k in ['uniform', 'gaussian', 'inverse_distance']:
        sa = aop.smooth_array(s, n_adj, kernel=k)
        assert sa.shape == (s.n_points,)

        sa2 = aop.smooth_array(s, 'NAdjCells', kernel=k)
        assert np.all(sa == sa2)

    # resample pointdata
    s2 = wrap_vtk(vtk.vtkSphereSource, phiResolution=20)
    s2.Update()
    s2 = wrap_vtk(s2.output)

    rd = aop.resample_pointdata(s, s2, 'NAdjCells')
    assert rd.shape == (s2.n_points,)

    rd2 = aop.resample_pointdata(s, s2, 'NAdjCells', red_func='mean')
    assert np.all(rd == rd2)
