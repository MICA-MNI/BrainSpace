"""Regression tests for contour geometry, transforms and surface immutability."""
import numpy as np
import pytest
from vtk.util.numpy_support import vtk_to_numpy

from brainspace.mesh.mesh_creation import build_polydata
from brainspace.plotting import (make_contour_actor, add_contours_to_plotter,
                                 build_plotter, plot_hemispheres)
from brainspace.plotting.base import Plotter
from brainspace.plotting.surface_plotting import orientations


def _surface():
    return build_polydata(np.array([[0., 0., 0.], [1., 0., 0.],
                                    [1., 1., 0.], [0., 1., 0.]]),
                          np.array([[0, 1, 2], [0, 2, 3]]))


def _matrix(actor):
    m = actor.GetMatrix()
    return np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)])


def test_contour_geometry_and_input_unchanged():
    surf = _surface()
    surf.append_array(np.arange(4.), name='original', at='p')
    before = list(surf.PointData.keys())
    actor = make_contour_actor(surf, [0, 1, 1, 0], color=(1, 0, 0),
                               line_width=4, alpha=0.6)
    actor.GetMapper().Update()
    output = actor.GetMapper().GetInput()
    points = vtk_to_numpy(output.GetPoints().GetData())
    np.testing.assert_allclose(points[:, 0], 0.5)
    assert output.GetNumberOfLines() == 2
    assert actor.GetProperty().GetOpacity() == 0.6
    assert actor.GetProperty().GetLineWidth() == 4
    assert list(surf.PointData.keys()) == before
    np.testing.assert_array_equal(surf.PointData['original'], np.arange(4.))


@pytest.mark.parametrize('view', list(orientations))
def test_contour_matches_custom_surface_transform_and_colorbar(view):
    surf = _surface()
    surf.append_array(np.arange(4.), name='map', at='p')
    p = build_plotter({'s': surf}, [['s', None]], array_name='map',
                      view=view, color_bar='right', label_text={'top': ['map', '']},
                      actor__position=(2, 3, 4), actor__scale=(2, 1, 3),
                      offscreen=True)
    try:
        actors = add_contours_to_plotter(p, {'s': [0, 1, 1, 0]}, alpha=0.25)
        assert len(actors) == 2
        for actor in actors:
            np.testing.assert_allclose(_matrix(actor),
                                       _matrix(p._surface_entries[0][3].VTKObject))
        actors[0].GetMapper().Update()
        fill = actors[0].GetMapper().GetInput()
        assert fill.GetNumberOfCells() > 0
        assert np.min(vtk_to_numpy(fill.GetPoints().GetData())[:, 0]) >= 0.5
    finally:
        p.close()


@pytest.mark.parametrize('layout_style', ['row', 'grid'])
@pytest.mark.parametrize('array_name', [None, 'map', ['map', 'map']])
def test_hemisphere_contours_all_rows_and_layouts(layout_style, array_name):
    left, right = _surface(), _surface()
    for surf in [left, right]:
        surf.append_array(np.arange(4.), name='map', at='p')
    p = plot_hemispheres(left, right, array_name=array_name,
                         layout_style=layout_style, contour=[0, 1, 1, 0] * 2,
                         contour_alpha=0.25, color_bar=True,
                         return_plotter=True, offscreen=True,
                         suppress_warnings=True)
    try:
        assert len(p._surface_entries) == (8 if isinstance(array_name, list) else 4)
        for _, _, renderer, _ in p._surface_entries:
            assert renderer.VTKObject.GetActors().GetNumberOfItems() == 3
    finally:
        p.close()


def test_invalid_contour_rejected_before_adding_actors():
    left, right = _surface(), _surface()
    p = build_plotter({'lh': left, 'rh': right}, [['lh', 'rh']], offscreen=True)
    try:
        with pytest.raises(ValueError, match='one value per'):
            add_contours_to_plotter(p, {'lh': [0, 1, 1, 0], 'rh': [0, 1]})
        for _, _, renderer, _ in p._surface_entries:
            assert renderer.VTKObject.GetActors().GetNumberOfItems() == 1
        with pytest.raises(ValueError, match='n_lh'):
            add_contours_to_plotter(p, [0, 1])
    finally:
        p.close()
    with pytest.raises(ValueError, match='finite'):
        make_contour_actor(left, [0, 1, np.nan, 0])
    with pytest.raises(ValueError, match='opacity'):
        make_contour_actor(left, [0, 1, 1, 0], alpha=2)


def test_explicit_hemisphere_helper_and_point_data():
    left, right = _surface(), _surface()
    for surf in [left, right]:
        surf.append_array(np.array([0., 1., 1., 0.]), name='mask', at='p')
    p = build_plotter({'lh': left, 'rh': right}, [['lh', 'rh']],
                      contour='mask', offscreen=True)
    try:
        actors = add_contours_to_plotter(p, left, right, [0, 1, 1, 0] * 2)
        assert len(actors) == 2
    finally:
        p.close()
