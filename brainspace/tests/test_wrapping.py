""" Test vtk wrapping interface """

import pytest

import numpy as np

import vtk
from vtk.util.vtkConstants import VTK_TRIANGLE, VTK_LINE, VTK_VERTEX

from brainspace.vtk_interface import checks, wrap_vtk, is_wrapper, is_vtk
from brainspace.vtk_interface.pipeline import serial_connect
from brainspace.vtk_interface.wrappers import (BSVTKObjectWrapper, BSPolyData,
                                               BSPolyDataMapper, BSActor)
from brainspace.vtk_interface.wrappers.algorithm import BSAlgorithm
from brainspace.vtk_interface.wrappers.data_object import BSDataSet
from brainspace.mesh import mesh_creation as mc


def test_cell_types():
    ss = vtk.vtkSphereSource()
    ss.Update()
    st = wrap_vtk(ss.GetOutput())
    sl = mc.to_lines(st)
    sv = mc.to_vertex(st)

    assert checks.get_cell_types(st) == np.array([VTK_TRIANGLE])
    assert checks.get_cell_types(st.VTKObject) == np.array([VTK_TRIANGLE])
    assert checks.get_cell_types(sl) == np.array([VTK_LINE])
    assert checks.get_cell_types(sv) == np.array([VTK_VERTEX])

    assert checks.get_number_of_cell_types(st) == 1
    assert checks.get_number_of_cell_types(st.VTKObject) == 1
    assert checks.get_number_of_cell_types(sl) == 1
    assert checks.get_number_of_cell_types(sv) == 1

    assert checks.has_unique_cell_type(st)
    assert checks.has_unique_cell_type(st.VTKObject)
    assert checks.has_unique_cell_type(sl)
    assert checks.has_unique_cell_type(sv)

    assert checks.has_only_triangle(st)
    assert checks.has_only_triangle(st.VTKObject)
    assert checks.has_only_line(sl)
    assert checks.has_only_vertex(sv)

    ss2 = vtk.vtkSphereSource()
    ss2.SetRadius(3)
    ss2.Update()
    s2 = ss2.GetOutput()

    app = vtk.vtkAppendPolyData()
    app.AddInputData(sl.VTKObject)
    app.AddInputData(s2)
    app.Update()
    spl = wrap_vtk(app.GetOutput())

    cell_types = np.sort([VTK_TRIANGLE, VTK_LINE])
    assert np.all(checks.get_cell_types(spl) == cell_types)
    assert checks.get_number_of_cell_types(spl) == cell_types.size
    assert checks.has_unique_cell_type(spl) is False
    assert checks.has_only_triangle(spl) is False
    assert checks.has_only_line(spl) is False
    assert checks.has_only_vertex(spl) is False


def test_basic_wrapping():

    assert is_vtk(vtk.vtkPolyData) is False
    assert is_vtk(vtk.vtkPolyData()) is True
    assert is_vtk(None) is False

    ws = wrap_vtk(vtk.vtkPolyData())
    assert is_wrapper(ws.VTKObject) is False
    assert is_wrapper(ws) is True
    assert is_wrapper(None) is False

    assert wrap_vtk(None) is None
    assert wrap_vtk(ws) is ws

    # test source
    s = wrap_vtk(vtk.vtkSphereSource, radius=3)
    assert isinstance(s, BSAlgorithm)
    assert s.is_source
    assert s.VTKObject.GetRadius() == 3

    s.setVTK(radius=4.5)
    assert s.VTKObject.GetRadius() == 4.5

    s.radius = 2.5
    assert s.VTKObject.GetRadius() == 2.5

    with pytest.raises(Exception):
        s.radius2 = 0

    # test filter (no source no sink)
    s = wrap_vtk(vtk.vtkSmoothPolyDataFilter, numberOfIterations=2)
    assert isinstance(s, BSAlgorithm)
    assert s.is_filter
    assert s.VTKObject.GetNumberOfIterations() == 2

    # test sink
    s = wrap_vtk(vtk.vtkXMLPolyDataWriter, filename='some/path')
    assert isinstance(s, BSAlgorithm)
    assert s.is_sink
    assert s.VTKObject.GetFileName() == 'some/path'

    # test vtkPolyDataMapper
    s = wrap_vtk(vtk.vtkPolyDataMapper, arrayName='array_name',
                 scalarMode='UseCellData')
    assert isinstance(s, BSAlgorithm)
    assert isinstance(s, BSPolyDataMapper)
    assert s.is_sink
    assert s.VTKObject.GetScalarModeAsString() == 'UseCellData'
    assert s.VTKObject.GetArrayName() == 'array_name'
    assert s.VTKObject.GetArrayAccessMode() == 1

    # test change in access mode
    s.arrayid = 3
    assert s.VTKObject.GetArrayId() == 3
    assert s.VTKObject.GetArrayAccessMode() == 0

    # test actor access to property
    s = wrap_vtk(vtk.vtkActor, opacity=0.5, interpolation='phong')
    assert isinstance(s, BSActor)
    assert s.VTKObject.GetProperty().GetOpacity() == 0.5
    assert s.property.opacity == 0.5
    assert s.opacity == 0.5

    # test implemented wrapper
    s = wrap_vtk(vtk.vtkPolyData)
    assert isinstance(s, BSPolyData)

    # test not implemented --> default to superclass
    s = wrap_vtk(vtk.vtkImageData)
    assert isinstance(s, BSDataSet)

    # test wrappers to create objects
    with pytest.raises(TypeError):
        w = BSVTKObjectWrapper()

    with pytest.raises(AttributeError):
        w = BSVTKObjectWrapper(None)

    pd = BSPolyData()
    assert isinstance(pd.VTKObject, vtk.vtkPolyData)
    a = BSActor()
    assert isinstance(a.VTKObject, vtk.vtkActor)
    m = BSPolyDataMapper()
    assert isinstance(m.VTKObject, vtk.vtkPolyDataMapper)


def test_pipeline():
    # check defaults
    s = vtk.vtkSphereSource()
    f = vtk.vtkSmoothPolyDataFilter()
    out = serial_connect(s, f)
    assert isinstance(out, BSPolyData)
    assert out.n_points > 0

    # check update filter
    s = vtk.vtkSphereSource()
    f = vtk.vtkSmoothPolyDataFilter()
    out = serial_connect(s, f, as_data=False)
    assert isinstance(out, BSAlgorithm)
    assert out.GetOutput().GetNumberOfPoints() > 0

    # check filter no update
    s = vtk.vtkSphereSource()
    f = vtk.vtkSmoothPolyDataFilter()
    out = serial_connect(s, f, as_data=False, update=False)
    assert isinstance(out, BSAlgorithm)
    assert out.GetOutput().GetNumberOfPoints() == 0

    # check non-existing port
    s = vtk.vtkSphereSource()
    f = vtk.vtkSmoothPolyDataFilter()
    out = serial_connect(s, f, port=1)
    assert out is None

    # check get all possible ports
    s = vtk.vtkSphereSource()
    f = vtk.vtkSmoothPolyDataFilter()
    out = serial_connect(s, f, port=-1)
    assert isinstance(out, list)
    assert len(out) == f.GetNumberOfOutputPorts()
    assert isinstance(out[0], BSPolyData)
    assert out[0].n_points > 0

    # check accept wrappers
    s = wrap_vtk(vtk.vtkSphereSource)
    f = wrap_vtk(vtk.vtkSmoothPolyDataFilter)
    assert isinstance(serial_connect(s, f), BSPolyData)
