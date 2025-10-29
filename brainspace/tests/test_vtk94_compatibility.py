"""Test VTK 9.4+ compatibility.

This module tests compatibility with VTK 9.4+ where the __vtkname__ attribute
is not available on some classes like PointSet.

See: https://github.com/MICA-MNI/BrainSpace/issues/128
"""

import pytest
import numpy as np

import vtk

from brainspace.vtk_interface import wrap_vtk, is_wrapper, is_vtk
from brainspace.vtk_interface.wrappers import BSVTKObjectWrapper, BSPolyData
from brainspace.vtk_interface.wrappers.utils import get_vtk_name
from brainspace.vtk_interface.pipeline import serial_connect


def test_get_vtk_name_basic():
    """Test get_vtk_name with various VTK objects."""
    # Test with VTK class
    name = get_vtk_name(vtk.vtkPolyData)
    assert name == 'vtkPolyData'

    # Test with VTK instance
    pd = vtk.vtkPolyData()
    name = get_vtk_name(pd)
    assert name == 'vtkPolyData'

    # Test with VTK algorithm
    sphere = vtk.vtkSphereSource()
    name = get_vtk_name(sphere)
    assert name == 'vtkSphereSource'

    # Test with VTK mapper - note: may return subclass name like vtkOpenGLPolyDataMapper
    mapper = vtk.vtkPolyDataMapper()
    name = get_vtk_name(mapper)
    assert 'Mapper' in name and name.startswith('vtk')


def test_get_vtk_name_without_vtkname():
    """Test get_vtk_name with classes that may not have __vtkname__."""
    # Create a mock VTK-like class without __vtkname__
    # Note: In Python 3, the class __name__ attribute is set automatically
    # and cannot be overridden in the class definition
    class vtkMockClass:
        pass

    mock_obj = vtkMockClass()
    name = get_vtk_name(mock_obj)
    assert name == 'vtkMockClass'

    # Test with class itself
    name = get_vtk_name(vtkMockClass)
    assert name == 'vtkMockClass'


def test_wrapping_with_vtk94():
    """Test basic wrapping functionality with VTK 9.4+ compatibility."""
    # Test wrapping various VTK objects
    pd = vtk.vtkPolyData()
    wrapped_pd = wrap_vtk(pd)
    assert isinstance(wrapped_pd, BSPolyData)
    assert is_wrapper(wrapped_pd)

    # Test with sphere source
    sphere = wrap_vtk(vtk.vtkSphereSource, radius=5.0)
    assert sphere.VTKObject.GetRadius() == 5.0

    # Test with filter
    smooth = wrap_vtk(vtk.vtkSmoothPolyDataFilter, numberOfIterations=10)
    assert smooth.VTKObject.GetNumberOfIterations() == 10


def test_vtk_map_access():
    """Test vtk_map property access with VTK 9.4+ compatibility."""
    sphere = wrap_vtk(vtk.vtkSphereSource)

    # Test that vtk_map is accessible
    vtk_map = sphere.vtk_map
    assert 'set' in vtk_map
    assert 'get' in vtk_map

    # Test that we can access setter methods
    assert 'radius' in vtk_map['set']

    # Test that we can access getter methods
    assert 'radius' in vtk_map['get']


def test_wrapping_hierarchy():
    """Test wrapping objects with class hierarchy traversal."""
    # Create a VTK object and wrap it
    pd = vtk.vtkPolyData()
    wrapped = wrap_vtk(pd)

    assert isinstance(wrapped, BSPolyData)
    assert isinstance(wrapped, BSVTKObjectWrapper)

    # Test with a subclass
    ugrid = vtk.vtkUnstructuredGrid()
    wrapped_ugrid = wrap_vtk(ugrid)
    assert is_wrapper(wrapped_ugrid)


def test_pipeline_with_vtk94():
    """Test pipeline operations with VTK 9.4+ compatibility."""
    # Create a simple pipeline
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(3.0)

    smooth = vtk.vtkSmoothPolyDataFilter()
    smooth.SetNumberOfIterations(20)

    # Test serial connection
    output = serial_connect(sphere, smooth)
    assert isinstance(output, BSPolyData)
    assert output.n_points > 0

    # Test with wrapped objects
    sphere_wrapped = wrap_vtk(vtk.vtkSphereSource, radius=2.5)
    smooth_wrapped = wrap_vtk(vtk.vtkSmoothPolyDataFilter, numberOfIterations=15)

    output = serial_connect(sphere_wrapped, smooth_wrapped)
    assert isinstance(output, BSPolyData)
    assert output.n_points > 0


def test_error_messages_with_vtk94():
    """Test that error messages work correctly with VTK 9.4+ compatibility."""
    from brainspace.vtk_interface.pipeline import connect

    sphere = wrap_vtk(vtk.vtkSphereSource)
    smooth = wrap_vtk(vtk.vtkSmoothPolyDataFilter)

    # Test error with invalid port numbers
    with pytest.raises(ValueError) as excinfo:
        connect(sphere, smooth, port0=10)  # Invalid output port
    assert "output ports" in str(excinfo.value).lower()

    with pytest.raises(ValueError) as excinfo:
        connect(sphere, smooth, port1=10)  # Invalid input port
    assert "input ports" in str(excinfo.value).lower()


def test_setvtk_getvtk_with_vtk94():
    """Test setVTK and getVTK methods with VTK 9.4+ compatibility."""
    sphere = wrap_vtk(vtk.vtkSphereSource)

    # Test setVTK
    sphere.setVTK(radius=7.5, thetaResolution=16, phiResolution=12)
    assert sphere.VTKObject.GetRadius() == 7.5
    assert sphere.VTKObject.GetThetaResolution() == 16
    assert sphere.VTKObject.GetPhiResolution() == 12

    # Test getVTK
    values = sphere.getVTK('radius', 'thetaResolution', phiResolution=None)
    assert values['radius'] == 7.5
    assert values['thetaResolution'] == 16
    assert values['phiResolution'] == 12


def test_multiple_vtk_objects():
    """Test wrapping multiple different VTK objects."""
    vtk_objects = [
        vtk.vtkPolyData(),
        vtk.vtkSphereSource(),
        vtk.vtkSmoothPolyDataFilter(),
        vtk.vtkPolyDataMapper(),
        vtk.vtkActor(),
        vtk.vtkImageData(),
        vtk.vtkUnstructuredGrid(),
    ]

    for obj in vtk_objects:
        wrapped = wrap_vtk(obj)
        assert is_wrapper(wrapped)
        # Verify vtk_name can be retrieved
        name = get_vtk_name(obj)
        assert name.startswith('vtk')
        # Verify vtk_map is accessible
        vtk_map = wrapped.vtk_map
        assert isinstance(vtk_map, dict)


def test_backwards_compatibility():
    """Test that existing code still works (backwards compatibility)."""
    # Test that code written for older VTK versions still works
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(4.0)
    sphere.Update()

    # Wrap the output
    output = wrap_vtk(sphere.GetOutput())
    assert isinstance(output, BSPolyData)

    # Test property access
    assert output.n_points > 0
    assert output.n_cells > 0

    # Test that we can get points and cells
    points = output.GetPoints()
    assert points is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
