"""Test copy methods for VTK wrappers."""

import copy
import pytest
import numpy as np
import vtk

from brainspace.vtk_interface import wrap_vtk
from brainspace.vtk_interface.wrappers import BSPolyData


def test_copy_method_basic():
    """Test basic .copy() method functionality."""
    # Create a PolyData object with some points
    pd = wrap_vtk(vtk.vtkPolyData())
    
    points = vtk.vtkPoints()
    points.InsertNextPoint(0, 0, 0)
    points.InsertNextPoint(1, 0, 0)
    points.InsertNextPoint(1, 1, 0)
    pd.SetPoints(points)
    
    # Create a copy
    pd_copy = pd.copy()
    
    # Verify it's a new object
    assert pd is not pd_copy
    assert pd.VTKObject is not pd_copy.VTKObject
    
    # Verify it has the same type
    assert type(pd) == type(pd_copy)
    assert isinstance(pd_copy, BSPolyData)
    
    # Verify it has the same data
    assert pd.GetNumberOfPoints() == pd_copy.GetNumberOfPoints()
    assert pd_copy.GetNumberOfPoints() == 3


def test_copy_module_copy():
    """Test Python's copy.copy() function."""
    pd = wrap_vtk(vtk.vtkPolyData())
    
    points = vtk.vtkPoints()
    points.InsertNextPoint(0, 0, 0)
    points.InsertNextPoint(1, 0, 0)
    pd.SetPoints(points)
    
    # Use copy.copy()
    pd_copy = copy.copy(pd)
    
    assert pd is not pd_copy
    assert type(pd) == type(pd_copy)
    assert pd.GetNumberOfPoints() == pd_copy.GetNumberOfPoints()


def test_copy_module_deepcopy():
    """Test Python's copy.deepcopy() function."""
    pd = wrap_vtk(vtk.vtkPolyData())
    
    points = vtk.vtkPoints()
    points.InsertNextPoint(0, 0, 0)
    points.InsertNextPoint(1, 0, 0)
    pd.SetPoints(points)
    
    # Use copy.deepcopy()
    pd_copy = copy.deepcopy(pd)
    
    assert pd is not pd_copy
    assert type(pd) == type(pd_copy)
    assert pd.GetNumberOfPoints() == pd_copy.GetNumberOfPoints()


def test_shallow_copy_shares_data():
    """Test that shallow copy shares data with original."""
    pd = wrap_vtk(vtk.vtkPolyData())
    
    # Add points
    points = vtk.vtkPoints()
    points.InsertNextPoint(0, 0, 0)
    points.InsertNextPoint(1, 0, 0)
    points.InsertNextPoint(1, 1, 0)
    pd.SetPoints(points)
    
    # Add point data
    point_data = vtk.vtkFloatArray()
    point_data.SetName('test_data')
    point_data.SetNumberOfTuples(3)
    point_data.SetValue(0, 1.0)
    point_data.SetValue(1, 2.0)
    point_data.SetValue(2, 3.0)
    pd.GetPointData().AddArray(point_data)
    
    # Create shallow copy
    pd_shallow = pd.copy(deep=False)
    
    # Verify initial values match
    assert pd_shallow.GetPointData().GetArray('test_data').GetValue(0) == 1.0
    
    # Modify original - shallow copy should see the change
    pd.GetPointData().GetArray('test_data').SetValue(0, 999.0)
    
    assert pd.GetPointData().GetArray('test_data').GetValue(0) == 999.0
    assert pd_shallow.GetPointData().GetArray('test_data').GetValue(0) == 999.0


def test_deep_copy_independent_data():
    """Test that deep copy has independent data from original."""
    pd = wrap_vtk(vtk.vtkPolyData())
    
    # Add points
    points = vtk.vtkPoints()
    points.InsertNextPoint(0, 0, 0)
    points.InsertNextPoint(1, 0, 0)
    points.InsertNextPoint(1, 1, 0)
    pd.SetPoints(points)
    
    # Add point data
    point_data = vtk.vtkFloatArray()
    point_data.SetName('test_data')
    point_data.SetNumberOfTuples(3)
    point_data.SetValue(0, 1.0)
    point_data.SetValue(1, 2.0)
    point_data.SetValue(2, 3.0)
    pd.GetPointData().AddArray(point_data)
    
    # Create deep copy
    pd_deep = pd.copy(deep=True)
    
    # Verify initial values match
    assert pd_deep.GetPointData().GetArray('test_data').GetValue(0) == 1.0
    
    # Modify original - deep copy should NOT see the change
    pd.GetPointData().GetArray('test_data').SetValue(0, 888.0)
    
    assert pd.GetPointData().GetArray('test_data').GetValue(0) == 888.0
    assert pd_deep.GetPointData().GetArray('test_data').GetValue(0) == 1.0


def test_copy_with_cells():
    """Test copying PolyData with cells."""
    pd = wrap_vtk(vtk.vtkPolyData())
    
    # Add points
    points = vtk.vtkPoints()
    points.InsertNextPoint(0, 0, 0)
    points.InsertNextPoint(1, 0, 0)
    points.InsertNextPoint(0.5, 1, 0)
    pd.SetPoints(points)
    
    # Add a triangle
    triangle = vtk.vtkTriangle()
    triangle.GetPointIds().SetId(0, 0)
    triangle.GetPointIds().SetId(1, 1)
    triangle.GetPointIds().SetId(2, 2)
    
    triangles = vtk.vtkCellArray()
    triangles.InsertNextCell(triangle)
    pd.SetPolys(triangles)
    
    # Copy
    pd_copy = pd.copy()
    
    # Verify cells are copied
    assert pd.GetNumberOfCells() == pd_copy.GetNumberOfCells()
    assert pd_copy.GetNumberOfCells() == 1
    assert pd.GetNumberOfPoints() == pd_copy.GetNumberOfPoints()


def test_copy_different_vtk_types():
    """Test copying different VTK object types."""
    # Test with vtkSphereSource output
    sphere = vtk.vtkSphereSource()
    sphere.Update()
    pd = wrap_vtk(sphere.GetOutput())
    
    pd_copy = pd.copy()
    
    assert pd is not pd_copy
    assert pd.GetNumberOfPoints() == pd_copy.GetNumberOfPoints()
    assert pd.GetNumberOfCells() == pd_copy.GetNumberOfCells()
    
    # Test with other data types
    ug = wrap_vtk(vtk.vtkUnstructuredGrid())
    ug_copy = ug.copy()
    
    assert ug is not ug_copy
    assert type(ug) == type(ug_copy)
