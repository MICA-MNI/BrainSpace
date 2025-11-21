
import pytest
import os
import shutil
import vtk
from brainspace.vtk_interface import wrap_vtk
from brainspace.mesh import mesh_io as mio

def _generate_sphere():
    s = vtk.vtkSphereSource()
    s.Update()
    return wrap_vtk(s.GetOutput())

@pytest.mark.parametrize('ext', ['pial', 'white', 'orig', 'sphere', 'inflated', 'smoothwm'])
def test_read_freesurfer_extensions(tmp_path, ext):
    # Generate a dummy surface
    s = _generate_sphere()
    
    # Write to a .fs file first (since write_surface supports .fs)
    # We use a temporary directory provided by pytest
    temp_dir = tmp_path
    fs_path = temp_dir / "test_surf.fs"
    target_path = temp_dir / f"test_surf.{ext}"
    
    # Write as 'fs' type
    mio.write_surface(s, str(fs_path), otype='fs')
    
    # Rename to the target extension
    shutil.move(str(fs_path), str(target_path))
    
    # Try reading it back
    # This should verify that read_surface accepts the extension and uses the correct reader
    s_read = mio.read_surface(str(target_path))
    
    assert s_read is not None
    assert s_read.GetNumberOfPoints() == s.GetNumberOfPoints()
    assert s_read.GetNumberOfCells() == s.GetNumberOfCells()
