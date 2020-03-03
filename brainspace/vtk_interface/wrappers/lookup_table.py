"""
Wrappers for VTK lookup tables.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


from vtk.util.vtkConstants import VTK_STRING, VTK_UNSIGNED_CHAR

from .base import BSVTKObjectWrapper
from ..decorators import unwrap_input


class BSScalarsToColors(BSVTKObjectWrapper):
    """Wrapper for vtkScalarsToColors."""

    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject, **kwargs)

    @unwrap_input(1, 2, vtype={1: None, 2: VTK_STRING})
    def SetAnnotations(self, values, annotations):
        self.VTKObject.SetAnnotations(values, annotations)


class BSLookupTable(BSScalarsToColors):
    """Wrapper for vtkLookupTable."""

    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)

    @unwrap_input(1, vtype={1: VTK_UNSIGNED_CHAR})
    def SetTable(self, table):
        self.VTKObject.SetTable(table)

    def SetNumberOfColors(self, n):
        # SetNumberOfColors() has no effect after the table has been built
        # Use SetNumberOfTableValues() instead
        self.VTKObject.SetNumberOfTableValues(n)

    def GetNumberOfColors(self):
        # SetNumberOfColors() has no effect after the table has been built
        # Use SetNumberOfTableValues() instead
        return self.VTKObject.GetNumberOfTableValues()

    @property
    def n_values(self):
        """int: Returns number of table values."""
        return self.VTKObject.GetNumberOfTableValues()

    @n_values.setter
    def n_values(self, n):
        self.VTKObject.SetNumberOfTableValues(n)


class BSLookupTableWithEnabling(BSLookupTable):
    """Wrapper for vtkLookupTableWithEnabling."""

    @unwrap_input(1, vtype=None)
    def SetEnabledArray(self, array):
        self.VTKObject.SetEnabledArray(array)


class BSWindowLevelLookupTable(BSLookupTable):
    """Wrapper for vtkWindowLevelLookupTable."""
    pass


class BSColorTransferFunction(BSScalarsToColors):
    """Wrapper for vtkColorTransferFunction."""
    pass


class BSDiscretizableColorTransferFunction(BSColorTransferFunction):
    """Wrapper for vtkDiscretizableColorTransferFunction."""
    pass
