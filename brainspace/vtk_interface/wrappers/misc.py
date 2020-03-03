"""
Misc wrappers for some VTK classes.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


from vtk.util.vtkConstants import VTK_ID_TYPE

from .base import BSVTKObjectWrapper
from ..decorators import unwrap_input, wrap_output


class BSCollection(BSVTKObjectWrapper):
    """Wrapper for vtkCollection."""
    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject, **kwargs)

    @property
    def n_items(self):
        return self.VTKObject.GetNumberOfItems()

    @wrap_output
    def __getitem__(self, i):
        if i < 0:
            i += self.n_items
        return self.VTKObject.GetItemAsObject(i)

    def __setitem__(self, i, obj):
        if i < 0:
            i += self.n_items
        self.VTKObject.ReplaceItem(i, obj)


class BSPropCollection(BSCollection):
    """Wrapper for vtkPropCollection."""
    pass


class BSActor2DCollection(BSPropCollection):
    """Wrapper for vtkActor2DCollection."""
    pass


class BSActorCollection(BSPropCollection):
    """Wrapper for vtkActorCollection."""
    pass


class BSProp3DCollection(BSPropCollection):
    """Wrapper for vtkProp3DCollection."""
    pass


class BSMapperCollection(BSCollection):
    """Wrapper for vtkMapperCollection."""
    pass


class BSRendererCollection(BSCollection):
    """Wrapper for vtkRendererCollection."""
    pass


class BSPolyDataCollection(BSCollection):
    """Wrapper for vtkPolyDataCollection."""
    pass


class BSTextPropertyCollection(BSCollection):
    """Wrapper for vtkTextPropertyCollection."""
    pass


############################################################
# Coordinate object
############################################################
class BSCoordinate(BSVTKObjectWrapper):
    """Wrapper for vtkCoordinate."""

    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject, **kwargs)


############################################################
# Cell Array
############################################################
class BSCellArray(BSVTKObjectWrapper):
    """Wrapper for vtkCellArray."""

    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)

    @unwrap_input(2, vtype={2: VTK_ID_TYPE})
    def SetCells(self, n_cells, cells):
        self.VTKObject.SetCells(n_cells, cells)


############################################################
# GL2PS Exporter
############################################################
class BSGL2PSExporter(BSVTKObjectWrapper):
    """Wrapper for vtkGL2PSExporter."""

    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)

