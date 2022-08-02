"""
Wrappers for VTK algorithms and mappers.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


from vtk.util.vtkConstants import VTK_DOUBLE

from .base import BSVTKObjectWrapper, wrap_vtk
from .property import BSTextProperty
from .lookup_table import (BSLookupTable, BSLookupTableWithEnabling,
                           BSWindowLevelLookupTable, BSColorTransferFunction,
                           BSDiscretizableColorTransferFunction)
from ..decorators import unwrap_input


class BSAlgorithm(BSVTKObjectWrapper):
    """Wrapper for vtkAlgorithm."""
    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)

    @property
    def nip(self):
        """int: Returns number of input ports"""
        return self.GetNumberOfInputPorts()

    @property
    def nop(self):
        """int: Returns number of output ports"""
        return self.GetNumberOfOutputPorts()

    @property
    def nic(self):
        """int: Returns number of total input connections"""
        return self.GetTotalNumberOfInputConnections()

    @property
    def is_source(self):
        """bool: Returns True if self is a source. False, otherwise."""
        return self.nip == 0

    @property
    def is_sink(self):
        """bool: Returns True if self is a sink. False, otherwise."""
        return self.nop == 0

    @property
    def is_filter(self):
        """bool: Returns True if self is a filter. False, otherwise.

        A filter that is not a source nor a sink.
        """
        return not (self.is_source and self.is_sink)


class LUTMixin:

    def SetLookupTable(self, obj=None, **kwargs):
        """Set lookup table.

        Wraps the `SetLookupTable` method of `vtkMapper` to accept a
        `vtkLookupTable` or BSLookupTable.

        Parameters
        ----------
        obj : vtkLookupTable or BSLookupTable, optional
            Lookup table. If None, a LookupTable is created.
            Default is None.
        kwargs : optional keyword arguments
            Arguments are used to set the lookup table.
        """

        if obj is None:
            obj = BSLookupTable(**kwargs)
        else:
            obj = wrap_vtk(obj, **kwargs)
        self.VTKObject.SetLookupTable(obj.VTKObject)
        return obj

    def SetLookupTableWithEnabling(self, obj=None, **kwargs):
        """Set lookup table using a LookupTableWithEnabling.

        Parameters
        ----------
        obj : vtkLookupTableWithEnabling or BSLookupTableWithEnabling, optional
            Lookup table. If None, the lut is created.
            Default is None.
        kwargs : optional keyword arguments
            Arguments are used to set the lookup table.
        """

        obj = BSLookupTableWithEnabling(vtkobject=obj, **kwargs)
        self.VTKObject.SetLookupTable(obj.VTKObject)
        return obj

    def SetWindowLevelLookupTable(self, obj=None, **kwargs):
        """Set lookup table using a WindowLevelLookupTable.

        Parameters
        ----------
        obj : vtkWindowLevelLookupTable or BSWindowLevelLookupTable, optional
            Lookup table. If None, the lut is created.
            Default is None.
        kwargs : optional keyword arguments
            Arguments are used to set the lookup table.
        """

        obj = BSWindowLevelLookupTable(vtkobject=obj, **kwargs)
        self.VTKObject.SetLookupTable(obj.VTKObject)
        return obj

    def SetColorTransferFunction(self, obj=None, **kwargs):
        """Set lookup table using a ColorTransferFunction.

        Parameters
        ----------
        obj : vtkColorTransferFunction or BSColorTransferFunction, optional
            Lookup table. If None, the color transfer function is created.
            Default is None.
        kwargs : optional keyword arguments
            Arguments are used to set the lookup table.
        """

        obj = BSColorTransferFunction(vtkobject=obj, **kwargs)
        self.VTKObject.SetLookupTable(obj.VTKObject)
        return obj

    def SetDiscretizableColorTransferFunction(self, obj=None, **kwargs):
        """Set lookup table using a DiscretizableColorTransferFunction.

        Parameters
        ----------
        obj : vtkDiscretizableColorTransferFunction or
            BSDiscretizableColorTransferFunction, optional
            Lookup table. If None, the discretizable color transfer function
            is created. Default is None.
        kwargs : optional keyword arguments
            Arguments are used to set the lookup table.
        """

        obj = BSDiscretizableColorTransferFunction(vtkobject=obj, **kwargs)
        self.VTKObject.SetLookupTable(obj.VTKObject)
        return obj


class BSAbstractMapper(BSAlgorithm):
    """Wrapper for vtkAbstractMapper."""
    def __init__(self, vtkobject, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)


class BSAbstractMapper3D(BSAbstractMapper):
    """Wrapper for vtkAbstractMapper."""
    pass


class BSMapper(BSAbstractMapper3D, LUTMixin):
    """Wrapper for vtkMapper."""
    def __init__(self, vtkobject, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)

    def SetArrayName(self, name):
        """Set array id.

        Wraps the `SetArrayName` method of `vtkMapper` such that the access
        mode is changed to accept setting the array name.

        Parameters
        ----------
        name : str
            Array name.
        """
        self.VTKObject.SetArrayAccessMode(1)
        self.VTKObject.SetArrayName(name)

    def SetArrayId(self, idx):
        """Set array id.

        Wraps the `SetArrayId` method of `vtkMapper` such that the access
        mode is changed to accept setting the array id.

        Parameters
        ----------
        idx : int
            Array id.
        """
        self.VTKObject.SetArrayAccessMode(0)
        self.VTKObject.SetArrayId(idx)


class BSDataSetMapper(BSMapper):
    """Wrapper for vtkDataSetMapper."""
    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)


class BSPolyDataMapper(BSMapper):
    """Wrapper for vtkPolyDataMapper."""
    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)


class BSLabeledContourMapper(BSMapper):
    """Wrapper for vtkLabeledContourMapper."""
    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)

    def SetTextProperty(self, obj=None, **kwargs):
        """Set text property.

        Wraps the `SetTextProperty` method of `vtkLabeledContourMapper` to
        accept a `vtkTextProperty` or BSTextProperty.

        Parameters
        ----------
        obj : vtkTextProperty or BSTextProperty, optional
            Label text property. If None, the property is created.
            Default is None.
        kwargs : optional keyword arguments
            Arguments are used to set the property.
        """

        obj = BSTextProperty(vtkobject=obj, **kwargs)
        self.VTKObject.SetTextProperty(obj.VTKObject)
        return obj

    @unwrap_input(1, vtype={1: VTK_DOUBLE})
    def SetTextPropertyMapping(self, mapping):
        self.VTKObject.SetTextPropertyMapping(mapping)


class BSMapper2D(BSAbstractMapper):
    """Wrapper for vtkMapper2D."""
    def __init__(self, vtkobject, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)


class BSLabeledDataMapper(BSMapper2D):
    """Wrapper for vtkLabeledDataMapper."""
    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)


class BSLabelPlacementMapper(BSMapper2D):
    """Wrapper for vtkLabelPlacementMapper."""
    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)


class BSPolyDataMapper2D(BSMapper2D, LUTMixin):
    """Wrapper for vtkPolyDataMapper2D."""
    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)


class BSTextMapper2D(BSMapper2D):
    """Wrapper for vtkPolyDataMapper2D."""
    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)

    def SetTextProperty(self, obj=None, **kwargs):
        """Set text property.

        Wraps the `SetTextProperty` method of `vtkTextActor` to
        accept a `vtkTextProperty` or BSTextProperty.

        Parameters
        ----------
        obj : vtkTextProperty or BSTextProperty, optional
            Label text property. If None, the property is created.
            Default is None.
        kwargs : optional keyword arguments
            Arguments are used to set the property.
        """

        if obj is None:
            obj = self.VTKObject.GetTextProperty()
        obj = BSTextProperty(vtkobject=obj, **kwargs)
        self.VTKObject.SetTextProperty(obj.VTKObject)
        return obj


class BSPolyDataAlgorithm(BSAlgorithm):
    """Wrapper for vtkPolyDataAlgorithm."""
    pass


class BSWindowToImageFilter(BSAlgorithm):
    """Wrapper for vtkWindowToImageFilter."""
    pass


class BSImageAlgorithm(BSAlgorithm):
    """Wrapper for vtkImageAlgorithm."""
    pass


class BSImageWriter(BSImageAlgorithm):
    """Wrapper for vtkImageWriter."""
    pass


class BSBMPWriter(BSImageWriter):
    """Wrapper for vtkBMPWriter."""
    pass


class BSJPEGWriter(BSImageWriter):
    """Wrapper for vtkJPEGWriter."""
    pass


class BSPNGWriter(BSImageWriter):
    """Wrapper for vtkPNGWriter."""
    pass


class BSPostScriptWriter(BSImageWriter):
    """Wrapper for vtkPostScriptWriter."""
    pass


class BSTIFFWriter(BSImageWriter):
    """Wrapper for vtkTIFFWriter."""
    pass
