"""
Wrappers for VTK actors.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


from .base import BSVTKObjectWrapper, wrap_vtk
from .property import BSProperty, BSProperty2D, BSTextProperty
from .algorithm import (BSPolyDataMapper, BSDataSetMapper,
                        BSLabeledContourMapper)


class BSProp(BSVTKObjectWrapper):
    """Wrapper for vtkProp."""
    def __init__(self, vtkobject, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)


class BSProp3D(BSProp):
    """Wrapper for vtkProp3D."""
    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)


class BSActor2D(BSProp):
    """Wrapper for vtkActor2D.

    Unresolved requests are forwarded to its 2D property.

    """
    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)
        self._property = BSProperty2D(vtkobject=self.VTKObject.GetProperty())

    def _handle_call(self, key, name, args):
        try:
            return super()._handle_call(key, name, args)
        except (AttributeError, KeyError):
            return self._property._handle_call(key, name, args)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except (AttributeError, KeyError):
            return self._property.__getattr__(name)

    def __setattr__(self, name, value):
        try:
            super().__setattr__(name, value)
        except (AttributeError, KeyError):
            self._property.__setattr__(name, value)

    def GetProperty(self):
        """Get property.

        Wraps the `GetProperty` method of `vtkActor2D` to return a wrapped
        property.

        Returns
        -------
        prop : BSProperty2D
            Actor's property.
        """
        return self._property


class BSScalarBarActor(BSActor2D):
    """Wrapper for vtkScalarBarActor.

    Unresolved requests are forwarded to its 2D property.

    """

    def SetTitleTextProperty(self, obj=None, **kwargs):
        """Set title text property.

        Wraps the `SetTitleTextProperty` method of `vtkScalarBarActor` to
        accept a `vtkTextProperty` or BSTextProperty.

        Parameters
        ----------
        obj : vtkTextProperty or BSTextProperty, optional
            Title text property. If None, the property is created.
            Default is None.
        kwargs : optional keyword arguments
            Arguments are use to set the property.
        """

        if obj is None:
            obj = self.VTKObject.GetTitleTextProperty()
        obj = BSTextProperty(vtkobject=obj, **kwargs)
        self.VTKObject.SetTitleTextProperty(obj.VTKObject)
        return obj

    def SetLabelTextProperty(self, obj=None, **kwargs):
        """Set label text property.

        Wraps the `SetLabelTextProperty` method of `vtkScalarBarActor` to
        accept a `vtkTextProperty` or BSTextProperty.

        Parameters
        ----------
        obj : vtkTextProperty or BSTextProperty, optional
            Label text property. If None, the property is created.
            Default is None.
        kwargs : optional keyword arguments
            Arguments are use to set the property.
        """

        if obj is None:
            obj = self.VTKObject.GetLabelTextProperty()
        obj = BSTextProperty(vtkobject=obj, **kwargs)
        self.VTKObject.SetLabelTextProperty(obj.VTKObject)
        return obj

    def SetAnnotationTextProperty(self, obj=None, **kwargs):
        """Set annotation text property.

        Wraps the `SetAnnotationTextProperty` method of `vtkScalarBarActor` to
        accept a `vtkTextProperty` or BSTextProperty.

        Parameters
        ----------
        obj : vtkTextProperty or BSTextProperty, optional
            Annotation text property. If None, the property is created.
            Default is None.
        kwargs : optional keyword arguments
            Arguments are use to set the property.
        """

        if obj is None:
            obj = self.VTKObject.GetAnnotationTextProperty()
        obj = BSTextProperty(vtkobject=obj, **kwargs)
        self.VTKObject.SetAnnotationTextProperty(obj.VTKObject)
        return obj

    def SetBackgroundProperty(self, obj=None, **kwargs):
        """Set background property.

        Wraps the `SetBackgroundProperty` method of `vtkScalarBarActor` to
        accept a `vtkProperty2D` or BSProperty2D.

        Parameters
        ----------
        obj : vtkProperty2D or BSProperty2D, optional
            Background property. If None, the property is created.
            Default is None.
        kwargs : optional keyword arguments
            Arguments are use to set the property.
        """

        if obj is None:
            obj = self.VTKObject.GetBackgroundProperty()
        obj = BSProperty2D(vtkobject=obj, **kwargs)
        self.VTKObject.SetBackgroundProperty(obj.VTKObject)
        return obj

    def SetFrameProperty(self, obj=None, **kwargs):
        """Set frame property.

        Wraps the `SetFrameProperty` method of `vtkScalarBarActor` to
        accept a `vtkProperty2D` or BSProperty2D.

        Parameters
        ----------
        obj : vtkProperty2D or BSProperty2D, optional
            Frame property. If None, the property is created.
            Default is None.
        kwargs : optional keyword arguments
            Arguments are use to set the property.
        """

        if obj is None:
            obj = self.VTKObject.GetFrameProperty()
        obj = BSProperty2D(vtkobject=obj, **kwargs)
        self.VTKObject.SetFrameProperty(obj.VTKObject)
        return obj


class BSTexturedActor2D(BSActor2D):
    """Wrapper for vtkTexturedActor2D."""
    pass


class BSTextActor(BSTexturedActor2D):
    """Wrapper for vtkTextActor."""

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
            Arguments are use to set the property.
        """

        if obj is None:
            obj = self.VTKObject.GetTextProperty()
        obj = BSTextProperty(vtkobject=obj, **kwargs)
        self.VTKObject.SetTextProperty(obj.VTKObject)
        return obj


class BSActor(BSProp3D):
    """Wrapper for vtkActor.

    Unresolved requests are forwarded to its property.

    Examples
    --------
    >>> from brainspace.vtk_interface.wrappers import BSActor
    >>> a = BSActor()
    >>> a.GetProperty().GetOpacity()
    1.0
    >>> a.GetOpacity() # It is forwarded to the property
    1.0
    >>> a.opacity = .5
    >>> a.VTKObject.GetProperty().GetOpacity()
    0.5
    """

    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)
        self._property = BSProperty(self.VTKObject.GetProperty())

    def _handle_call(self, key, name, args):
        try:
            return super()._handle_call(key, name, args)
        except (AttributeError, KeyError):
            return self._property._handle_call(key, name, args)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except (AttributeError, KeyError):
            return self._property.__getattr__(name)

    def __setattr__(self, name, value):
        try:
            super().__setattr__(name, value)
        except (AttributeError, KeyError):
            self._property.__setattr__(name, value)

    def SetMapper(self, obj=None, **kwargs):
        """Set mapper.

        Wraps the `SetMapper` method of `vtkActor` to accept a
        `vtkMapper` or BSMapper.

        Parameters
        ----------
        obj : vtkMapper or BSMapper, optional
            Mapper. If None, a PolyDataMapper is created. Default is None.
        kwargs : optional keyword arguments
            Arguments are used to set the mapper.
        """

        if obj is None:
            return self.SetPolyDataMapper(**kwargs)
        obj = wrap_vtk(obj, **kwargs)
        self.VTKObject.SetMapper(obj.VTKObject)
        return obj

    def SetPolyDataMapper(self, obj=None, **kwargs):
        """Set a PolyDataMapper.

        Parameters
        ----------
        obj : vtkMapper or BSMapper, optional
            Mapper. If None, the mapper is created. Default is None.
        kwargs : optional keyword arguments
            Arguments are used to set the mapper.
        """

        obj = BSPolyDataMapper(vtkobject=obj, **kwargs)
        self.VTKObject.SetMapper(obj.VTKObject)
        return obj

    def SetDataSetMapper(self, obj=None, **kwargs):
        """Set DataSetMapper.

        Parameters
        ----------
        obj : vtkMapper or BSMapper, optional
            Mapper. If None, the mapper is created. Default is None.
        kwargs : optional keyword arguments
            Arguments are used to set the mapper.
        """

        obj = BSDataSetMapper(vtkobject=obj, **kwargs)
        self.VTKObject.SetMapper(obj.VTKObject)
        return obj

    def SetLabeledContourMapper(self, obj=None, **kwargs):
        """Set LabeledContourMapper.

        Parameters
        ----------
        obj : vtkMapper or BSMapper, optional
            Mapper. If None, the mapper is created. Default is None.
        kwargs : optional keyword arguments
            Arguments are used to set the mapper.
        """

        obj = BSLabeledContourMapper(vtkobject=obj, **kwargs)
        self.VTKObject.SetMapper(obj.VTKObject)
        return obj

    def GetProperty(self):
        """Get property.

        Wraps the `GetProperty` method of `vtkActor` to return a wrapped
        property.

        Returns
        -------
        prop : BSProperty
            Actor's property.
        """
        return self._property
