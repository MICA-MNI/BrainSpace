"""
Wrappers for VTK classes needed for rendering.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


from .base import BSVTKObjectWrapper, wrap_vtk
from .actor import BSActor, BSActor2D, BSScalarBarActor, BSTextActor


###############################################################
# Renderer
###############################################################
class BSViewport(BSVTKObjectWrapper):
    """Wrapper for vtkViewport."""

    def __init__(self, vtkobject, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)

    def AddActor2D(self, obj=None, **kwargs):
        """Set mapper.

        Wraps the `AddActor2D` method of `vtkViewport` to accept a
        `vtkActor2D` or BSActor2D.

        Parameters
        ----------
        obj : vtkActor or BSActor
            2D Actor.
        kwargs : optional keyword arguments
            Arguments are used to set the actor.
        """

        actor = BSActor2D(vtkobject=obj, **kwargs)
        self.VTKObject.AddActor2D(actor.VTKObject)
        return actor

    def AddScalarBarActor(self, obj=None, **kwargs):
        actor = BSScalarBarActor(vtkobject=obj, **kwargs)
        self.VTKObject.AddActor2D(actor.VTKObject)
        return actor

    def AddTextActor(self, obj=None, **kwargs):
        actor = BSTextActor(vtkobject=obj, **kwargs)
        self.VTKObject.AddActor2D(actor.VTKObject)
        return actor


class BSRenderer(BSViewport):
    """Wrapper for vtkRenderer."""

    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)

    def AddActor(self, obj=None, **kwargs):
        """Set mapper.

        Wraps the `AddActor` method of `vtkRenderer` to accept a
        `vtkActor` or BSActor.

        Parameters
        ----------
        obj : vtkActor or BSActor, optional
            Actor. If None, the actor is created. Default is None.
        kwargs : optional keyword arguments
            Arguments are used to set the actor.
        """

        actor = BSActor(vtkobject=obj, **kwargs)
        self.VTKObject.AddActor(actor.VTKObject)
        return actor


###############################################################
# Interactor style
###############################################################
class BSInteractorObserver(BSVTKObjectWrapper):
    """Wrapper for vtkInteractorObserver."""
    def __init__(self, vtkobject, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)


class BSInteractorStyle(BSInteractorObserver):
    """Wrapper for vtkInteractorStyle."""
    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)


class BSInteractorStyleJoystickCamera(BSInteractorStyle):
    """Wrapper for vtkInteractorStyleJoystickCamera."""
    pass


class BSInteractorStyleJoystickActor(BSInteractorStyle):
    """Wrapper for vtkInteractorStyleJoystickActor."""
    pass


class BSInteractorStyleTerrain(BSInteractorStyle):
    """Wrapper for vtkInteractorStyleTerrain."""
    pass


class BSInteractorStyleRubberBandZoom(BSInteractorStyle):
    """Wrapper for vtkInteractorStyleRubberBandZoom."""
    pass


class BSInteractorStyleTrackballActor(BSInteractorStyle):
    """Wrapper for vtkInteractorStyleTrackballActor."""
    pass


class BSInteractorStyleTrackballCamera(BSInteractorStyle):
    """Wrapper for vtkInteractorStyleTrackballCamera."""
    pass


class BSInteractorStyleImage(BSInteractorStyleTrackballCamera):
    """Wrapper for vtkInteractorStyleImage."""
    pass


class BSInteractorStyleRubberBandPick(BSInteractorStyleTrackballCamera):
    """Wrapper for vtkInteractorStyleRubberBandPick."""
    pass


class BSInteractorStyleSwitchBase(BSInteractorStyle):
    """Wrapper for vtkInteractorStyleSwitchBase."""
    pass


class BSInteractorStyleSwitch(BSInteractorStyleSwitchBase):
    """Wrapper for vtkInteractorStyleSwitch."""
    pass


###############################################################
# Window Interactor
###############################################################
class BSRenderWindowInteractor(BSVTKObjectWrapper):
    """Wrapper for vtkRenderWindowInteractor."""
    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)

    def SetInteractorStyle(self, obj=None, **kwargs):
        if obj is None and len(kwargs) == 0:
            self.VTKObject.SetInteractorStyle(None)
            return None
        if obj is None:
            style = BSInteractorStyleSwitch(vtkobject=obj, **kwargs)
        else:
            style = wrap_vtk(obj, **kwargs)
        self.VTKObject.SetInteractorStyle(style.VTKObject)
        return style

    def SetInteractorStyleNone(self):
        return self.VTKObject.SetInteractorStyle(None)

    def SetInteractorStyleSwitch(self, obj=None, **kwargs):
        style = BSInteractorStyleSwitch(vtkobject=obj, **kwargs)
        self.VTKObject.SetInteractorStyle(style.VTKObject)
        return style

    def SetInteractorStyleTrackBallCamera(self, obj=None, **kwargs):
        style = BSInteractorStyleTrackballCamera(vtkobject=obj, **kwargs)
        self.VTKObject.SetInteractorStyle(style.VTKObject)
        return style

    def SetInteractorStyleJoystickCamera(self, obj=None, **kwargs):
        style = BSInteractorStyleJoystickCamera(vtkobject=obj, **kwargs)
        self.VTKObject.SetInteractorStyle(style.VTKObject)
        return style

    def SetInteractorStyleTrackballActor(self, obj=None, **kwargs):
        style = BSInteractorStyleTrackballActor(vtkobject=obj, **kwargs)
        self.VTKObject.SetInteractorStyle(style.VTKObject)
        return style

    def SetInteractorStyleJoystickActor(self, obj=None, **kwargs):
        style = BSInteractorStyleJoystickActor(vtkobject=obj, **kwargs)
        self.VTKObject.SetInteractorStyle(style.VTKObject)
        return style

    def SetInteractorStyleRubberBandZoom(self, obj=None, **kwargs):
        style = BSInteractorStyleRubberBandZoom(vtkobject=obj, **kwargs)
        self.VTKObject.SetInteractorStyle(style.VTKObject)
        return style

    def SetInteractorStyleRubberBandPick(self, obj=None, **kwargs):
        style = BSInteractorStyleRubberBandPick(vtkobject=obj, **kwargs)
        self.VTKObject.SetInteractorStyle(style.VTKObject)
        return style

    def SetInteractorStyleImage(self, obj=None, **kwargs):
        style = BSInteractorStyleImage(vtkobject=obj, **kwargs)
        self.VTKObject.SetInteractorStyle(style.VTKObject)
        return style

    def SetInteractorStyleTerrain(self, obj=None, **kwargs):
        style = BSInteractorStyleTerrain(vtkobject=obj, **kwargs)
        self.VTKObject.SetInteractorStyle(style.VTKObject)
        return style


class BSGenericRenderWindowInteractor(BSRenderWindowInteractor):
    """Wrapper for vtkGenericRenderWindowInteractor."""
    pass


###############################################################
# Window
###############################################################
class BSWindow(BSVTKObjectWrapper):
    """Wrapper for vtkWindow."""
    def __init__(self, vtkobject, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)


class BSRenderWindow(BSWindow):
    """Wrapper for vtkRenderWindow."""

    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)

    def AddRenderer(self, obj=None, **kwargs):
        ren = BSRenderer(vtkobject=obj, **kwargs)
        self.VTKObject.AddRenderer(ren.VTKObject)
        return ren

    def SetInteractor(self, obj=None, **kwargs):
        if obj is None and len(kwargs) == 0:
            self.VTKObject.SetInteractor(None)
            return None
        iren = BSRenderWindowInteractor(vtkobject=obj, **kwargs)
        self.VTKObject.SetInteractor(iren.VTKObject)
        return iren


###############################################################
# Camera
###############################################################
class BSCamera(BSVTKObjectWrapper):
    """Wrapper for vtkCamera."""

    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)
