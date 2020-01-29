"""
Wrappers for some VTK properties.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


from .base import BSVTKObjectWrapper


class BSProperty(BSVTKObjectWrapper):
    """Wrapper for vtkProperty."""

    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject, **kwargs)


class BSProperty2D(BSVTKObjectWrapper):
    """Wrapper for vtkProperty2D."""

    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject, **kwargs)


class BSTextProperty(BSVTKObjectWrapper):
    """Wrapper for vtkTextProperty."""

    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject, **kwargs)
