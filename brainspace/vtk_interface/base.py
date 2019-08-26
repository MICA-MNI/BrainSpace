"""
Base wrapper class for VTK objects.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import re
from collections import defaultdict

import vtk
# from vtkmodules.numpy_interface.dataset_adapter import VTKObjectWrapper
from vtk.numpy_interface.dataset_adapter import VTKObjectWrapper


re_state = 'Set(?P<state>(?P<root>[A-Z0-9].*)To(?P<value>[A-Z0-9].*))'
re_set = 'Set(?P<setter>[A-Z0-9].*)'
re_get = 'Get(?P<getter>[A-Z0-9].*)'
re_method = re.compile('|'.join([re_state, re_set, re_get]))


def get_vtk_methods(obj):
    """ Retrieve Set and Get methods from vtk class or instance.

    Parameters
    ----------
    obj : type or object
        VTK class or object.

    Returns
    -------
    methods : dict
        Dictionary with set and get methods.

    Notes
    -----
    State methods (see vtkMethodParser) can also be used with options.

    Examples
    --------
    >>> from vtkmodules.vtkRenderingCorePython import vtkPolyDataMapper
    >>> from brainspace.vtk_interface.base import get_vtk_methods
    >>> vtk_map = get_vtk_methods(vtkPolyDataMapper)
    >>> vtk_map.keys()
    dict_keys(['set', 'get'])

    Check setter (state) methods for color mode:

    >>> vtk_map['set']['colormode']
    {'name': 'SetColorMode',
     'options': {
        'default': 'SetColorModeToDefault',
        'directscalars': 'SetColorModeToDirectScalars',
        'mapscalars': 'SetColorModeToMapScalars'}}

    Check getter methods for array name:

    >>> vtk_map['get']['arrayname']
    'GetArrayName'

    """

    lm = {k: dict() for k in ['set', 'get']}
    state_methods = defaultdict(dict)
    for m in dir(obj):
        r = re_method.match(m)
        if r is None:
            continue

        gd = {k: v.lower() for k, v in r.groupdict().items() if v is not None}
        if 'state' in gd:
            lm['set'][gd['state']] = m
            state_methods[gd['root']][gd['value']] = m
        elif 'setter' in gd:
            lm['set'][gd['setter']] = m
        elif 'getter' in gd:
            lm['get'][gd['getter']] = m

    for sm, options in state_methods.items():
        if len(options) == 1:
            continue
        if sm in lm['set']:
            lm['set'][sm] = {'name': lm['set'][sm], 'options': options}
        else:
            lm['set'][sm] = {'name': None, 'options': options}

    return lm


def call_vtk(obj, method, args=None):
    """ Invoke a method on a vtk object.

    Parameters
    ----------
    obj : object
        VTK object.
    method : str
        Method name.
    args : None ot tuple or list
        Arguments to be passed to the method.
        If None, the method is called with no arguments.

    Returns
    -------
    result : Any
        Return the results of invoking `method` with `args` on `obj`.

    Notes
    -----
    Use a tuple to pass a None to the method: (None,).

    Examples
    --------
    >>> from vtkmodules.vtkRenderingCorePython import vtkPolyDataMapper
    >>> from brainspace.vtk_interface.base import call_vtk
    >>> m = vtkPolyDataMapper()

    Get array id of the mapper:

    >>> call_vtk(m, 'GetArrayId', args=None)
    -1
    >>> m.GetArrayId()
    -1

    Set array id of the mapper to 2:

    >>> call_vtk(m, 'SetArrayId', args=(2,)) # same as m.SetArrayId(2)
    >>> m.GetArrayId()
    2

    """

    # Function takes no args -> use None
    # e.g., SetColorModeToMapScalars() -> colorModeToMapScalars=None
    if args is None:
        return getattr(obj, method)()

    # If iterable try first with multiple arguments
    if isinstance(args, (tuple, list)):
        try:
            return getattr(obj, method)(*args)
        except TypeError:
            pass

    return getattr(obj, method)(args)


class BSVTKObjectWrapperMeta(type):
    """ Metaclass for our VTK wrapper

        BSVTKObjectWrapper __setattr__ does not allow creating attributes
        This metaclass, hides __setattr__ (delegates to object.__setattr__)
        during __init__

        Postpones setting VTK kwds after __init__ because some subclasses
        may forward them to other vtkobjects within.
        See for example BSActor, which forwards to its property (GetProperty()).
        But this is not known until the actor is created.
        E.g.:    actor = BSActor(visibility=1, opacity=.2)
        Here visibility is forwarded to vtkActor. But we cannot forward
        opacity because it belongs to the actor's property and this is created
        after BSVTKObjectWrapper __init__.


    """

    # def __new__(mcs, name, bases, dic):
    #     cls = type.__new__(mcs, name, bases, dic)
    #     cls._override = []
    #     cls._set_override = False
    #     return cls

    def __call__(cls, *args, **kwargs):
        real_setattr = cls.__setattr__
        cls.__setattr__ = object.__setattr__

        self = super().__call__(*args, **kwargs)

        # To capture vtk methods reimplemented by the wrapper subclasses
        # if not cls._set_override:
        #     override_vtk = [m for m in dir(self) if not m.startswith('__')
        #                     and hasattr(self.VTKObject, m)]
        #     cls._override = override_vtk
        #     cls._set_override = True

        # Cannot use kwargs directly cause subclasses may define additional
        # kwargs. This just captures kwargs in BSVTKObjectWrapper
        self.setVTK(**self._vtk_kwargs)
        del self._vtk_kwargs

        cls.__setattr__ = real_setattr
        return self


class BSVTKObjectWrapper(VTKObjectWrapper, metaclass=BSVTKObjectWrapperMeta):
    """Base class for all classes that wrap VTK objects.

    Adapted from dataset_adapter, with additional setVTK and getVTK methods.
    Create an instance if class is passed instead of object.

    This class holds a reference to the wrapped VTK object. It also
    forwards unresolved methods to the underlying object by overloading
    __getattr__. This class also supports all VTK setters and getters to be
    used like properties/attributes dropping the get/set prefix. This is case
    insensitive.

    Parameters
    ----------
    vtkobject : type or object
        VTK class or object.
    kwargs : optional keyword parameters
        Parameters used to invoke set methods on the vtk object.

    Attributes
    ----------
    VTKObject : vtkObject
        A VTK object.

    Examples
    --------
    >>> from vtkmodules.vtkRenderingCorePython import vtkPolyDataMapper
    >>> from brainspace.vtk_interface.base import BSVTKObjectWrapper
    >>> m1 = BSVTKObjectWrapper(vtkPolyDataMapper())
    >>> m1
    <brainspace.vtk_interface.base.BSVTKObjectWrapper at 0x7f38a4b70198>
    >>> m1.VTKObject
    (vtkRenderingOpenGL2Python.vtkOpenGLPolyDataMapper)0x7f38a4bee888

    Passing class and additional keyword arguments:

    >>> m2 = BSVTKObjectWrapper(vtkPolyDataMapper, arrayId=3,
    ...                         colorMode='mapScalars')
    >>> # Get color name, these are all the same
    >>> m2.VTKObject.GetColorModeAsString()
    'MapScalars'
    >>> m2.GetColorModeAsString()
    'MapScalars'
    >>> m2.colorModeAsString
    'MapScalars'
    >>> # Get array id
    >>> m2.VTKObject.GetArrayId()
    3
    >>> m2.GetArrayId()
    3
    >>> m2.arrayId
    3

    We can change array id and color mode as follows:

    >>> m2.arrayId = 0
    >>> m2.VTKObject.GetArrayId()
    0
    >>> m2.colorMode = 'default'
    >>> m2.VTKObject.GetColorModeAsString()
    'Default'

    """

    _vtk_map = dict()

    def __init__(self, vtkobject, **kwargs):

        if vtkobject is None:
            name = type(self).__name__.replace('BS', 'vtk', 1)
            vtkobject = getattr(vtk, name)()
        elif type(vtkobject) == type:
            vtkobject = vtkobject()

        if isinstance(vtkobject, type(self)):
            vtkobject = vtkobject.VTKObject

        super().__init__(vtkobject)

        if self.VTKObject.__vtkname__ not in self._vtk_map:
            self._vtk_map[self.VTKObject.__vtkname__] = \
                get_vtk_methods(self.VTKObject)

        # Has to be postponed (see metaclass) cause we may forward some kwds
        # to a different vtk object (e.g., BSActor forwards to its BSProperty)
        # that is not defined at this moment
        self._vtk_kwargs = kwargs

    def _handle_call(self, key, name, args):
        method = self.vtk_map[key][name.lower()]
        # obj = self.VTKObject
        # if method in self._override:
        #     obj = self
        if isinstance(method, dict):
            if isinstance(args, str) and args.lower() in method['options']:
                return call_vtk(self, method['options'][args.lower()])
            elif 'name' in method:
                return call_vtk(self, method['name'], args)
            else:
                raise AttributeError("Cannot find VTK name '%s'" % name)

        return call_vtk(self, method, args)

    def __getattr__(self, name):
        """ Forwards unknown attribute requests to vtk object.

        Examples
        --------
        >>> from vtkmodules.vtkRenderingCorePython import vtkPolyDataMapper
        >>> from brainspace.vtk_interface.base import BSVTKObjectWrapper
        >>> m1 = BSVTKObjectWrapper(vtkPolyDataMapper())
        >>> m1.GetArrayId() # same as self.VTKObject.GetArrayId()
        -1
        >>> self.arrayId  # same as self.VTKObject.GetArrayId()
        -1

        """

        # We are here cause name is not in self
        # First forward to vtkobject
        # If it doesn't exist, look for it in vtk_map, find its corresponding
        # vtk name and forward again
        try:
            return super().__getattr__(name)
        except:
            return self._handle_call('get', name, None)

    def __setattr__(self, name, value):
        """ Forwards unknown set requests to vtk object.

        Examples
        --------
        >>> from vtkmodules.vtkRenderingCorePython import vtkPolyDataMapper
        >>> from brainspace.vtk_interface.base import BSVTKObjectWrapper
        >>> m1 = BSVTKObjectWrapper(vtkPolyDataMapper())
        >>> m1.GetArrayId()
        -1
        >>> self.arrayId = 3  # same as self.VTKObject.SetArrayId(3)
        >>> m1.GetArrayId()
        3

        """

        # Check self attributes first
        # Note: With this we cannot create attributes dynamically
        if name in self.__dict__:
            super().__setattr__(name, value)
        else:
            self._handle_call('set', name, value)

    def setVTK(self, *args, **kwargs):
        """ Invoke set methods on the vtk object.

        Parameters
        ----------
        args : list of str
            Setter methods that require no arguments.
        kwargs : list of keyword-value arguments
            key-word arguments can be use for methods that require arguments.
            When several arguments are required, use a tuple.
            Methods that require no arguments can also be used here using
            None as the argument.

        Returns
        -------
        self : BSVTKObjectWrapper object
            Return self.

        Examples
        --------
        >>> from vtkmodules.vtkRenderingCorePython import vtkPolyDataMapper
        >>> from brainspace.vtk_interface.base import BSVTKObjectWrapper
        >>> m1 = BSVTKObjectWrapper(vtkPolyDataMapper())
        >>> m1.setVTK(arrayId=3, colorMode='mapScalars')
        <brainspace.vtk_interface.base.BSVTKObjectWrapper at 0x7f38a4ace320>
        >>> m1.arrayId
        3
        >>> m1.colorModeAsString
        'MapScalars'

        """

        kwargs = dict(zip(args, [None] * len(args)), **kwargs)
        for k, v in kwargs.items():
            self._handle_call('set', k, v)

        return self

    def getVTK(self, *args, **kwargs):
        """ Invoke get methods on the vtk object.

        Parameters
        ----------
        args : list of str
            Method that require no arguments.
        kwargs : list of keyword-value arguments
            key-word arguments can be use for methods that require arguments.
            When several arguments are required, use a tuple.
            Methods that require no arguments can also be used here using
            None as the argument.

        Returns
        -------
        results : dict
            Dictionary of results where the keys are the method names and
            the values the results.

        Examples
        --------
        >>> from vtkmodules.vtkRenderingCorePython import vtkPolyDataMapper
        >>> from brainspace.vtk_interface.base import BSVTKObjectWrapper
        >>> m1 = BSVTKObjectWrapper(vtkPolyDataMapper())
        >>> m1.getVTK('arrayId', colorModeAsString=None)
        {'arrayId': -1, 'colorModeAsString': 'Default'}
        >>> m1.getVTK('colorModeAsString', arrayId=None)
        {'colorModeAsString': 'Default', 'arrayId': -1}
        >>> m1.getVTK(numberOfInputConnections=0)
        {'numberOfInputConnections': 0}

        """

        kwargs = dict(zip(args, [None] * len(args)), **kwargs)
        output = {}
        for k, v in kwargs.items():
            output[k] = self._handle_call('get', k, v)
        return output

    @property
    def vtk_map(self):
        """dict: Dictionary of vtk setter and getter methods."""
        return self._vtk_map[self.VTKObject.__vtkname__]
