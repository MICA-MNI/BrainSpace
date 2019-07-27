"""
Base wrapper class for VTK objects.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import re
from collections import defaultdict

import vtk
from vtkmodules.numpy_interface.dataset_adapter import VTKObjectWrapper


re_state = 'Set(?P<state>(?P<root>[A-Z0-9].*)To(?P<value>[A-Z0-9].*))'
re_set = 'Set(?P<setter>[A-Z0-9].*)'
re_get = 'Get(?P<getter>[A-Z0-9].*)'
re_method = re.compile('|'.join([re_state, re_set, re_get]))


def get_vtk_methods(cls):
    """ Retrieve Set and Get methods from vtk class.

    Parameters
    ----------
    cls : type
        VTK class.

    Returns
    -------
    methods : dict
        Dictionary with set and get methods
        E.g., {{'set': {'colormode': 'SetColorMode'}, ...}, {'get': ...}
        The user can use colormode=1 as a kwd arg and it translates to
        obj.SetColorMode(1)

    Notes
    -----
    State methods (see vtkMethodParser) can be used with options. For example,
    from vtk.vtkPolyDataMapper we also include the base method with the options:
    ...
    colormodetodefault: 'SetColorModeToDefault',
    colormodetodirectscalars: 'SetColorModeToDirectScalars',
    colormodetomapscalars: 'SetColorModeToMapScalars',
    'colormode': {'name': 'SetColorMode',
                  'options':
                        {'default': 'SetColorModeToDefault',
                        'directscalars': 'SetColorModeToDirectScalars',
                        'mapscalars': 'SetColorModeToMapScalars'}
    ...

    This can be used as follows:
        Methods that accept no args can be used as strings or key-word args:
            colormodetomapscalars=None -> obj.SetColorModeToMapScalars()
            'colormodetomapscalars' -> obj.SetColorModeToMapScalars()
        For state methods, there is an additional way:
            colormode='mapscalars'     -> obj.SetColorModeToMapScalars()
        When there is also a method that accepts the mode, we can use:
            colormode=1             -> obj.SetColorMode(1)

    """

    lm = {k: dict() for k in ['set', 'get']}
    state_methods = defaultdict(dict)
    for m in dir(cls):
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
        When None is needed to be passed to the method, use a tuple: (None,)
        method=(None,)

    Returns
    -------
    result : Any
        Return the results of invoking `method` with `args` on `obj`.

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
    """Superclass for classes that wrap VTK objects with Python objects.
    This class holds a reference to the wrapped VTK object. It also
    forwards unresolved methods to the underlying object by overloading
    __getattr__.

    Adapted from dataset_adapter, with additional setVTK and getVTK methods.
    Create an instance if class is passed instead of object.

    This class also supports all VTK setters and getters to be used like
    properties dropping the 'get'/'set' prefix, e.g., self.opacity = 0.5
    And it is case insensitive.

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

        Two possible options:
            1. e.g., self.GetOpacity(), normal behaviour. Becomes:
                self.VTKObject.GetOpacity() -> forwards to VTKObject
            2. self.opacity. Becomes:
                self.VTKObject.GetOpacity()

        """

        # We are here cause name is not in self
        # First forward to vtkobject
        # If it doesn't exist, look for it in vtkmap, find its corresponding
        # vtk name and forward again
        try:
            return super().__getattr__(name)
        except:
            return self._handle_call('get', name, None)

    def __setattr__(self, name, value):
        """ Forwards unknown set requests to vtk object.

        For example:
            self.opacity = .5, becomes self.VTKObject.SetOpacity(.5)

        """

        # Check self attributes first
        # Note: With this we cannot create attributes dynamically
        if name in self.__dict__:
            super().__setattr__(name, value)
        else:
            self._handle_call('set', name, value)

    def setVTK(self, *args, **kwargs):
        """ Invoke VTK set methods on the current `VTKObject`.

        Parameters
        ----------
        args : str
            Method that require no arguments.
        kwargs : dict
            key-word arguments can be use for methods that require arguments.
            When several arguments are required, use a tuple.
            Methods that require no arguments can also be used here using
            None as the argument.

            self.setVTK(opacity=.3) --> self.VTKObject.SetOpacity(.3)
            self.setVTK(numberofiterations=10) -->
                            self.VTKObject.SetNumberOfIterations(10)

        Returns
        -------
        self : BSVTKObjectWrapper object
            Return self.

        """

        kwargs = dict(zip(args, [None] * len(args)), **kwargs)
        for k, v in kwargs.items():
            self._handle_call('set', k, v)

        return self

    def getVTK(self, *args, **kwargs):
        """ Invoke VTK get methods on the vtk object.

        Parameters
        ----------
        args : str
            Method that require no arguments.

            self.getVTK('opacity') --> self.VTKObject.GetOpacity()

        kwargs : dict
            key-word arguments can be use for methods that require arguments.
            When several arguments are required, use a tuple.
            Methods that require no arguments can also be used here using
            None as the argument.

            self.getVTK(opacity=None) --> self.VTKObject.GetOpacity()
            self.getVTK(inputPortInformation=1) -->
                            self.VTKObject.GetGetInputPortInformation(1)

        Returns
        -------
        results : dict
            Dictionary of results where the keys are the method names and
            the values the results.

        """

        kwargs = dict(zip(args, [None] * len(args)), **kwargs)
        output = {}
        for k, v in kwargs.items():
            output[k] = self._handle_call('get', k, v)
        return output

    @property
    def vtk_map(self):
        return self._vtk_map[self.VTKObject.__vtkname__]
