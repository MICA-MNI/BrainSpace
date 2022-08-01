"""
Base wrapper for VTK objects.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import numpy as np

import vtk
from vtk import (vtkAbstractArray, vtkStringArray, vtkIdList, vtkVariantArray,
                 vtkDataArray)
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.util.vtkConstants import VTK_STRING

from .utils import call_vtk, get_vtk_methods, is_numpy_string, is_vtk_string


try:
    from vtk import vtkUnicodeStringArray
except ImportError:
    vtkUnicodeStringArray = vtkStringArray


class VTKMethodWrapper:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name.__repr__()

    def __call__(self, *args, **kwargs):
        args, kwargs = _unwrap_input_data(args, kwargs)
        return _wrap_output_data(self.name(*args, **kwargs))


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

    entries = {}

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        BSVTKObjectWrapperMeta.entries[cls.__name__[2:]] = cls

    def __call__(cls, *args, **kwargs):
        real_setattr = cls.__setattr__
        cls.__setattr__ = object.__setattr__

        self = super().__call__(*args, **kwargs)

        # Cannot use kwargs directly cause subclasses may define additional
        # kwargs. This just captures kwargs in BSVTKObjectWrapper
        self.setVTK(**self._vtk_kwargs)
        del self._vtk_kwargs

        cls.__setattr__ = real_setattr
        return self


class BSVTKObjectWrapper(dsa.VTKObjectWrapper,
                         metaclass=BSVTKObjectWrapperMeta):
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
    >>> from brainspace.vtk_interface.wrappers import BSVTKObjectWrapper
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

        if isinstance(args, dict) and key == 'set':
            try:
                obj = args.pop('obj', None)
                if obj is None or name.lower() not in self.vtk_map[key]:
                    obj = self._handle_call('get', name, None)
                    return obj.setVTK(**args)
                args = wrap_vtk(obj, **args)
            except:
                pass

        method = self.vtk_map[key][name.lower()]

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
        >>> import vtk
        >>> from brainspace.vtk_interface.wrappers import BSVTKObjectWrapper
        >>> m1 = BSVTKObjectWrapper(vtk.vtkPolyDataMapper())
        >>> m1.GetArrayId()  # same as self.VTKObject.GetArrayId()
        -1
        >>> self.arrayId  # same as self.VTKObject.GetArrayId()
        -1

        """

        # We are here cause name is not in self
        # First forward to vtkobject
        # If it doesn't exist, look for it in vtk_map, find its corresponding
        # vtk name and forward again
        try:
            return VTKMethodWrapper(super().__getattr__(name))
        except:
            return self._handle_call('get', name, None)

    def __setattr__(self, name, value):
        """ Forwards unknown set requests to vtk object.

        Examples
        --------
        >>> import vtk
        >>> from brainspace.vtk_interface.wrappers import BSVTKObjectWrapper
        >>> m1 = BSVTKObjectWrapper(vtk.vtkPolyDataMapper())
        >>> m1.GetArrayId()
        -1
        >>> self.arrayId = 3  # same as self.VTKObject.SetArrayId(3)
        >>> m1.GetArrayId()
        3

        """

        # Check self attributes first
        # Note: With this we cannot create attributes dynamically
        if name in self.__dict__:
            VTKMethodWrapper(super().__setattr__(name, value))
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
        >>> import vtk
        >>> from brainspace.vtk_interface.wrappers import BSVTKObjectWrapper
        >>> m1 = BSVTKObjectWrapper(vtk.vtkPolyDataMapper())
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
        >>> import vtk
        >>> from brainspace.vtk_interface.wrappers import BSVTKObjectWrapper
        >>> m1 = BSVTKObjectWrapper(vtk.vtkPolyDataMapper())
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

    def __repr__(self):
        r = super().__repr__()[:-1].split('.')[-1]
        vr = self.VTKObject.__repr__()[1:].split(')')[0]
        return '<{0} [Wrapping a {1}]>'.format(r, vr)

    @property
    def vtk_map(self):
        """dict: Dictionary of vtk setter and getter methods."""
        return self._vtk_map[self.VTKObject.__vtkname__]


def is_wrapper(obj):
    """ Check if `obj` is a wrapper.

    Parameters
    ----------
    obj : object
        Any object.

    Returns
    -------
    res : bool
        True if `obj` is a VTK wrapper. False, otherwise.
    """
    return isinstance(obj, BSVTKObjectWrapper)


def is_vtk(obj):
    """ Check if `obj` is a vtk object.

    Parameters
    ----------
    obj : object
        Any object.

    Returns
    -------
    res : bool
        True if `obj` is a VTK object. False, otherwise.
    """
    return isinstance(obj, vtk.vtkObject)


def BSWrapVTKObject(obj):
    """Wraps a vtk object.

    Parameters
    ----------
    obj : object
        A vtk class, object or None. If class, the object is created.

    Returns
    -------
    wrapped : None or BSVTKObjectWrapper
        Wrapped object. Returns None if `obj` is None.
    """

    if obj is None or is_wrapper(obj):
        return obj

    if type(obj) == type:
        obj = obj()

    if not is_vtk(obj):
        raise ValueError('Unknown object type: {0}'.format(type(obj)))

    # Is this really needed? is there a vtk class that doesn't start with vtk?
    if not obj.__vtkname__.startswith('vtk'):
        raise ValueError('Unknown object type: {0}'.format(type(obj)))

    # Find wrapper for vtk class or one of its superclasses
    for c in [sc.__vtkname__[3:] for sc in obj.__class__.mro()[:-3]]:
        if c in BSVTKObjectWrapperMeta.entries:
            bs_cls = BSVTKObjectWrapperMeta.entries[c]
            return bs_cls(obj)

    # Fall back to generic wrapper
    return BSVTKObjectWrapper(obj)


def _string_to_numpy(a):
    dtype = np.string_
    if isinstance(a, vtkUnicodeStringArray) \
            and vtkUnicodeStringArray != vtkStringArray:
        dtype = np.unicode_
    shape = a.GetNumberOfTuples(), a.GetNumberOfComponents()
    an = [a.GetValue(i) for i in range(a.GetNumberOfValues())]
    return np.asarray(an, dtype=dtype).reshape(shape)


def _numpy_to_string(a, array_type=None):
    if np.issubdtype(a.dtype, np.string_) or array_type == VTK_STRING:
        av = vtkStringArray()
    else:
        av = vtkUnicodeStringArray()
    av.SetNumberOfComponents(1 if a.ndim == 1 else a.shape[1])
    av.SetNumberOfValues(a.size)
    for i, s in enumerate(a.ravel()):
        av.SetValue(i, s)
    return av


def _variant_to_numpy(a):
    shape = a.GetNumberOfTuples(), a.GetNumberOfComponents()
    an = [a.GetValue(i) for i in range(a.GetNumberOfValues())]
    return np.asarray(an, dtype=np.object_).reshape(shape)


def _numpy_to_variant(a):
    av = vtkVariantArray()
    av.SetNumberOfComponents(1 if a.ndim == 1 else a.shape[1])
    av.SetNumberOfValues(a.size)
    for i, s in enumerate(a.ravel()):
        av.SetValue(i, s)
    return av


def _idlist_to_numpy(a):
    n = a.GetNumberOfIds()
    return np.array([a.GetId(i) for i in range(n)])


def wrap_vtk_array(a):
    if isinstance(a, vtkIdList):
        return _idlist_to_numpy(a)
    if isinstance(a, (vtkStringArray, vtkUnicodeStringArray)):
        return _string_to_numpy(a)
    if isinstance(a, vtkVariantArray):
        return _variant_to_numpy(a)
    if isinstance(a, vtkDataArray):
        return dsa.vtkDataArrayToVTKArray(a)
    raise ValueError('Unsupported array type: {0}'.format(type(a)))


def unwrap_vtk_array(a, array_type=None):
    if is_numpy_string(a.dtype) or is_vtk_string(array_type):
        return _numpy_to_string(a, array_type=array_type)
    if any([np.issubdtype(a.dtype, d) for d in [np.integer, np.floating]]):
        return dsa.numpyTovtkDataArray(a, array_type=array_type)
    if np.issubdtype(a.dtype, np.object_):
        return _numpy_to_variant(a)
    raise ValueError('Unsupported array type: {0}'.format(type(a)))


def wrap_vtk(obj, **kwargs):
    """Wrap input object to BSVTKObjectWrapper or one of its subclasses.

    Parameters
    ----------
    obj : vtkObject or BSVTKObjectWrapper
        Input object.
    kwargs : kwds, optional
        Additional keyword parameters are passed to vtk object.

    Returns
    -------
    wrapper: BSVTKObjectWrapper
        The wrapped object.
    """

    wobj = BSWrapVTKObject(obj)
    if len(kwargs) > 0:
        wobj.setVTK(**kwargs)

    if isinstance(obj, (vtkAbstractArray, vtkIdList)):
        try:
            return wrap_vtk_array(obj)
        except:
            pass

    return wobj


def unwrap_vtk(obj, vtype=None):
    if vtype is not False and isinstance(obj, np.ndarray) and obj.ndim < 3:
        vtype = None if vtype is True else vtype
        return unwrap_vtk_array(obj, array_type=vtype)
    if is_wrapper(obj):
        return obj.VTKObject
    raise ValueError('Unknown object type: {0}'.format(type(obj)))


def _wrap_output_data(data):
    """ Wraps the output of a function or method.

    This won't work if function returns multiples objects.

    Parameters
    ----------
    data : any
        Data returned by some function.

    Returns
    -------
    wrapped_data : BSVTKObjectWrapper
        Wrapped data.

    """

    if is_vtk(data):
        return wrap_vtk(data)
    return data


def _unwrap_output_data(data, vtype=False):
    """ Unwraps the output of a function or method.

    This won't work if function returns multiples objects.

    Parameters
    ----------
    data : any
        Data returned by some function.

    Returns
    -------
    unwrapped_data : instance of vtkObject
        Unwrapped data.

    """

    # if is_wrapper(data) or isinstance(data, np.ndarray) and data.ndim < 3:
    #     return unwrap_vtk(data, vtype=vtype)
    # return data
    try:
        return unwrap_vtk(data, vtype=vtype)
    except:
        pass
    return data


def _wrap_input_data(args, kwargs, *xargs, skip=False):
    """ Wrap vtk objects in `args` and `kwargs`.

    E.g., xargs=(0, 2, 'key1') wrap positional arguments in positions 0
    and 2, and keyword arg 'key1'.

    Parameters
    ----------
    args : tuple
        Function args.
    kwargs : dict
        Keyword args.
    xargs : sequence of int and str
        Positional indices (integers) and keys as strings (for keyword
        args) to wrap. If not specified, try to wrap all arguments.
        If ``skip == True``, wrap all arguments except these ones.
    skip : bool, optional
        Wrap all arguments except those in `xargs`. Default is False.

    Returns
    -------
    wrapped_args : args
         Return args with the wrapped vtk objects wrapped.
    wrapped_kwargs: kwargs
         Return keyword args with wrapped vtk objects.

    """

    list_args = list(range(len(args))) + list(kwargs.keys())
    if len(xargs) == 0:
        xargs = list_args
    if skip:
        xargs = [a for a in list_args if a not in xargs]

    new_args = list(args)
    for i, a in enumerate(new_args):
        if i in xargs:
            new_args[i] = _wrap_output_data(a)

    for k, v in kwargs.items():
        if k in xargs:
            kwargs[k] = _wrap_output_data(v)
    return new_args, kwargs


def _unwrap_input_data(args, kwargs, *xargs, vtype=False, skip=False):
    """ Unwrap (return the wrapped vtk object) wrappers in `args` and `kwargs`.

    E.g., ``xargs=(0, 2, 'key1')`` unwrap positional arguments in
    positions 0 and 2, and keyword arg 'key1'.

    Parameters
    ----------
    args : tuple
        Function args.
    kwargs : dict
        Keyword args.
    xargs : sequence of int and str
        Positional indices (integers) and keys as strings (for keyword
        args) to unwrap. If not specified, try to unwrap all arguments.
        If ``skip == True``, unwrap all arguments except these ones.
    skip : bool, optional
        Unwrap all arguments except those in `wrap_args`. Default is False.

    Returns
    -------
    unwrapped_args : args
         Return args with unwrapped vtk objects.
    unwrapped_kwargs: kwargs
         Return keyword args with unwrapped vtk objects.

    """

    dv = False
    if not isinstance(vtype, dict):
        if vtype in [True, None]:
            dv = None
        vtype = {}

    list_args = list(range(len(args))) + list(kwargs.keys())
    if len(xargs) == 0:
        xargs = list_args
    if skip:
        xargs = [a for a in list_args if a not in xargs]

    new_args = list(args)
    for i, a in enumerate(new_args):
        if i in xargs:
            new_args[i] = _unwrap_output_data(a, vtype=vtype.get(i, dv))

    for k, v in kwargs.items():
        if k in xargs:
            kwargs[k] = _unwrap_output_data(v, vtype=vtype.get(k, dv))
    return new_args, kwargs
