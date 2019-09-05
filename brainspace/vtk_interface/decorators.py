"""
Decorators for wrapping/unwrapping vtk objects passed/returned by a function.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import inspect
import functools


from . import wrappers


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
        if i in xargs and wrappers.is_vtk(a):
            new_args[i] = wrappers.wrap_vtk(a)

    for k, v in kwargs.items():
        if k in xargs and wrappers.is_vtk(v):
            kwargs[k] = wrappers.wrap_vtk(v)
    return new_args, kwargs


def _unwrap_input_data(args, kwargs, *xargs, skip=False):
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

    list_args = list(range(len(args))) + list(kwargs.keys())
    if len(xargs) == 0:
        xargs = list_args
    if skip:
        xargs = [a for a in list_args if a not in xargs]

    new_args = list(args)
    for i, a in enumerate(new_args):
        if i in xargs and wrappers.is_wrapper(a):
            new_args[i] = a.VTKObject

    for k, v in kwargs.items():
        if k in xargs and wrappers.is_wrapper(v):
            kwargs[k] = v.VTKObject
    return new_args, kwargs


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

    if wrappers.is_vtk(data):
        return wrappers.wrap_vtk(data)
    return data


def _unwrap_output_data(data):
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

    if wrappers.is_wrapper(data):
        return data.VTKObject
    return data


def wrap_input(*xargs, skip=False):
    """Decorator to wrap the arguments of a function.

    An object is wrapped only if it is an instance of :class:`vtkObject`
    or one of its subclasses.

    Parameters
    ----------
    xargs : sequence of int and str
        Positional indices (integers) and keys as strings (for keyword
        args) to wrap. If no specified, try to wrap all args.
    skip : bool, optional
        Wrap all arguments except those in `xargs`. Default is False.

    See Also
    --------
    :func:`wrap_output`
    :func:`wrap_func`
    :func:`unwrap_input`
    """

    def _wrapper_decorator(func):
        @functools.wraps(func)
        def _wrapper_wrap(*args, **kwds):
            args, kwds = _wrap_input_data(args, kwds, *xargs, skip=skip)
            data = func(*args, **kwds)
            return data
        return _wrapper_wrap
    return _wrapper_decorator


def wrap_output(func):
    """Decorator to wrap output of function.

    The output is wrapped only if is an instance of :class:`vtkObject`
    or one of its subclasses.

    Parameters
    ----------
    func : callable
        Function whose output is wrapped.

    See Also
    --------
    :func:`wrap_input`
    :func:`wrap_func`
    :func:`unwrap_output`

    """

    @functools.wraps(func)
    def _wrapper_wrap(*args, **kwds):
        data = func(*args, **kwds)
        return _wrap_output_data(data)
    return _wrapper_wrap


def unwrap_input(*xargs, skip=False):
    """Decorator to unwrap input arguments of function.

    An object is unwrapped only if it is an instance of
    :class:`.BSVTKObjectWrapper` or one of its subclasses.

    Parameters
    ----------
    xargs : sequence of int and str
        Positional indices (integers) and keys as strings (for keyword
        args) to unwrap. If no specified, try to unwrap all args.
    skip : bool, optional
        Unwrap all arguments except those in `xargs`. Default is False.

    See Also
    --------
    :func:`unwrap_output`
    :func:`unwrap_func`
    :func:`wrap_input`
    """

    def _wrapper_decorator(func):
        @functools.wraps(func)
        def _wrapper_wrap(*args, **kwds):
            args, kwds = _unwrap_input_data(args, kwds, *xargs, skip=skip)
            data = func(*args, **kwds)
            return data
        return _wrapper_wrap
    return _wrapper_decorator


def unwrap_output(func):
    """Decorator to unwrap output of function.

    The output is unwrapped only if is an instance of
    :class:`.BSVTKObjectWrapper` or one of its subclasses.

    Parameters
    ----------
    func : callable
        Function whose output is unwrapped.

    See Also
    --------
    :func:`unwrap_input`
    :func:`unwrap_func`
    :func:`wrap_output`
    """

    @functools.wraps(func)
    def _wrapper_wrap(*args, **kwds):
        data = func(*args, **kwds)
        return _unwrap_output_data(data)
    return _wrapper_wrap


def wrap_func(*xargs, inp=True, out=True, skip=False):
    """Decorator to wrap both arguments and output of a function.

    An object is wrapped only if it is an instance of :class:`vtkObject`
    or one of its subclasses.

    Parameters
    ----------
    xargs : sequence of int and str
        Positional indices (integers) and keys as strings (for keyword
        args) to wrap. If no specified, try to wrap all args.
    inp : bool, optional
        If True, wrap input arguments. Default is True.
    out : bool, optional
        If True, wrap output. Default is True.
    skip : bool, optional
        Wrap all arguments except those in `xargs`. Default is False.

    See Also
    --------
    :func:`wrap_input`
    :func:`wrap_output`
    :func:`unwrap_func`
    """

    def _wrapper_decorator(func):
        @functools.wraps(func)
        def _wrapper_wrap(*args, **kwds):
            if inp:
                args, kwds = _wrap_input_data(args, kwds, xargs, skip=skip)
            data = func(*args, **kwds)

            if out:
                return _wrap_output_data(data)
            return data
        return _wrapper_wrap

    return _wrapper_decorator


def unwrap_func(inp=True, out=True, largs='*', skip=False):
    """Decorator to unwrap both arguments and output of a function.

    An object is unwrapped only if it is an instance of
    :class:`.BSVTKObjectWrapper` or one of its subclasses.

    Parameters
    ----------
    largs : sequence of int and str
        Positional indices (integers) and keys as strings (for keyword
        args) to unwrap. If no specified, try to unwrap all args.
    inp : bool, optional
        If True, unwrap input arguments. Default is True.
    out : bool, optional
        If True, unwrap output. Default is True.
    skip : bool, optional
        Unwrap all arguments except those in `largs`. Default is False.

    See Also
    --------
    :func:`unwrap_input`
    :func:`unwrap_output`
    :func:`wrap_func`
    """

    def _wrapper_decorator(func):
        @functools.wraps(func)
        def _wrapper_wrap(*args, **kwds):
            if inp:
                args, kwds = _unwrap_input_data(*args, **kwds, largs=largs,
                                                  skip=skip)
            data = func(*args, **kwds)

            if out:
                return _unwrap_output_data(data)
            return data
        return _wrapper_wrap
    return _wrapper_decorator


def _get_default_args(func):
    sig = inspect.signature(func)
    kwds = {k: v.default for k, v in sig.parameters.items()
            if v.default is not inspect.Parameter.empty}
    return kwds


def append_vtk(to='point'):
    """Decorator to append data to surface.

    Parameters
    ----------
    to : {'point', 'cell', 'field'}, optional
        Append data to PointData, CellData or FieldData. Default is 'point'.

    Returns
    -------
    wrapped_func : callable
        Wrapped function.

    Notes
    -----
    All functions using this decorator must:

    - Return an ndarray. The size of the array must be consistent with
      the data it will be appended to (e.g., number of points if
      ``to == 'point'``), except for FieldData.

    - Have the following 2 key-value arguments:

      #. append (bool, optional)
          If True, append data to surface. Otherwise, return data.

      #. key (str, optional)
          Array names of data.

    See Also
    --------
    :func:`.compute_cell_area`
    :func:`.get_n_adjacent_cells`

    """

    def _wrapper_decorator(func):
        @functools.wraps(func)
        def _wrapper_append(surf, *args, **kwargs):
            kwds = _get_default_args(func)
            kwds.update(kwargs)
            ws = wrappers.wrap_vtk(surf)
            data = func(ws, *args, **kwds)
            if not kwds['append']:
                return data
            if kwds['key'] is None:
                raise ValueError('Key is None. Cannot append data.')
            ws.append_array(data, name=kwds['key'], at=to)
            return surf
        return _wrapper_append
    return _wrapper_decorator
