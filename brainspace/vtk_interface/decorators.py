"""
Decorators for wrapping/unwrapping vtk objects passed/returned by a function.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import inspect
import functools


from . import wrappers


def _wrap_input_data(*args, only_args=None, skip_args=None, **kwargs):
    """ Wrap all vtk objects in `args` and `kwargs` except those in `skip_args`.

    E.g., skip_args=[0, 2, 'key1'] to skip positional arguments in positions 0
    and 2, and keyword arg 'key1' from wrapping.

    Parameters
    ----------
    args : args
        Function args.
    only_args : int, str or list of int and str, optional
        List of positional indices (integers) and keys as strings (for keyword
        args) to wrap. This has preference over `skip_args`.
        Default is None.
    skip_args : int, str or list of int and str, optional
        List of positional indices (integers) and keys as strings (for keyword
        args) to skip from wrapping. Not used if `only_args` is not None.
        Default is None. If both are None, try to wrap all arguments.
    kwargs : kwargs
        Keyword args.

    Returns
    -------
    wrapped_args : args
         Return args with all vtk objects wrapped if not in `skip_args`.
    wrapped_kwargs: kwargs
         Return keyword args with all vtk objects wrapped if not in `skip_args`.

    """

    if only_args is None and skip_args is None:
        only_args = list(range(len(args))) + list(kwargs.keys())
    elif only_args is None:
        if not isinstance(skip_args, list):
            skip_args = [skip_args]
        only_args = [i for i in range(len(args)) if i not in skip_args]
        only_args += [k for k in kwargs.keys() if k not in skip_args]
    elif not isinstance(only_args, list):
        only_args = [only_args]

    new_args = list(args)
    for i, a in enumerate(new_args):
        if i in only_args and wrappers.is_vtk(a):
            new_args[i] = wrappers.wrap_vtk(a)

    for k, v in kwargs.items():
        if k in only_args and wrappers.is_vtk(v):
            kwargs[k] = wrappers.wrap_vtk(v)

    return new_args, kwargs


def _unwrap_input_data(*args, only_args=None, skip_args=None, **kwargs):
    """ Unwrap (return the wrapped vtk object) all wrappers in `args` and
    `kwargs` except those in `skip_args`.

    E.g., skip_args=[0, 2, 'key1'] to skip positional arguments in positions 0
    and 2, and keyword arg 'key1' from unwrapping.

    Parameters
    ----------
    args : args
        Function args.
    only_args : int, str or list of int and str, optional
        List of positional indices (integers) and keys as strings (for keyword
        args) to unwrap. This has preference over `skip_args`.
        Default is None.
    skip_args : int, str or list of int and str, optional
        List of positional indices (integers) and keys as strings (for keyword
        args) to skip from unwrapping. Not used if `only_args` is not None.
        Default is None. If both are None, try to unwrap all arguments.
    kwargs : kwargs
        Keyword args.

    Returns
    -------
    unwrapped_args : args
         Return args unwrapped vtk objects if not in `skip_args`.
    unwrapped_kwargs: kwargs
         Return keyword args with unwrapped vtk objects if not in `skip_args`.

    """

    if only_args is None and skip_args is None:
        only_args = list(range(len(args))) + list(kwargs.keys())
    elif only_args is None:
        if not isinstance(skip_args, list):
            skip_args = [skip_args]
        only_args = [i for i in range(len(args)) if i not in skip_args]
        only_args += [k for k in kwargs.keys() if k not in skip_args]
    elif not isinstance(only_args, list):
        only_args = [only_args]

    new_args = list(args)
    for i, a in enumerate(new_args):
        if i in only_args and wrappers.is_wrapper(a):
            new_args[i] = a.VTKObject

    for k, v in kwargs.items():
        if k in only_args and wrappers.is_wrapper(v):
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


def wrap_input(only_args=None, skip_args=None):
    """Decorator to wrap the arguments of a function.

    An object is wrapped only if it is an instance of :class:`vtkObject`
    or one of its subclasses.

    Parameters
    ----------
    only_args : int, str or list of int and str, optional
        List of positional indices (integers) and keys as strings (for keyword
        args) to wrap. This has preference over `skip_args`.
        Default is None.
    skip_args : int, str or list of int and str, optional
        List of positional indices (integers) and keys as strings (for keyword
        args) to skip from wrapping. Not used if `only_args` is not None.
        Default is None. If both are None, try to wrap all arguments.

    See Also
    --------
    :func:`wrap_output`
    :func:`wrap_func`
    :func:`unwrap_input`
    """

    def _wrapper_decorator(func):
        @functools.wraps(func)
        def _wrapper_wrap(*args, **kwargs):
            args, kwargs = _wrap_input_data(*args, **kwargs,
                                            only_args=only_args,
                                            skip_args=skip_args)
            data = func(*args, **kwargs)
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
    def _wrapper_wrap(*args, **kwargs):
        data = func(*args, **kwargs)
        return _wrap_output_data(data)
    return _wrapper_wrap


def unwrap_input(only_args=None, skip_args=None):
    """Decorator to unwrap input arguments of function.

    An object is unwrapped only if it is an instance of
    :class:`.BSVTKObjectWrapper` or one of its subclasses.

    Parameters
    ----------
    only_args : int, str or list of int and str, optional
        List of positional indices (integers) and keys as strings (for keyword
        args) to unwrap. This has preference over `skip_args`.
        Default is None.
    skip_args : int, str or list of int and str, optional
        List of positional indices (integers) and keys as strings (for keyword
        args) to skip from unwrapping. Not used if `only_args` is not None.
        Default is None. If both are None, try to unwrap all arguments.

    See Also
    --------
    :func:`unwrap_output`
    :func:`unwrap_func`
    :func:`wrap_input`
    """

    def _wrapper_decorator(func):
        @functools.wraps(func)
        def _wrapper_wrap(*args, **kwargs):
            args, kwargs = _unwrap_input_data(*args, **kwargs,
                                              only_args=only_args,
                                              skip_args=skip_args)
            data = func(*args, **kwargs)
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
    def _wrapper_wrap(*args, **kwargs):
        data = func(*args, **kwargs)
        return _unwrap_output_data(data)
    return _wrapper_wrap


def wrap_func(inp=True, out=True, only_args=None, skip_args=None):
    """Decorator to wrap both arguments and output of a function.

    An object is wrapped only if it is an instance of :class:`vtkObject`
    or one of its subclasses.

    Parameters
    ----------
    inp : bool, optional
        If True, wrap input arguments. Default is True.
    out : bool, optional
        If True, wrap output. Default is True.
    only_args : int, str or list of int and str, optional
        List of positional indices (integers) and keys as strings (for keyword
        args) to wrap. This has preference over `skip_args`.
        Default is None.
    skip_args : int, str or list of int and str, optional
        List of positional indices (integers) and keys as strings (for keyword
        args) to skip from wrapping. Not used if `only_args` is not None.
        Default is None. If both are None, try to wrap all arguments.

    See Also
    --------
    :func:`wrap_input`
    :func:`wrap_output`
    :func:`unwrap_func`
    """

    def _wrapper_decorator(func):
        @functools.wraps(func)
        def _wrapper_wrap(*args, **kwargs):
            if inp:
                args, kwargs = _wrap_input_data(*args, **kwargs,
                                                only_args=only_args,
                                                skip_args=skip_args)
            data = func(*args, **kwargs)

            if out:
                return _wrap_output_data(data)
            return data
        return _wrapper_wrap

    return _wrapper_decorator


def unwrap_func(inp=True, out=True, only_args=None, skip_args=None):
    """Decorator to unwrap both arguments and output of a function.

    An object is unwrapped only if it is an instance of
    :class:`.BSVTKObjectWrapper` or one of its subclasses.

    Parameters
    ----------
    inp : bool, optional
        If True, unwrap input arguments. Default is True.
    out : bool, optional
        If True, unwrap output. Default is True.
    only_args : int, str or list of int and str, optional
        List of positional indices (integers) and keys as strings (for keyword
        args) to unwrap. This has preference over `skip_args`.
        Default is None.
    skip_args : int, str or list of int and str, optional
        List of positional indices (integers) and keys as strings (for keyword
        args) to skip from unwrapping. Not used if `only_args` is not None.
        Default is None. If both are None, try to unwrap all arguments.

    See Also
    --------
    :func:`unwrap_input`
    :func:`unwrap_output`
    :func:`wrap_func`
    """

    def _wrapper_decorator(func):
        @functools.wraps(func)
        def _wrapper_wrap(*args, **kwargs):
            if inp:
                args, kwargs = _unwrap_input_data(*args, **kwargs,
                                                  only_args=only_args,
                                                  skip_args=skip_args)
            data = func(*args, **kwargs)

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

      #. array_name (str, optional)
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
            if kwds['array_name'] is None:
                raise ValueError('Array name is None. Cannot append data.')
            ws.append_array(data, name=kwds['array_name'], at=to)
            return surf
        return _wrapper_append
    return _wrapper_decorator
