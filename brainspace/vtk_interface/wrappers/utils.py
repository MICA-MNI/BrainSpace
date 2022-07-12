"""
Utility functions for vtk wrappers.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import re
import string
from collections import defaultdict

import numpy as np

from vtk.util.vtkConstants import VTK_BIT, VTK_STRING

# VTK_UNICODE_STRING has been removed since VTK 9.2.
# See: https://gitlab.kitware.com/vtk/vtk/-/blob/master/Documentation/release/9.2.md
try:
    from vtk.util.vtkConstants import VTK_UNICODE_STRING
except ImportError:
    VTK_UNICODE_STRING = VTK_STRING


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
    >>> import vtk
    >>> from brainspace.vtk_interface.wrappers.base import get_vtk_methods
    >>> vtk_map = get_vtk_methods(vtk.vtkPolyDataMapper)
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
    >>> import vtk
    >>> from brainspace.vtk_interface.wrappers.base import call_vtk
    >>> m = vtk.vtkPolyDataMapper()

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
        try:
            return getattr(obj, method)()
        except:
            return getattr(obj, method)(None)

    if isinstance(args, dict):
        return getattr(obj, method)(**args)

    # If iterable try first with multiple arguments
    if isinstance(args, (tuple, list)):
        try:
            return getattr(obj, method)(*args)
        except TypeError:
            pass

    return getattr(obj, method)(args)


def generate_random_string(size=20, n_reps=10, exclude_list=None,
                           random_state=None):
    """Generate random string.

    Parameters
    ----------
    size : int, optional
        String length. Default is 20.
    n_reps : int, optional
        Number of attempts to generate string that in not in `exclude_list`.
        Default is 10.
    exclude_list : list of str, optional
        List of string to exclude. Default is None.
    random_state : int or None, optional
        Random state. Default is None.

    Returns
    -------
    str
        Random string.
    """

    if isinstance(random_state, np.random.RandomState):
        rs = random_state
    else:
        rs = np.random.RandomState(random_state)

    choices = list(string.ascii_letters + string.digits)
    if exclude_list is None:
        return ''.join(rs.choice(choices, size=size))

    for i in range(n_reps):
        s = ''.join(rs.choice(choices, size=size))
        if s not in exclude_list:
            return s

    return None


def is_numpy_string(dtype):
    if np.issubdtype(dtype, np.string_) or np.issubdtype(dtype, np.unicode_):
        return True
    return False


def is_vtk_string(vtype):
    return vtype in [VTK_STRING, VTK_UNICODE_STRING]
