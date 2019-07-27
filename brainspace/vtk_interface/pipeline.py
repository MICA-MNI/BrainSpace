"""
VTK pipeline wrapper.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause

from vtkmodules.vtkCommonExecutionModelPython import vtkAlgorithm

from .wrappers import BSDataObject, BSAlgorithm
from .decorators import wrap_input


# From https://vtk.org/Wiki/VTK/Tutorials/New_Pipeline
# Outputs are referred to by port number while
# inputs are referred to by both their port number and connection number
# (because a single input port can have more than one connection)


@wrap_input(only_args=[0, 1])
def connect(ftr0, ftr1, port0=0, port1=0, add_conn=False):
    """Connection of two filters.

    Connects the output port `port0` of filter `ftr0` with the input port
    `port1` of filter `ftr1`.

    Parameters
    ----------
    ftr0 : vtkAlgorithm, vtkDataSet, BSAlgorithm or BSDataSet
        The input filter. May be a filter or dataset.
    ftr1 : vtkAlgorithm or BSAlgorithm
        The output filter.
    port0 : int, optional
        Output port of `ftr0`. Not used if `ftr0` is a dataset. Default is 0.
    port1 : int, optional
        Input port of `ftr1`. Default is 0.
    add_conn : bool or int, optional
        Connect to specific connection of `port1`.  If False, use
        `SetInputConnection` or `SetInputData` (all other added connections to
        `port1` are removed). Otherwise, use `AddInputConnection` or
        `AddInputData`. If int, add to given connection (e.g.,
        `SetInputConnectionByNumber` or `SetInputDataByNumber`). Only used if
        `port1` accepts more than one connection (i.e., repeatable).
        Default is False.

    Returns
    -------
    ftr1 : BSAlgorithm
        Returns (wrapped) `frt1` after connecting it with the input filter.

    """

    if isinstance(ftr0, BSAlgorithm):
        if port0 >= ftr0.nop:
            raise ValueError("'{0}' only has {1} output ports."
                             .format(ftr0.__vtkname__, ftr0.nop))

    if port1 >= ftr1.nip:
        raise ValueError("'{0}' only accepts {1} input ports.".
                         format(ftr1.__vtkname__, ftr1.nip))

    if type(add_conn) == bool and add_conn or type(add_conn) == int:
        if ftr1.nip > 1:
            raise ValueError("No support yet for 'add_conn' when filter "
                             "has more than 1 input ports.")

        pinfo = ftr1.GetInputPortInformation(port1)
        if pinfo.Get(ftr1.INPUT_IS_REPEATABLE()) == 0:
            raise ValueError("Input port {0} of '{1}' does not accept multiple "
                             "connections.".format(ftr1.nip, ftr1.__vtkname__))

        if type(add_conn) == int:
            if not hasattr(ftr1, 'GetUserManagedInputs') or \
                    ftr1.GetUserManagedInputs() == 0:
                raise ValueError("Input port {0} of '{1}' does not accept "
                                 "connection number.".format(ftr1.nip,
                                                             ftr1.__vtkname__))

    if isinstance(ftr0, BSAlgorithm):
        if type(add_conn) == bool and add_conn:
            # Connection for only 1 input port. Not tested.
            ftr1.AddInputConnection(port1, ftr0.GetOutputPort(port0))
        elif type(add_conn) == int:
            # Connection for only 1 input port. Not tested.
            ftr1.SetInputConnectionByNumber(add_conn, ftr0.GetOutputPort(port0))
        else:
            ftr1.SetInputConnection(port1, ftr0.GetOutputPort(port0))

    elif isinstance(ftr0, BSDataObject):
        ftr0 = ftr0.VTKObject
        if type(add_conn) == bool and add_conn:
            ftr1.AddInputData(ftr0)
        elif type(add_conn) == int:
            ftr1.SetInputDataByNumber(add_conn, ftr0)
        else:
            ftr1.SetInputDataObject(port1, ftr0)

    else:
        raise ValueError('Unknown input filter type: {0}'.format(type(ftr0)))

    return ftr1


@wrap_input(only_args=0)
def to_data(ftr, port=0):
    """Extract data from filter.

    Parameters
    ----------
    ftr : vtkAlgorithm or BSAlgorithm
        Input filter.
    port : int, optional
        Port to get data from. When port is -1, refers to all ports.
        Default is 0.

    Returns
    -------
    data : BSDataObject or list of BSDataObject
        Returns the output of the filter. If port is -1 and number of output
        ports > 1, then return list of outputs.


    Notes
    -----
    Filters are automatically updated to get the output.

    """

    n_ports = 1 if port > -1 else ftr.nop
    out = [None] * n_ports
    for i in range(n_ports):
        ftr.Update(i)
        out[i] = ftr.GetOutputDataObject(i)

    if port > -1:
        return out[0]
    return out


@wrap_input(only_args=0)
def get_output(ftr, as_data=True, update=True, port=0):
    """Get output from filter.

    Parameters
    ----------
    ftr : vtkAlgorithm or BSAlgorithm
        Input filter.
    as_data : bool, optional
        Return data as BSDataObject instead of BSAlgorithm. If True, the filter
        is automatically updated. Default is True.
    update : bool, optional
        Update filter. Only used when `as_data` is False. Default is True.
    port : int or None, optional
        Output port to update or get data from. Only used when input is
        vtkAlgorithm. When port is -1, refers to all ports. When None, call
        Update() with no argument. Not used, when `ftr` is a sink
        (i.e., 0 output ports), call Update(). Default is 0.

    Returns
    -------
    poly : BSAlgorithm or BSDataObject
        Returns filter or its output. If port is -1, returns all outputs in a
        list if ``as_data=True``.

    """

    if as_data:
        return to_data(ftr, port=port)

    if update:
        if port is None or ftr.is_sink:
            try:
                ftr.Write()
            except AttributeError:
                ftr.Update()

        elif port > -1:
            ftr.Update(port)
        else:
            # In vtkAlgorithm.cxx, lines 1474-1482
            # vtkAlgorithm()->Update() defaults to vtkExecutive()->Update(0)
            # if the number of output ports of the algorithm > 0. Otherwise,
            # vtkExecutive()->Update(-1)
            # In vtkExecutive.cxx lines 310-318, is the same
            # In vtkDemandDrivenPipeline.cxx, NeedToExecuteData function,
            # lines 1051-1064
            # if(outputPort >= 0) Update port outputPort!
            # // No port is specified.  Check all ports.
            # for(int i=0; i < this->Algorithm->GetNumberOfOutputPorts(); ++i)
            # When No input port is specified, they go through all ports

            # Update individually, just in case
            for i in range(ftr.nop):
                ftr.Update(i)

    return ftr


def _map_input_filter(f):
    if not isinstance(f, (list, tuple)):
        return f, 0  # assume is only filter

    if isinstance(f, list):
        f = tuple(f)

    if len(f) == 1:  # (fn,)
        f += (0,)
    elif len(f) > 2:
        raise ValueError('Cannot recognize input filter {0}.'.format(f))

    return f


def _map_output_filter(f):
    if not isinstance(f, (list, tuple)):
        return False, 0, f  # assume is only filter

    if isinstance(f, list):
        f = tuple(f)

    if len(f) == 1:         # (fn,)
        f = (False, 0) + f
    elif len(f) == 2:       # (ip, fn)
        f = (False,) + f
    elif len(f) > 3:
        raise ValueError('Cannot recognize input filter {0}.'.format(f))

    return f


def _map_intermediate_filter(f):
    if not isinstance(f, (list, tuple)):
        return False, 0, f, 0  # assume is only filter

    if isinstance(f[-1], int):     # (..., op)
        return _map_output_filter(f[:-1]) + (f[-1],)
    return _map_output_filter(f) + (0,)


def serial_connect(*filters, as_data=True, update=True, port=0):
    """Connect filters serially.

    Parameters
    ----------
    *filters : sequence of tuple or list
        Input filters to serially connect. Each input takes the one of the
        following formats:
        0. First filter in sequence: (f0, op=0)
            0.0. `f0` is the first filter: vtkAlgorithm, BSAlgorithm,
            vtkDataObject or BSDataObject
            0.1. `op` is the output port of `f0`: int, optional. Default is 0.
        1. Last filter in sequence: (ic=None, ip=0, fn)
            1.0. `ic` is the input connection index: int, optional.
                Default is False.
            1.1. `ip` is the input port: int, optional. Must be specified when
                `ic` is not None. Default is 0.
            1.2. `fn` is the last filter: vtkAlgorithm or BSAlgorithm
        2. Intermediate filters: (ic=None, ip=0, fi, op=0)
    as_data : bool, optional
        Return data instead of filter. If True, last filter is automatically
        updated. Default is True.
    update : bool, optional
        Update last filter. Only used when `as_data` is False, Default is True.
    port : int, optional
        Port to update or get data from. When port is -1, refers to all ports.
        Default is 0.

    Returns
    -------
    output : BSAlgorithm, BSDataObject
        Last filter or its output.

    """

    prev_f, prev_op = _map_input_filter(filters[0])

    for i, f1 in enumerate(filters[1:-1]):
        ic, ip, fi, op = _map_intermediate_filter(f1)
        prev_f = connect(prev_f, fi, port0=prev_op, port1=ip, add_conn=ic)
        prev_op = op

    ic, ip, fo = _map_output_filter(filters[-1])
    fo = connect(prev_f, fo, port0=prev_op, port1=ip, add_conn=ic)

    return get_output(fo, as_data=as_data, update=update, port=port)
