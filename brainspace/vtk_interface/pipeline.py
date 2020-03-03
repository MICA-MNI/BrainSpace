"""
Pipeline for VTK filters.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


from .decorators import wrap_input
from .wrappers.algorithm import BSAlgorithm
from .wrappers.data_object import BSDataObject


# From https://vtk.org/Wiki/VTK/Tutorials/New_Pipeline
# Outputs are referred to by port number while
# inputs are referred to by both their port number and connection number
# (because a single input port can have more than one connection)


@wrap_input(0, 1)
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

    if isinstance(ftr0, BSAlgorithm) and port0 >= ftr0.nop:
        raise ValueError("'{0}' only has {1} output ports.".
                         format(ftr0.__vtkname__, ftr0.nop))

    if port1 >= ftr1.nip:
        raise ValueError("'{0}' only accepts {1} input ports.".
                         format(ftr1.__vtkname__, ftr1.nip))

    if add_conn is True or type(add_conn) == int:
        if ftr1.nip > 1:
            raise ValueError("No support yet for 'add_conn' when filter "
                             "has more than 1 input ports.")

        pinfo = ftr1.GetInputPortInformation(port1)
        if pinfo.Get(ftr1.INPUT_IS_REPEATABLE()) == 0:
            raise ValueError("Input port {0} of '{1}' does not "
                             "accept multiple connections.".
                             format(ftr1.nip, ftr1.__vtkname__))

        if type(add_conn) == int:
            if not hasattr(ftr1, 'GetUserManagedInputs') or \
                    ftr1.GetUserManagedInputs() == 0:
                raise ValueError("Input port {0} of '{1}' does not accept "
                                 "connection number."
                                 .format(ftr1.nip, ftr1.__vtkname__))

    if isinstance(ftr0, BSAlgorithm):
        op = ftr0.GetOutputPort(port0)
        if add_conn is True:
            # Connection for only 1 input port. Not tested.
            ftr1.AddInputConnection(port1, op)
        elif type(add_conn) == int:
            # Connection for only 1 input port. Not tested.
            ftr1.SetInputConnectionByNumber(add_conn, op)
        else:
            ftr1.SetInputConnection(port1, op)

    elif isinstance(ftr0, BSDataObject):
        ftr0 = ftr0.VTKObject
        if add_conn is True:
            ftr1.AddInputData(ftr0)
        elif type(add_conn) == int:
            ftr1.SetInputDataByNumber(add_conn, ftr0)
        else:
            ftr1.SetInputDataObject(port1, ftr0)

    else:
        raise ValueError('Unknown input filter type: {0}'.format(type(ftr0)))

    return ftr1


@wrap_input(0)
def to_data(ftr, port=0):
    """Extract data from filter.

    Parameters
    ----------
    ftr : vtkAlgorithm or :class:`.BSAlgorithm`
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

    list_ports = [port] if port > -1 else range(ftr.nop)
    n_ports = len(list_ports)
    out = [None] * n_ports
    for i, port_id in enumerate(list_ports):
        ftr.Update(port_id)
        out[i] = ftr.GetOutputDataObject(port_id)

    if port > -1:
        return out[0]
    return out


@wrap_input(0)
def get_output(ftr, as_data=True, update=True, port=0):
    """Get output from filter.

    Parameters
    ----------
    ftr : vtkAlgorithm or :class:`.BSAlgorithm`
        Input filter.
    as_data : bool, optional
        Return data as BSDataObject instead of :class:`.BSAlgorithm`. If True,
        the filter is automatically updated. Default is True.
    update : bool, optional
        Update filter. Only used when `as_data` is False. Default is True.
    port : int or None, optional
        Output port to update or get data from. Only used when input is
        vtkAlgorithm. When port is -1, refers to all ports. When None, call
        Update() with no arguments. Not used, when `ftr` is a sink
        (i.e., 0 output ports), call Update(). Default is 0.

    Returns
    -------
    poly : BSAlgorithm or BSDataObject
        Returns filter or its output. If port is -1, returns all outputs in a
        list if ``as_data == True``.

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
        Input filters to serially connect. Each input takes one of the
        following formats:

        #. First filter in sequence: ``(f0, op=0)``

            * `f0` (vtkAlgorithm, :class:`.BSAlgorithm`, vtkDataObject or
              :class:`.BSDataObject`) - This is the first filter.
            * `op` (int, optional) - This is the output port of `f0`.
              Default is 0.

        #. Last filter in sequence: ``(ic=None, ip=0, fn)``

            * `ic` (int, optional) - This is the input connection of the
              input port `ip` of filter `fn`. Default is None.
            * `ip` (int, optional) - This is the input port of `fn`. Must be
              specified when `ic` is not None. Default is 0.
            * `fn` (vtkAlgorithm or :class:`.BSAlgorithm`) - This is the last
              filter.

        #. Intermediate filters: ``(ic=None, ip=0, fi, op=0)``

            * `ic` (int, optional) - This is the input connection of the
              input port `ip` of filter `fi`. Default is None.
            * `ip` (int, optional) - This is the input port of `fi`. Must be
              specified when `ic` is not None. Default is 0.
            * `fi` (vtkAlgorithm or :class:`.BSAlgorithm`) - This is a filter.
            * `op` (int, optional) - This is the output port of `fi`.
              Default is 0.

    as_data : bool, optional
        Return data instead of filter. If True, last filter is automatically
        updated. Default is True.
    update : bool, optional
        Update last filter. Only used when ``as_data == False``.
        Default is True.
    port : int, optional
        Port to update or get data from. When port is -1, refers to all ports.
        Default is 0.

    Returns
    -------
    output : BSAlgorithm or BSDataObject
        Last filter or its output.

    Examples
    --------

    In VTK:

    >>> # point source
    >>> ps = vtk.vtkPointSource()
    >>> ps.SetNumberOfPoints(100)
    >>> # delauny
    >>> dn = vtk.vtkDelaunay2D()
    >>> dn.SetTolerance(0.01)
    >>> dn.SetInputConnection(0, ps.GetOutputPort(0))
    >>> # smooth
    >>> sf = vtk.vtkWindowedSincPolyDataFilter()
    >>> sf.SetInputConnection(0, dn.GetOutputPort(0))
    >>> sf.SetNumberOfIterations(20)
    >>> # update and get output
    >>> sf.Update()
    >>> sf.GetOutput(0)
    (vtkCommonDataModelPython.vtkPolyData)0x7f0134fffb28

    With `serial_connect` function:

    >>> from brainspace.vtk_interface.pipeline import serial_connect
    >>> # point source
    >>> ps = vtk.vtkPointSource()
    >>> ps.SetNumberOfPoints(100)
    >>> # delauny
    >>> dn = vtk.vtkDelaunay2D()
    >>> dn.SetTolerance(0.01)
    >>> # smooth
    >>> sf = vtk.vtkWindowedSincPolyDataFilter()
    >>> sf.SetNumberOfIterations(20)
    >>> # Connection
    >>> serial_connect((ps, 0), (None, 0, dn, 0), (None, 0, sf), as_data=True,
    ...                port=0)
    <brainspace.vtk_interface.wrappers.BSPolyData at 0x7f0134efb048>
    >>> # This can be shortened, since no input connection is needed
    >>> serial_connect((ps, 0), (0, dn, 0), (0, sf), as_data=True, port=0)
    <brainspace.vtk_interface.wrappers.BSPolyData at 0x7f0134ee9128>
    >>> # And shortened even further since the default input and output
    >>> # ports are 0
    >>> serial_connect((ps,), (dn,), (sf,), as_data=True, port=0)
    <brainspace.vtk_interface.wrappers.BSPolyData at 0x7f0134ee92b0>
    >>> # This is the same
    >>> serial_connect(ps, dn, sf)
    <brainspace.vtk_interface.wrappers.BSPolyData at 0x7f0134eee898>
    """

    prev_f, prev_op = _map_input_filter(filters[0])

    for i, f1 in enumerate(filters[1:-1]):
        ic, ip, fi, op = _map_intermediate_filter(f1)
        prev_f = connect(prev_f, fi, port0=prev_op, port1=ip, add_conn=ic)
        prev_op = op

    ic, ip, fo = _map_output_filter(filters[-1])
    fo = connect(prev_f, fo, port0=prev_op, port1=ip, add_conn=ic)

    return get_output(fo, as_data=as_data, update=update, port=port)
