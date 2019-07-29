"""
Wrappers for some VTK objects.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause

import string
import warnings
import numpy as np

from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.vtkCommonExecutionModelPython import vtkAlgorithm
from vtkmodules.vtkCommonCorePython import vtkObjectBase, vtkLookupTable
from vtkmodules.vtkRenderingCorePython import vtkMapper

from .checks import (get_cell_types, get_number_of_cell_types,
                     has_unique_cell_type, has_only_triangle,
                     has_only_line, has_only_vertex)

from .base import BSVTKObjectWrapper
from .decorators import wrap_output, unwrap_input


def _generate_random_string(size=20, n_reps=10, exclude_list=None,
                            random_state=None):

    if isinstance(random_state, np.random.RandomState):
        r = random_state
    else:
        r = np.random.RandomState(random_state)

    choices = list(string.ascii_letters + string.digits)
    if exclude_list is None:
        return ''.join(r.choice(choices, size=size))

    for i in range(n_reps):
        s = ''.join(r.choice(choices, size=size))
        if s not in exclude_list:
            return s

    return None


class BSAlgorithm(BSVTKObjectWrapper):

    def __init__(self, vtkobject, **kwargs):
        super().__init__(vtkobject, **kwargs)

    @property
    def nip(self):
        return self.GetNumberOfInputPorts()

    @property
    def nop(self):
        return self.GetNumberOfOutputPorts()

    @property
    def nic(self):
        return self.GetTotalNumberOfInputConnections()

    @property
    def is_source(self):
        return self.nip == 0

    @property
    def is_sink(self):
        return self.nop == 0

    @property
    def is_filter(self):
        return not (self.is_source and self.is_sink)

    @wrap_output
    def GetInputDataObject(self, *args):
        return self.VTKObject.GetInputDataObject(*args)

    @wrap_output
    def GetOutputDataObject(self, *args):
        return self.VTKObject.GetOutputDataObject(*args)

    @unwrap_input(skip_args=0)
    def SetInputDataObject(self, *args):
        self.VTKObject.SetInputDataObject(*args)

    @unwrap_input(skip_args=0)
    def AddInputDataObject(self, *args):
        self.VTKObject.AddInputDataObject(*args)


# Wrap vtk data objects from dataset_adapter
class BSDataObject(dsa.DataObject, BSVTKObjectWrapper):
    def __init__(self, vtkobject=None):
        super().__init__(vtkobject)

    @property
    def field_keys(self):
        return self.FieldData.keys()

    @property
    def n_field_data(self):
        return len(self.FieldData.keys())


class BSTable(dsa.Table, BSDataObject):
    pass


class BSCompositeDataSet(dsa.CompositeDataSet, BSDataObject):
    pass


class BSDataSet(dsa.DataSet, BSDataObject):

    @property
    def point_keys(self):
        return self.PointData.keys()

    @property
    def cell_keys(self):
        return self.CellData.keys()

    @property
    def n_point_data(self):
        return len(self.PointData.keys())

    @property
    def n_cell_data(self):
        return len(self.CellData.keys())

    @property
    def n_points(self):
        return self.GetNumberOfPoints()

    @property
    def n_cells(self):
        return self.GetNumberOfCells()

    @property
    def cell_types(self):
        return get_cell_types(self)

    @property
    def number_of_cell_types(self):
        return get_number_of_cell_types(self)

    @property
    def has_unique_cell_type(self):
        return has_unique_cell_type(self)

    @property
    def has_only_triangle(self):
        return has_only_triangle(self)

    @property
    def has_only_line(self):
        return has_only_line(self)

    @property
    def has_only_vertex(self):
        return has_only_vertex(self)

    def append_array(self, array, name=None, at=None, convert_bool='warn',
                     overwrite='warn'):
        """Append array to attributes.

        Parameters
        ----------
        array : 1D or 2D ndarray
            Array to append to the dataset.
        name : str or None, optional
            Array name. If None, a random string is generated and returned.
            Default is None.
        at : {'point', 'cell', 'field', 'p', 'c', 'f'} or None, optional.
            Attribute to append data to. Points (i.e., 'point' or 'p'),
            cells (i.e., 'cell' or 'c') or field (i.e., 'field' or 'f') data.
            If None, it will attempt to append data to the attributes with
            the same number of elements. Only considers points and cells.
            If both have the same number of elements or the size of the array
            does not coincide with any of them, it raises an exception.
            Default is None.
        convert_bool : bool or {'warn', 'raise'}, optional
            If True append array after conversion to uint8. If False,
            array is not appended. If 'warn', issue a warning but append
            the array. If raise, raise an exception.
        overwrite : bool or {'warn', 'raise'}, optional
            If True append array even if its name already exists. If False,
            array is not appended, issue a warning. If 'warn', issue a warning
            but append the array. If raise, raise an exception.

        Returns
        -------
        name : str
            Array name used to append the array to the dataset.

        """

        # Check bool
        if np.issubdtype(array.dtype, np.bool_):
            if convert_bool == 'raise':
                raise ValueError('VTK does not accept boolean arrays.')
            if convert_bool in ['warn', True]:
                array = array.astype(np.uint8)
                if convert_bool == 'warn':
                    warnings.warn('Input array is boolean. Casting to uint8.')
            else:
                warnings.warn('Array was not appended. Input array is boolean.')
                return None

        # Check array name
        if name is None:
            exclude_list = self.point_keys + self.cell_keys + self.field_keys
            name = _generate_random_string(size=20, exclude_list=exclude_list)
            if name is None:
                raise ValueError('Cannot generate an name for this array. '
                                 'Please provide an array name.')

        # Check shapes
        shape = np.array(array.shape)
        to_point = np.any(shape == self.n_points)
        to_cell = np.any(shape == self.n_cells)

        if at is None:
            if to_cell and to_point:
                raise ValueError(
                    'Cannot figure out the attributes to append the '
                    'data to. Please provide the attributes to use.')
            if to_point:
                at = 'point'
            elif to_cell:
                at = 'cell'
            else:
                raise ValueError('Array shape is not valid. Please provide the '
                                 'attributes to use.')

        def _array_overwrite(attributes, has_same_shape):
            if has_same_shape in [True, None]:
                if name is not attributes.keys() or overwrite is True:
                    attributes.append(array, name)
                elif overwrite == 'warn':
                    warnings.warn('Array name already exists. Updating data.')
                    attributes.append(array, name)
                elif overwrite == 'raise':
                    raise ValueError('Array name already exists.')
                else:
                    warnings.warn('Array was not appended. Array name already '
                                  'exists.')
            else:
                raise ValueError('Array shape is not valid.')

        if at in ['point', 'p']:
            _array_overwrite(self.PointData, to_point)
        elif at in ['cell', 'c']:
            _array_overwrite(self.CellData, to_cell)
        elif at in ['field', 'f']:
            _array_overwrite(self.FieldData, None)
        else:
            raise ValueError('Unknown PolyData attributes: \'{0}\''.format(at))

        return name

    def remove_array(self, name=None, at=None):
        """Remove array from vtk dataset.

        Parameters
        ----------
        surf : vtkDataset or BSDataSet
            Input surface.
        name : str, list of str or None, optional
            Array name to remove. If None, remove all arrays. Default is None.
        at : {'point', 'cell', 'field', 'p', 'c', 'f'} or None, optional.
            Attributes to remove the array from. Points (i.e., 'point' or 'p'),
            cells (i.e., 'cell' or 'c') or field (i.e., 'field' or 'f').
            If None, remove array name from all attributes. Default is None.

        """

        if name is None:
            name = []
            if at in ['point', 'p', None]:
                name += self.point_keys
            if at in ['cell', 'c', None]:
                name += self.cell_keys
            if at in ['field', 'f', None]:
                name += self.field_keys

        if not isinstance(name, list):
            name = [name]

        for k in name:
            if at in ['point', 'p', None] and k in self.point_keys:
                self.GetPointData().RemoveArray(k)
            if at in ['cell', 'c', None] and k in self.cell_keys:
                self.GetCellData().RemoveArray(k)
            if at in ['field', 'f', None] and k in self.field_keys:
                self.GetFieldData().RemoveArray(k)

    def get_array(self, name=None, at=None, return_name=False):
        """Return array in attributes.

            Parameters
            ----------
            name : str, list of str or None, optional
                Array names. If None, return all arrays. Cannot be None
                if ``at=None``. Default is None.
            at : {'point', 'cell', 'field', 'p', 'c', 'f'} or None, optional.
                Attributes to get the array from. Points (i.e., 'point' or 'p'),
                cells (i.e., 'cell' or 'c') or field (i.e., 'field' or 'f').
                If None, get array name from all attributes that have an array
                with the same array name. Cannot be None if ``name=None``.
                Default is None.
            return_name : bool, optional
                Whether to return array names too. Default is False.

            Returns
            -------
            arrays : VTKArray or list of VTKArray
                Data arrays. None is returned if `name` does not exist.

            names : str or list of str
                Names of returned arrays. Only if ``return_name=True``.

            """

        if name is None and at is None:
            raise ValueError('Please specify \'name\' or \'at\'.')

        if name is None:
            if at in ['point', 'p']:
                name = self.point_keys
            elif at in ['cell', 'c']:
                name = self.cell_keys
            else:
                name = self.field_keys

        is_list = True
        if not isinstance(name, list):
            is_list = False
            name = [name]

        arrays = [None] * len(name)
        for i, k in enumerate(name):
            out = []
            if at in ['point', 'p', None] and k in self.point_keys:
                out.append(self.PointData[k])
            if at in ['cell', 'c', None] and k in self.cell_keys:
                out.append(self.CellData[k])
            if at in ['field', 'f', None] and k in self.field_keys:
                out.append(self.FieldData[k])

            if len(out) == 1:
                arrays[i] = out[0]
            elif len(out) > 1:
                raise ValueError(
                    "Array name is present in more than one attribute."
                    "Please specify 'at'.")

        if not is_list:
            arrays, name = arrays[0], name[0]

        if return_name:
            return arrays, name
        return arrays


class BSPointSet(dsa.PointSet, BSDataSet):
    pass


class BSPolyData(dsa.PolyData, BSPointSet):

    def get_cells2D(self):
        if not self.has_unique_cell_type:
            raise ValueError('PolyData has different types of cells.')
        cells = self.Polygons
        return cells.reshape(-1, cells[0] + 1)[:, 1:]


class BSUnstructuredGrid(dsa.UnstructuredGrid, BSPointSet):
    pass


class BSGraph(dsa.Graph, BSDataObject):
    pass


class BSMolecule(dsa.Molecule, BSDataObject):
    pass


class BSLookupTable(BSVTKObjectWrapper):
    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)

    def SetTable(self, table):
        if isinstance(table, np.ndarray):
            table = numpy_to_vtk(table)
        self.VTKObject.SetTable(table)

    def GetTable(self):
        table = self.VTKObject.GetTable()
        return vtk_to_numpy(table)

    @property
    def n_values(self):
        # return self.numberoftablevalues
        return self.VTKObject.GetNumberOfTableValues()

    @n_values.setter
    def n_values(self, n):
        self.VTKObject.SetNumberOfTableValues(n)

    @property
    def n_colors(self):
        return self.VTKObject.GetNumberOfColors()

    @n_colors.setter
    def n_colors(self, n):
        self.VTKObject.SetNumberOfColors(n)


class BSMapper(BSAlgorithm):

    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)

    @wrap_output
    def GetInput(self):
        return self.VTKObject.GetInput()

    @wrap_output
    def GetInputAsDataSet(self):
        return self.VTKObject.GetInputAsDataSet()

    def SetLookupTable(self, lut=None, **kwargs):
        if lut is None:
            lut = BSLookupTable(**kwargs)
        else:
            lut = wrap_vtk(lut)
            lut.setVTK(**kwargs)
        self.VTKObject.SetLookupTable(lut.VTKObject)
        return lut

    @wrap_output
    def GetLookupTable(self):
        return self.VTKObject.GetLookupTable()

    def SetArrayName(self, name):
        self.VTKObject.SetArrayAccessMode(1)
        self.VTKObject.SetArrayName(name)

    def SetArrayId(self, idx):
        self.VTKObject.SetArrayAccessMode(0)
        self.VTKObject.SetArrayId(idx)


class BSPolyDataMapper(BSMapper):

    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)

    @unwrap_input(skip_args=0)
    def SetInputData(self, polyData):
        self.VTKObject.SetInputData(polyData)


class BSActor(BSVTKObjectWrapper):

    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)
        self._property = BSWrapVTKObject(self.VTKObject.GetProperty())

    def _handle_call(self, key, name, args):
        try:
            return super()._handle_call(key, name, args)
        except (AttributeError, KeyError):
            return self._property._handle_call(key, name, args)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except (AttributeError, KeyError):
            return self._property.__getattr__(name)

    def __setattr__(self, name, value):
        try:
            super().__setattr__(name, value)
        except (AttributeError, KeyError):
            self._property.__setattr__(name, value)

    def SetMapper(self, mapper=None, **kwargs):
        if mapper is None:
            mapper = BSPolyDataMapper(vtkobject=mapper, **kwargs)
        else:
            mapper = BSWrapVTKObject(mapper)
            mapper.setVTK(**kwargs)
        self.VTKObject.SetMapper(mapper.VTKObject)
        return mapper

    @wrap_output
    def GetMapper(self):
        return self.VTKObject.GetMapper()

    def GetProperty(self):
        return self._property


class BSRenderer(BSVTKObjectWrapper):
    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)

    def AddActor(self, actor=None, **kwargs):
        actor = BSActor(vtkobject=actor, **kwargs)
        self.VTKObject.AddActor(actor.VTKObject)
        return actor


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
    return isinstance(obj, vtkObjectBase)


def BSWrapVTKObject(obj):
    """Returns a Numpy friendly wrapper of a vtk objects."""

    if obj is None or is_wrapper(obj):
        return obj
    if not is_vtk(obj):
        raise ValueError('Unknown object type: {0}'.format(type(obj)))

    cls_name = obj.__vtkname__

    # Is this really needed? is there a vtk class that doesn't start with vtk?
    if not cls_name.startswith('vtk'):
        raise ValueError('Unknown object type: {0}'.format(type(obj)))

    cls_name = cls_name[3:]
    try:  # First, see if wrapper class is implemented
        return eval('BS' + cls_name)(obj)
    except:
        pass

    # Try to handle abstract classes such as vtkActor, vtkPolyDataMapper
    # That instantiate to vtkOpenGLActor, vtkOpenGLPolyDataMapper
    if cls_name.startswith('OpenGL'):
        cls_name = cls_name[6:]
        try:
            return eval('BS' + cls_name)(obj)
        except:
            pass

    if isinstance(obj, vtkMapper):
        return BSMapper(obj)

    if isinstance(obj, vtkLookupTable):
        return BSLookupTable(obj)

    if isinstance(obj, vtkAlgorithm):
        return BSAlgorithm(obj)

    # Fall back to generic wrapper
    return BSVTKObjectWrapper(obj)


def wrap_vtk(obj, return_input=False, **kwargs):
    """Wrap input object to BSVTKObjectWrapper or one of its subclasses.

    Parameters
    ----------
    obj : vtkObjctect or BSVTKObjectWrapper
        Input object.
    return_input : bool, optional
        If True, also return original data. Otherwise, only return warped
        object. Default is False.
    kwargs : kwds, optional
        Additional keyword parameters are passed to vtk object.

    Returns
    -------
    wrapper : BSVTKObjectWrapper
        The wrapped object.
    obj : vtkObject or BSVTKObjectWrapper
        The input object. Only provided if `return_input` is True.

    """

    wobj = BSWrapVTKObject(obj)
    if len(kwargs) > 0:
        wobj.setVTK(**kwargs)
    return (wobj, obj) if return_input else wobj
