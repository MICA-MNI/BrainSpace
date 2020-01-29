"""
Wrappers for VTK data objects.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import warnings

import numpy as np

from vtk.numpy_interface import dataset_adapter as dsa
from vtk.util.vtkConstants import (VTK_ID_TYPE, VTK_POLY_VERTEX, VTK_POLY_LINE,
                                   VTK_TRIANGLE, VTK_POLYGON, VTK_QUAD)

from .base import BSVTKObjectWrapper
from .misc import BSCellArray
from .utils import generate_random_string
from ..checks import (get_cell_types, get_number_of_cell_types, has_only_line,
                      has_only_vertex, has_only_triangle, has_unique_cell_type,
                      has_only_quad)


# Wrap vtk data objects from dataset_adapter
class BSDataObject(BSVTKObjectWrapper, dsa.DataObject):
    """Wrapper for vtkDataObject."""

    def __init__(self, vtkobject=None, **kwargs):
        super().__init__(vtkobject=vtkobject, **kwargs)

    @property
    def field_keys(self):
        """list of str: Returns keys of field data."""
        return self.FieldData.keys()

    @property
    def n_field_data(self):
        """int: Returns number of entries in field data."""
        return len(self.FieldData.keys())


class BSTable(BSDataObject, dsa.Table):
    """Wrapper for vtkTable."""
    pass


class BSCompositeDataSet(BSDataObject, dsa.CompositeDataSet):
    """Wrapper for vtkCompositeDataSet."""
    pass


class BSDataSet(BSDataObject, dsa.DataSet):
    """Wrapper for vtkDataSet."""

    @property
    def point_keys(self):
        """list of str: Returns keys of point data."""
        return self.PointData.keys()

    @property
    def cell_keys(self):
        """list of str: Returns keys of cell data."""
        return self.CellData.keys()

    @property
    def n_point_data(self):
        """int: Returns number of entries in point data."""
        return len(self.PointData.keys())

    @property
    def n_cell_data(self):
        """int: Returns number of entries in cell data."""
        return len(self.CellData.keys())

    @property
    def n_points(self):
        """int: Returns number of points."""
        return self.GetNumberOfPoints()

    @property
    def n_cells(self):
        """int: Returns number of cells."""
        return self.GetNumberOfCells()

    @property
    def cell_types(self):
        """list of int: Returns cell types of the object."""
        return get_cell_types(self)

    @property
    def number_of_cell_types(self):
        """int: Returns number of cell types."""
        return get_number_of_cell_types(self)

    @property
    def has_unique_cell_type(self):
        """bool: Returns True if object has a unique cell type.
        False, otherwise."""
        return has_unique_cell_type(self)

    @property
    def has_only_quad(self):
        """bool: Returns True if object has only quad cells.
        False, otherwise."""
        return has_only_quad(self)

    @property
    def has_only_triangle(self):
        """bool: Returns True if object has only triangles.
        False, otherwise."""
        return has_only_triangle(self)

    @property
    def has_only_line(self):
        """bool: Returns True if object has only lines. False, otherwise."""
        return has_only_line(self)

    @property
    def has_only_vertex(self):
        """bool: Returns True if object has only vertex cells.
        False, otherwise."""
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
                warnings.warn('Array was not appended. Input array is '
                              'boolean.')
                return None

        # Check array name
        if name is None:
            exclude_list = self.point_keys + self.cell_keys + self.field_keys
            name = generate_random_string(size=20, exclude_list=exclude_list)
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
                raise ValueError('Array shape is not valid. Please provide '
                                 'the attributes to use.')

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
                if ``at == None``. Default is None.
            at : {'point', 'cell', 'field', 'p', 'c', 'f'} or None, optional.
                Attributes to get the array from. Points (i.e., 'point' or
                'p'), cells (i.e., 'cell' or 'c') or field (i.e., 'field' or
                'f'). If None, get array name from all attributes that have an
                array with the same array name. Cannot be None
                if ``name == None``. Default is None.
            return_name : bool, optional
                Whether to return array names too. Default is False.

            Returns
            -------
            arrays : VTKArray or list of VTKArray
                Data arrays. None is returned if `name` does not exist.

            names : str or list of str
                Names of returned arrays. Only if ``return_name == True``.

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


class BSPointSet(BSDataSet, dsa.PointSet):
    """Wrapper for vtkPointSet."""
    pass


class BSPolyData(BSPointSet, dsa.PolyData):
    """Wrapper for vtkPolyData."""

    def GetCells2D(self):
        """Return cells as a 2D ndarray.

        Returns
        -------
        cells : 2D ndarray, shape = (n_points, n)
            PolyData cells.

        Raises
        ------
        ValueError
            If PolyData has different cell types.
        """
        if self.has_only_triangle or self.has_only_quad:
            return self.GetPolys2D()
        if self.has_only_line:
            return self.GetLines2D()
        if self.has_only_vertex:
            return self.GetVerts2D()
        raise ValueError('Cell type not supported.')

    def GetVerts(self):
        """Returns the verts as a 1D VTKArray instance."""
        if not self.VTKObject.GetVerts():
            return None
        return dsa.vtkDataArrayToVTKArray(
            self.VTKObject.GetVerts().GetData(), self)

    def GetLines(self):
        """Returns the lines as a 1D VTKArray instance."""
        if not self.VTKObject.GetLines():
            return None
        return dsa.vtkDataArrayToVTKArray(
            self.VTKObject.GetLines().GetData(), self)

    def GetPolys(self):
        """Returns the polys as a 1D VTKArray instance."""
        return self.GetPolygons()

    def GetVerts2D(self):
        """Returns the verts as a 2D VTKArray instance.

        Returns
        -------
        verts : 2D ndarray, shape = (n_points, n)
            PolyData verts.

        Raises
        ------
        ValueError
            If PolyData has different vertex types.
        """

        v = self.GetVerts()
        if v is None:
            return v

        if VTK_POLY_VERTEX not in self.cell_types:
            return v.reshape(-1, v[0] + 1)[:, 1:]
        raise ValueError('PolyData contains different vertex types')

    def GetLines2D(self):
        """Returns the lines as a 2D VTKArray instance.

        Returns
        -------
        lines : 2D ndarray, shape = (n_points, n)
            PolyData lines.

        Raises
        ------
        ValueError
            If PolyData has different line types.
        """

        v = self.GetLines()
        if v is None:
            return v

        if VTK_POLY_LINE not in self.cell_types:
            return v.reshape(-1, v[0] + 1)[:, 1:]
        raise ValueError('PolyData contains different line types')

    def GetPolys2D(self):
        """Returns the polys as a 2D VTKArray instance.

        Returns
        -------
        polys : 2D ndarray, shape = (n_points, n)
            PolyData polys.

        Raises
        ------
        ValueError
            If PolyData has different poly types.
        """

        v = self.GetPolys()
        if v is None:
            return v

        ct = self.cell_types
        if np.isin([VTK_QUAD, VTK_TRIANGLE], ct).all() or VTK_POLYGON in ct:
            raise ValueError('PolyData contains different poly types')
        return v.reshape(-1, v[0] + 1)[:, 1:]

    @staticmethod
    def _numpy2cells(cells):
        if cells.ndim == 1:
            offset = 0
            n_cells = 0
            while offset < cells.size:
                offset += cells[offset] + 1
                n_cells += 1
            vtk_cells = cells
        else:
            n_cells, n_points_cell = cells.shape
            vtk_cells = np.empty((n_cells, n_points_cell + 1),
                                 dtype=np.uintp)
            vtk_cells[:, 0] = n_points_cell
            vtk_cells[:, 1:] = cells
            vtk_cells = vtk_cells.ravel()

        # cells = dsa.numpyTovtkDataArray(vtk_cells, array_type=VTK_ID_TYPE)
        ca = BSCellArray()
        ca.SetCells(n_cells, vtk_cells)
        return ca.VTKObject

    def SetVerts(self, verts):
        """Set verts.

        Parameters
        ----------
        verts : 1D or 2D ndarray
            If 2D, shape = (n_points, n), and n is the number of points per
            vertex. All verts must use the same number of points.

        """

        if isinstance(verts, np.ndarray):
            verts = self._numpy2cells(verts)
        self.VTKObject.SetVerts(verts)

    def SetLines(self, lines):
        """Set lines.

        Parameters
        ----------
        lines : 1D or 2D ndarray
            If 2D, shape = (n_points, n), and n is the number of points per
            line. All lines must use the same number of points.

        """
        if isinstance(lines, np.ndarray):
            lines = self._numpy2cells(lines)
        self.VTKObject.SetLines(lines)

    def SetPolys(self, polys):
        """Set polys.

        Parameters
        ----------
        polys : 1D or 2D ndarray
            If 2D, shape = (n_points, n), and n is the number of points per
            poly. All polys must use the same number of points.

        """
        if isinstance(polys, np.ndarray):
            polys = self._numpy2cells(polys)
        self.VTKObject.SetPolys(polys)

    @property
    def polys(self):
        """Return polys as a 1D VTKArray."""
        return self.GetPolys()

    @property
    def polys2D(self):
        """Return polys as a 2D VTKArray if possible."""
        return self.GetPolys2D()

    @polys.setter
    def polys(self, polys):
        self.SetPolys(polys)

    @property
    def lines(self):
        """Return lines as a 1D VTKArray."""
        return self.GetLines()

    @property
    def lines2D(self):
        """Return lines as a 2D VTKArray if possible."""
        return self.GetLines2D()

    @lines.setter
    def lines(self, lines):
        self.SetLines(lines)

    @property
    def verts(self):
        """Return verts as a 1D VTKArray."""
        return self.GetVerts()

    @property
    def verts2D(self):
        """Return verts as a 2D VTKArray if possible."""
        return self.GetVerts2D()

    @verts.setter
    def verts(self, verts):
        self.SetVerts(verts)


class BSUnstructuredGrid(BSPointSet, dsa.UnstructuredGrid):
    """Wrapper for vtkUnstructuredGrid."""
    pass


# Not available in VTK 8.1.2
# class BSGraph(BSDataObject, dsa.Graph):
#     """Wrapper for vtkUnstructuredGrid."""
#     pass


# class BSMolecule(BSDataObject, dsa.Molecule):
#     """Wrapper for vtkMolecule."""
#     pass
