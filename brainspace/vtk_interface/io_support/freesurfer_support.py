"""
VTK read/write filters for FreeSurfer geometry files.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import re
import numpy as np

from vtk import vtkPolyData
from vtk.util.vtkAlgorithm import VTKPythonAlgorithmBase

from ..checks import has_only_triangle
from ..decorators import wrap_input
from ...mesh.mesh_creation import build_polydata


TRIANGLE_MAGIC = 16777214
QUAD_MAGIC = 16777215
NEW_QUAD_MAGIC = 16777213


def _fread3(fobj):
    """Read a 3-byte int from an open binary file object
    Parameters
    ----------
    fobj : file
        File descriptor
    Returns
    -------
    n : int
        A 3 byte int
    """
    b1, b2, b3 = np.fromfile(fobj, ">u1", 3)
    return (b1 << 16) + (b2 << 8) + b3


def _fread3_many(fobj, n):
    """Read 3-byte ints from an open binary file object.
    Parameters
    ----------
    fobj : file
        File descriptor
    Returns
    -------
    out : 1D array
        An array of 3 byte int
    """
    b1, b2, b3 = np.fromfile(fobj, ">u1", 3 * n).reshape(-1, 3).astype(np.int64).T
    return (b1 << 16) + (b2 << 8) + b3


def _read_geometry_fs(ipth, is_ascii=False):
    """Adapted from nibabel. Add ascii support."""

    if is_ascii:
        with open(ipth) as fh:
            re_header = re.compile('^#!ascii version (.*)$')
            fname_header = re_header.match(fh.readline()).group(1)

            re_npoints_cells = re.compile('[\s]*(\d+)[\s]*(\d+)[\s]*$')
            re_n = re_npoints_cells.match(fh.readline())
            n_points, n_cells = int(re_n.group(1)), int(re_n.group(2))

            x_points = np.zeros((n_points, 3))
            for i in range(n_points):
                x_points[i, :] = [float(v) for v in fh.readline().split()[:3]]

            x_cells = np.zeros((n_cells, 3), dtype=np.uintp)
            for i in range(n_cells):
                x_cells[i] = [np.uintp(v) for v in fh.readline().split()[:3]]

    else:
        with open(ipth, 'rb') as fh:
            magic = _fread3(fh)
            if magic not in [TRIANGLE_MAGIC, QUAD_MAGIC, NEW_QUAD_MAGIC]:
                raise IOError('File does not appear to be a '
                              'FreeSurfer surface.')

            if magic in (QUAD_MAGIC, NEW_QUAD_MAGIC):  # Quad file
                n_points, n_quad = _fread3(fh), _fread3(fh)

                (fmt, div) = ('>i2', 100) if magic == QUAD_MAGIC else ('>f4', 1)
                x_points = np.fromfile(fh, fmt, n_points * 3).astype(np.float64)
                x_points /= div
                x_points = x_points.reshape(-1, 3)

                quads = _fread3_many(fh, n_quad * 4)
                quads = quads.reshape(n_quad, 4)
                n_cells = 2 * n_quad
                x_cells = np.zeros((n_cells, 3), dtype=np.uintp)

                # Face splitting follows (Remove loop in nib) -> Not tested!
                m0 = (quads[:, 0] % 2) == 0
                m0d = np.repeat(m0, 2)
                x_cells[m0d].flat[:] = quads[m0][:, [0, 1, 3, 2, 3, 1]]
                x_cells[~m0d].flat[:] = quads[~m0][:, [0, 1, 2, 0, 2, 3]]

            elif magic == TRIANGLE_MAGIC:  # Triangle file
                # create_stamp = fh.readline().rstrip(b'\n').decode('utf-8')
                fh.readline()
                fh.readline()

                n_points, n_cells = np.fromfile(fh, '>i4', 2)
                x_points = np.fromfile(fh, '>f4', n_points * 3)
                x_points = x_points.reshape(n_points, 3).astype(np.float64)

                x_cells = np.zeros((n_cells, 3), dtype=np.uintp)
                x_cells.flat[:] = np.fromfile(fh, '>i4', n_cells * 3)

    return build_polydata(x_points, cells=x_cells).VTKObject


@wrap_input(0)
def _write_geometry_fs(pd, opth, fname_header=None, is_ascii=False):
    """Adapted from nibabel. Add ascii support."""

    if not has_only_triangle(pd):
        raise ValueError('FreeSurfer writer only accepts triangles.')

    n_points, n_cells = pd.GetNumberOfPoints(), pd.GetNumberOfCells()
    x_points = np.zeros((n_points, 4), dtype=np.float32)
    x_points[:, :3] = pd.GetPoints()
    x_cells = np.zeros((n_cells, 4), dtype=np.uintp)
    x_cells[:, :3] = pd.GetPolygons().reshape(-1, 4)[:, 1:]

    if is_ascii:
        header = '#!ascii version of {fname}\n'.\
            format(fname='...' if fname_header is None else fname_header)
        npoints_cells = '{npoints} {ncells}\n'.\
            format(npoints=n_points, ncells=n_cells)

        with open(opth, 'w') as fh:
            fh.write(header)
            fh.write(npoints_cells)
            np.savetxt(fh, x_points, fmt=['%.6f', '%.6f', '%.6f', '%d'],
                       delimiter='  ')
            np.savetxt(fh, x_cells, fmt='%d', delimiter='  ')

    else:
        magic_bytes = np.array([255, 255, 254], dtype=np.uint8)
        create_stamp = 'created by {0}'.\
            format('...' if fname_header is None else fname_header)

        with open(opth, 'wb') as fobj:
            magic_bytes.tofile(fobj)
            fobj.write('{0}%s\n\n'.format(create_stamp).encode('utf-8'))

            np.array([n_points, n_cells], dtype='>i4').tofile(fobj)

            # Coerce types, just to be safe
            x_points[:, :3].astype('>f4').reshape(-1).tofile(fobj)
            x_cells[:, :3].astype('>i4').reshape(-1).tofile(fobj)


###############################################################################
# VTK Reader and Writer for FreeSurfer surfaces
###############################################################################
class vtkFSReader(VTKPythonAlgorithmBase):
    """VTK-like FreeSurfer surface geometry reader.

    Supports both binary and ASCII files. Default is binary.
    """

    def __init__(self):
        super().__init__(nInputPorts=0, nOutputPorts=1,
                         outputType='vtkPolyData')
        self.__FileName = ''
        self.__is_ascii = False

    def RequestData(self, request, inInfo, outInfo):
        opt = vtkPolyData.GetData(outInfo, 0)
        if self.__is_ascii or self.__FileName.split('.')[-1] == 'asc':
            s = _read_geometry_fs(self.__FileName, is_ascii=True)
        else:
            s = _read_geometry_fs(self.__FileName, is_ascii=False)
        opt.ShallowCopy(s)
        return 1

    def SetFileTypeToBinary(self):
        if self.__is_ascii:
            self.__is_ascii = False
            self.Modified()

    def SetFileTypeToASCII(self):
        if not self.__is_ascii:
            self.__is_ascii = True
            self.Modified()

    def SetFileName(self, fname):
        if fname != self.__FileName:
            self.__FileName = fname
            self.Modified()

    def GetFileName(self):
        return self.__FileName

    def GetOutput(self, p_int=0):
        return self.GetOutputDataObject(p_int)


class vtkFSWriter(VTKPythonAlgorithmBase):
    """VTK-like FreeSurfer surface geometry writer.

    Only writes surface geometry/topology (points and cells).
    Supports both binary and ASCII files. Default is binary.
    """

    def __init__(self):
        super().__init__(nInputPorts=1, inputType='vtkPolyData', nOutputPorts=0)
        self.__FileName = ''
        self.__is_ascii = False

    def RequestData(self, request, inInfo, outInfo):
        _write_geometry_fs(vtkPolyData.GetData(inInfo[0], 0),
                           self.__FileName, fname_header=None,
                           is_ascii=self.__is_ascii)
        return 1

    def SetFileName(self, fname):
        if fname != self.__FileName:
            self.__FileName = fname
            self.Modified()

    def GetFileName(self):
        return self.__FileName

    def SetFileTypeToBinary(self):
        if self.__is_ascii:
            self.__is_ascii = False
            self.Modified()

    def SetFileTypeToASCII(self):
        if not self.__is_ascii:
            self.__is_ascii = True
            self.Modified()

    def Write(self):
        self.Update()

    def SetInputData(self, *args):
        # Signature is SetInputData(self, port, vtkDataObject) or simply
        # SetInputData(self, vtkDataObject)
        # A way to manage overloading in C++, because port is optional
        self.SetInputDataObject(*args)
