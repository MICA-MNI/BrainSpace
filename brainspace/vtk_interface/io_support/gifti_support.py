"""
VTK read/write filters for Gifti (.surf.gii).
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


from vtk import vtkPolyData
from vtk.util.vtkAlgorithm import VTKPythonAlgorithmBase

from ..decorators import wrap_input
from ...mesh.mesh_creation import build_polydata


try:
    import nibabel as nb
    INTENT_POINTS = nb.nifti1.intent_codes['NIFTI_INTENT_POINTSET']
    INTENT_CELLS = nb.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']
    INTENT_POINTDATA = nb.nifti1.intent_codes['NIFTI_INTENT_ESTIMATE']
    has_nibabel = True

except:
    has_nibabel = False


def _read_gifti(ipth, ipths_pointdata):
    g = nb.load(ipth)

    points = g.get_arrays_from_intent(INTENT_POINTS)[0].data
    cells = g.get_arrays_from_intent(INTENT_CELLS)[0].data
    s = build_polydata(points, cells=cells)

    # for pth_data, keys in ipths_pointdata.items():
    #     d = nb.load(pth_data)
    #     if len(d.darrays) != len(keys):
    #         raise ValueError("Number of arrays in '%s' does not coincide with "
    #                          "the number of keys %s" % pth_data, keys)
    #     for i, d1 in enumerate(d.darrays):
    #         if d1.intent in [INTENT_POINTS, INTENT_CELLS]:
    #             raise ValueError("File '%s' contains coord/top data" %
    #                              pth_data, keys)
    #
    #         s.append_array(d1.data, name=keys[i], at='p')

    # for a1 in g.darrays:
    #     if a1.intent not in [INTENT_POINTS, INTENT_CELLS]:
    #         # random array names
    #         s.append_array(a1.data, name=None, at='p')
    return s.VTKObject


@wrap_input(0)
def _write_gifti(pd, opth):
    # TODO: what about pointdata?
    from nibabel.gifti.gifti import GiftiDataArray

    if not pd.has_only_triangle:
        raise ValueError('GIFTI writer only accepts triangles.')

    points = GiftiDataArray(data=pd.Points, intent=INTENT_POINTS)
    cells = GiftiDataArray(data=pd.GetCells2D(), intent=INTENT_CELLS)
    # if data is not None:
    #     data_array = GiftiDataArray(data=data, intent=INTENT_POINTDATA)
    #     gii = nb.gifti.GiftiImage(darrays=[points, cells, data_array])
    # else:
    g = nb.gifti.GiftiImage(darrays=[points, cells])
    nb.save(g, opth)


###############################################################################
# VTK Reader and Writer for GIFTI surfaces
###############################################################################
class vtkGIFTIReader(VTKPythonAlgorithmBase):
    """VTK-like GIFTI surface reader.

    """

    def __init__(self):
        if not has_nibabel:
            raise AssertionError('vtkGIFTIReader requires nibabel.')
        super().__init__(nInputPorts=0, nOutputPorts=1,
                         outputType='vtkPolyData')
        self._FileName = ''
        self._fnames_pointdata = {}

    def RequestData(self, request, inInfo, outInfo):
        opt = vtkPolyData.GetData(outInfo, 0)
        s = _read_gifti(self._FileName, self._fnames_pointdata)
        opt.ShallowCopy(s)
        return 1

    def SetFileName(self, fname):
        if fname != self._FileName:
            self._FileName = fname
            self.Modified()

    # def AddFileNamePointData(self, fname, key):
    #     if fname not in self._filenames_pointdata:
    #         key = key if isinstance(key, list) else [key]
    #         self._fnames_pointdata[fname] = key
    #         self.Modified()

    def GetFileName(self):
        return self._FileName

    def GetOutput(self, p_int=0):
        return self.GetOutputDataObject(p_int)


class vtkGIFTIWriter(VTKPythonAlgorithmBase):
    """VTK-like GIFTI surface writer.

    """

    def __init__(self):
        if not has_nibabel:
            raise AssertionError('vtkGIFTIWriter requires nibabel.')
        super().__init__(nInputPorts=1, inputType='vtkPolyData', nOutputPorts=0)
        self.__FileName = ''

    def RequestData(self, request, inInfo, outInfo):
        _write_gifti(vtkPolyData.GetData(inInfo[0], 0), self.__FileName)
        return 1

    def SetFileName(self, fname):
        if fname != self.__FileName:
            self.__FileName = fname
            self.Modified()

    def GetFileName(self):
        return self.__FileName

    def Write(self):
        self.Update()

    def SetInputData(self, *args):
        # Signature is SetInputData(self, port, vtkDataObject) or simply
        # SetInputData(self, vtkDataObject)
        # A way to manage overloading in C++, because port is optional
        self.SetInputDataObject(*args)
