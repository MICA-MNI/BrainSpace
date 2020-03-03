import os
import numpy as np

from vtk import vtkPolyDataNormals

from ..mesh.mesh_io import read_surface
from ..mesh.mesh_operations import combine_surfaces
from ..utils.parcellation import reduce_by_labels
from ..vtk_interface import wrap_vtk, serial_connect


def load_group_fc(parcellation, scale=400, group='main'):
    """ Load group level connectivity matrix for a given parcellation.

    Connectivity is derived from a subset of HCP data.

    Parameters
    ----------
    parcellation : {'schaefer', 'vosdewael'}
        Parcellation name, either 'schaefer' for Schaefer (functional)
        parcellations or 'vosdewael' for a subparcellation of aparc.
    scale : {100, 200, 300, 400}, optional
        Number of parcels. Default is 400.
    group : {'main', 'holdout'}
        Group of subjects used to derive the connectivity matrix.
        Default is 'main'.

    Returns
    -------
    conn : 2D ndarray, shape = (scale, scale)
        Connectivity matrix.
    """

    root_pth = os.path.dirname(__file__)
    fname = '{0}_{1}_mean_connectivity_matrix.csv'.format(parcellation, scale)
    ipth = os.path.join(root_pth, 'matrices', '{0}_group', fname).format(group)
    return np.loadtxt(ipth, dtype=np.float, delimiter=',')


def load_group_mpc(parcellation, scale=400):
    """ Load group level connectivity matrix for a given parcellation.

    Connectivity is derived from a subset of HCP data.

    Parameters
    ----------
    parcellation : {'schaefer', 'vosdewael'}
        Parcellation name, either 'schaefer' for Schaefer (functional)
        parcellations or 'vosdewael' for a subparcellation of aparc.
    scale : {100, 200, 300, 400}, optional
        Number of parcels. Default is 400.

    Returns
    -------
    conn : 2D ndarray, shape = (scale, scale)
        Connectivity matrix.
    """

    if parcellation != 'vosdewael':
        raise ValueError("Only 'vosdewael' parcellation is accepted at the "
                         "moment.")

    if scale != 200:
        raise ValueError("Only a scale of 200 is accepted at the moment.")

    root_pth = os.path.dirname(__file__)
    fname = '{0}_{1}_mpc_matrix.csv'.format(parcellation, scale)
    ipth = os.path.join(root_pth, 'matrices/fusion_tutorial', fname)
    return np.loadtxt(ipth, dtype=np.float, delimiter=',')


def load_parcellation(name, scale=400, join=False):
    """ Load parcellation for conte69.

    Parameters
    ----------
    name : {'schaefer', 'vosdewael'}
        Parcellation name, either 'schaefer' for Schaefer (functional)
        parcellations or 'vosdewael' for a subparcellation of aparc.
    scale : {100, 200, 300, 400}, optional
        Number of parcels. Default is 400.
    join : bool, optional
        If False, return one array for each hemisphere. Otherwise,
        return a single array for both left and right hemisphere.
        Default is False.

    Returns
    -------
    parcellation : tuple of ndarrays or ndarray
        Parcellations for left and right hemispheres. If ``join == True``, one
        parcellation with both hemispheres.
    """

    root_pth = os.path.dirname(__file__)
    fname = '{name}_{np}_conte69.csv'.format(name=name, np=scale)
    ipth = os.path.join(root_pth, 'parcellations', fname)
    x = np.loadtxt(ipth, dtype=np.int)
    if join:
        return x
    return x[:x.size//2], x[x.size//2:]


def load_mask(name='midline', join=False):
    """ Load mask for conte69.

    Parameters
    ----------
    name : {'midline', 'temporal'} or None, optional
        Region name. If 'midline', load mask for all cortex.
        Default is 'midline'.
    join : bool, optional
        If False, return one array for each hemisphere. Otherwise,
        return a single array for both left and right hemispheres.
        Default is False.

    Returns
    -------
    mask : tuple of ndarrays or ndarray
        Boolean masks for left and right hemispheres. If ``join == True``, one
        mask with both hemispheres.
    """

    root_pth = os.path.dirname(__file__)
    ipth = os.path.join(root_pth, 'surfaces', 'conte69_32k_{0}{1}_mask.csv')
    if name == 'midline':
        name = ''
    else:
        name = '_' + name
    mask_lh = np.loadtxt(ipth.format('lh', name), dtype=np.bool)
    mask_rh = np.loadtxt(ipth.format('rh', name), dtype=np.bool)
    if join:
        return np.concatenate([mask_lh, mask_rh])
    return mask_lh, mask_rh


def load_conte69(as_sphere=False, with_normals=True, join=False):
    """ Load conte69 surfaces.

    Parameters
    ----------
    as_sphere : bool, optional
        Return spheres instead of cortical surfaces. Default is False.
    with_normals : bool, optional
        Whether to compute surface normals. Default is True.
    join : bool, optional
        If False, return one surface for left and right hemispheres. Otherwise,
        return a single surface as a combination of both left and right
        surfaces. Default is False.

    Returns
    -------
    surf : tuple of BSPolyData or BSPolyData
        Surfaces for left and right hemispheres. If ``join == True``, one
        surface with both hemispheres.
    """

    root_pth = os.path.dirname(__file__)
    if as_sphere:
        fname = 'conte69_32k_{}_sphere.gii'
    else:
        fname = 'conte69_32k_{}.gii'

    ipth = os.path.join(root_pth, 'surfaces', fname)
    surfs = [None] * 2
    for i, side in enumerate(['lh', 'rh']):
        surfs[i] = read_surface(ipth.format(side))
        if with_normals:
            nf = wrap_vtk(vtkPolyDataNormals, splitting=False,
                          featureAngle=0.1)
            surfs[i] = serial_connect(surfs[i], nf)

    if join:
        return combine_surfaces(*surfs)
    return surfs[0], surfs[1]


def load_fsa5(with_normals=True, join=False):
    """ Load fsaverage5 surfaces.

    Parameters
    ----------
    with_normals : bool, optional
        Whether to compute surface normals. Default is True.
    join : bool, optional
        If False, return one surface for left and right hemispheres. Otherwise,
        return a single surface as a combination of both left and right
        surfaces. Default is False.

    Returns
    -------
    surf : tuple of BSPolyData or BSPolyData
        Surfaces for left and right hemispheres. If ``join == True``, one
        surface with both hemispheres.
    """

    root_pth = os.path.dirname(__file__)
    fname = 'fsa5.pial.{}.gii'

    ipth = os.path.join(root_pth, 'surfaces', fname)
    surfs = [None] * 2
    for i, side in enumerate(['lh', 'rh']):
        surfs[i] = read_surface(ipth.format(side))
        if with_normals:
            nf = wrap_vtk(vtkPolyDataNormals, splitting=False,
                          featureAngle=0.1)
            surfs[i] = serial_connect(surfs[i], nf)

    if join:
        return combine_surfaces(*surfs)
    return surfs[0], surfs[1]


def load_confounds_preprocessing():
    """Load counfounds for preprocessing tutorial.

    Returns
    -------
    confounds : ndarray

    """

    root_pth = os.path.dirname(__file__)
    fname = 'sub-010188_ses-02_task-rest_acq-AP_run-01_confounds.txt'
    ipth = os.path.join(root_pth, 'preprocessing', fname)
    return np.loadtxt(ipth)


def fetch_timeseries_preprocessing():
    """Fetch timeseries to deconfound for preprocessing tutorial.

    Returns
    -------
    timeseries : 2D ndarray, shape = (nodes (lh, rh), timepoints)
        Timeseries for left and right hemispheres.

    """

    import nibabel as nib

    root_pth = os.path.dirname(__file__)
    fname = 'sub-010188_ses-02_task-rest_acq-AP_run-01.fsa5.{}.mgz'
    ipth = os.path.join(root_pth, 'preprocessing', fname)

    ts = [None] * 2
    for i, h in enumerate(['lh', 'rh']):
        ts[i] = nib.load(ipth.format(h)).get_fdata().squeeze()
    ts = np.vstack(ts)
    return ts


def _load_feat(feat_name, parcellation=None, mask=None):
    root_pth = os.path.dirname(__file__)
    ipth = os.path.join(root_pth, 'matrices', 'main_group',
                        '{0}.csv'.format(feat_name))
    x = np.loadtxt(ipth, dtype=np.float)
    if mask is not None:
        x = x[mask]

    if parcellation is not None:
        if mask is not None:
            parcellation = parcellation[mask]
        x = reduce_by_labels(x, parcellation, red_op='mean')
    return x


def load_marker(name, join=False):
    """ Load cortical data for conte69.

    Parameters
    ----------
    name : {'curvature', 'thickness', 't1wt2w'}
        Marker name.
    join : bool, optional
        If False, return one array for each hemisphere. Otherwise,
        return a single array for both left and right hemispheres.
        Default is False.

    Returns
    -------
    marker : tuple of ndarrays or ndarray
        Marker data for left and right hemispheres. If ``join == True``, one
        array with both hemispheres.
    """

    feat_name = 'conte69_32k_{0}'.format(name)
    x = _load_feat(feat_name)
    if join:
        return x
    return x[:x.size//2], x[x.size//2:]


def load_gradient(name, idx=0, join=False):
    """ Load gradient for conte69.

    Parameters
    ----------
    name : {'fc', 'mpc'}
        The type of gradient, either 'fc' for functional connectivity or 'mpc'
        for microstructural profile covariance.
    idx : int, optional
        Gradient index. Default is 0 (first gradient).
    join : bool, optional
        If False, return one array for each hemisphere. Otherwise,
        return a single array for both left and right hemispheres.
        Default is False.

    Returns
    -------
    marker : tuple of ndarrays or ndarray
        Gradients for left and right hemispheres. If ``join == True``, one
        gradient array with both hemispheres.
    """

    feat_name = 'conte69_32k_{0}_gradient{1}'.format(name, idx)
    x = _load_feat(feat_name)
    if join:
        return x
    return x[:x.size//2], x[x.size//2:]
