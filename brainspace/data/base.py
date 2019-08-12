from os.path import dirname, join
import numpy as np

from ..mesh.mesh_io import load_surface
from ..utils.parcellation import reduce_by_labels


def load_holdout_hcp(name, n_parcels=400):
    root_pth = dirname(__file__)
    fname = '{name}_{np}_mean_connectivity_matrix.csv'.format(name=name,
                                                              np=n_parcels)
    ipth = join(root_pth, 'matrices/holdout_group', fname)
    return np.loadtxt(ipth, dtype=np.float, delimiter=',')


def load_individuals_hcp(name, n_parcels=400):
    pass


def load_group_hcp(name, n_parcels=400):
    root_pth = dirname(__file__)
    fname = '{name}_{np}_mean_connectivity_matrix.csv'.format(name=name,
                                                              np=n_parcels)
    ipth = join(root_pth, 'matrices/main_group', fname)
    return np.loadtxt(ipth, dtype=np.float, delimiter=',')


def load_parcellation(name, n_parcels=400):
    root_pth = dirname(__file__)
    fname = '{name}_{np}_conte69.csv'.format(name=name, np=n_parcels)
    ipth = join(root_pth, 'parcellations', fname)
    return np.loadtxt(ipth, dtype=np.int)


def load_conte69(as_sphere=False):
    root_pth = dirname(__file__)
    if as_sphere:
        fname_lh = 'conte69_32k_left_sphere.gii'
        fname_rh = 'conte69_32k_right_sphere.gii'
    else:
        fname_lh = 'conte69_32k_left_hemisphere.gii'
        fname_rh = 'conte69_32k_right_hemisphere.gii'

    ipth_lh = join(root_pth, 'surfaces', fname_lh)
    ipth_rh = join(root_pth, 'surfaces', fname_rh)

    surf_lh = load_surface(ipth_lh)
    surf_rh = load_surface(ipth_rh)

    return surf_lh, surf_rh


def load_thickness(parcellation=None, mask=None):
    root_pth = dirname(__file__)
    ipth = join(root_pth, 'matrices/main_group/conte69_32k_t1wt2w.csv')
    x = np.loadtxt(ipth, dtype=np.float)
    if mask is not None:
        x = x[mask]

    if parcellation is not None:
        if mask is not None:
            parcellation = parcellation[mask]
        x = reduce_by_labels(x, parcellation, red_op='mean')
    return x


def load_t1t2(parcellation=None, mask=None):
    root_pth = dirname(__file__)
    ipth = join(root_pth, 'matrices/main_group/conte69_32k_thickness.csv')
    x = np.loadtxt(ipth, dtype=np.float)

    if mask is not None:
        x = x[mask]

    if parcellation is not None:
        if mask is not None:
            parcellation = parcellation[mask]
        x = reduce_by_labels(x, parcellation, red_op='mean')

    return x
