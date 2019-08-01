from os.path import dirname, join
import numpy as np


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


def load_conte69():
    root_pth = dirname(__file__)
    fname_lh = 'conte69_32k_left_hemisphere.gii'
    fname_rh = 'conte69_32k_right_hemisphere.gii'
    ipth_lh = join(root_pth, 'surfaces', fname_lh)
    ipth_rh = join(root_pth, 'surfaces', fname_rh)
    from ..mesh.mesh_io import load_surface
    surf_lh = load_surface(ipth_lh)
    surf_rh = load_surface(ipth_rh)

    return surf_lh, surf_rh




