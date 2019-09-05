"""
Implementation of Spin permutations.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator

from ..vtk_interface.wrappers import BSPolyData
from ..mesh import mesh_elements as me


def _generate_spins(points_lh, points_rh=None, unique=False, n_rep=100,
                    random_state=None, surface_algorithm='FreeSurfer'):
    """ Generate rotational spins based on points that lie on a sphere.

    Parameters
    ----------
    points_lh : BSPolyData or ndarray, shape = (n_lh, 3)
        Array of points in a sphere, where `n_lh` is the number of points.
    points_rh : BSPolyData or ndarray, shape = (n_rh, 3), optional
        Array of points in a sphere, where `n_rh` is the number of points. If
        provided, rotations are derived from the rotations computed for
        `points_lh` by reflecting the rotation matrix across the Y-Z plane.
        Default is None.
    unique : bool, optional
        Whether to enforce a one-to-one correspondence between original points
        and rotated ones. If true, the Hungarian algorithm is used.
        Default is False.
    n_rep : int, optional
        Number of random rotations. Default is 100.
    surface_algorithm : {'FreeSurfer', 'CIVET'}
        For 'CIVET', no flip is required to generate the spins for the right
        hemisphere. Only used when ``points_rh is not None``.
        Default is 'FreeSurfer'.
    random_state : int or None, optional
        Random state. Default is None.

    Returns
    -------
    result : dict[str, ndarray]
        Spin indices for left points (and also right, if provided).

    References
    ----------
    * Alexander-Bloch A, Shou H, Liu S, Satterthwaite TD, Glahn DC,
      Shinohara RT, Vandekar SN and Raznahan A (2018). On testing for spatial
      correspondence between maps of human brain structure and function.
      NeuroImage, 178:540-51.
    * Blaser R and Fryzlewicz P (2016). Random Rotation Ensembles.
      Journal of Machine Learning Research, 17(4): 1–26.
    * https://netneurotools.readthedocs.io

    """

    # Handle if user provides spheres
    if not isinstance(points_lh, np.ndarray):
        points_lh = me.get_points(points_lh)

    if points_rh is not None:
        if not isinstance(points_rh, np.ndarray):
            points_rh = me.get_points(points_rh)

    pts = {'lh': points_lh}
    if points_rh is not None:
        pts['rh'] = points_rh

        # for reflecting across Y-Z plane
        reflect = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])

    idx = {k: np.arange(p.shape[0]) for k, p in pts.items()}
    spin = {k: np.empty((n_rep, p.shape[0]), dtype=int)
            for k, p in pts.items()}
    if not unique:
        tree = {k: cKDTree(p, leafsize=20) for k, p in pts.items()}

    rs = check_random_state(random_state)

    rot = {}
    for i in range(n_rep):

        # generate rotation for left
        rot['lh'], temp = np.linalg.qr(rs.normal(size=(3, 3)))
        rot['lh'] *= np.sign(np.diag(temp))
        rot['lh'][:, 0] *= np.sign(np.linalg.det(rot['lh']))

        # reflect the left rotation across Y-Z plane
        if 'rh' in pts:
            if surface_algorithm.lower() == 'freesurfer':
                rot['rh'] = reflect @ rot['lh'] @ reflect
            else:
                rot['rh'] = rot['lh']

        for k, p in pts.items():
            if unique:
                dist = cdist(p, p @ rot[k])
                row, col = linear_sum_assignment(dist)
                spin[k][i, idx[k]] = idx[k][col]
            else:
                _, spin[k][i] = tree[k].query(p @ rot[k], k=1, n_jobs=1)

    return spin


def spin_permutations(spheres, data, unique=False, n_rep=100,
                      random_state=None, surface_algorithm='FreeSurfer'):
    """ Generate null data using spin permutations.

    Parameters
    ----------
    spheres : dict[str, ndarray or BSPolyData], BSPolyData or ndarray
        Dictionary of points in a sphere, for left ('lh' key) and
        right ('rh' key) hemispheres. The right hemisphere is optional. If
        provided, rotations are derived from the rotations computed for
        `points_lh` by reflecting the rotation matrix across the Y-Z plane.
    data : dict[str, ndarray] or ndarray
        Dictionary of data to randomize. Array of variables arranged in
        columns for each hemisphere.
    unique : bool, optional
        Whether to enforce a one-to-one correspondence between original points
        and rotated ones. If true, the Hungarian algorithm is used.
        Default is False.
    n_rep : int, optional
        Number of random rotations. Default is 100.
    surface_algorithm : {'FreeSurfer', 'CIVET'}
        For 'CIVET', no flip is required to generate the spins for the right
        hemisphere. Only used when ``points_rh is not None``.
        Default is 'FreeSurfer'.
    random_state : int or None, optional
        Random state. Default is None.

    Returns
    -------
    rand_lh : ndarray, shape = (n_rep, n_lh, n_feat)
        Permutations of data in left hemisphere.
    rand_rh : ndarray, shape = (n_rep, n_rh, n_feat)
        Permutations of data in right hemisphere. Only if right data and
        sphere are provided.

    See Also
    --------
    :class:`.SpinPermutations`

    References
    ----------
    * Alexander-Bloch A, Shou H, Liu S, Satterthwaite TD, Glahn DC,
      Shinohara RT, Vandekar SN and Raznahan A (2018). On testing for spatial
      correspondence between maps of human brain structure and function.
      NeuroImage, 178:540-51.
    * Blaser R and Fryzlewicz P (2016). Random Rotation Ensembles.
      Journal of Machine Learning Research, 17(4): 1–26.
    * https://netneurotools.readthedocs.io

    """

    if isinstance(data, np.ndarray):
        data = {'lh': data}

    if isinstance(spheres, BSPolyData) or isinstance(spheres, np.ndarray):
        spheres = {'lh': spheres}

    if data.keys() != spheres.keys():
        raise ValueError("Keys for data and spheres do not coincide.")

    if len(data) > 2:
        raise ValueError("Unknown keys. Possible keys: {'lh', 'rh'}.")

    if len(data) == 1 and 'lh' not in data.keys():
        raise ValueError("Key must be 'lh'.")
    elif len(data) == 2 and ('lh' not in data or 'rh' not in data):
        raise ValueError("Unknown keys. Possible keys: {'lh', 'rh'}.")

    points_lh = spheres['lh']
    points_rh = spheres.pop('rh', None)

    spin_idx = _generate_spins(points_lh, points_rh=points_rh, unique=unique,
                               n_rep=n_rep, random_state=random_state,
                               surface_algorithm=surface_algorithm)

    spin_lh = spin_idx['lh']
    spin_rh = spin_idx.pop('rh', None)

    x_lh = data['lh']
    rand_lh = x_lh[spin_lh]
    if spin_rh is None:
        return rand_lh

    x_rh = data.pop('rh', None)
    rand_rh = None
    if x_rh is not None:
        rand_rh = x_rh[spin_rh]

    return rand_lh, rand_rh


class SpinPermutations(BaseEstimator):
    """ Spin permutations.

    Parameters
    ----------
    unique : bool, optional
        Whether to enforce a one-to-one correspondence between original points
        and rotated ones. If True, the Hungarian algorithm is used.
        Default is False.
    n_rep : int, optional
        Number of randomizations. Default is 100.
    random_state : int or None, optional
        Random state. Default is None.
    surface_algorithm : {'FreeSurfer', 'CIVET'}
        For 'CIVET', no flip is required to generate the spins for the right
        hemisphere. Only used when ``points_rh is not None``.
        Default is 'FreeSurfer'.

    Attributes
    ----------
    spin_lh_ : ndarray, shape (n_rep, n_lh)
        Spin indices for points in left hemisphere.
    spin_rh_ : ndarray, shape (n_rep, n_rh)
        Spin indices for points in right hemisphere. Only if user provides
        right hemisphere points. None, otherwise.

    See Also
    --------
    :func:`.spin_permutations`
    :class:`.MoranRandomization`

    Notes
    -----
    Right hemisphere permutations are generated by reflecting the rotation
    matrix used for the left hemisphere.

    """

    def __init__(self, unique=False, n_rep=100, random_state=None,
                 surface_algorithm='FreeSurfer'):
        self.unique = unique
        self.n_rep = n_rep
        self.random_state = random_state
        self.surface_algorithm = surface_algorithm

    def fit(self, points_lh, points_rh=None):
        """ Compute spin indices by random rotation.

        Parameters
        ----------
        points_lh : BSPolyData or ndarray, shape = (n_lh, 3)
            Sphere for the left hemisphere. If ndarray, each row must
            represent a vertex in the sphere.
        points_rh : BSPolyData or ndarray, shape = (n_rh, 3), optional
            Sphere for the right hemisphere. If ndarray, row must
            represent a vertex in the sphere. Default is None.

        Returns
        -------
        self : object
            Returns self.

        """

        spin_idx = _generate_spins(points_lh, points_rh=points_rh,
                                   unique=self.unique, n_rep=self.n_rep,
                                   random_state=self.random_state,
                                   surface_algorithm=self.surface_algorithm)

        self.spin_lh_ = spin_idx['lh']
        self.spin_rh_ = spin_idx.pop('rh', None)
        return self

    def randomize(self, x_lh, x_rh=None):
        """ Generate random samples from `x_lh` and `x_rh`.

        Parameters
        ----------
        x_lh : ndarray, shape = (n_lh,) or (n_lh, n_feat)
            Array of variables arranged in columns, where `n_feat` is the
            number of variables.
        x_rh : ndarray, shape = (n_rh,) or (n_rh, n_feat), optional
            Array of variables arranged in columns for the right hemisphere.
            Default is None.

        Returns
        -------
        rand_lh : ndarray, shape = (n_rep, n_lh, n_feat)
            Permutations of `x_rh`. If ``n_feat == 1``, shape = (n_rep, n_lh).
        rand_lh : ndarray, shape = (n_rep, n_rh, n_feat)
            Permutations of `x_rh`. If ``n_feat == 1``, shape = (n_rep, n_rh).
            None if `x_rh` is None. Only if `spin_rh_` is not None.

        """

        rand_lh = x_lh[self.spin_lh_]
        if self.spin_rh_ is None:
            return rand_lh

        rand_rh = None
        if x_rh is not None:
            rand_rh = x_rh[self.spin_rh_]

        return rand_lh, rand_rh
