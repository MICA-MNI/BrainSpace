"""
Clustering and sampling of surface mesh points.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import numpy as np

from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import k_means

# from vtkmodules.vtkFiltersCorePython import vtkDecimatePro
from vtk import vtkDecimatePro

from . import mesh_elements as me
from .mesh_correspondence import find_point_correspondence
from ..vtk_interface.pipeline import serial_connect
from ..utils.parcellation import map_to_mask, reduce_by_labels
from ..gradient import diffusion_mapping


def cluster_points(surf, n_clusters=100, is_size=False, mask=None,
                   with_centers=True, random_state=None, approach='kmeans',
                   n_init=3, n_jobs=1):
    """Clustering of surface points.

    Parameters
    ----------
    surf : vtkPolyData or BSPolyData
        Input surface.
    n_clusters : int, optional
        Number of clusters. Default is 100.
    is_size : bool, optional
        If True, interpret `n_clusters` as cluster size. Default is False.
    mask : 1D ndarray, optional
        Mask for surface points. Points outside the mask (i.e., False) are
        discarded from clustering. Default is None.
    with_centers : bool, optional
        If True, an array of labels with the closest points to the centroid of
        each cluster is returned. Default is True.
    random_state : int, RandomState instance or None, optional
         Random state. Default is None.
    approach : {'kmeans', 'ward'}, optional
        Clustering method: k-means or hierarchical with ward linkage.
        Hierarchical clustering is faster but k-means provides better results.
        Default is 'kmeans'.
    n_init : int, optional
        Number of k-means repetitions. Only used when ``approach == 'kmeans'``.
        Default is 3.
    n_jobs : int or None, optional
        The number of parallel jobs. Only used when ``approach == 'kmeans'``.
        Default is 1.

    Returns
    -------
    cluster_labels : 1D ndarray, shape (n_points,)
        Array of cluster labels. If `mask` is provided, points out of the mask
        are assigned label 0.
    center_labels : 1D ndarray, shape (n_points,)
        Array with centers labeled with their corresponding cluster label.
        The rest of points is assigned label 0. Returned only if
        ``with_centers=True``.

    Notes
    -----
    Valid cluster labels start from 1. If the mask is provided, zeros are
    assigned to the points outside the mask.

    """

    # Get immediate geodesic distance
    a = me.get_ring_distance(surf, n_ring=1, metric='geodesic')
    if mask is not None:
        a = a[mask, :][:, mask]
    a.data = np.exp(-a.data/np.median(a.data))
    a.tolil().setdiag(1)

    # Embedding
    evs, _ = diffusion_mapping(a, n_components=30, alpha=0, diffusion_time=1,
                               random_state=random_state)
    evs = normalize(evs)  # To find spherical clusters

    if is_size:
        n_clusters = evs.shape[0] // n_clusters

    # Find clusters
    if approach == 'kmeans':
        _, cluster_labs, _ = k_means(evs, n_clusters=n_clusters,
                                     random_state=random_state, n_jobs=n_jobs,
                                     n_init=n_init)
    else:
        conn = me.get_immediate_adjacency(surf, include_self=False)
        if mask is not None:
            conn = conn[mask, :][:, mask]
        hc = AgglomerativeClustering(n_clusters=n_clusters, connectivity=conn,
                                     linkage='ward')
        cluster_labs = hc.fit(evs).labels_

    # Valid clusters start from 1
    cluster_labs += 1

    # Find centers
    if with_centers:
        points = surf.Points if mask is None else surf.Points[mask]
        centroids = reduce_by_labels(points, cluster_labs, red_op='mean',
                                     axis=1)

        centroid_labs = np.zeros_like(cluster_labs)
        idx_samples = np.arange(points.shape[0])
        for i, lab in enumerate(range(1, n_clusters+1)):
            mask_cl = cluster_labs == lab
            dif = centroids[i] - points[mask_cl]
            idx = np.einsum('ij,ij->i', dif, dif).argmin()
            idx_centroid = idx_samples[mask_cl][idx]
            centroid_labs[idx_centroid] = lab

        if mask is not None:
            centroid_labs = map_to_mask(centroid_labs, mask)

    if mask is not None:
        cluster_labs = map_to_mask(cluster_labs, mask)

    if with_centers:
        return cluster_labs, centroid_labs
    return cluster_labs


def sample_points_clustering(surf, keep=0.1, mask=None, random_state=None,
                             approach='kmeans', n_init=3, n_jobs=1):
    """Sample equidistant points from surface based on clustering.

    Parameters
    ----------
    surf : vtkPolyData or BSPolyData
        Input surface.
    keep : float or int, optional
        If float, percentage of points to sample. Must be ``0 < keep < 1``.
        If int, number of points to sample. Default is 0.1.
    mask : 1D ndarray, optional
        Mask for surface points. Points outside the mask (i.e., False) are
        discarded from sampling. Default is None.
    random_state : int, RandomState instance or None, optional
         Random state. Default is None.
    approach : {'kmeans', 'ward'}, optional
        Clustering approach: k-means or hierarchical with ward linkage.
        Hierarchical is faster but k-means provides better results.
        Default is 'kmeans'.
    n_init : int, optional
        Number of k-means repetitions. Only used when ``approach == 'kmeans'``.
        Default is 3.
    n_jobs : int or None, optional
        The number of parallel jobs. Only used when ``approach == 'kmeans'``.
        Default is 1.

    Returns
    -------
    sampled : 1D ndarray, shape (n_points,)
        Array with sampled points marked with 1 in their corresponding
        positions. The rest is 0.

    See Also
    --------
    :func:`cluster_points`
    :func:`sample_points_decimation`

    Notes
    -----
    This method first clusters the surface points and then selects the points
    closest to the centroids as the sampled points.

    """

    if isinstance(keep, int):
        n_clusters = keep
    elif 0 < keep < 1:
        n_clusters = int(keep * surf.GetNumberOfPoints())
    else:
        ValueError('The value of \'keep\' is not valid.')

    sampled_points = cluster_points(surf, n_clusters=n_clusters, mask=mask,
                                    random_state=random_state,
                                    with_centers=True, approach=approach,
                                    n_init=n_init, n_jobs=n_jobs)[1]

    return (sampled_points > 0).astype(np.uint8)


def sample_points_decimation(surf, keep=0.1, mask=None, n_jobs=1):
    """Sample points from surface based on surface decimation.

    Parameters
    ----------
    surf : vtkPolyData or BSPolyData
        Input surface.
    keep : float or int, optional
        If float, percentage of points to sample. Must be ``0 < keep < 1``.
        If int, number of points to sample. Default is 0.1.
    mask : 1D ndarray, optional
        Mask for surface points. Points outside the mask (i.e., False) are
        discarded from sampling. Default is None.
    n_jobs : int, optional
        Number of jobs. Default is 1.

    Returns
    -------
    sampled : 1D ndarray, shape (n_points,)
        Array with sampled points marked with 1 in their corresponding
        positions. The rest is 0.

    Notes
    -----
    In this method, sampling is based on mesh decimation. Number of sampled
    points will most probably not coincide with the requested number.

    See Also
    --------
    :func:`sample_points_clustering`

    """

    n_pts = surf.GetNumberOfPoints()

    # Increase to account for points outside mask
    if mask is not None:
        keep *= (1 + np.count_nonzero(~mask) / n_pts)

    if isinstance(keep, int):
        factor = (n_pts - keep) / n_pts
    elif 0 < keep < 1:
        factor = 1 - keep
    else:
        ValueError('The value of \'keep\' is not valid.')

    cf = vtkDecimatePro()
    cf.SetTargetReduction(factor)
    decimated_surf = serial_connect(surf, cf)

    idx = find_point_correspondence(decimated_surf, surf, eps=0, n_jobs=n_jobs)
    sampled_points = np.zeros(n_pts, dtype=np.uint8)
    sampled_points[idx] = 1

    # Remove points sampled outside mask
    if mask is not None:
        sampled_points[~mask] = 0

    return sampled_points
