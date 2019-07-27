"""
Utility functions for parcellations/labelings.
"""

# Author: Oualid Benkarim <oualid.benkarim@mcgill.ca>
# License: BSD 3 clause


import numpy as np
from scipy.stats import mode
from scipy.optimize import linear_sum_assignment

from sklearn.utils.extmath import weighted_mode


def relabel_consecutive(lab, start_from=0):
    """Relabel array with consecutive values.

    Parameters
    ----------
    lab : array_like
        Array to relabel.
    start_from : int, optional
        Initial label. The default is 0.

    Returns
    -------
    new_lab : ndarray
        Array with consecutive labels.

    """

    _, new_lab = np.unique(lab, return_inverse=True)
    return new_lab + start_from


def relabel(lab, new_labels=None):
    """Relabel array.

    Parameters
    ----------
    lab : array_like
        Array to relabel.
    new_labels: array_like or dict, optional
        New labels. If dict, provide new label for each label in input array.
        If array_like, mapping is performed in ascending order. If None, relabel
        consecutively, starting from 0. Default is None.

    Returns
    -------
    new_lab : ndarray
        Array with new labels.

    """

    if isinstance(new_labels, dict):
        # new_lab = np.empty_like(lab)
        new_lab = lab.copy()
        for l1, l2 in new_labels.items():
            new_lab[lab == l1] = l2
        return new_lab

    if new_labels is None:
        return relabel_consecutive(lab)

    return relabel(lab, dict(zip(np.unique(lab), new_labels)))


def find_label_correspondence(lab1, lab2):
    """Find label correspondences.


    Parameters
    ----------
    lab1 : array_like
        First array of labels.
    lab2 : array_like
        Second array of labels.

    Returns
    -------
    map_labels : dict
        Dictionary with label correspondences between first and second arrays.

    Notes
    -----
    Correspondences are based on largest overlap using the Hungarian algorithm.

    """

    u1, idx1 = np.unique(lab1, return_inverse=True)
    u2, idx2 = np.unique(lab2, return_inverse=True)

    upairs, n_overlap = np.unique(list(zip(idx1, idx2)), axis=0,
                                  return_counts=True)

    cost = np.full((u1.size, u2.size), max(lab1.size, lab2.size),
                   dtype=np.float32)
    cost[tuple([*upairs.T])] /= n_overlap
    ridx, cidx = linear_sum_assignment(cost)

    return dict(zip(u1[ridx], u2[cidx]))


# def find_label_correspondence_old(lab1, lab2):
#     """Find label correspondences.
#
#
#     Parameters
#     ----------
#     lab1 : array_like
#         First array of labels.
#     lab2 : array_like
#         Second array of labels.
#
#     Returns
#     -------
#     map_labels : dict
#         Dictionary with label correspondences between first and second arrays.
#
#     Notes
#     -----
#     Correspondences are based on largest overlap using the Hungarian algorithm.
#
#     """
#
#     u1, idx1 = np.unique(lab1, return_inverse=True)
#     u2, idx2 = np.unique(lab2, return_inverse=True)
#
#     if u1.size != u2.size:
#         raise ValueError('Arrays do not have the same number of labels.')
#
#     upairs, n_overlap = np.unique(list(zip(idx1, idx2)), axis=0,
#                                   return_counts=True)
#
#     cost = np.full((u1.size, u1.size), len(lab1), dtype=np.float32)
#     cost[tuple([*upairs.T])] /= n_overlap
#     ridx, cidx = linear_sum_assignment(cost)
#
#     return dict(zip(u1[ridx], u2[cidx]))


def relabel_by_overlap(lab, ref_lab):
    """Relabel according to overlap with reference.

    Parameters
    ----------
    lab : array_like
        Array of labels.
    ref_lab : array_like
        Reference array of labels.

    Returns
    -------
    new_lab : ndarray
        Array relabeled using the reference array.

    Notes
    -----
    Correspondences between labels are based on largest overlap using the
    Hungarian algorithm.

    """

    u1 = np.unique(lab)
    u2 = np.unique(ref_lab)
    if u1.size > u2.size:
        thresh = lab.max() + 1
        lab_shifted = lab + thresh

        lab_corr = find_label_correspondence(lab_shifted, ref_lab)
        lab_shifted = relabel(lab_shifted, new_labels=lab_corr)

        ulab = np.unique(lab_shifted)
        ulab = ulab[ulab >= thresh]
        map_seq = dict(zip(ulab, np.arange(ulab.size) + ref_lab.max() + 1))
        return relabel(lab_shifted, new_labels=map_seq)

    lab_corr = find_label_correspondence(lab, ref_lab)
    return relabel(lab, new_labels=lab_corr)


def map_to_mask(values, mask, fill=0, axis=0):
    """Assign data to mask.

    Parameters
    ----------
    values : 1D or 2D ndarray
        Source array of values.
    mask : 1D ndarray
        Mask of boolean values. Data is mapped to True positions.
        If `values` is 2D, the mask is applied for each row.
    fill : float, optional
        Value used to fill elements outside the mask.
        Default is 0.
    axis : {0, 1}, optional
        If `axis=0` map rows. Otherwise, map columns.
        Default is 0.

    Returns
    -------
    output : ndarray
        Values mapped to mask. If `values` is 1D, shape (n_mask,).
        When `values` is 2D, shape (n_rows, n_mask) if axis=0 and
        (n_mask, n_cols) otherwise. Where n_rows and n_cols are the
        number of rows and columns of `values` and n_mask is the size
        of the mask.

    """

    if values.ndim == 1:
        axis = 0

    values2d = np.atleast_2d(values)
    n = values2d.shape[axis]
    mapped = np.full((n, mask.size), fill, dtype=values.dtype)
    mapped[:, mask] = values2d if axis == 0 else values2d.T

    if values.ndim == 1:
        return mapped[0]
    if axis == 1:
        return mapped.T
    return mapped


def map_to_labels(source_val, target_lab, source_lab=None):
    """Map data in source to target according to their labels.

    Target labels are sorted in ascending order, such that the smallest label
    indexes value at position 0

    Parameters
    ----------
    source_val : array_like
        Source array of values.
    target_lab : array_like
        Target labels.
    source_lab : array_like, optional
        Source labels. If None, it takes the same unique labels
        as the target label in ascending order. Default is None.

    Returns
    -------
    target_val : ndarray
        Target array with corresponding source values.

    """

    source_val = np.asarray(source_val)
    uq_tl, idx_tl = np.unique(target_lab, return_inverse=True)

    if source_lab is not None:
        source_lab = np.asarray(source_lab)
        if source_lab.size != source_val.size:
            raise ValueError('Source values and labels must have same size.')

        uq_sl, idx_sl = np.unique(source_lab, return_inverse=True)

        if source_lab.size != uq_sl.size:
            raise ValueError('Source labels must have distinct labels.')
        if np.setdiff1d(uq_tl, uq_sl).size > 0:
            raise ValueError('Source and target labels do not coincide.')

        source_val = source_val[idx_sl]

    if idx_tl.max() >= source_val.size:
        raise ValueError('There are more labels than values.')

    return source_val[idx_tl]


def reduce_by_labels(values, labels, weights=None, target_labels=None,
                     red_op='mean', axis=0, dtype=np.float):
    """Summarize data in source according to its labels.

    Parameters
    ----------
    values : 1D or 2D ndarray
        Array of values.
    labels : 1D ndarray
        Labels to group by values.
    weights : 1D ndarray, optional
        Weights associated with labels. Only used when `red_op` is
        'average', 'mean', 'sum' and 'mode'. Default is None.
    target_labels : 1D ndarray, optional
        Target labels. Arrange new array following the ordering of labels
        in the `target_labeles`. When None, new array is arranged in ascending
        order of source labels. Default is None.
    red_op : str or callable, optional
        How to summarize data. If str, options are: {'min', 'max', 'sum',
        'mean', 'median', 'mode', 'average'}. If callable, it should receive
        a 1D array of values, an array of weights (or None), and return a
        scalar value. Default is 'mean'.
    dtype : dtype
        Dtype of output array. Default is float.
    axis : {0, 1}, optional
        If `axis=0` apply to each row (therefore, reducing number of columns
        per row). Otherwise, to each column (reducing number of rows per
        column). Default is 0.

    Returns
    -------
    target_values : ndarray
        Summarized target values.

    Notes
    -----
    This method is similar to scipy's labeled_comprehension, although it does
    not accept weights.

    Examples
    --------
    TODO: examples and check axis arg
    TODO: return 1D array when input is 1D
    """

    values2d = np.atleast_2d(values)
    if target_labels is None:
        target_labels = np.unique(labels)
        idx_back = None
    else:
        target_labels, idx_back = np.unique(target_labels, return_inverse=True)

    if weights is not None:
        weights = np.atleast_2d(weights)

    if isinstance(red_op, str):
        if red_op in ['mean', 'average']:
            if weights is None:
                fred = lambda x, w, ax: np.mean(x, axis=ax)
            else:
                fred = lambda x, w, ax: np.average(x, weights=w, axis=ax)
        elif red_op == 'median':
            fred = lambda x, w, ax: np.median(x, axis=ax)
        elif red_op == 'mode':
            if weights is None:
                fred = lambda x, w, ax: mode(x, axis=ax)[0].ravel()
            else:
                fred = lambda x, w, ax: weighted_mode(x, w, axis=ax)
                # np.apply_along_axis(lambda x:
                # np.bincount(x, weights=w.ravel()).argmax(), 0, x)
        elif red_op == 'sum':
            fred = lambda x, w, ax: np.sum(x if w is None else w*x, axis=ax)
        elif red_op == 'max':
            fred = lambda x, w, ax: np.max(x, axis=ax)
        elif red_op == 'min':
            fred = lambda x, w, ax: np.min(x, axis=ax)
        else:
            raise ValueError('Unknown reduction operation \'{0}\''.
                             format(red_op))
    else:
        fred = red_op

    nr, nc = values2d.shape
    new_shape = (nr, target_labels.size) if axis == 0 \
        else (target_labels.size, nc)
    mapped = np.empty(new_shape, dtype=dtype)
    for ilab, lab in enumerate(target_labels):
        mask = labels == lab
        wm = None
        if weights is not None:
            wm = weights[:, mask] if axis == 0 else weights[:, mask].T

        if red_op in ['min', 'max', 'sum', 'mean', 'average', 'median', 'mode']:
            if axis == 0:
                mapped[:, ilab] = fred(values2d[:, mask], wm, ax=1-axis)
            else:
                mapped[ilab, :] = fred(values2d[mask, :], wm, ax=1-axis)

        else:
            for idx in range(values2d.shape[axis]):
                if axis == 0:
                    mapped[idx, ilab] = fred(values2d[idx, mask], wm, ax=1-axis)
                else:
                    mapped[ilab, idx] = fred(values2d[mask, idx], wm, ax=1-axis)

    if idx_back is not None:
        if axis == 0:
            return mapped[:, idx_back]
        return mapped[idx_back, :]
    return mapped
