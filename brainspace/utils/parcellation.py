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
    lab : ndarray
        Array to relabel.
    start_from : int, optional
        Initial label. The default is 0.

    Returns
    -------
    new_lab : ndarray
        Array with consecutive labels.

    """

    new_lab = np.empty_like(lab)
    new_lab[:] = np.unique(lab, return_inverse=True)[1]
    new_lab += start_from
    return new_lab


def relabel(lab, new_labels=None):
    """Relabel array.

    Parameters
    ----------
    lab : array_like
        Array to relabel.
    new_labels: array_like or dict, optional
        New labels. If dict, provide new label for each label in input array.
        If array_like, mapping is performed in ascending order. If None,
        relabel consecutively, starting from 0. Default is None.

    Returns
    -------
    new_lab : ndarray
        Array with new labels.

    """

    if isinstance(new_labels, dict):
        new_lab = lab.copy()
        for l1, l2 in new_labels.items():
            new_lab[lab == l1] = l2
        return new_lab

    if new_labels is None:
        return relabel_consecutive(lab)

    keys = np.unique(lab)[:new_labels.size]
    return relabel(lab, dict(zip(keys, new_labels)))


def find_label_correspondence(lab1, lab2):
    """Find label correspondences.


    Parameters
    ----------
    lab1 : ndarray, shape = (n_lab,)
        First array of labels.
    lab2 : ndarray, shape = (n_lab,)
        Second array of labels.

    Returns
    -------
    dict
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
    cost[tuple([*upairs.T])] -= n_overlap
    ridx, cidx = linear_sum_assignment(cost)

    return dict(zip(u1[ridx], u2[cidx]))


def relabel_by_overlap(lab, ref_lab):
    """Relabel according to overlap with reference.

    Parameters
    ----------
    lab : ndarray, shape = (n_lab,)
        Array of labels.
    ref_lab : ndarray, shape = (n_lab,)
        Reference array of labels.

    Returns
    -------
    new_lab : ndarray, shape = (n_lab,)
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
    values : ndarray, shape = (n_rows, n_cols) or (n_cols,)
        Source array of values.
    mask : ndarray, shape = (n_mask,)
        Mask of boolean values. Data is mapped to mask.
        If `values` is 2D, the mask is applied according to `axis`.
    fill : float, optional
        Value used to fill elements outside the mask. Default is 0.
    axis : {0, 1}, optional
        If ``axis == 0`` map rows. Otherwise, map columns. Default is 0.

    Returns
    -------
    output : ndarray
        Values mapped to mask. If `values` is 1D, shape (n_mask,).
        When `values` is 2D, shape (n_rows, n_mask) if ``axis == 0`` and
        (n_mask, n_cols) otherwise.

    """

    if np.issubdtype(values.dtype, np.integer) and not np.isfinite(fill):
        raise ValueError("Cannot use non-finite 'fill' with integer arrays.")

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


def map_to_labels(source_val, target_lab, mask=None, fill=0, source_lab=None):
    """Map data in source to target according to their labels.

    Target labels are sorted in ascending order, such that the smallest label
    indexes the value at position 0 in `source_val`. If `source_lab` is
    specified, any label in `target_lab` must be in `source_lab`.

    Parameters
    ----------
    source_val : ndarray, shape = (n_val,)
        Source array of values.
    target_lab : ndarray, shape = (n_lab,)
        Target labels.
    mask : ndarray, shape = (n_lab,), optional
        If mask is not None, only consider target labels in mask.
        Default is None.
    fill : float, optional
        Value used to fill elements outside the mask. Only used if mask is not
        None. Default is 0.
    source_lab : ndarray, shape = (n_val,), optional
        Source labels for source values. If None, use unique labels in
        `target_lab` in ascending order. Default is None.

    Returns
    -------
    target_val : ndarray, shape = (n_lab,)
        Target array with corresponding source values.

    """

    if mask is not None:
        target_lab2 = target_lab[mask]
        labs2 = map_to_labels(source_val, target_lab2, source_lab=source_lab)
        return map_to_mask(labs2, mask, fill=fill)

    if source_lab is None:
        uq_tl, idx_tl = np.unique(target_lab, return_inverse=True)
        return source_val[idx_tl]

    if source_lab.size != source_val.size:
        raise ValueError('Source values and labels must have same size.')

    uq_sl, idx_sl = np.unique(source_lab, return_inverse=True)
    if source_lab.size != uq_sl.size:
        raise ValueError('Source labels must have distinct labels.')

    source_val = source_val[idx_sl]
    return source_val[target_lab]


def _get_redop(red_op, weights=None, axis=None):
    if red_op in ['mean', 'average']:
        if weights is None:
            def fred(x, w): return np.mean(x, axis=axis)
        else:
            def fred(x, w): return np.average(x, weights=w, axis=axis)
    elif red_op == 'median':
        def fred(x, w): return np.median(x, axis=axis)
    elif red_op == 'mode':
        if weights is None:
            def fred(x, w): return mode(x, axis=axis)[0].ravel()
        else:
            def fred(x, w): return weighted_mode(x, w, axis=axis)
    elif red_op == 'sum':
        def fred(x, w): return np.sum(x if w is None else w * x, axis=axis)
    elif red_op == 'max':
        def fred(x, w): return np.max(x, axis=axis)
    elif red_op == 'min':
        def fred(x, w): return np.min(x, axis=axis)
    else:
        raise ValueError("Unknown reduction operation '{0}'".format(red_op))
    return fred


def reduce_by_labels(values, labels, weights=None, target_labels=None,
                     red_op='mean', axis=0, dtype=np.float):
    """Summarize data in `values` according to `labels`.

    Parameters
    ----------
    values : 1D or 2D ndarray
        Array of values.
    labels : 1D ndarray, shape = (n_lab,)
        Labels used summarize values.
    weights : 1D ndarray, shape = (n_lab,), optional
        Weights associated with labels. Only used when `red_op` is
        'average', 'mean', 'sum' or 'mode'. Weights are not normalized.
        Default is None.
    target_labels : 1D ndarray, optional
        Target labels. Arrange new array following the ordering of labels
        in the `target_labels`. When None, new array is arranged in ascending
        order of `labels`. Default is None.
    red_op : str or callable, optional
        How to summarize data. If str, options are: {'min', 'max', 'sum',
        'mean', 'median', 'mode', 'average'}. If callable, it should receive
        a 1D array of values, array of weights (or None) and return a scalar
        value. Default is 'mean'.
    dtype : dtype, optional
        Data type of output array. Default is float.
    axis : {0, 1}, optional
        If ``axis == 0``, apply to each row (reduce number of columns per row).
        Otherwise, apply to each column (reduce number of rows per column).
        Default is 0.

    Returns
    -------
    target_values : ndarray
        Summarized target values.
    """

    if axis == 1 and values.ndim == 1:
        axis = 0

    if target_labels is None:
        uq_tl = np.unique(labels)
        idx_back = None
    else:
        uq_tl, idx_back = np.unique(target_labels, return_inverse=True)

    if weights is not None:
        weights = np.atleast_2d(weights)

    v2d = np.atleast_2d(values)
    if axis == 1:
        v2d = v2d.T

    if isinstance(red_op, str):
        fred = _get_redop(red_op, weights=weights, axis=1)
    else:
        fred = red_op

    mapped = np.empty((v2d.shape[0], uq_tl.size), dtype=dtype)
    for ilab, lab in enumerate(uq_tl):
        mask = labels == lab
        wm = None if weights is None else weights[:, mask]

        if isinstance(red_op, str):
            mapped[:, ilab] = fred(v2d[:, mask], wm)

        else:
            for idx in range(v2d.shape[0]):
                mapped[idx, ilab] = fred(v2d[idx, mask], wm)

    if idx_back is not None:
        mapped = mapped[:, idx_back]

    if axis == 1:
        mapped = mapped.T

    if values.ndim == 1:
        return mapped[0]
    return mapped
