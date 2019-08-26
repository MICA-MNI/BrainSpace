""" Test brainspace.utils.parcellation """


import pytest

import numpy as np

from brainspace.utils import parcellation as parc


parametrize = pytest.mark.parametrize


testdata_consecutive = [
    # default start_from = 0 and dtype
    (np.array([1, 3, 3, 2, 2, 2], dtype=np.int),
     {},
     np.array([0, 2, 2, 1, 1, 1], dtype=np.int)),

    # default start_from = 0 and dtype
    (np.array([1, 3, 3, 2, 2, 2], dtype=np.uint8),
     {'start_from': 0},
     np.array([0, 2, 2, 1, 1, 1], dtype=np.uint8)),

    # default start_from = 1 and dtype
    (np.array([1, 3, 3, 2, 2, 2], dtype=np.float),
     {'start_from': 1},
     np.array([1, 3, 3, 2, 2, 2], dtype=np.float)),
]

testdata_relabel = [
    # default new_labels = None => consecutive
    (np.array([1, 3, 3, 2, 2, 2], dtype=np.int),
     {},
     np.array([0, 2, 2, 1, 1, 1], dtype=np.int)),

    # with new_labels as array
    (np.array([1, 3, 3, 2, 2, 2], dtype=np.uint8),
     {'new_labels': np.array([2, 2, 3])},
     np.array([2, 3, 3, 2, 2, 2], dtype=np.uint8)),

    # without some labels
    (np.array([1, 3, 3, 2, 2, 2], dtype=np.uint8),
     {'new_labels': np.array([2, 3])},
     np.array([2, 3, 3, 3, 3, 3], dtype=np.uint8)),

    # with new_labels as dict
    (np.array([1, 3, 3, 2, 2, 2], dtype=np.float),
     {'new_labels': {1: 0, 2: 4, 3: 1}},
     np.array([0, 1, 1, 4, 4, 4], dtype=np.float)),

    # without some labels
    (np.array([1, 3, 3, 2, 2, 2], dtype=np.float),
     {'new_labels': {1: 0, 3: 1}},
     np.array([0, 1, 1, 2, 2, 2], dtype=np.float)),
]


testdata_correspondence = [
    # dict correspondence
    (np.array([1, 3, 3, 2, 2, 2], dtype=np.int),
     np.array([0, 2, 2, 1, 1, 1], dtype=np.int),
     {1: 0, 3: 2, 2: 1}),

    # dict correspondence with more input labels
    (np.array([3, 1, 1, 2, 2, 2], dtype=np.uint8),
     np.array([2, 3, 3, 2, 2, 2], dtype=np.uint8),
     {1: 3, 2: 2}),

    # dict correspondence with more ref labels
    (np.array([3, 1, 1, 2, 2, 2], dtype=np.float),
     np.array([4, 3, 3, 6, 1, 1], dtype=np.float),
     {1: 3, 2: 1, 3: 4}),
]


testdata_overlap = [
    # overlap
    (np.array([1, 3, 3, 2, 2, 2], dtype=np.int),
     np.array([0, 2, 2, 1, 1, 1], dtype=np.int),
     np.array([0, 2, 2, 1, 1, 1], dtype=np.int)),

    # overlap with more input labels -> remaining with consecutive
    (np.array([3, 1, 1, 2, 2, 2], dtype=np.uint8),
     np.array([2, 3, 3, 2, 2, 2], dtype=np.uint8),
     np.array([4, 3, 3, 2, 2, 2], dtype=np.uint8)),

    # overlap with more ref labels
    (np.array([3, 1, 1, 2, 2, 2], dtype=np.float),
     np.array([4, 3, 3, 6, 1, 1], dtype=np.float),
     np.array([4, 3, 3, 1, 1, 1], dtype=np.float))
]


testdata_map_mask = [
    # with default fill=0
    (np.array([1, 3, 3, 2], dtype=np.int),
     np.array([0, 0, 1, 1, 1, 1], dtype=np.bool),
     {},
     np.array([0, 0, 1, 3, 3, 2], dtype=np.int),
     None),

    # raises ValueError is integer and fill=nan
    (np.array([1, 3, 3, 2], dtype=np.int),
     np.array([0, 0, 1, 1, 1, 1], dtype=np.bool),
     {'fill': np.nan},
     np.array([0, 0, 1, 3, 3, 2], dtype=np.int),
     ValueError),

    # test default axis=0
    (np.array([[1, 3, 3, 2], [3, 4, 4, 0]], dtype=np.float),
     np.array([1, 0, 0, 1, 1, 1], dtype=np.bool),
     {'fill': np.nan},
     np.array([[1, np.nan, np.nan, 3, 3, 2],
               [3, np.nan, np.nan, 4, 4, 0]], dtype=np.float),
     None),

    # test axis=1
    (np.array([[1, 3, 3, 2], [3, 4, 4, 0]], dtype=np.float),
     np.array([1, 0, 1], dtype=np.bool),
     {'fill': np.nan, 'axis': 1},
     np.array([[1, 3, 3, 2],
               [np.nan, np.nan, np.nan, np.nan],
               [3, 4, 4, 0]], dtype=np.float),
     None),
]


testdata_map_labels = [
    # test defaults
    (np.array([1, 2, 3], dtype=np.float),
     np.array([1, 1, 2, 2, 0, 0], dtype=np.int),
     {},
     np.array([2, 2, 3, 3, 1, 1], dtype=np.float),
     None),

    # test defaults small labels
    (np.array([1, 2, 3], dtype=np.float),
     np.array([5, 6], dtype=np.int),
     {},
     np.array([1, 2], dtype=np.float),
     None),

    # test default fill=0
    (np.array([2, 1, 3], dtype=np.float),
     np.array([1, 1, 2, 2, 0, 0], dtype=np.int),
     {'mask': np.array([1, 1, 1, 0, 0, 1], dtype=np.bool)},
     np.array([1, 1, 3, 0, 0, 2], dtype=np.float),
     None),

    # test default fill=np.nan with int
    (np.array([2, 1, 3], dtype=np.int),
     np.array([1, 1, 2, 2, 0, 0], dtype=np.int),
     {'mask': np.array([1, 1, 1, 0, 0, 1], dtype=np.bool), 'fill': np.nan},
     np.array([1, 1, 3, 0, 0, 2], dtype=np.int),
     ValueError),

    # test source_lab
    (np.array([2, 1, 3], dtype=np.float),
     np.array([1, 1, 2, 2, 0, 0], dtype=np.int),
     {'mask': np.array([1, 1, 1, 0, 0, 1], dtype=np.bool), 'fill': np.nan,
      'source_lab': np.array([2, 1, 0])},
     np.array([1, 1, 2, np.nan, np.nan, 3], dtype=np.float),
     None),

    # test source_lab.size != source_val.size
    (np.array([2, 1, 3], dtype=np.float),
     np.array([1, 1, 2, 2, 0, 0], dtype=np.int),
     {'mask': np.array([1, 1, 1, 0, 0, 1], dtype=np.bool), 'fill': np.nan,
      'source_lab': np.array([2, 1])},
     np.array([1, 1, 2, np.nan, np.nan, 3], dtype=np.float),
     ValueError),

    # test (unique source_lab).size != source_val.size
    (np.array([2, 1, 3], dtype=np.float),
     np.array([1, 1, 2, 2, 0, 0], dtype=np.int),
     {'mask': np.array([1, 1, 1, 0, 0, 1], dtype=np.bool), 'fill': np.nan,
      'source_lab': np.array([2, 1, 2])},
     np.array([1, 1, 2, np.nan, np.nan, 3], dtype=np.float),
     ValueError),

    # test (unique source_lab).size != source_val.size
    pytest.param(np.array([2, 1, 3], dtype=np.float),
                 np.array([1, 1, 2, 2, 1, 0], dtype=np.int),
                 {'mask': np.array([1, 1, 1, 0, 0, 1], dtype=np.bool),
                  'fill': np.nan,
                  'source_lab': np.array([2, 1, 0])},
                 np.array([1, 1, 2, np.nan, np.nan, 1], dtype=np.float),
                 None,
                 marks=pytest.mark.xfail),
]


testdata_reduce = [
    # test defaults
    (np.array([1, 2, 3, 4, 5, 6], dtype=np.float),
     np.array([1, 1, 2, 2, 0, 0], dtype=np.int),
     {},
     np.array([5.5, 1.5, 3.5], dtype=np.float),
     None),

    # test weights
    (np.array([1, 2, 3, 4, 5, 6], dtype=np.float),
     np.array([1, 1, 2, 2, 0, 0], dtype=np.int),
     {'weights': np.array([1, 1, 2, 1, 1, 2])},
     np.array([17/3, 1.5, 10/3], dtype=np.float),
     None),

    # Test target labels
    (np.array([1, 2, 3, 4, 5, 6], dtype=np.float),
     np.array([1, 1, 2, 2, 0, 0], dtype=np.int),
     {'target_labels': np.array([2, 1, 0])},
     np.array([3.5, 1.5, 5.5], dtype=np.float),
     None),

    # Test target labels small
    (np.array([1, 2, 3, 4, 5, 6], dtype=np.float),
     np.array([1, 1, 2, 2, 0, 0], dtype=np.int),
     {'target_labels': np.array([2, 1])},
     np.array([3.5, 1.5], dtype=np.float),
     None),

    # Test red_op
    (np.array([1, 2, 2, 5, 5, 6], dtype=np.int),
     np.array([1, 1, 1, 0, 0, 0], dtype=np.int),
     {'red_op': 'mode', 'dtype': np.int},
     np.array([5, 2], dtype=np.int),
     None),

    # Test default axis=0
    (np.array([[1, 2, 2, 5], [6, 6, 7, 8]], dtype=np.int),
     np.array([1, 1, 1, 0], dtype=np.int),
     {'red_op': 'mode', 'dtype': np.int},
     np.array([[5, 2], [8, 6]], dtype=np.int),
     None),

    # Test default axis=1
    (np.array([[1, 2, 2, 5], [6, 4, 7, 8], [6, 4, 7, 5]], dtype=np.int),
     np.array([0, 0, 0], dtype=np.int),
     {'red_op': 'mode', 'dtype': np.int, 'axis': 1},
     np.array([[6, 4, 7, 5]], dtype=np.int),
     None),

    # Test red_op callable
    (np.array([[1, 2, 2, 5], [6, 4, 7, 8], [6, 4, 7, 5]], dtype=np.int),
     np.array([0, 0, 0], dtype=np.int),
     {'red_op': lambda x, w: np.mean(x), 'axis': 1},
     np.array([[13/3, 10/3, 16/3, 18/3]], dtype=np.float),
     None),

]


@parametrize('lab, kwds, out', testdata_consecutive)
def test_consecutive(lab, kwds, out):
    res = parc.relabel_consecutive(lab, **kwds)
    assert np.all(res == out)
    assert res.dtype == out.dtype


@parametrize('lab, kwds, out', testdata_relabel)
def test_relabel(lab, kwds, out):
    res = parc.relabel(lab, **kwds)
    assert np.all(res == out)
    assert res.dtype == out.dtype


@parametrize('lab1, lab2, out', testdata_correspondence)
def test_label_correspondence(lab1, lab2, out):
    res = parc.find_label_correspondence(lab1, lab2)
    assert res == out


@parametrize('lab, ref_lab, out', testdata_overlap)
def test_overlap(lab, ref_lab, out):
    res = parc.relabel_by_overlap(lab, ref_lab)
    assert np.all(res == out)
    assert res.dtype == out.dtype


@parametrize('lab, mask, kwds, out, expects', testdata_map_mask)
def test_map_to_mask(lab, mask, kwds, out, expects):
    if expects:
        with pytest.raises(expects):
            parc.map_to_mask(lab, mask, **kwds)
    else:
        res = parc.map_to_mask(lab, mask, **kwds)
        assert np.all((res == out) | (np.isnan(out) & np.isnan(out)))
        assert res.dtype == out.dtype
        assert res.shape == out.shape


@parametrize('source_lab, target_lab, kwds, out, expects', testdata_map_labels)
def test_map_to_labels(source_lab, target_lab, kwds, out, expects):
    if expects:
        with pytest.raises(expects):
            parc.map_to_labels(source_lab, target_lab, **kwds)
    else:
        res = parc.map_to_labels(source_lab, target_lab, **kwds)
        assert np.all((res == out) | (np.isnan(out) & np.isnan(out)))
        assert res.dtype == out.dtype


@parametrize('values, labels, kwds, out, expects', testdata_reduce)
def test_reduce(values, labels, kwds, out, expects):
    if expects:
        with pytest.raises(expects):
            parc.reduce_by_labels(values, labels, **kwds)
    else:
        res = parc.reduce_by_labels(values, labels, **kwds)
        assert np.allclose(res, out)
        assert res.dtype == out.dtype
        assert res.shape == out.shape
