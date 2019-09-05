
.. _pypage-python_intro:

Python Package Introduction
===========================
This document gives a basic walkthrough of BrainSpace python package.

- :ref:`pysec-python_intro-gradients`


.. _pysec-python_intro-gradients:

Working with gradients
----------------------

:class:`.GradientMaps` is the main class that offers the functionality to
work with gradients. This class builds the affinity matrix, performs the
embedding and aligns the gradients. This class follows closely the `API of scikit-learn
objects <https://scikit-learn.org/dev/developers/contributing.html#apis-of-scikit-learn-objects>`_.

#. Let's first generate two random symmetric matrices using scikit-learn :func:`~sklearn.datasets.make_spd_matrix`::

    >>> from sklearn.datasets import make_spd_matrix

    >>> x1 = make_spd_matrix(100)
    >>> x2 = make_spd_matrix(100)
    >>> x1.shape
    (100, 100)


#. Next, we build a :class:`.GradientMaps` object::

    >>> from brainspace.gradient import GradientMaps

    >>> # We build the affinity matrix using 'normalized_angle',
    >>> # use Laplacian eigenmaps (i.e., 'le') to find the gradients
    >>> # and align gradients using procrustes
    >>> gm = GradientMaps(n_gradients=2, approach='le', kernel='normalized_angle',
    ...                   alignment='procrustes')


#. Now we can compute the gradients for the two datasets by invoking the :meth:`~.GradientMaps.fit` method::

    >>> # Note that multiple datasets are passed as a list
    >>> gm.fit([x1, x2])
    GradientMaps(alignment='procrustes', approach='le', kernel='normalized_angle',
       n_gradients=2, random_state=None)

#. The object has 3 important attributes: eigenvalues, gradients, and aligned gradients::

    >>> # The eigenvalues for x1
    >>> gm.lambdas_[0]
    array([0.76390278, 0.99411812])

    >>> # and x2
    >>> gm.lambdas_[1]
    array([0.77444778, 0.99058541])

    >>> # The gradients for x1
    >>> gm.gradients_[0].shape
    (100, 2)

    >>> # and the gradients after alignment
    >>> gm.aligned_[0].shape
    (100, 2)

#. To illustrate the effect of alignment, we can check the distance between the gradients::

    >>> import numpy as np

    >>> # Disparity between the original gradients
    >>> np.sum(np.square(gm.gradients_[0] - gm.gradients_[1]))
    0.07000481706312509

    >>> # disparity is decreased after alignment
    >>> np.sum(np.square(gm.aligned_[0] - gm.aligned_[1]))
    1.4615624326798128e-05

#. We can also change the embedding approach using 'dm' or an object of :class:`.DiffusionMaps`::

    >>> # In this case we will pass an object
    >>> from brainspace.gradient import DiffusionMaps
    >>> dm = DiffusionMaps(alpha=1, diffusion_time=0)

    >>> # let's create a new gm object with the new embedding approach
    >>> gm2 = GradientMaps(n_gradients=2, approach=dm, kernel='normalized_angle',
    ...                    alignment='procrustes')

    >>> # and fit to the data
    >>> gm2.fit([x1, x2])
    GradientMaps(alignment='procrustes',
                 approach=DiffusionMaps(alpha=1, diffusion_time=0,
                                        n_components=2, random_state=None),
                 kernel='normalized_angle', n_gradients=2, random_state=None)

    >>> # the disparity between the gradients
    >>> np.sum(np.square(gm2.gradients_[0] - gm2.gradients_[1]))
    21.815792454516334

    >>> # and after alignment
    >>> np.sum(np.square(gm2.aligned_[0] - gm2.aligned_[1]))
    3.326408646218633e-05


#. If we try a different alignment method::

    >>> gm3 = GradientMaps(n_gradients=2, approach='le', kernel='normalized_angle',
    ...                    alignment='joint')
    >>> gm3.fit([x1, x2])
    GradientMaps(alignment='joint', approach='le', kernel='normalized_angle',
                 n_gradients=2, random_state=None)

    >>> # the disparity between the gradients
    >>> np.sum(np.square(gm3.gradients_[0] - gm3.gradients_[1]))
    0.019346449795655286

    >>> # with 'manifold', the embedding and alignment are performed simultaneously
    >>> np.sum(np.square(gm3.aligned_[0] - gm3.aligned_[1]))
    0.019346449795655286



