
.. _pymod-gradient:

Gradient Maps
===========================


- :ref:`pysec-gradient-gradient`
- :ref:`pysec-gradient-embedding`
- :ref:`pysec-gradient-alignment`
- :ref:`pysec-gradient-kernels`
- :ref:`pysec-gradient-utils`



.. _pysec-gradient-gradient:

Gradients
------------


.. currentmodule:: brainspace.gradient.gradient

.. autosummary::
   :toctree: ../../generated/

    GradientMaps


.. _pysec-gradient-embedding:

Embedding
------------


.. currentmodule:: brainspace.gradient.embedding

.. autosummary::
   :toctree: ../../generated/

    Embedding
    DiffusionMaps
    LaplacianEigenmaps
    PCAMaps
    diffusion_mapping
    laplacian_eigenmaps


.. _pysec-gradient-alignment:

Alignment
---------

.. currentmodule:: brainspace.gradient.alignment

.. autosummary::
   :toctree: ../../generated/

    ProcrustesAlignment
    procrustes_alignment
    procrustes


.. _pysec-gradient-kernels:

Kernels
---------

.. currentmodule:: brainspace.gradient.kernels

.. autosummary::
   :toctree: ../../generated/

    compute_affinity


.. _pysec-gradient-utils:

Utility functions
------------------

.. currentmodule:: brainspace.gradient.utils

.. autosummary::
   :toctree: ../../generated/

    dominant_set
    is_symmetric
    make_symmetric