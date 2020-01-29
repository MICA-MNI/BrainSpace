.. _load_group_mpc_matlab:

=======================
load_group_mpc
=======================

------------------
Synopsis
------------------

Loads group level microstructural profile covariance matrices (`source code <https://github.com/MICA-MNI/BrainSpace/blob/master/matlab/example_data_loaders/load_group_mpc.m>`_). 

------------------
Usage
------------------

::

    conn_matrices = load_group_mpc(parcellation,scale,group)

- *parcellation*: Name of the parcellation, either 'schaefer' for Schaefer (functional) parcellations or 'vosdewael' for a subparcellation of the `Desikan-Killiany atlas`__. Both may be provided as a cell or string array. 
- *scale*: Scale of the parcellation. Either 100, 200, 300, or 400. Multiple may be provided as a vector.
- *group*: Loads data from the main group if set to 'main' (default) or the holdout group if set to 'holdout'. 
- *conn_matrices*: Structure of all requested data. 

.. _DK: https://surfer.nmr.mgh.harvard.edu/ftp/articles/desikan06-parcellation.pdf

__ DK_

.. note ::
    The mpc matrix presented here match the subject cohort of `(Paquola et al.,
    2019)
    <https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000284>`_.
    Other matrices in this package match the subject groups used by `(Vos de Wael et
    al., 2018) <https://www.pnas.org/content/115/40/10154.short>`_. We make direct
    comparisons in our tutorial for didactic purposes only. 