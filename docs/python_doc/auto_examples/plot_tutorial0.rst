.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_python_doc_auto_examples_plot_tutorial0.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_python_doc_auto_examples_plot_tutorial0.py:


Tutorial 0: Preparing your data for gradient analysis
=====================================================
In this example, we will introduce how to preprocess raw MRI data and how
to prepare it for subsequent gradient analysis in the next tutorials.

Preprocessing
-------------
Begin with an MRI dataset that is organized in `BIDS
<https://bids.neuroimaging.io/>`_ format. We recommend preprocessing your data
using `fmriprep <http://fmriprep.readthedocs.io/>`_, as described below, but
any preprocessing pipeline will work.

Following is example code to run `fmriprep <http://fmriprep.readthedocs.io/>`_
using docker from the command line::

    docker run -ti --rm \
      -v <local_BIDS_data_dir>:/data:ro \
      -v <local_output_dir>:/out poldracklab/fmriprep:latest \
      --output-spaces fsaverage5 \
      --fs-license-file license.txt \
      /data /out participant

.. note::
    For this tutorial, it is crucial to output the data onto a cortical surface
    template space.

Confound regression
++++++++++++++++++++++++
To remove confound regressors from the output of the fmriprep pipeline, first
extract the confound columns. For example::

   from brainspace.utils.confound_loader import load_confounds
   confounds_out = load_confounds("path to confound file",
                              strategy='minimal',
                              n_components=0.95,
                              motion_model='6params')

Otherwise, simply read in:


.. code-block:: default

    from brainspace.datasets import load_confounds_preprocessing

    confounds_out = load_confounds_preprocessing()









Then regress these confounds from the preprocessed data using `nilearn
<https://nilearn.github.io/auto_examples/03_connectivity/
plot_signal_extraction.html#extract-signals-on-a-parcellation-
defined-by-labels/>`_


.. code-block:: default


    import numpy as np
    from nilearn import datasets

    atlas = datasets.fetch_atlas_surf_destrieux()

    # Remove non-cortex regions
    regions = atlas['labels'].copy()
    masked_regions = [b'Medial_wall', b'Unknown']
    masked_labels = [regions.index(r) for r in masked_regions]
    for r in masked_regions:
        regions.remove(r)

    # Build Destrieux parcellation and mask
    labeling = np.concatenate([atlas['map_left'], atlas['map_right']])
    mask = ~np.isin(labeling, masked_labels)

    # Distinct labels for left and right hemispheres
    lab_lh = atlas['map_left']
    labeling[lab_lh.size:] += lab_lh.max() + 1









Do the confound regression


.. code-block:: default


    from brainspace.datasets import fetch_timeseries_preprocessing
    from brainspace.utils.parcellation import reduce_by_labels
    from nilearn import signal

    # Fetch timeseries
    timeseries = fetch_timeseries_preprocessing()


    # Remove confounds
    clean_ts = [None] * 2
    for i, ts in enumerate(timeseries):
        clean_ts[i] = signal.clean(ts.T, confounds=confounds_out).T

    seed_ts = np.vstack(clean_ts)
    seed_ts = reduce_by_labels(seed_ts[mask], labeling[mask], axis=1, red_op='mean')









Calculate the functional connectivity matrix using
`nilearn <https://nilearn.github.io/auto_examples/03_connectivity/plot_
signal_extraction.html#compute-and-display-a-correlation-matrix/>`_:


.. code-block:: default


    from nilearn.connectome import ConnectivityMeasure

    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([seed_ts.T])[0]









Plot the correlation matrix:


.. code-block:: default


    from nilearn import plotting

    # Reduce matrix size, only for visualization purposes
    mat_mask = np.where(np.std(correlation_matrix, axis=1) > 0.2)[0]
    c = correlation_matrix[mat_mask][:, mat_mask]

    # Create corresponding region names
    regions_list = ['%s_%s' % (h, r.decode()) for h in ['L', 'R'] for r in regions]
    masked_regions = [regions_list[i] for i in mat_mask]


    corr_plot = plotting.plot_matrix(c, figure=(15, 15), labels=masked_regions,
                                     vmax=0.8, vmin=-0.8, reorder=True)





.. image:: /python_doc/auto_examples/images/sphx_glr_plot_tutorial0_001.png
    :class: sphx-glr-single-img





Run gradient analysis and visualize
-----------------------------------

Run gradient analysis


.. code-block:: default


    from brainspace.gradient import GradientMaps

    gm = GradientMaps(n_components=2, random_state=0)
    gm.fit(correlation_matrix)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /media/oualid/hd500/oualid/BrainSpace/brainspace/gradient/embedding.py:70: UserWarning: Affinity is not symmetric. Making symmetric.
      warnings.warn('Affinity is not symmetric. Making symmetric.')

    GradientMaps(alignment=None, approach='dm', kernel=None, n_components=2,
                 random_state=0)



Visualize results


.. code-block:: default

    from brainspace.datasets import load_fsa5
    from brainspace.plotting import plot_hemispheres
    from brainspace.utils.parcellation import map_to_labels

    # Map gradients to original parcels
    grad = [None] * 2
    for i, g in enumerate(gm.gradients_.T):
        grad[i] = map_to_labels(g, labeling, mask=mask, fill=np.nan)


    # Load fsaverage5 surfaces
    surf_lh, surf_rh = load_fsa5()

    # sphinx_gallery_thumbnail_number = 2
    plot_hemispheres(surf_lh, surf_rh, array_name=grad, size=(1200, 600),
                     cmap='viridis_r', color_bar=True, label_text=['Grad1', 'Grad2'])





.. image:: /python_doc/auto_examples/images/sphx_glr_plot_tutorial0_002.png
    :class: sphx-glr-single-img





This concludes the setup tutorial. The following tutorials can be run using
either the output generated here or the example data.


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  4.067 seconds)


.. _sphx_glr_download_python_doc_auto_examples_plot_tutorial0.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_tutorial0.py <plot_tutorial0.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_tutorial0.ipynb <plot_tutorial0.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
