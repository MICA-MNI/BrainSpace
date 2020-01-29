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


    import numpy as np
    confounds_out = np.loadtxt('../../shared/data/preprocessing/sub-010188_ses-02_'
                               'task-rest_acq-AP_run-01_confounds.txt')









Then regress these confounds from the preprocessed data using `nilearn
<https://nilearn.github.io/auto_examples/03_connectivity/
plot_signal_extraction.html#extract-signals-on-a-parcellation-
defined-by-labels/>`_


.. code-block:: default


    from nilearn import datasets

    destrieux_atlas = datasets.fetch_atlas_surf_destrieux()

    # Remove non-cortex regions
    regions = destrieux_atlas['labels'].copy()
    masked_regions = [b'Medial_wall', b'Unknown']
    masked_labels = [regions.index(r) for r in masked_regions]
    for r in masked_regions:
        regions.remove(r)

    # Build Destrieux parcellation and mask
    labeling = np.concatenate([destrieux_atlas['map_left'],
                               destrieux_atlas['map_right']])
    mask = ~np.isin(labeling, masked_labels)

    # Distinct labels for left and right hemispheres
    lab_lh = destrieux_atlas['map_left']
    labeling[lab_lh.size:] += lab_lh.max() + 1









Do the confound regression


.. code-block:: default


    from brainspace.utils.parcellation import reduce_by_labels
    from nilearn import signal
    import nibabel as nib

    timeseries_clean = [None] * 2
    for i, h in enumerate(['lh', 'rh']):

        timeseries = nib.load('../../shared/data/preprocessing/sub-010188_ses-02_'
                              'task-rest_acq-AP_run-01.fsa5.%s.'
                              'mgz' % h).get_fdata().squeeze()

        # remove confounds
        # timeseries_clean = signal.clean(timeseries.T, confounds=confounds_out).T
        timeseries_clean[i] = timeseries.copy()

    seed_timeseries = np.vstack(timeseries_clean)
    seed_timeseries = reduce_by_labels(seed_timeseries[mask], labeling[mask],
                                       axis=1, red_op='mean')









Calculate the functional connectivity matrix using
`nilearn <https://nilearn.github.io/auto_examples/03_connectivity/plot_
signal_extraction.html#compute-and-display-a-correlation-matrix/>`_.


.. code-block:: default


    from nilearn.connectome import ConnectivityMeasure

    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([seed_timeseries.T])[0]

    # save correlation matrix
    # np.save('../../shared/data/preprocessing/correlation_matrix.npy',
    #         correlation_matrix)









Plot the correlation matrix


.. code-block:: default


    from nilearn import plotting

    # Reduce matrix size, only visualization purposes
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
Load fsaverage5 surfaces


.. code-block:: default


    from brainspace.mesh.mesh_io import read_surface

    surf_lh = read_surface('../../shared/surfaces/fsa5.pial.lh.gii')
    surf_rh = read_surface('../../shared/surfaces/fsa5.pial.rh.gii')









Run gradient analysis


.. code-block:: default


    from brainspace.gradient import GradientMaps

    gm = GradientMaps(n_components=2, random_state=0)
    gm.fit(correlation_matrix)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    /media/oualid/hd500/oualid/BrainSpace/brainspace/gradient/embedding.py:66: UserWarning: Affinity is not symmetric. Making symmetric.
      warnings.warn('Affinity is not symmetric. Making symmetric.')

    GradientMaps(alignment=None, approach='dm', kernel=None, n_components=2,
                 random_state=0)



Visualize results


.. code-block:: default

    from brainspace.plotting import plot_hemispheres
    from brainspace.utils.parcellation import map_to_labels

    # remove_labels = np.unique(labeling[mask])[~mat_mask]
    # mask &= np.isin(labeling, remove_labels)

    grad = [None] * 2
    for i in range(2):
        # map the gradient to the parcels
        grad[i] = map_to_labels(gm.gradients_[:, i], labeling, mask=mask,
                                fill=np.nan)

    # sphinx_gallery_thumbnail_number = 2
    plot_hemispheres(surf_lh, surf_rh, array_name=grad, size=(1200, 600),
                     cmap='viridis_r', color_bar=True,
                     label_text=['Grad1', 'Grad2'], embed_nb=True,
                     interactive=False)





.. image:: /python_doc/auto_examples/images/sphx_glr_plot_tutorial0_002.png
    :class: sphx-glr-single-img





This concludes the setup tutorial. The following tutorials can be run using
either the output generated here or the example data.


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  2.383 seconds)


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
