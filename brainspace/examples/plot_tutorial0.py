"""
Tutorial 0: Preparing your data for gradient analysis
=================================================
In this example, we will introduce how to preprocess raw MRI data and how
to prepare it for subsequent gradient analysis in the next tutorials.

Preprocessing
----------------------
Begin with an MRI dataset that is organized in `BIDS <https://bids.neuroimaging.io/>`_ format. We recommend preprocessing your data using `fmriprep <http://fmriprep.readthedocs.io/>`_, as described below, but any preprocessing pipeline will work.

Following is example code to run `fmriprep <http://fmriprep.readthedocs.io/>`_ using docker from the command line::

    docker run -ti --rm \\
      -v <local_BIDS_data_dir>:/data:ro \\
      -v <local_output_dir>:/out poldracklab/fmriprep:latest \\
      --output-spaces fsaverage5 \\
      --fs-license-file license.txt \\
      /data /out participant

*Note: For this tutorial, it is crucial to output the data onto a cortical surface template space.*

Confound regression
++++++++++++++++++++++++
To remove confound regressors from the output of the fmriprep pipeline, first extract the confound columns. For example::

    from brainspace.utils.confound_loader import load_confounds
    confounds_out = load_confounds("path to confound file",
                                   strategy=["minimal"],
                                   n_components=0.95,
                                   motion_model="6params")

Then regress these confounds from the preprocessed data using `nilearn <https://nilearn.github.io/auto_examples/03_connectivity/plot_signal_extraction.html#extract-signals-on-a-parcellation-defined-by-labels/>`_::

    from nilearn import datasets
    fsaverage = datasets.fetch_surf_fsaverage()
    destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
    labels = destrieux_atlas['labels']

    parcellation = destrieux_atlas['map_left']


    timeseries = nib.load('../../shared/data/preprocessing/sub-01_task-rest.fsa5.lh.mgz').get_data()

    from nilearn.input_data import NiftiLabelsMasker
    masker = NiftiLabelsMasker(labels_img=destrieux_atlas, standardize=True)
    time_series = masker.fit_transform(fmri_filenames, confounds=confounds_out)

    import numpy as np

    for i in ....:
        roi_ind = np.where(parcellation == labels[i][0])[0]
        seed_timeseries = np.mean(timeseries[roi_ind], axis=0)


Calculate the functional connectivity matrix using `nilearn <https://nilearn.github.io/auto_examples/03_connectivity/plot_signal_extraction.html#compute-and-display-a-correlation-matrix/>`_::

    from nilearn.connectome import ConnectivityMeasure
    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]

Plot the correlation matrix::

    import numpy as np
    from nilearn import plotting
    np.fill_diagonal(correlation_matrix, 0)
    plotting.plot_matrix(correlation_matrix, figure=(10, 8), labels=labels[1:],
                         vmax=0.8, vmin=-0.8, reorder=True)

In summary
----------------------

Load surfaces::

    fsaverage = datasets.fetch_surf_fsaverage()

Load labels::

    from nilearn import datasets
    destrieux_atlas = datasets.fetch_atlas_surf_destrieux()


Load matrix::


"""

###############################################################################
# This concludes the setup tutorial. The following tutorials can be run using either the output generated here or the example data.
