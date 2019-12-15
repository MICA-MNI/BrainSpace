"""
Tutorial 0: Preparing your data for gradient analysis
=================================================
In this example, we will introduce how to preprocess raw MRI data and how
to prepare it for subsequent gradient analysis in the next tutorials.

Preprocessing
----------------------
Begin with an MRI dataset that is organized in `BIDS
<https://bids.neuroimaging.io/>`_ format. We recommend preprocessing your data using `fmriprep
<http://fmriprep.readthedocs.io/>`_, as described below, but any preprocessing pipeline will work.

Following is example code to run `fmriprep <http://fmriprep.readthedocs.io/>`_ using docker from the command line::

    docker run -ti --rm \\
      -v <local_BIDS_data_dir>:/data:ro \\
      -v <local_output_dir>:/out poldracklab/fmriprep:latest \\
      --output-spaces fsaverage5 \\
      --fs-license-file license.txt \\
      /data /out participant

*Note: For this tutorial, it is crucial to output the data onto a cortical surface template space.*
"""

import warnings
warnings.simplefilter('ignore')

################################################################################
# Confound regression
# ++++++++++++++++++++++++
# To remove confound regressors from the output of the fmriprep pipeline, first extract the confound columns.
# For example::
#
#    from brainspace.utils.confound_loader import load_confounds
#    confounds_out = load_confounds("path to confound file",
#                               strategy=["minimal"],
#                               n_components=0.95,
#                               motion_model="6params")

################################################################################
# Otherwise, simply read in:


import numpy as np

confounds_out = np.loadtxt('../../shared/data/preprocessing/sub-010188_ses-02_task-rest_acq-AP_run-01_confounds.txt')

# ############################################################################### Then regress these confounds from
# the preprocessed data using `nilearn
# <https://nilearn.github.io/auto_examples/03_connectivity/plot_signal_extraction.html#extract-signals-on-a
# -parcellation-defined-by-labels/>`_


from nilearn import datasets

destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
medial_wall_ind = destrieux_atlas['labels'].index(b'Medial_wall')
labels = destrieux_atlas['labels']
labels.remove(b'Medial_wall')
labels.remove(b'Unknown')
lh_labels = ['L_' + i.decode() for i in labels]
rh_labels = ['R_' + i.decode() for i in labels]
label_list = list(np.concatenate((lh_labels, rh_labels)))

################################################################################
# Do the confound regression

from nilearn import signal
import nibabel as nib

seed_timeseries = []
hemi = ['left', 'right']
hh = ['lh', 'rh']

for h in [0, 1]:
    timeseries = nib.load('../../shared/data/preprocessing/sub-010188_ses-02_task-rest_acq-AP_run-01.fsa5.%s.mgz' %
                          hh[h]).get_data().squeeze()

    # remove confounds
    # timeseries_clean = signal.clean(timeseries.T,confounds=confounds_out).T
    timeseries_clean = timeseries.copy()

    parcellation = destrieux_atlas['map_%s' % hemi[h]]
    parcel_ind = np.unique(parcellation)
    parcel_ind = np.setdiff1d(parcel_ind, medial_wall_ind)

    for i in parcel_ind:
        roi_ind = np.where(parcellation == i)[0]
        seed_timeseries.append(np.nanmean(timeseries_clean[roi_ind], axis=0))

seed_timeseries = np.asarray(seed_timeseries)
seed_timeseries[np.isnan(seed_timeseries)] = 0

################################################################################
# Calculate the functional connectivity matrix using
# `nilearn <https://nilearn.github.io/auto_examples/03_connectivity/plot_signal_extraction
# .html#compute-and-display-a-correlation-matrix/>`_

from nilearn.connectome import ConnectivityMeasure

correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrix = correlation_measure.fit_transform([seed_timeseries.T])[0]
# save correlation matrix
np.save('../../shared/data/preprocessing/correlation_matrix.npy', correlation_matrix)

################################################################################
# Plot the correlation matrix

import numpy as np
from nilearn import plotting

mat_mask = np.where(np.std(correlation_matrix, axis=1) > 0.1)[0]
masked_labels = [label_list[i] for i in mat_mask]
# np.fill_diagonal(correlation_matrix, 0)
c = correlation_matrix[mat_mask, :][:, mat_mask]
plotting.plot_matrix(c, figure=(15, 15),
                     labels=masked_labels,
                     vmax=0.8, vmin=-0.8,
                     reorder=True)

################################################################################
# Load data and run gradient analysis
# ----------------------
# Load fsaverage5 surfaces

from brainspace.mesh.mesh_io import read_surface

surf_lh = read_surface('../../shared/surfaces/fsa5.pial.lh.gii')
surf_rh = read_surface('../../shared/surfaces/fsa5.pial.rh.gii')

################################################################################
# Load labels

import numpy as np
from nilearn import datasets
from brainspace.utils.parcellation import map_to_labels

destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
labeling = np.concatenate((destrieux_atlas['map_left'],
                           destrieux_atlas['map_right'] + max(destrieux_atlas['map_left']) + 1))
mask = (labeling != 0) * (labeling != medial_wall_ind)
mask = mask * (labeling != medial_wall_ind +
               max(destrieux_atlas['map_left']) + 1)

################################################################################
# Run gradient analysis

from brainspace.gradient import GradientMaps

gm = GradientMaps(n_components=5, random_state=0)
mat_mask = np.where(np.std(correlation_matrix, axis=1) > 0.1)[0]
gm.fit(correlation_matrix[mat_mask, :][:, mat_mask])

################################################################################
# Visualize results

# this only works because of left hemisphere
removed_labels = list(set(label_list) - set(masked_labels))
rm_label_ind = []
for i in range(len(removed_labels)):
    removed_label_ind = destrieux_atlas['labels'].index(removed_labels[i][2:].encode())
    mask = mask * (labeling != removed_label_ind)
    rm_label_ind.append(removed_label_ind)

emb = np.zeros((152, 5))
rm_label_ind.append(0)
rm_label_ind.append(76)
rm_label_ind.append(medial_wall_ind)
rm_label_ind.append(medial_wall_ind +
                    max(destrieux_atlas['map_left']) + 1)
ind = np.setdiff1d(range(76 * 2), rm_label_ind)
emb[ind, :] = gm.gradients_

grad = [None] * 2
for i in range(2):
    # map the gradient to the parcels
    grad[i] = map_to_labels(emb[:, i], labeling, mask=mask, fill=np.nan)

from brainspace.plotting import plot_hemispheres

plot_hemispheres(surf_lh, surf_rh, array_name=grad, size=(1200, 600), cmap='viridis_r',
                 color_bar=True, label_text=['Grad1', 'Grad2'])

###############################################################################
# This concludes the setup tutorial.
# The following tutorials can be run using either the output generated here or the example data.
