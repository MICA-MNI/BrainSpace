"""
Tutorial 1: Building your first gradient
=================================================
In this example, we will derive a gradient and do some basic inspections to
determine which gradients may be of interest and what the multidimensional
organization of the gradients looks like.
"""


###############################################################################
# We’ll first start by loading some sample data. Note that we’re using
# parcellated data for computational efficiency.

import warnings
warnings.simplefilter('ignore')

from brainspace.datasets import load_group_hcp, load_parcellation, load_conte69

# First load mean connectivity matrix and Schaefer parcellation
conn_matrix = load_group_hcp('schaefer', n_parcels=400)
labeling = load_parcellation('schaefer', n_parcels=400)

# and load the conte69 hemisphere surfaces
surf_lh, surf_rh = load_conte69()


###############################################################################
# Let’s first look at the parcellation scheme we’re using.

from brainspace.plotting import plot_hemispheres

plot_hemispheres(surf_lh, surf_rh, array_name=labeling,
                 size=(800, 150), cmap_name='tab20')


###############################################################################
# and let’s construct our gradients.

from brainspace.gradient import GradientMaps

# Construct the gradients
gm = GradientMaps(random_state=0)
gm.fit(conn_matrix)


###############################################################################
# Note that the default parameters are normalized angle kernel, diffusion
# embedding approach, 10 components. Once you have your gradients, a good first
# step is to simply inspect what they look like. Let’s have a look at the first
# two gradients.

import numpy as np

from brainspace.utils.parcellation import map_to_labels

mask = labeling != 0

gradients = [None] * 2
for i in range(2):
    # map the gradient to the parcels
    gradients[i] = map_to_labels(gm.gradients_[:, i], labeling,
                                 mask=mask, fill=np.nan)


plot_hemispheres(surf_lh, surf_rh, array_name=gradients,
                 size=(800, 300), cmap_name='viridis')


###############################################################################
# But which gradients should you keep for your analysis? In some cases you may
# have an a priori interest in some previously defined set of gradients. When
# you do not have a pre-defined set, you can instead look at the lambdas
# (eigenvalues) of each component in a scree plot. Higher eigenvalues (or lower
# in Laplacian eigenmaps) are more important, so one can choose a cut-off based
# on a scree plot.

import matplotlib.pyplot as plt

plt.scatter(range(gm.lambdas_.size), gm.lambdas_)


###############################################################################
# This concludes the first tutorial. In the next tutorial we will have a look
# at how to customize the methods of gradient estimation, as well as gradient
# alignments.
