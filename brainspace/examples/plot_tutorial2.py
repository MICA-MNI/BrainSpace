"""
Tutorial 2: Customizing and aligning gradients
=================================================
In this tutorial you’ll learn about the methods available within the
GradientMaps class. The flexible usage of this class allows for the
customization of gradient computation with different kernels and dimensionality
reductions, as well as aligning gradients from different datasets.
"""


###############################################################################
# As before, we’ll start by loading the sample data.


from brainspace.datasets import load_group_hcp, load_parcellation, load_conte69

# First load mean connectivity matrix and Schaefer parcellation
conn_matrix = load_group_hcp('schaefer', n_parcels=400)
labeling = load_parcellation('schaefer', n_parcels=400)
mask = labeling != 0

# and load the conte69 hemisphere surfaces
surf_lh, surf_rh = load_conte69()


###############################################################################
# The GradientMaps object allows for many different kernels and dimensionality
# reduction techniques. Let’s have a look at three different kernels.

import numpy as np

from brainspace.gradient import GradientMaps
from brainspace.plotting import plot_hemispheres
from brainspace.utils.parcellation import map_to_labels

kernels = ['pearson', 'spearman', 'normalized_angle']

gradients_kernel = [None] * len(kernels)
for i, k in enumerate(kernels):
    gm = GradientMaps(kernel=k, approach='dm', random_state=0)
    gm.fit(conn_matrix)

    gradients_kernel[i] = map_to_labels(gm.gradients_[:, i], labeling,
                                        mask=mask, fill=np.nan)


plot_hemispheres(surf_lh, surf_rh, array_name=gradients_kernel,
                 size=(800, 450), cmap_name='viridis')


###############################################################################
# It seems the gradients provided by these kernels are quite similar although
# their scaling is quite different. Do note that the gradients are in arbitrary
# units, so the smaller/larger axes across kernels do not imply anything.
# Similar to using different kernels, we can also use different dimensionality
# reduction techniques.

embeddings = ['pca', 'le', 'dm']

gradients_embedding = [None] * len(embeddings)
for i, emb in enumerate(embeddings):
    gm = GradientMaps(kernel='normalized_angle', approach=emb, random_state=0)
    gm.fit(conn_matrix)

    gradients_embedding[i] = map_to_labels(gm.gradients_[:, i], labeling,
                                           mask=mask, fill=np.nan)


plot_hemispheres(surf_lh, surf_rh, array_name=gradients_embedding,
                 size=(800, 450), cmap_name='viridis')


###############################################################################
# Here we do see some substantial differences: PCA appears to find a slightly
# different axis, with the somatomotor in the middle between default mode and
# visual, whereas LE and DM both find the canonical first gradient but their
# signs are flipped. Fortunately, the sign of gradients is arbitrary, so we
# could simply multiply either the LM and DM gradient by -1 to make them more
# comparable.
#
# A more principled way of increasing comparability across gradients are
# alignment techniques. BrainSpace provides two alignment techniques:
# Procrustes analysis, and joint alignment. For this example we will load
# functional connectivity data of a second subject group and align it with the
# first group.

from brainspace.datasets import load_holdout_hcp

conn_matrix2 = load_holdout_hcp('schaefer', 400)
gp = GradientMaps(kernel='normalized_angle', alignment='procrustes')
gj = GradientMaps(kernel='normalized_angle', alignment='joint')

gp.fit([conn_matrix2, conn_matrix])
gj.fit([conn_matrix2, conn_matrix])


###############################################################################
# Here, `gp` contains the Procrustes aligned data and `gj` contains the joint
# aligned data. Let’s plot them, but in separate figures to keep things
# organized.

# First gradient from holdout and original data
gradients_unaligned = [None] * 2
for i in range(2):
    gradients_unaligned[i] = map_to_labels(gp.gradients_[i][:, 0], labeling,
                                           mask=mask, fill=np.nan)

plot_hemispheres(surf_lh, surf_rh, array_name=gradients_unaligned,
                 size=(800, 300), cmap_name='viridis')


###############################################################################

gradients_procrustes = [None] * 2
for i in range(2):
    gradients_procrustes[i] = map_to_labels(gp.aligned_[i][:, 0], labeling,
                                            mask=mask, fill=np.nan)

plot_hemispheres(surf_lh, surf_rh, array_name=gradients_procrustes,
                 size=(800, 300), cmap_name='viridis')


###############################################################################

gradients_joint = [None] * 2
for i in range(2):
    gradients_joint[i] = map_to_labels(gj.aligned_[i][:, 0], labeling,
                                       mask=mask, fill=np.nan)

plot_hemispheres(surf_lh, surf_rh, array_name=gradients_joint,
                 size=(800, 300), cmap_name='viridis')


###############################################################################
# Before gradient alignment, the first gradient is reversed, but both
# alignments resolve this issue. If the input data was less similar, alignments
# may also resolve changes in the order of the gradients. However, you should
# always inspect the output of an alignment; if the input data are sufficiently
# dissimilar then the alignment may produce odd results.
#
# That concludes the second tutorial. In the third tutorial we will consider
# null hypothesis testing of comparisons between gradients and other markers.
