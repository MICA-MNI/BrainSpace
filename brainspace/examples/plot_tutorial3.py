"""
Tutorial 3: Null models for gradient significance
==================================================
In this tutorial we assess the significance of correlations between the first
canonical gradient and data from other modalities (curvature, cortical
thickness and T1w/T2w image intensity). A normal test of the significance of
the correlation cannot be used, because the spatial auto-correlation in MRI
data may bias the test statistic. In this tutorial we will show two approaches
for null hypothesis testing: spin permutations and Moran spectral
randomization.

.. note::
    When using either approach to compare gradients to non-gradient markers,
    we recommend randomizing the non-gradient markers as these randomizations
    need not maintain the statistical independence between gradients.

"""

"""
Spin Permutations
--------------------
"""

###############################################################################
# Here, we use the spin permutations approach previously proposed in
# `(Alexander-Bloch et al., 2018)
# <https://www.sciencedirect.com/science/article/pii/S1053811918304968>`_,
# which preserves the auto-correlation of the permuted feature(s) by rotating
# the feature data on the spherical domain.
# We will start by loading the conte69 surfaces for left and right hemispheres,
# their corresponding spheres, midline mask, and t1w/t2w intensity as well as
# cortical thickness data, and a template functional gradient.


from brainspace.datasets import (load_gradient, load_t1t2, load_thickness,
                                 load_conte69)

# load the conte69 hemisphere surfaces and spheres
surf_lh, surf_rh = load_conte69()
sphere_lh, sphere_rh = load_conte69(as_sphere=True)

n_pts_lh = surf_lh.n_points

# Load the data
t1wt2w = load_t1t2()
t1wt2w_lh, t1wt2w_rh = t1wt2w[:n_pts_lh], t1wt2w[n_pts_lh:]

thickness = load_thickness()
thickness_lh, thickness_rh = thickness[:n_pts_lh], thickness[n_pts_lh:]

# Template functional gradient
embedding = load_gradient('fc', idx=0)


###############################################################################
# Let’s first generate some null data using spintest.

import numpy as np

from brainspace.null_models import SpinRandomization
from brainspace.plotting import plot_hemispheres

# Let's create some rotations

n_permutations = 10

sp = SpinRandomization(n_rep=n_permutations, random_state=0)
sp.fit(sphere_lh, points_rh=sphere_rh)

t1wt2w_rotated = np.hstack(sp.randomize(t1wt2w_lh, t1wt2w_rh))
thickness_rotated = np.hstack(sp.randomize(thickness_lh, thickness_rh))


###############################################################################
# As an illustration of the rotation, let’s plot the original t1w/t2w data


plot_hemispheres(surf_lh, surf_rh, array_name=t1wt2w,
                 size=(800, 150), cmap_name='viridis',
                 nan_color=(0.5, 0.5, 0.5, 1))


###############################################################################
# as well as a few rotated versions.

plot_hemispheres(surf_lh, surf_rh, array_name=t1wt2w_rotated[:3],
                 size=(800, 450), cmap_name='viridis',
                 nan_color=(0.5, 0.5, 0.5, 1))


###############################################################################
# Now we simply compute the correlations between the first gradient and the
# original data, as well as all rotated data.

from scipy.stats import pearsonr

feats = {'t1wt2w': t1wt2w, 'thickness': thickness}
rotated = {'t1wt2w': t1wt2w_rotated, 'thickness': thickness_rotated}

mask = ~np.isnan(thickness)
r_spin = {k: np.empty(n_permutations) for k in feats.keys()}
for fn, feat in feats.items():

    r_spin = np.empty(n_permutations)
    for i in range(n_permutations):
        # Remove non-cortex
        mask_rot = mask & ~np.isnan(rotated[fn][i])
        emb = embedding[mask_rot]
        r_spin[i] = pearsonr(rotated[fn][i][mask_rot], emb)[0]

    r_orig, pv_orig = pearsonr(feat[mask], embedding[mask])
    pv_spin = (np.count_nonzero(r_spin > r_orig) + 1) / (n_permutations + 1)

    print('{0}:\n Orig: {1:.5e}\n Spin: {2:.5e}'.format(fn.capitalize(),
                                                        pv_orig, pv_spin))
    print()


###############################################################################
# It is interesting to see that both p-values increase when taking into
# consideration the auto-correlation present in the surfaces. Also, we can see
# that the correlation with thickness is no longer statistically significant
# after spin permutations.



"""
Moran Spectral Randomization
------------------------------
"""

###############################################################################
# Moran Spectral Randomization (MSR) computes Moran's I, a metric for spatial
# auto-correlation and generates normally distributed data with similar
# auto-correlation. MSR relies on a weight matrix denoting the spatial
# proximity of features to one another. Within neuroimaging, one
# straightforward example of this is inverse geodesic distance i.e. distance
# along the cortical surface.
#
# In this example we will show how to use MSR to assess statistical
# significance between cortical markers (here curvature and cortical t1wt2w
# intensity) and the first functional connectivity gradient. We will start by
# loading the left temporal lobe mask, t1w/t2w intensity as well as cortical
# thickness data, and a template functional gradient


from brainspace.datasets import load_curvature, load_mask
from brainspace.mesh import mesh_elements as me

mask_tl = load_mask(region='temporal')[:n_pts_lh]

# Keep only the temporal lobe.
embedding_tl = embedding[:n_pts_lh][mask_tl]
t1wt2w_tl = t1wt2w[:n_pts_lh][mask_tl]
curv_tl = load_curvature()[:n_pts_lh][mask_tl]


###############################################################################
# We will now compute the Moran eigenvectors. This can be done either by
# providing a weight matrix of spatial proximity between each vertex, or by
# providing a cortical surface. Here we’ll use a cortical surface.

from brainspace.null_models import MoranSpectralRandomization

# compute spatial weight matrix
w = me.get_ring_distance(surf_lh, n_ring=1)
w = w[mask_tl][:, mask_tl]
w.data **= -1

n_rand = 1000

msr = MoranSpectralRandomization(n_rep=n_rand, tol=1e-6, random_state=43)
msr.fit(w)


###############################################################################
# Using the Moran eigenvectors we can now compute the randomized data.

curv_rand = msr.randomize(curv_tl)
t1wt2w_rand = msr.randomize(t1wt2w_tl)


###############################################################################
# Now that we have the randomized data, we can compute correlations between
# the gradient and the real/randomised data.

from scipy.stats import pearsonr
from scipy.spatial.distance import cdist

r_orig_curv = pearsonr(curv_tl, embedding_tl)[0]
r_rand_curv = 1 - cdist(curv_rand, embedding_tl[None], metric='correlation')

r_orig_t1wt2w = pearsonr(t1wt2w_tl, embedding_tl)[0]
r_rand_t1wt2w = 1 - cdist(t1wt2w_rand, embedding_tl[None], metric='correlation')


###############################################################################
# Finally, the p-values can be computed using the same approach used with
# spin permutations.
