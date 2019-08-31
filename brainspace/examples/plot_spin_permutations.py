"""
Null models based on spin permutations
=================================================

In this example we assess the significance of correlations between the first
canonical gradient and data from other modalities (cortical thickness and
T1w/T2w image intensity). We use the spin test approach previously proposed in
(Alexander-Bloch et al., 2018), which preserve the spatial auto-correlation of the
permuted feature(s) by rotating the feature data on the spherical domain.

"""


###############################################################################
# We will start by loading the conte69 surfaces for left and right hemispheres
# and their corresponding spheres.

from brainspace.data.base import load_conte69, load_mask


# load the conte69 hemisphere surfaces and spheres
surf_lh, surf_rh = load_conte69()
sphere_lh, sphere_rh = load_conte69(as_sphere=True)

# and the mask
mask = load_mask()


###############################################################################
# We are going to work with Schaefer parcellations to reduced the computational
# burden.

from brainspace.gradient import GradientMaps
from brainspace.data.base import (load_t1t2, load_thickness, load_parcellation,
                                  load_group_hcp)

parcel_name = 'schaefer'
n_parcels = 400
parcellation = load_parcellation(parcel_name, n_parcels=n_parcels)


# load  myelin and thickness data
myelin = load_t1t2(parcellation, mask=mask)
thickness = load_thickness(parcellation, mask=mask)

# and connectivity matrix to compute first canonical gradient
cm = load_group_hcp(parcel_name, n_parcels=n_parcels)

gm = GradientMaps(n_gradients=1, approach='dm', kernel='normalized_angle',
                  random_state=16)
gradient = gm.fit(cm).gradients_[:, 0]


###############################################################################
# Let's visualize the the data. We are going to append the data to both hemispheres
# surfaces and spheres.

import numpy as np
from brainspace.plotting import plot_hemispheres
from brainspace.utils.parcellation import map_to_labels

# Append data to surfaces and spheres
map_feat = dict(zip(['myelin', 'thickness', 'gradient'],
                    [myelin, thickness, gradient]))

n_pts_lh = surf_lh.n_points
for fn, feat_parc in map_feat.items():
    feat = map_to_labels(feat_parc, parcellation, mask=mask, fill=np.nan)

    surf_lh.append_array(feat[:n_pts_lh], name=fn, at='p')
    surf_rh.append_array(feat[n_pts_lh:], name=fn, at='p')

    sphere_lh.append_array(feat[:n_pts_lh], name=fn, at='p')
    sphere_rh.append_array(feat[n_pts_lh:], name=fn, at='p')

plot_hemispheres(surf_lh, surf_rh,
                 array_name=['myelin', 'thickness', 'gradient'],
                 interactive=False, embed_nb=True, size=(800, 600),
                 cmap_name=['YlOrBr_r', 'PuOr_r', 'viridis'])


###############################################################################
# We can also display the data on the spheres.

plot_hemispheres(sphere_lh, sphere_rh,
                 array_name=['myelin', 'thickness', 'gradient'],
                 interactive=False, embed_nb=True, size=(800, 600),
                 cmap_name=['YlOrBr_r', 'PuOr_r', 'viridis'])


###############################################################################
# Because we are using a parcellation, we need to compute the centroids for each
# parcels and used them as the sphere coordinates

from brainspace.mesh import array_operations as aop

mask_lh = mask[:n_pts_lh]
mask_rh = mask[n_pts_lh:]

parcellation_lh = parcellation[:n_pts_lh]
parcellation_rh = parcellation[n_pts_lh:]


# Compute parcellation centroids and append to spheres
aop.get_parcellation_centroids(sphere_lh, parcellation_lh, mask=mask_lh,
                               non_centroid=0, append=True,
                               array_name='centroids')
aop.get_parcellation_centroids(sphere_rh, parcellation_rh, mask=mask_rh,
                               non_centroid=0, append=True,
                               array_name='centroids')

mask_centroids_lh = sphere_lh.get_array('centroids') > 0
mask_centroids_rh = sphere_rh.get_array('centroids') > 0

centroids_lh = sphere_lh.Points[mask_centroids_lh]
centroids_rh = sphere_lh.Points[mask_centroids_rh]

# We can see the centroids on the sphere surfaces
plot_hemispheres(sphere_lh, sphere_rh, array_name='centroids',
                 interactive=False, embed_nb=True, size=(800, 200),
                 cmap_name='binary')


###############################################################################
# Now, let's generate 2000 random samples using spin permutations.

from brainspace.null_models import SpinRandomization

n_spins = 2000
sp = SpinRandomization(n_rep=n_spins, random_state=0)
sp.fit(centroids_lh, points_rh=centroids_rh)
gradient_spins_lh, gradient_spins_rh = sp.randomize(gradient[:200],
                                                    x_rh=gradient[200:])


###############################################################################
# Let's check the 3 first spin permutations

# First, append randomized data to spheres
for i in range(3):
    array_name = 'gradient_spins{i}'.format(i=i)
    gs2 = map_to_labels(gradient_spins_lh[i], parcellation_lh, mask=mask_lh,
                        fill=np.nan)
    sphere_lh.append_array(gs2, name=array_name, at='p')

    gs2 = map_to_labels(gradient_spins_rh[i], parcellation_rh, mask=mask_rh,
                        fill=np.nan)
    sphere_rh.append_array(gs2, name=array_name, at='p')


# and plot original data and the 3 first randomizations
array_names = ['gradient', 'gradient_spins0', 'gradient_spins1',
               'gradient_spins2']
plot_hemispheres(sphere_lh, sphere_rh, array_name=array_names,
                 interactive=False, embed_nb=True, size=(800, 800),
                 cmap_name='viridis_r')


###############################################################################
# Finally, we assess the correlation significance between myelin/thickness and
# the first canonical gradient without considering the spatial auto-correlation
# in and after accounting for this using spin permutations.

from scipy.stats import pearsonr
from scipy.spatial.distance import cdist

feats = {'myelin': myelin, 'thickness': thickness}

for fn, feat in feats.items():
    corr, pv = pearsonr(gradient, feat)

    gradient_spins = np.hstack([gradient_spins_lh, gradient_spins_rh])
    corr_spin = 1 - cdist(gradient_spins, feat[None],
                          metric='correlation').squeeze()
    pv_spin = (np.count_nonzero(corr_spin > corr) + 1) / (corr_spin.size + 1)

    print('{0}:\n Orig: {1:.5e}\n Spin: {2:.5e}'.format(fn.capitalize(), pv,
                                                        pv_spin))
    print()


###############################################################################
# It is interesting to see that both p-values increase when taking into
# consideration the auto-correlation present in the surfaces. Also, we can see
# that the correlation with thickness is no longer statistically significant
# after spin permutations.