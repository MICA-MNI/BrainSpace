from brainspace.data.base import load_group_hcp, load_parcellation, load_conte69

# First load mean connectivity matrix and Schaefer parcellation
conn_matrix = load_group_hcp('schaefer', n_parcels=100)
labeling  = load_parcellation('schaefer', n_parcels=100)

# and load the conte69 hemisphere surfaces
surf_lh, surf_rh = load_conte69()




from brainspace.plotting import plot_hemispheres

# Let's see the parcellation
# first we are going to append the parcellation to the hemispheres
n_pts_lh = surf_lh.n_points

surf_lh.append_array(labeling[:n_pts_lh], name='Schaefer100', at='point')
surf_rh.append_array(labeling[n_pts_lh:], name='Schaefer100', at='point')

# Then plot the data on the surface
plot_hemispheres(surf_lh, surf_rh, array_name='Schaefer100', interactive=False, embed_nb=True,
                 size=(800, 200), cmap_name='nipy_spectral')



import matplotlib.pyplot as plt

# The mean connectivity matrix built from the HCP data
plt.imshow(conn_matrix, cmap='hot', interpolation='bilinear')
ax = plt.gca()
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

from brainspace.gradient import GradientMaps
from brainspace.utils.parcellation import map_to_labels, map_to_mask

# Let's see now the gradients using different embedding approaches
list_embedding = ['pca', 'le', 'dm']

for i, emb in enumerate(list_embedding):
    # We ask for 2 gradients
    gm = GradientMaps(n_gradients=2, approach=emb, kernel='normalized_angle',
                      random_state=0)

    # fit to the connectivity matrix
    gm.fit(conn_matrix)

    # append gradients to the surfaces
    for k in range(2):
        array_name = '{emb}_grad{k}'.format(emb=emb, k=k)
        grad = gm.gradients_[:, k]
        print("Appending '%s'" % array_name)

        # map the gradient to the parcels
        grad = map_to_labels(grad, labeling[labeling != 0])
        grad = map_to_mask(grad, labeling != 0)

        # append to hemispheres
        surf_lh.append_array(grad[:n_pts_lh], name=array_name, at='point')
        surf_rh.append_array(grad[n_pts_lh:], name=array_name, at='point')


from brainspace.plotting import plot_surf
import numpy as np
# To check the gradients
surfs = {'lh': surf_lh, 'rh': surf_rh}
layout = [['lh', 'lh', 'rh', 'rh']] * 6
views = ['lateral', 'medial', 'medial', 'lateral'] # will be broadcasted for each row
array_names = np.asarray(['pca_grad0', 'pca_grad1', 'le_grad0', 'le_grad1', 'dm_grad0', 'dm_grad1'])[:, None] # will be broadcasted for each col
plot_surf(surfs, layout, array_name=array_names, view=views, cmap_name='autumn_r',
          interactive=False, embed_nb=True, size=(400, 600))



