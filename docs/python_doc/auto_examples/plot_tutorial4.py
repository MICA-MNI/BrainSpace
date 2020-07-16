"""
Tutorial 4: Subcortical surface visualization
==================================================
In this tutorial, we will display subcortical surface data.

"""


###############################################################################

# The subcortical viewer includes 16 segmented subcortical structures obtained
# from the Desikan-Killiany atlas (aparc+aseg.mgz). Subcortical regions are:
# bilateral accumbens, amygdala, caudate, hippocampus, pallidum, putamen, thalamus,
# and ventricles.


import numpy as np
from brainspace.datasets import load_subcortical
from brainspace.plotting import plot_hemispheres
from brainspace.utils.parcellation import subcorticalvertices

# Transform subcortical values (one per subcortical structure) to vertices
# Input values (i.e., subcortical_values) are ordered as follows:
#     np.array([left-accumbens, left-amygdala, left-caudate, left-hippocampus,
#               left-pallidum, left-putamen, left-thalamus, left-ventricles,
#               right-accumbens, right-amygdala, right-caudate, right-hippocampus,
#               right-pallidum, right-putamen, right-thalamus, right-ventricles])
data = subcorticalvertices(subcortical_values=np.array(range(16)))

# Load subcortical surfaces
surf_lh, surf_rh = load_subcortical()

# Plot subcortical values
plot_hemispheres(surf_lh, surf_rh, array_name=data, size=(800, 400),
                 cmap='viridis', color_range=(0,15), color_bar=True)
