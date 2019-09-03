.. _load_mask_matlab:

=======================
load_mask
=======================

------------------
Synopsis
------------------

Loads cortical masks (`source code <https://github.com/MICA-MNI/BrainSpace/blob/master/matlab/example_data_loaders/load_mask.m>`_). 

------------------
Usage
------------------

::

    [mask_lh,mask_rh] = load_mask(name)

- *name*: Type of mask: either 'midline' for the midline or 'temporal' for the temporal lobe.
- *mask_lh*: Left hemispheric mask. 
- *mask_rh*: Right hemispheric mask.
