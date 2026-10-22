====================
BrainSpace
====================

BrainSpace is a lightweight cross-platform toolbox primarily intended 
for macroscale gradient mapping and analysis of 
neuroimaging and connectome level data. The current version 
of BrainSpace is available in Python and MATLAB, programming 
languages widely used by the neuroimaging and network neuroscience 
communities. The toolbox also contains several maps that allow for 
exploratory analysis of gradient correspondence with other 
brain-derived features, together with tools to generate spatial null models.

For installation instructions, examples and documentation of BrainSpace see
our `documentation <https://brainspace.readthedocs.io>`_.

Display-independent surface reports
-----------------------------------

BrainSpace can export browser-readable HTML surface reports without opening a
VTK render window. This is useful on remote servers, CI runners, notebooks, and
containers where interactive VTK rendering is unavailable.

`Watch the BrainSpace threemica bridge preview video
<https://raw.githubusercontent.com/MICA-MNI/BrainSpace/2636d2e30d64ea7499f939096f74a32c13b314b8/docs/_static/brainspace_threemica_bridge_preview.mp4>`_.
The preview is generated from BrainSpace's bundled Conte69/fsLR-32k reference
surfaces and maps; it does not contain participant data.

The native BrainSpace HTML writer keeps the dependency surface small:

.. code-block:: python

   from brainspace.plotting import write_surface_report

   report = write_surface_report(
       surfaces={"lh": surf_lh, "rh": surf_rh},
       arrays={"Gradient 1": gradient_1},
       filename="surface_report.html",
       title="BrainSpace surface report",
   )

The optional threemica bridge preserves Zhengchen Cai's original viewer and
converts BrainSpace surfaces/maps into the BIDS-style derivative layout that
threemica reads:

.. code-block:: python

   from brainspace.plotting import write_brainspace_threemica_report

   reports = write_brainspace_threemica_report(
       "brainspace_threemica_report",
       maps=["thickness", "fc_gradient1"],
       custom_maps={
           "aligned_gradient": {
               "values": aligned_gradients,
               "label": "Aligned gradient",
               "unit": "score",
               "cmap": "diverging",
           },
       },
   )

Happy gradient analysis! 

License
-----------

The BrainSpace source code is available under the BSD (3-Clause) license.

Support
-----------

If you have problems installing the software or questions about usage 
and documentation, or something else related to BrainSpace, 
you can post to the Issues section of our `repository <https://github.com/MICA-MNI/BrainSpace/issues>`_.

Paper
-----------

If you consider using BrainSpace, please cite our manuscript: 
Vos de Wael R, Benkarim O, Paquola C, Lariviere S, Royer J, Tavakol S, Xu T, Hong S, Langs G, Valk S, Misic B, Milham M, Margulies D, Smallwood J, Bernhardt B (2020). BrainSpace: a toolbox for the analysis of macroscale gradients in neuroimaging and connectomics datasets. Commun Biol 3, 103.

Core development team
-----------------------

* Reinder Vos de Wael, MICA Lab - Montreal Neurological Institute
* Oualid Benkarim, MICA Lab - Montreal Neurological Institute
* Boris Bernhardt, MICA Lab - Montreal Neurological Institute
