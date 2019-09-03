.. _gettingstarted:

Getting Started
==============================


BrainSpace is a compact and flexible toolbox that implements a wide variety of
approaches to build macroscale gradients from neuroimaging and connectome data.
The toolbox allows for (i) the identication of gradients (using dimensionality
reduction techniques), (ii) their alignment (across subjects or modalities),
and (iii) their visualization (in embedding or cortical space). The toolbox is
implemented in both matlab and python. The steps below will help you to get
started and to build your first gradients. Please also see the tutorials for
further examples.


The packages comes with the conte69 surface, and several cortical features and
parcellations. Let's start by loading the conte69 surfaces:

.. tabs::

   .. code-tab:: py

        >>> from brainspace.datasets import load_conte69

        >>> # Load left and right hemispheres
        >>> surf_lh, surf_rh = load_conte69()
        >>> surf_lh.n_points
        32492

        >>> surf_rh.n_points
        32492

   .. code-tab:: matlab

        addpath('/path/to/micasoft/BrainSpace/matlab');

        % Load left and right hemispheres
        [surf_lh, surf_rh] = load_conte69();


To load your own surfaces, you can use our Matlab :func:`.convert_surface`
function and :func:`.load_surface` when using Python. BrainSpace also provides
surface plotting functionality. We can plot the conte69 hemisphere surfaces as
follows:

.. tabs::

   .. code-tab:: py

        >>> from brainspace.plotting import plot_hemispheres
        >>> plot_hemispheres(surf_lh, surf_rh, interactive=False,
        ...                  embed_nb=True, size=(800, 200))

   .. code-tab:: matlab

        plot_hemispheres(ones(64984,1),{surf_lh,surf_rh}); 


.. image:: ./matlab_doc/examples/example_figs/gettingstarted1.png
   :scale: 70%
   :align: center


<<<<<<< HEAD
And also load the mean connectivity matrix built from a subset of the human
connectome project (HCP). The package comes with several example matrices,
downsampled using the Schaefer parcellations `(Schaefer et al., 2017) <https://academic.oup.com/cercor/article/28/9/3095/3978804>`_.
Let's load one of them.

.. tabs::

   .. code-tab:: py

        >>> from brainspace.datasets import load_group_hcp
        >>> m = load_group_hcp('schaefer', n_parcels=400)
        >>> m.shape
        (400, 400)

   .. code-tab:: matlab

        labeling = load_parcellation('schaefer',400);
        conn_matices = load_group_hcp('schaefer',400);
        m = conn_matices.schaefer_400; 

To compute the gradients of our connectivity matrix `m` we create the
`GradientMaps` object and fit the model to our data:


.. tabs::

   .. code-tab:: py

        >>> from brainspace.gradient import GradientMaps

        >>> # Build gradients using diffusion maps and normalized angle
        >>> gm = GradientMaps(n_gradients=2, approach='dm',
        ...                   kernel='normalized_angle', random_state=0)

        >>> # and fit to the data
        >>> gm = gm.fit(m)
        GradientMaps(alignment=None, approach='dm', kernel='normalized_angle',
                     n_gradients=2, random_state=0)

        >>> # The gradients are in
        >>> gm.gradients_.shape
        (400, 2)

   .. code-tab:: matlab

        % Build gradients using diffusion maps and normalized angle
        gm = GradientMaps('kernel','na','approach','dm','n_components',2);

        % and fit to the data
        gm = gm.fit(m);


Now we can visually inspect the gradients. Let's plot the first gradient:

.. tabs::

   .. code-tab:: py

        >>> # Plot first gradient on the cortical surface.
        >>> plot_hemispheres(surf_lh, surf_rh, array_name=gm.gradients_[:, 0],
        ...                  size=(800, 200))


   .. code-tab:: matlab

        % Plot the first gradient on the cortical surface.
        plot_hemispheres(gm.gradients{1}(:,1), {surf_lh,surf_rh});


.. image:: ./matlab_doc/examples/example_figs/gettingstarted2.png
   :scale: 70%
   :align: center
