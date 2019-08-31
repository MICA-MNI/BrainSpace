.. _getting_started:

Getting Started
==============================

BrainSpace is a compact and flexible toolbox that implements a wide variety of approaches to build macroscale gradients from neuroimaging and connectome data. The toolbox allows for (i) the identication of gradients (using dimensionality reduction techniques), (ii) their alignment (across
subjects or modalities), and (iii) their visualization (in embedding or cortical
space). The toolbox is implemented in both matlab and python. The steps below will help you to get stated and to build your first gradients. Please also see the tutorials for furtuer examples. 


Let's start by loading the data:

.. tabs::

   .. code-tab:: py

        >>> from brainspace.data.base import load_conte69

        >>> # Load left and right hemisphere
        >>> surf_lh, surf_rh = load_conte69()
        >>> surf_lh.n_points
        32492

        >>> surf_rh.n_points
        32492

   .. code-tab:: matlab

         addpath('/path/to/micasoft/BrainSpace/matlab');
         [surf_lh, surf_rh] = load_conte69()

We can plot the surfaces:

.. tabs::

   .. code-tab:: py

        >>> from brainspace.plotting import plot_hemispheres
        >>> plot_hemispheres(surf_lh, surf_rh, interactive=False,
        ...                  embed_nb=True, size=(800, 200),
        ...                  color=(0, 0.5, 0.9))

   .. code-tab:: matlab

        plot_hemispheres(ones(64984,1),{surf_lh,surf_rh}); 


.. image:: ../_static/getting_started00.png
   :scale: 70%
   :align: center


And also load the mean connectivity matrix built from a subset of the human connectome project (HCP). The package comes with several example matrices, downsampled using the Schaefer parcellations `(Schaefer et al., 2017) <https://academic.oup.com/cercor/article/28/9/3095/3978804>`_. Lets load one of them. 

.. tabs::

   .. code-tab:: py

        >>> from brainspace.data.base import load_hcp_group
        >>> m = load_group_hcp('schaefer', n_parcels=400)
        >>> m.shape
        (400, 400)

   .. code-tab:: matlab

        conn_matices = load_group_hcp('schaefer',400);
        m = conn_matices.schaefer_400; 

To compute the gradients of `m`. Next, we create the `GradientMaps` object and
fit the model to our data:

.. tabs::

   .. code-tab:: py

        >>> from brainspace.gradient import GradientMaps

        >>> # create gradient mapper using diffusion maps and normalized angle
        >>> # gradients will be aligned using procrustes analysis
        >>> gm = GradientMaps(n_gradients=2, approach='dm', kernel='normalized_angle',
        ...                   align=None, random_state=0)

        >>> # and fit to the data
        >>> gm = gm.fit(m)
        GradientMaps(align=None, approach='dm', kernel='normalized_angle',
                     n_gradients=2, random_state=0)

        >>> # The gradients are in
        >>> gm.gradients_.shape
        (400, 2)

   .. code-tab:: matlab

        % Create gradient mapper using diffusion maps and normalized angle
        gm = GradientMaps('kernel','na','manifold',dm','n_components',2);

        % Fit the data with this gradient mapper.
        gm = gm.fit(m);


We can visually inspect the gradients:

.. tabs::

   .. code-tab:: py

        >>> n_pts_lh = surf_lh.n_points

        >>> # We need to append the first gradient to the left hemisphere
        >>> surf_lh.append_array(gm.gradients_[:n_pts_lh, 0], name='gradient1', at='points')

        >>> # and right hemisphere
        >>> surf_rh.append_array(gm.gradients_[n_pts_lh:, 0], name='gradient1', at='points')

        >>> # now, plotting
        >>> plot_hemispheres(surf_lh, surf_rh, array_name='gradient1',
        ...                  interactive=False, embed_nb=True, size=(800, 200))


   .. code-tab:: matlab
        % Plot the first gradient on the cortical surface. 
        plot_hemispheres(gm.gradients{1}(:,1), {surf_lh,surf_rh});


.. image:: ../_static/getting_started00.png
   :scale: 70%
   :align: center
