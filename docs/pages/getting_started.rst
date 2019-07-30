.. _getting_started:

Getting Started
==============================

BrainSpace a wide variety of approaches to build gradients. Here we are going to
its main features and teh basics to start using BrainSpace.


Let's start by loading the data:

.. tabs::

   .. code-tab:: py

        >>> from os.path import join
        >>> from brainspace.mesh import mesh_io as mio

        >>> # Load left and right hemisphere
        >>> pth = '/media/oualid/hd500/oualid/BrainSpace/brainspace_data/surfaces'
        >>> surf_lh = mio.load_surface(join(pth, 'conte69_64k_left_hemisphere.gii'))
        >>> surf_rh = mio.load_surface(join(pth, 'conte69_64k_right_hemisphere.gii'))

        >>> surf_lh.n_points
        33809

        >>> surf_rh.n_points
        33809

   .. code-tab:: matlab

         addpath('/path/to/micasoft/BrainSpace/matlab');


We can plot the surfaces:

.. tabs::

   .. code-tab:: py

        >>> from brainspace.plotting import plot_hemispheres
        >>> plot_hemispheres(surf_lh, surf_rh, interactive=False,
        ...                  embed_nb=True, size=(800, 200),
        ...                  color=(0, 0.5, 0.9))

   .. code-tab:: matlab

        addpath('/path/to/micasoft/BrainSpace/matlab');


.. image:: ../_static/getting_started00.png
   :scale: 70%
   :align: center


And also load the input matrix:

.. tabs::

   .. code-tab:: py

        >>> from brainspace.datasets import load_data
        >>> m = load_data('something')
        >>> m.shape
        (n, n)

   .. code-tab:: matlab

        addpath('/path/to/micasoft/BrainSpace/matlab');


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
        (n, 2)

   .. code-tab:: matlab

        addpath('/path/to/micasoft/BrainSpace/matlab');


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

        addpath('/path/to/micasoft/BrainSpace/matlab');


.. image:: ../_static/getting_started00.png
   :scale: 70%
   :align: center
