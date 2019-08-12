Functional gradients across spatial scales
=================================================
To assess consistency of gradient mapping, in this example we will compute the
gradients across different spatial scales. Specifically we subdivide the conte69
surface into 100, 200, 300, and 400 parcels based on a functional clustering
(Schaefer et al., 2017) and built functional gradients from these
representations.

First, let's load the different parcellations (Schaefer et al., 2017) and
their corresponding mean connectivity matrices. We also load the conte69
surface. These files are provided with BrainSpace.


.. code-block:: default


    addpath(genpath('/path/to/BrainSpace/matlab')); 

    # Different parcellations
    list_parcels = 100:100:400;

    # Load parcellations and mean connectivity matrices
    labelings = load_parcellation('schaefer',list_parcels); 
    conn_matices = load_group_hcp('schaefer',list_parcels);
    
    # and load the conte69 hemisphere surfaces
    [surf_lh,surf_rh] = load_conte69();


Let's see the different parcellations of the surface. We have to append the
parcellations to the left and right hemispheres first.


.. code-block:: default

    h = plot_hemispheres(labelings.schaefer_100, {surf_lh,surf_rh});
    colormap(h.figure,lines(101))


.. image:: examples_figs/schaefer_400.png
   :scale: 70%
   :align: center





We have 4 mean connectivity matrices built from each parcellation.


.. code-block:: default

    h = struct();
    parcel_names = fieldnames(conn_matices);
    titles = replace(parcel_names,'_','\_');

    % The mean connectivity matrix built from the HCP data for each parcellation
    h.fig = figure;
    for ii = 1:4
        h.ax(ii) = subplot(1,4,ii);
        h.img(ii) = imagesc(conn_matices.(parcel_names{ii}));
        title(titles{ii})
        axis square
        colormap hot
    end





.. * .. image:: /auto_examples/images/sphx_glr_plot_different_parcellations_001.png
.. *     :class: sphx-glr-single-img

.. image:: examples_figs/connectivity_matrices.png
   :scale: 70%
   :align: center


Now, we use our GradientMaps class to build one gradient for each connectivity
matrix. Gradients are the appended to the surfaces.


.. code-block:: default

    % Fit a gradient for each parcellation scheme. 
    gm = GradientMaps('kernel','normalized angle', ...
                      'approach','diffusionembedding');
    gm = gm.fit(struct2cell(conn_matices));


Finally, we plot the first gradient of Schaefer_400 as follows:


.. code-block:: default
    % Note that {1} = schaefer_100, {2} = schaefer_200, etc...
    plot_hemispheres(G.gradients{4}(:,1),{surf_lh,surf_rh},labelings.schaefer_400);


.. image:: examples_figs/schaefer_400_G1.png
   :scale: 70%
   :align: center
