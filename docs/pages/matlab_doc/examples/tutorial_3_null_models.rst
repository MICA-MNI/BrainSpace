Tutorial 3: Null models
=================================================

TODO: Update with gradient comparison rather than thickness + t1w/t2w

In this tutorial we assess the significance of correlations between the first canonical gradient and data from other modalities (cortical thickness and T1w/T2w image intensity). A normal test of the significance of the correlation cannot be used, because the spatial auto-correlation in MRI data may bias the test statistic. In this tutorial we will show two approaches for null hypothesis testing: spin permutations and Moran spectral randomization. 

Spin Permutations
--------------------
Here, we use the spin permutations approach previously proposed in `(Alexander-Bloch et al., 2018) <https://www.sciencedirect.com/science/article/pii/S1053811918304968>`_, which preserves the auto-correlation of the permuted feature(s) by rotating the feature data on the spherical domain. Note that when comparing gradients to non-gradient markers, we recommend permuting the non-gradient markers. 
We will start by loading the conte69 surfaces for left and right hemispheres, their corresponding spheres, midline mask, and t1w/t2w intensity as well as cortical thickness data, and a template functional gradient.

.. code-block:: matlab

    addpath(genpath('/path/to/BrainSpace/matlab')); 

    % load the conte69 hemisphere surfaces and spheres
    [surf_lh, surf_rh] = load_conte69;
    [sphere_lh, sphere_rh] = load_conte69('spheres');

    % Load the data 
    [t1wt2w_lh,t1wt2w_rh] = load_metric('t1wt2w');
    [thickness_lh,thickness_rh] = load_metric('thickness');
    
    % Template functional gradient
    embedding = load_template('fc');
    
Lets first generate some null data using spintest. 

.. code-block:: matlab

    % Lets create some rotations
    rng(0); % For replicability
    n_permutations = 1000;
    y_rand = spintest({[t1wt2w_lh,thickness_lh],[t1wt2w_rh,thickness_rh]}, ...
                      {sphere_lh,sphere_rh}, ...
                      n_permutations);

    % Merge the rotated data into single vectors
    t1wt2w_rotated = squeeze([y_rand{1}(:,1,:); y_rand{2}(:,1,:)]);
    thickness_rotated = squeeze([y_rand{1}(:,2,:); y_rand{2}(:,2,:)]);

As an illustration of the rotation, lets plot the original t1w/t2w data

.. code-block:: matlab
 
    % Plot original data
    h1 = plot_hemispheres([t1wt2w_lh;t1wt2w_rh],{surf_lh,surf_rh});

.. image:: ./example_figs/t1wt2w_original.png
   :scale: 50%
   :align: center

as well as a few rotated version.

.. code-block:: matlab

    % Plot a few of the rotations
    h2 = plot_hemispheres(t1wt2w_rotated(:,1:3),{surf_lh,surf_rh});

.. image:: ./example_figs/t1wt2w_rotated.png
   :scale: 50%
   :align: center

.. warning:: Depending on the overlap of midlines (i.e. NaNs) in the original data and in the rotation, statistical comparisons between them may compare different numbers of features. This can bias your test statistics. Therefore, if a large portion of the sphere is not used, we recommend using Moran spectral randomization instead.  

Now we simply compute the correlations between the first gradient and the original data, as well as all rotated data.

.. code-block:: matlab

    % Find correlation between thickness and T1w/T2w
    r_original = corr([t1wt2w_lh;t1wt2w_rh],[thickness_lh;thickness_rh], ...
                      'rows','pairwise','type','spearman');
    r_rand = corr([t1wt2w_lh;t1wt2w_rh],thickness_rotated, ...
                  'rows','pairwise','type','spearman');

To find a p-value, we simply compute the percentile rank of the true correlation in the distribution or random correlations. Assuming a threshold of p<0.05 for statistical significance and disregarding multiple comparison corrections, we consider the correlation to be significant if it is lower or higher than the 2.5th/97.5th percentile, respectively. 

.. code-block:: matlab

   % Compute percentile rank.
   prctile_rank = mean(r_original > r_rand);
   significant = prctile_rank < 0.025 || prctile_rank >= 0.975;

If significant is true, the we've found a statistically significant correlation. Alternatively, one could also test the one-tailed hypothesis whether the percentile rank is lower or higher than the 5th/95th percentile, respectively. 

Moran Spectral Randomization 
--------------------------------



