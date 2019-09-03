.. _gradientmaps_matlab:

==============================
GradientMaps
==============================

Synopsis
=============

The core object of the MATLAB BrainSpace package (`source code
<https://github.com/MICA-MNI/BrainSpace/blob/master/matlab/%40GradientMaps/GradientMaps.m>`_).

Usage 
=============
::

    gm = GradientMaps(varargin_1);
    gm = gm.fit({data_matrix_1,...,data_matrix_n},varargin_2);

- *gm*: the GradientMaps object. 
- *data_matrix_n*: the input feature matrix
- *varargin_1*: a set of name-value pairs (see below).
- *varargin_2*: a set of name-value pairs (see below).

Description
===============

Properties
--------------

The **method** property is a structure array which itself consists of four
fields. Each of these is set at the initalization of the object (see below) and
cannot be modifed afterwards. The fields are: "kernel", "approach", "alignment",
and "n_components". 

The **gradients** property is a cell array containing the (unaligned) gradients
of each input matrix. Each cell is an n-by-m matrix where n is the number of
datapoints and m the number of components. In joint embedding the gradients of
all data sets are computed simultaneously, and thus no unaligned gradients are
stored.

The **aligned** property is a cell array of identical dimensions to the
gradients property. If an alignment was requested, then the aligned data are
stored here. 

The **lambda** property stores the variance explained (for PCA) or the
eigenvalues (for LE and DM). Note that all computed lambdas are provided even if
this is more than the number of requested components. 

Initialization
---------------

A basic GradientMaps object can initialized by simply running it without
arguments i.e. ``gm = GradientMaps();``. However, several name-value pairs can
be provided to alter its behavior.  

'kernel'
   - 'none', ''
   - 'pearson', 'p'
   - 'spearman', 'sm'
   - 'gaussian', 'g'
   - 'cosine similarity', 'cs', 'cossim', 'cosine_similarity'
   - 'normalized angle', 'na', 'normangle', 'normalized_angle' (default)
   - a function handle (the function will be applied to the data matrix)
'approach'
   - 'principal component analysis', 'pca'
   - 'laplacian eigenmap', 'le'
   - 'diffusion embedding', 'dm' (default)
   - a function handle (the function will be applied to the post-kernel data matrix)
'alignment'
   - 'none', '' (default)
   - 'procrustes analysis', 'pa', 'procrustes'  
   - 'joint alignment', 'ja', 'joint'
   - a function handle (the function will be applied to the post-manifold data matrix)
'random_state' 
   - Any input accepted by MATLAB's ``rng`` function, or nan to use the current random state. (default: nan)
'n_components'
   - Any natural number in double format. (default: 10)

Putting it all together, an example initialization could be: ``gm =
GradientMaps('kernel','g','approach','pca','alignment','','random_state',10);``

Public Methods
---------------

Public methods are accesible to the end-user. Disregarding the constructor (see
initialization section), the GradientMaps class contains one public method. 

fit
   Uses the settings set in the methods to compute the gradients of all provided data matrices. ``varargin`` can be used to provide name-value pairs to modify the behavior of the fitting process. The following name-value pairs are allowed:
      - sparsity (default: 90)
       Sets the sparsity at which the data matrix is thresholded. 
      - tolerance (default: 1e-6)
       Floating point errors may cause the kernel to output asymmetric matrices. This number denotes the amount of asymmetry that is allowed before an error is thrown. 
      - gamma (default: 1 / number_of_data_points)
       The gamma parameter used in the Gaussian kernel. 
      - alpha (default: 0.5)
       The alpha paramter used in diffusion embedding.
      - diffusion_time (default: 0)
       The diffusion time used in diffusion embedding. Leave at 0 for automatic estimation.
      - niterations (default: 10)
       The number of iterations in Procrustes analysis.
      - reference (default: gradients of the first data matrix)
       The target for alignment for the first iteration of Procrustes analysis.
   Example usage: ``fit({data_matrix_1,data_matrix_2,...,data_matrix_n},'sparsity',75)``

Private Methods
-----------------

Private methods are not accesible to the user, but are called by other methods
i.e. GradientMaps initialization and GradientMaps.fit. The GradientMaps class
contains three private methods. As these methods are not intended for user
interaction, we only provide a basic explanation here. 

- *set(obj,varargin)*: used for setting properties of the GradientMaps class.
- *kernels(obj,data,varargin)*: performs kernel computations.
- *approaches(obj,data,varargin)*: performs dimensionality reduction.

