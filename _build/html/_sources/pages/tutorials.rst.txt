
===============
Tutorials
===============


.. tabs::
	.. tab:: MATLAB
		.. highlight:: matlab
		This toolbox has been tested with MATLAB versions R2017a and R2018b, it likely throws errors with MATLAB R2016a or older. An example script using sample data exists at /matlab/sample_run.m. What follows is an explanation of how to run standard gradient analyses.

		Start MATLAB and begin by adding the toolbox path.
		::
			addpath('/path/to/micasoft/BrainSpace/matlab');

		Next, we create the Gradient object and fit the model to our data. Assume that `my_control_group` is an input data matrix of which we want to compute the gradients.
		::
			G = Gradient( 'kernel','NA', ...
			              'manifold','DE', ...
			              'alignment',false, ...
			              'random_state', 0, ...
			              'n_components', 10);
			G = G.fit(my_control_group);

		This will run a gradient analysis using the normalized angle kernel and diffusion embedding as manifold learning. See `help Gradient` for a full description of all arguments and their options. The gradients are stored inside `G.gradients{1}.embeddings`, where the first column is the first gradient, the second column the second gradient etcetera.
		Imagine that we had a patient and a control group, with the patient data stored in `my_patient_group`, and wanted to compare these. We'd run this as follows:
		::
			G2 = Gradient( 'kernel','NA', ...
			               'manifold','DE', ...
			               'alignment',true, ...
			               'random_state', 0, ...
			               'n_components', 10);
			G = G.fit({my_control_group,my_patient_group});

		Note that we set the alignment option to `true` here to enable alignment across the groups. The unaligned gradients are stored in G.gradients{1}.embeddings for controls and G.gradients{2}.embeddings for patients (in the same order as they are provided) and the aligned results are stored in G.aligned{1} and G.aligned{2} for controls and patients, respectively.

		It is recommended to visually inspect gradients both before and after alignment, as alignments may fail when the unaligned gradients are vastly different. To do this, we need a surface on which the plot the gradient data. Lets assume our patient/control data matrices can be plotted on the same template surface, that our data is on both left and right hemispheres, and that left comes before right in our data matrix.
		::
			surface_left = convert_surface('/path/to/my/left/surface/template'); % Reads the surface and converts it to SurfStat format.
			surface_right = convert_surface('/path/to/my/right/surface/template');
			target_gradient = 1;
			h{1} = data_on_surface(G.aligned{1}(:,target_gradient), {surface_left,surface_right}); % Plots the gradients of the control group.
			h{2} = data_on_surface(G.aligned{2}(:,target_gradient), {surface_left,surface_right}); % Plots the gradients of the patient group.

		Note that convert_surface will also read MATLAB/Gifti surfaces. For advanced MATLAB figure manipulation, all generated handles are returned in a structure. This function can also handle parcellated data. Let `parcellation` be a vector where each unique number denotes a parcel.
		::
			h{1} = data_on_surface(G.aligned{1}(:,target_gradient), {surface_left,surface_right}, parcellation); % Plots the parcellated gradients of the control group.

		There are two other plotting functions. For simplicity we'll only show these for the control group. The first plots gradients in 3D space and colorcodes each point by their location
		::
			h = gradient_in_euclidean(G.aligned{1}(:,1:3));


		The second creates a scree plot of variance explained. This is only supported for principal component analysis and diffusion embedding.
		::
			h = scree_plot(G.gradients{1}.metadata);


		In case you want to create null data for significance testing, we provide two options: spin testing and Moran spectral randomization. To run spin-test we require the spheres of the left/right hemispheres and some feature vector (e.g. cortical thickness) for which we want to create null data. Let sphere_left and sphere_right be these spheres and feature_vector_left and feature_vector_right be the feature vectors for the left and right spheres.
		::
			null_data = G.spintest({feature_vector_left,feature_vector_right}, {sphere_left,sphere_right}, 1000);

		Note that this function also works with only one sphere and one feature vector, but not with more than 2.

		Similarly null data can be created with Moran spectral randomization as follows
		::
			surfaces = combine_surfaces(surface_left,surface_right);
			null_data = G.MSR([feature_vector_left;feature_vector_right],'surface',surfaces,'permutationnumber',1000);	
		blablabla
	.. tab:: Python3
		.. highlight:: python
		M4TL4B RUL3S



