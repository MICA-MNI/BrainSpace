%% Only modify things in this section.
% Note: Figures were created on a 2560 x 1440 resolution screen. 
% Relative positions of figure elements to each other may shift at other 
% resolutions. 

% Set this to the location of your BrainSpace directory.
brainspace_path = '/data/mica1/03_projects/reinder/micasoft/BrainSpace/';

% Set to true if you want to save .png files of the figures.
save_figures = true;

if save_figures
    % Set this to the location where you want to store your figures. 
    figure_path = '/data/mica1/03_projects/reinder/figures/2019-BrainSpace/figure_1';
    mkdir(figure_path)
end

% Gradients for all kernels/manifolds are stored in the online data. Set 
% this to true if you want to recompute them instead. These computations
% can take several minutes. 
recompute_gradients = false;

% Set the desired kernel and manifold for figure 1A and 1B. 
% Use P (Pearson), SM (Spearman), CS (Cosine Similarity), 
% NA (Normalized Angle), or G (Gaussian) for the kernel and 
% PCA (Principal Component Analysis), LE (Laplacian Eigenmap)
% or DM (diffusion map embedding) for the manifold.
target_kernel = 'CS';
target_manifold = 'DM';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Do not modify the code below this %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load data
% Add paths
addpath(genpath([brainspace_path filesep 'matlab']));

% Download the data if it doesn't exist. 
data_path = [brainspace_path filesep 'shared' filesep 'data' filesep 'online'];
data_file = [data_path filesep 'figure_data.mat'];
download_data(data_file)

% Load data
tmp = load(data_file,'figure1');
figure_data = tmp.figure1; 

% Load brain surfaces and mask
surface_path = [brainspace_path filesep 'shared' filesep 'surfaces' filesep];
left_surface = convert_surface([surface_path 'conte69_5k_left_hemisphere.gii']);
right_surface = convert_surface([surface_path 'conte69_5k_right_hemisphere.gii']);
mask = load([surface_path 'conte69_5k_midline_mask.csv']);

%% Perform the computations.

% Remove the midplane.
r_mask = figure_data.connectivity_matrix(~mask,~mask);

% Compute gradients - this can take a long time so we stored all
% kernel/manifold pairs in our online data. 
manifolds = {'PCA','LE','DM'};
if recompute_gradients
    for ii = 1:numel(manifolds)
        G{ii} = GradientMaps('kernel',target_kernel, ...
            'manifold',manifolds{ii}, ...
            'n_components',10);
        G{ii} = G{ii}.fit(r_mask);
    end
else
    kernels = figure_data.kernels;
    k_logi = kernels == string(target_kernel);
    G = figure_data.G(:,k_logi);
end

%% Connectivity matrix figure
ha.figure = figure('Color','white');
ha.axes = axes('DataAspectRatio',[1 1 1],'PlotBoxAspectRatio',[1 1 1]);
ha.img = imagesc(r_mask);
ha.axes.XTick = []; 
ha.axes.YTick = [];
ha.axes.CLim = [0 .10];
axis equal; axis off
if save_figures
    export_fig([figure_path filesep 'connectivity_matrix.png'], '-png', '-m2');
end
%% Sparse matrix figure
ha.figure = figure('Color','white');
ha.axes = axes('DataAspectRatio',[1 1 1],'PlotBoxAspectRatio',[1 1 1]);
ha.img = imagesc(r_mask .* (r_mask >  prctile(r_mask,90)));
ha.axes.XTick = []; 
ha.axes.YTick = [];
ha.axes.CLim = [0 .10];
axis equal; axis off
if save_figures
    export_fig([figure_path filesep 'connectivity_matrix_sparse.png'], '-png', '-m2');
end
%% Gradients on neocortex figure.
% This section creates and deletes figures. If you manually delete the
% generated figures then you have to run the section from the start again. 

% Attempt to homogenize gradient direction by correlation.
r(:,1) = cellfun(@(x)corr(x.gradients{1}(:,1), G{1}.gradients{1}(:,1)), G);
r(:,2) = cellfun(@(x)corr(x.gradients{1}(:,2), G{1}.gradients{1}(:,2)), G);

% Plot in individual figures with BrainSpace
for ii = 1:numel(manifolds)
    for jj = 1:2
        % Homogenize sign, z-score, and bring to the surface.
        sign_cor = G{ii}.gradients{1}(:,jj) .* sign(r(ii,jj));
        z_scored = (sign_cor - mean(sign_cor))/std(sign_cor); 
        plt = -inf(10000,1); 
        plt(~mask) = z_scored; 
        h{ii,jj} = plot_hemispheres(plt,{left_surface,right_surface});
    end
end

% Copy all manifold figures to a single one. 
hx.fig = figure('Color','w','Units','Normalized','Position',[0 0 1 1]);
for ii = 1:numel(manifolds) 
    for jj = 1:4 % Axes
        for kk = 1:2 % Gradient numbers. 
            hx.axes(kk,jj,ii) = copyobj(h{ii,kk}.axes(jj),hx.fig);
            hx.axes(kk,jj,ii).Position = [.1+.06*jj 1.1-ii*.18-kk*.08 .1 .1];
        end
    end
    close(h{ii,1}.figure);
    close(h{ii,2}.figure);
end

% Add colorbar
colormap([.7 .7 .7; parula(256)])
set(hx.axes                                  , ...
    'CLim'                  , [-2.5 2.5]         )

hx.cb = colorbar(hx.axes(1,4,1));
hx.cb.Position = [.43 .645 .005 .05];
hx.cb.Ticks = [-2.5 2.5];
hx.cb.FontName = 'DroidSans';
hx.cb.FontSize = 12;

% Export figure
if save_figures
    export_fig([figure_path filesep 'kernel_' kernels{ll} '_manifolds.png'], ...
        '-png', '-m2');
end
%% Manifolds in euclidean space figures
m_logi = manifolds == string(target_manifold);
gradients = G{m_logi}.gradients{1}(:,1:3);
Z = (gradients - mean(gradients)) ./ std(gradients);
h = gradient_in_euclidean(Z);
set(h.axes                              , ...
    'XLim'              , [-3 3]        , ...
    'YLim'              , [-3 3]        , ...
    'ZLim'              , [-3 3]        , ...
    'XTick'             , [-3 0 3]      , ...
    'YTick'             , [-3 0 3]      , ...
    'ZTick'             , [-3 0 3]      , ...
    'DataAspectRatio'   , [1 1 1]       , ...
    'PlotBoxAspectRatio', [1 1 1]       , ...
    'FontName'          , 'DroidSans'   , ...
    'FontSize'          , 18            , ...
    'View'              , [44 20]       ); 
xlabel('Gradient 1'); ylabel('Gradient 2'); zlabel('Gradient 3');
set(h.scatter3                          , ...
    'SizeData'         , 30            ) ;
if save_figures
    export_fig([figure_path filesep 'euclidean_kernel_' kernels{ll} ...
        '_manifold_' manifolds{ii} '.png'], '-png', '-m2');
end