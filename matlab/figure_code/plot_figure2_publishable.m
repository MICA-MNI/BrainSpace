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
    figure_path = '/data/mica1/03_projects/reinder/figures/2019-BrainSpace/figure_2';
    mkdir(figure_path)
end

% Only gradients for CS and DM are stored in the online data. Set 
% this to true if you want to recompute gradients instead.
recompute_gradients = true;

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

%% Load Data
addpath(genpath([brainspace_path filesep 'matlab']));

% Download the data if it doesn't exist. 
data_path = [brainspace_path filesep 'shared' filesep 'data' filesep 'online'];
data_file = [data_path filesep 'figure_data.mat'];
download_data(data_file)

% Load data
tmp = load(data_file,'figure2'); % We need some data also used by figure 1. 
fc = tmp.figure2.fc; 
mpc = tmp.figure2.mpc;

% Load brain surfaces and mask
surface_path = [brainspace_path filesep 'shared' filesep 'surfaces' filesep];
left_surface = convert_surface([surface_path 'conte69_5k_left_hemisphere.gii']);
right_surface = convert_surface([surface_path 'conte69_5k_right_hemisphere.gii']);
mask = load([surface_path 'conte69_5k_midline_mask.csv']);

%% Perform the computations
masked_mpc = mpc(~mask,~mask);
masked_fc = fc(~mask,~mask);

% Compute gradients with two different alignment methods. 
if recompute_gradients
    G_p = GradientMaps('kernel',target_kernel,'manifold',target_manifold, ...
        'alignment','pa','n_components',10);
    G_p = G_p.fit({masked_mpc,masked_fc});
    G_m = GradientMaps('kernel',target_kernel,'manifold',target_manifold, ...
        'alignment','j','n_components',10);
    G_m = G_m.fit({masked_mpc,masked_fc});
else
    if target_kernel ~= "CS" || target_manifold ~= "DM"
        error('Online data only has the ''CS'' kernel and ''DM'' manifold');
    end
    G_p = tmp.figure2.G_p;
    G_m = tmp.figure2.G_m;
end

%% Build the figure
allGradients = [G_p.gradients{1}(:,1),G_p.gradients{2}(:,1), ...
                G_p.aligned{1}(:,1),G_p.aligned{2}(:,1), ...
                G_m.aligned{1}(:,1),G_m.aligned{2}(:,1)];
allGradients = zscore(allGradients);            
allGradients_maskIncl = -inf(10000,6);
allGradients_maskIncl(~mask,:) = allGradients;

hx.fig = figure('Units','Normalized','Position',[0 0 1 1],'Color','White');
for ii = 1:size(allGradients,2)
    h{ii} = plot_hemispheres(allGradients_maskIncl(:,ii),{left_surface,right_surface});
    for jj = 1:4
        hx.axes{ii,jj} = copyobj(h{ii}.axes(jj),hx.fig);
        xshift = (jj-1)*.07;
        yshift = (ii-1)*-.08 - floor((ii-1)/2)*.05;
        hx.axes{ii,jj}.Position = [.1 .8 .1 .1] + [xshift yshift 0 0];
    end
    delete(h{ii}.figure);
end
colormap([.7 .7 .7; parula])
set([hx.axes{:}],'CLim',[-2.1,2.1]);
hx.cb = colorbar(hx.axes{1,1},'south');
hx.cb.Position = [.23 .70 .05 .007];
hx.cb.FontName = 'DroidSans';
hx.cb.FontSize = 14; 
hx.cb.Ticks = sort([hx.axes{1,1}.CLim 0]);

for ii = 1:2:5
    hx.text1(ii) = text(hx.axes{ii,1},-.2,.5,'MPC-G1','Units','Normalized'); 
    hx.text1(ii+1) = text(hx.axes{ii+1,1},-.2,.5,'FC-G1','Units','Normalized'); 
end

names = {'Unaligned','Procrustes','Joint'};
for ii = 1:3
    hx.text2(ii) = text(hx.axes{2*ii-1,1},-.6,.07,names{ii},'Units','Normalized');
end
set([hx.text1,hx.text2]                     , ...
    'HorizontalAlignment'   , 'Center'      , ...
    'FontName'              , 'DroidSans'   , ...
    'Rotation'              , 90            , ...
    'FontSize'              , 16            );

%%% Correlation plots

% Create Gaussian filter
[xG, yG] = meshgrid(-5:5); % Must evaluate to square matrix with odd length
sigma = 1;
g = exp(-xG.^2./(2.*sigma.^2)-yG.^2./(2.*sigma.^2));
g = g./sum(g(:));

nBins = 100;
for ii = 1:3
    % Get Spearman correlation
    x = allGradients(:,2*ii-1); y = allGradients(:,2*ii);
    rho(ii) = corr(x,y,'type','Spearman');
    
    % Compute smoothed data and new limits.
    [N,x_edges,y_edges] = histcounts2(x,y,nBins); 
    N_smooth = conv2(N,g,'full');
    
    x_step = mean(x_edges(2:end)-x_edges(1:end-1));
    y_step = mean(y_edges(2:end)-y_edges(1:end-1));   
    
    x_centroids = (x_edges(1:end-1) + x_edges(2:end))/2;
    y_centroids = (y_edges(1:end-1) + y_edges(2:end))/2;
    
    x_new = linspace(x_centroids(1)   - x_step*((size(g,1)-1)/2), ...
                     x_centroids(end) + x_step*((size(g,1)-1)/2), ...
                     size(N_smooth,1));    
    y_new = linspace(y_centroids(1)   - y_step*((size(g,1)-1)/2), ...
                     y_centroids(end) + y_step*((size(g,1)-1)/2), ...
                     size(N_smooth,2));    
    
    hx.ax_img(ii) = axes('Position',[.40 .75-(ii-1)*.21 .12 .12]);
    hx.pcolor = pcolor(x_new,y_new,N_smooth);
    
    colormap(hx.ax_img(ii),flipud(gray))
    
    floor1d = @(x)floor(x*10)/10; 
    ceil1d = @(x)ceil(x*10)/10; 
    
    xlim = [floor1d(x_new(1)),ceil2d(x_new(end))];
    ylim = [floor1d(y_new(1)),ceil2d(y_new(end))];
    set(hx.ax_img(ii)                           , ...
        'PlotBoxAspectRatio', [1 1 1]           , ...
        'CLim'              , [0 5]             , ...
        'XTick'             , xlim              , ...
        'XLim'              , xlim              , ...
        'YTick'             , ylim              , ...
        'YLim'              , ylim              , ...
        'XTickLabel'        , strsplit(num2str(xlim,2)), ...
        'YTickLabel'        , strsplit(num2str(ylim,2)), ...
        'FontName'          , 'DroidSans'       , ...
        'FontSize'          , 14                , ...
        'Box'               , 'off'             );
    hx.ax_img(ii).YAxis.Direction = 'normal';
    set(hx.ax_img(ii).XLabel                    , ...
        'String'            , 'MPC-G1'          , ...
        'Units'             , 'normalized'      , ...
        'Position'          , [.5 -.2 1]      );
    set(hx.ax_img(ii).YLabel                    , ...
        'String'            , 'FC-G1'           , ...
        'Units'             , 'normalized'      , ...
        'Position'          , [-.2 .5 1]      );
     set(hx.pcolor                               , ...
        'EdgeColor'         , 'None'            );
    hx.imgtext(ii) = text(hx.ax_img(ii),1.1,.1,['\rho = ' num2str(rho(ii),2)], ...
        'Units','Normalized', ...
        'FontName','DroidSans', ...
        'FontSize', 14);
end

export_fig([figure_path filesep 'figure2.png'], '-png', '-m2');