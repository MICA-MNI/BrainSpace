%% Load requisite data and perform expensive computations. 

% Include export_fig package.
addpath(genpath([getenv('MICASOFT_DIR') '/BrainSpace/matlab/']));

% Figure output path
figure_path = '/data/mica1/03_projects/reinder/figures/2019_brainspace';
mkdir(figure_path)
mkdir([figure_path '/figure_1']);

% Load surfaces
left_surface = convert_surface([getenv('MICASOFT_DIR') '/BrainSpace/shared/surfaces/conte69_10k_left_hemisphere.mat']);
right_surface = convert_surface([getenv('MICASOFT_DIR') '/BrainSpace/shared/surfaces/conte69_10k_right_hemisphere.mat']);
M = load([getenv('MICASOFT_DIR') '/BrainSpace/shared/surfaces/conte69_10k_mask.mat']);
surface = combine_surfaces(left_surface,right_surface);

% Load 10k data and convert back to r values. 
tmp = load('/host/fladgate/local_raid/reinder/temp/UR1QC_MT1QC_10k_corr.mat');
r_all = tanh(tmp.z_all);
r_mask = r_all(M.mask,M.mask);

% Compute gradients
recomputeGradients = false;
if recomputeGradients
    manifolds = {'PCA','LE','DE'};
    kernels = {'Pearson','Spearman','CS','NA','Gaussian'};
    for ll = 1:numel(kernels)
        for ii = 1:numel(manifolds)
            G{ii,ll} = GradientMaps( 'kernel',kernels{ll}, ...
                'manifold',manifolds{ii}, ...
                'n_components',10);
            G{ii,ll} = G{ii,ll}.fit(r_mask,'sparsity',90);
        end
    end
    save([figure_path '/figure_1/gradientmaps.mat'],'G','kernels','manifolds')
else
    load([figure_path '/figure_1/gradientmaps.mat']);
end

% Correlation matrices between different manifold methods.
pairs_manifold = nchoosek(1:3,2);
for ll = 1:size(pairs_manifold,1)
    for ii = 1:numel(manifolds)
        M1 = G{pairs_manifold(ii,1),ll}.gradients{1};
        M2 = G{pairs_manifold(ii,2),ll}.gradients{1};
        r_manifold(ii,ll,1,1) = abs(corr(M1(:,1),M2(:,1)));
        r_manifold(ii,ll,2,1) = abs(corr(M1(:,1),M2(:,2)));
        
        Ma = procrustes_alignment({M1,M2});
        r_manifold(ii,ll,1,2) = abs(corr(Ma{1}(:,1),Ma{2}(:,1)));
        r_manifold(ii,ll,2,2) = abs(corr(Ma{1}(:,1),Ma{2}(:,2)));
    end
end

% Correlation matrices between different kernel methods
pairs_kernel = nchoosek(1:5,2);
for ll = 1:size(pairs_kernel,1)
    for ii = 1:numel(manifolds)
        M1 = G{ii,pairs_kernel(ll,1)}.gradients{1};
        M2 = G{ii,pairs_kernel(ll,2)}.gradients{1};
        r_kernel(ii,ll,1,1) = abs(corr(M1(:,1),M2(:,1)));
        r_kernel(ii,ll,2,1) = abs(corr(M1(:,1),M2(:,2)));
        
        Ma = procrusteS_alignment({M1,M2});
        r_kernel(ii,ll,1,2) = abs(corr(Ma{1}(:,1),Ma{2}(:,1)));
        r_kernel(ii,ll,2,2) = abs(corr(Ma{1}(:,1),Ma{2}(:,2)));
    end
end
kernels_shorthand = replace(kernels,{'Pearson','Spearman','Gaussian'},{'P','SM','G'});
%% Connectivity matrix figure
ha.figure = figure('Color','white');
ha.axes = axes('DataAspectRatio',[1 1 1],'PlotBoxAspectRatio',[1 1 1]);
ha.img = imagesc(r_mask);
ha.axes.XTick = []; 
ha.axes.YTick = [];
ha.axes.CLim = [0 .10];
axis equal; axis off
export_fig([figure_path '/figure_1/connectivity_matrix.png'], '-png', '-m2');

%% Sparse matrix figure
ha.figure = figure('Color','white');
ha.axes = axes('DataAspectRatio',[1 1 1],'PlotBoxAspectRatio',[1 1 1]);
ha.img = imagesc(r_mask .* (r_mask >  prctile(r_mask,90)));
ha.axes.XTick = []; 
ha.axes.YTick = [];
ha.axes.CLim = [0 .10];
axis equal; axis off
export_fig([figure_path '/figure_1/connectivity_matrix_sparse.png'], '-png', '-m2');

%% Gradients on neocortex figure.

for ll = 1:numel(kernels)
    % Attempt to homogenize gradient direction by correlation.
    r(:,1) = cellfun(@(x)corr(x.gradients{1}(:,1), G{1,ll}.gradients{1}(:,1)), ...
        G(:,ll));
    r(:,2) = cellfun(@(x)corr(x.gradients{1}(:,2), G{1,ll}.gradients{1}(:,2)), ...
        G(:,ll));

    % Plot in individual figures with BrainSpace
    for ii = 1:numel(manifolds)
        for jj = 1:2
            % Homogenize sign, z-score, and bring to the surface.
            sign_cor = G{ii,ll}.gradients{1}(:,jj) .* sign(r(ii,jj));
            z_scored = (sign_cor - mean(sign_cor))/std(sign_cor); 
            plt = -inf(10000,1); 
            plt(M.mask) = z_scored; 
            h{ii,jj} = data_on_surface(plt,{left_surface,right_surface});
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
    
    % Add colorbars
    colormap([.7 .7 .7; parula(256)])
    set(hx.axes                                  , ...
        'CLim'                  , [-2.5 2.5]         )
    
    hx.cb = colorbar(hx.axes(1,4,1));
    hx.cb.Position = [.43 .645 .005 .05];
    hx.cb.Ticks = [-2.5 2.5];
    hx.cb.FontName = 'DroidSans';
    hx.cb.FontSize = 12;
    
    % Export figure
    export_fig([figure_path '/figure_1/kernel_' kernels{ll} '_manifolds.png'], ...
        '-png', '-m2');
end

%% Manifolds in euclidean space figures
for ii = 1:3
    for ll = 1:5
        gradients = G{ii,ll}.gradients{1}(:,1:3);
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
            'FontSize'          , 14            , ...
            'View'              , [44 20]       ); 
        xlabel('Gradient 1'); ylabel('Gradient 2'); zlabel('Gradient 3');
        set(h.scatter3                          , ...
            'SizeData'         , 30            ) ;
        export_fig([figure_path '/figure_1/euclidean_kernel_' kernels{ll} ...
            '_manifold_' manifolds{ii} '.png'], '-png', '-m2');
        close(h.figure);
    end
end


%% One heatmap figure to rule them all. 
kernelLabels = kernels_shorthand(pairs_kernel(:,1)) + "/" + kernels_shorthand(pairs_kernel(:,2));
manifoldLabels = manifolds(pairs_manifold(:,1)) + "/" + manifolds(pairs_manifold(:,2));
h.fig  = figure('Color','White','Units','Normalized','Position',[0 0 1 1]);

% Define axes in a loop so positions are easy.
for ii = 1:3
    for jj = 1:2
        h.axes(ii,jj) = axes('Position',[-0.1+jj*.18 1-ii*.19 .2 .1]);
    end
end

% Plot images
h.img(1,1) = imagesc(h.axes(1,1),r_kernel(:,:,1,1));
h.img(2,1) = imagesc(h.axes(2,1),r_kernel(:,:,2,1));
h.img(3,1) = imagesc(h.axes(3,1),r_kernel(:,:,1,2));

h.img(1,2) = imagesc(h.axes(1,2),r_manifold(:,:,1,1));
h.img(2,2) = imagesc(h.axes(2,2),r_manifold(:,:,2,1));
h.img(3,2) = imagesc(h.axes(3,2),r_manifold(:,:,1,2));

% Set axes properties
set(h.axes                                              , ...
            'CLim'                  , [0 1]             , ...   
            'XAxisLocation'         , 'top'             , ...
            'XTickLabelRotation'    , -45               , ...
            'TickLength'            , [0  0]            , ...
            'FontName'              , 'DroidSans'       , ...
            'FontSize'              , 18                , ...
            'DataAspectRatio'       , [1 1 1]           , ...
            'Box'                   , 'Off'             , ...
            'PlotBoxAspectRatio'    , [1 1 1]           );

set(h.axes(:,1)                                     , ...
        'YTick'                 , 1:3               , ...
        'XTick'                 , 1:10              , ...
        'YTickLabel'            , manifolds         , ...
        'XTickLabel'            , kernelLabels           );
    
    
set(h.axes(:,2)                                     , ...
        'XTick'                 , 1:5               , ...
        'YTick'                 , 1:3               , ...
        'XTickLabel'            , kernels_shorthand , ...
        'YTickLabel'            , manifoldLabels);
 
% Set colors.
cmap = gray(128);
cmap = cmap(1:110,:);
colormap(cmap);
h.cb = colorbar(h.axes(1));
set(h.cb                                                , ...
    'Position'                  , [.29 .74 .005 .05]   , ...
    'Ticks'                     , [0 1]                 , ...
    'FontName'                  , 'DroidSans'           , ...
    'Fontsize'                  , 18                    );

% Add title text
h.columnText(1) = text(h.axes(1,1),.5, 1.8, 'Affinity Comparisons', 'Units','Normalized');
h.columnText(2) = text(h.axes(1,2),.5, 1.8, 'Manifold Comparisons', 'Units','Normalized');
h.rowText(1) = text(h.axes(1,1), -.25, .5, 'G1/G1 correlation', 'Units','Normalized','Rotation',90);
h.rowText(2) = text(h.axes(2,1), -.25, .5, 'G1/G2 correlation', 'Units','Normalized','Rotation',90);
h.rowText(3) = text(h.axes(3,1), -.25, .5, {'Aligned', 'G1/G1 correlation'}, 'Units','Normalized','Rotation',90);
set([h.columnText,h.rowText]                        , ...
    'HorizontalAlignment'       , 'center'          , ...
    'FontName'                  , 'DroidSans'       , ...
    'FontSize'                  , 16                ); 

export_fig([figure_path '/figure_1/heatmaps.png'], '-m2', '-png')