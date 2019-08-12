%% Only modify things in this section.
% Note: Figures were created on a 2560 x 1440 resolution screen. 
% Relative positions of figure elements to each other may shift at other 
% resolutions. 

% Set this to the location of your BrainSpace directory.
brainspace_path = '/data/mica1/03_projects/reinder/micasoft/BrainSpace';

% Set to true if you want to save .png files of the figures.
save_figures = true;

if save_figures
    % Set this to the location where you want to store your figures. 
    figure_path = '/data/mica1/03_projects/reinder/figures/2019-BrainSpace/figure_4';
    mkdir(figure_path)
end

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

% Load brain surfaces and mask
surface_path = [brainspace_path filesep 'shared' filesep 'surfaces' filesep];
left_surface_32k = convert_surface([surface_path 'conte69_32k_left_hemisphere.gii']);
right_surface_32k = convert_surface([surface_path 'conte69_32k_right_hemisphere.gii']);
left_surface_5k = convert_surface([surface_path 'conte69_5k_left_hemisphere.gii']);
right_surface_5k = convert_surface([surface_path 'conte69_5k_right_hemisphere.gii']);
mask_5k = load([surface_path 'conte69_5k_midline_mask.csv']);

% Download the data if it doesn't exist. 
data_path = [brainspace_path filesep 'shared' filesep 'data' filesep 'online'];
data_file = [data_path filesep 'figure_data.mat'];
download_data(data_file)

% Load data
tmp = load(data_file,'figure1');
logi_m = tmp.figure1.manifolds == string(target_manifold);
logi_k = tmp.figure1.kernels == string(target_kernel);
vertex_gradients = -inf(10000,2);
vertex_gradients(~mask_5k,:) = zscore(tmp.figure1.G{logi_m,logi_k}.gradients{1}(:,1:2)); 

% Load all data and parcellations
shared_path = string(brainspace_path) + filesep + "shared";
parcellation_names = ["vosdewael_";"schaefer_"] + (400:-100:100);
parcellation_files = shared_path + filesep + "parcellations" + filesep + ...
                     parcellation_names + "_conte69.csv";
data_files = shared_path + filesep + "data" + filesep + "main_group" + ...
             filesep + parcellation_names + "_mean_connectivity_matrix.csv";
for ii = 1:numel(parcellation_names)
    parcellations.(parcellation_names{ii}) = load(parcellation_files{ii});
    data.(parcellation_names{ii}) = load(data_files{ii});
end

% Load Mesulam
mesulam_32k = load(shared_path + filesep + "parcellations" + filesep + "mesulam_conte69.csv");
vertex_indices = load(shared_path + filesep + "surfaces" + filesep + "conte69_5k_vertex_indices.csv");
mesulam_5k = mesulam_32k(vertex_indices);
%% Do computations
% Compute gradients.
for ii = 1:numel(parcellation_names)
    G{ii} = GradientMaps('kernel',target_kernel,'manifold',target_manifold, ...
                     'n_components', 2); 
    G{ii} = G{ii}.fit(data.(parcellation_names{ii}));
    
    gradients_full{ii,1} = parcel2full(G{ii}.gradients{1}(:,1), ...
                           parcellations.(parcellation_names{ii}));
    gradients_full{ii,2} = parcel2full(G{ii}.gradients{1}(:,2), ...
                           parcellations.(parcellation_names{ii}));

    gradients_z{ii} = zscore(G{ii}.gradients{1});    
end

% Attempt to homogenize signs across gradients.
signs = zeros(numel(parcellation_names),2);
signs(:,1) = cellfun(@(x)sign(corr(x,gradients_full{8,1},'rows','complete')),gradients_full(:,1));
signs(:,2) = cellfun(@(x)sign(corr(x,gradients_full{8,2},'rows','complete')),gradients_full(:,2));

% Post-hoc sign modification.
signs(:,2) = signs(:,2).*-1;

for ii = 1:numel(gradients_z)
    gradients_z{ii} = gradients_z{ii} .* signs(ii,:);
end

% Average mesulam values
for ii = 1:numel(parcellation_names)
    g1z = parcel2full(gradients_z{ii}(:,1), ...
        parcellations.(parcellation_names{ii}));
    mesulam_g(:,ii) = labelmean(g1z',mesulam_32k','ignorewarning');
end

vg = vertex_gradients(:,1);
vg(isinf(vg)) = nan; 
mesulam_gv = labelmean(vg(:,1)',mesulam_5k','ignorewarning');

%% Plot the data

% Plot all parcellated surfaces
hx.fig = figure('Color','w','Units','Normalized','Position',[0 0 1 1]);
for ii = 1:numel(parcellation_names)
    % Plot parcellated data to separate figures.
    h{ii,1} = plot_hemispheres(gradients_z{ii}(:,1).*signs(ii,1), ...
        {left_surface_32k,right_surface_32k}, parcellations.(parcellation_names{ii}));
   
    h{ii,2} = plot_hemispheres(gradients_z{ii}(:,2).*signs(ii,2), ...
        {left_surface_32k,right_surface_32k}, parcellations.(parcellation_names{ii}));

    % Bring all parcellated data to one figure.
    if mod(ii,2)==1; jmin=1;jmax=2; else; jmin=3;jmax=4; end
    ipos = ceil(ii/2);
    for jj = jmin:jmax % Axes
        for kk = 1:2 % Gradient numbers.
            hx.axes(kk,jj,ii) = copyobj(h{ii,kk}.axes(jj),hx.fig);
            hx.axes(kk,jj,ii).Position = [.1 + .06 * jj + floor((jj-1)/2) * .03, ...
                                          .92 - ipos * .18 - kk * .08, ...
                                          .1 .1];
        end
    end
    close(h{ii,1}.figure);
    close(h{ii,2}.figure);
end

% Plot vertex-wise gradients
for ii = 1:2
    h_tmp = plot_hemispheres(vertex_gradients(:,ii), ...
                            {left_surface_5k,right_surface_5k});
    hx.axesv(:,ii) = copyobj(h_tmp.axes,hx.fig);
    delete(h_tmp.figure);
    for jj = 1:4
        hx.axesv(jj,ii).Position = [.115 + .06 * jj, .95-.08*ii, .1 .1];
    end
end

% Set colormap and colorbar
all_axes = [hx.axesv(:);hx.axes(isgraphics(hx.axes))];
colormap([.7 .7 .7; parula])
set(all_axes                                                , ...
    'CLim'                      , [-2.5 2.5]                );
hx.cb = colorbar;
hx.cb.Position = [.31 .55 .005 .05];
hx.cb.Ticks = [-2.5 2.5];
hx.cb.FontName = 'DroidSans';
hx.cb.FontSize = 12;

% Add Mesulam plots
hx.axesmv = axes();
hx.axesmv.Position(1) = hx.axes(2,2,1).Position(1) + .25;
hx.axesmv.Position(2) = hx.axesv(1,2).Position(2) + .03;
hx.axesmv.Position(3:4) = [.12 .12];
hx.plotmv = plot(mesulam_gv);
ylabel('Mean Gradient Score');xlabel('Mesulam Class');
for ii = 1:4
    hx.axesm(ii) = axes();
    hx.axesm(ii).Position(1) = hx.axes(2,2,1).Position(1) + .25;
    hx.axesm(ii).Position(2) = hx.axes(2,4,ii*2).Position(2) + .03;
    hx.axesm(ii).Position(3:4) = [.12 .12];
    hold on 
    struct_gradient = (ii-1)*2+1;
    func_gradient = ii*2;
    hx.plotm(ii,1) = plot(mesulam_g(:,struct_gradient));
    ylabel('Mean Gradient Score');xlabel('Mesulam Class');
    hx.plotm(ii,2) = plot(mesulam_g(:,func_gradient));
    ylabel('Mean Gradient Score');xlabel('Mesulam Class');
end

% Set Mesulam plot properties
set([hx.axesmv;hx.axesm(:)]                             , ...
    'PlotBoxAspectRatio'    , [1 1 1]                   , ...
    'Box'                   , 'off'                     , ...
    'XLim'                  , [0 5]                     , ...
    'YLim'                  , [-1.2 1.2]                , ...
    'XTick'                 , 1:4                       , ...
    'YTick'                 , [-1.2 0 1.2]              , ...
    'FontName'              , 'DroidSans'               , ...
    'FontSize'              , 13                        );
set([hx.plotm(:);hx.plotmv]                             , ...
    'LineStyle'             , '--'                      , ...
    'Marker'                , 'o'                       );
set(hx.plotm(:,1),'Color',[.7 .7 .7]);
set(hx.plotm(:,2),'Color',[0 0 0]);
set(hx.plotmv,'Color',[.45 .45 .45])

% Add labels
hx.columnText(1) = text(hx.axes(1,1,1),1.1,1.2,'Structural Parcellation', 'Units', 'Normalized');
hx.columnText(2) = text(hx.axes(1,3,2),1.1,1.2,'Functional Parcellation', 'Units', 'Normalized');
hx.rowText(1) = text(hx.axes(1,1,1),-.5,.1,'400 ROIs','Rotation',90','Units','Normalized');
hx.rowText(2) = text(hx.axes(1,1,3),-.5,.1,'300 ROIs','Rotation',90','Units','Normalized');
hx.rowText(3) = text(hx.axes(1,1,5),-.5,.1,'200 ROIs','Rotation',90','Units','Normalized');
hx.rowText(4) = text(hx.axes(1,1,7),-.5,.1,'100 ROIs','Rotation',90','Units','Normalized');

for ii = 1:4
    hx.gText(ii,1) = text(hx.axes(1,1,ii*2-1),-.2,.5,'G1','Rotation',90,'Units','Normalized');
    hx.gText(ii,2) = text(hx.axes(2,1,ii*2-1),-.2,.5,'G2','Rotation',90,'Units','Normalized');
end
hx.gTextv(1) = text(hx.axesv(1,1),-.2,.5,'G1','Rotation',90,'Units','Normalized');
hx.gTextv(2) = text(hx.axesv(1,2),-.2,.5,'G2','Rotation',90,'Units','Normalized');
hx.vertexText = text(hx.axesv(2),1.1,1.2,'Vertexwise','Units','Normalized');

% Set text properties
set([hx.columnText(:);hx.rowText(:);hx.gText(:);hx.vertexText(:);hx.gTextv(:)], ...
    'HorizontalAlignment'   , 'Center'                  , ...
    'Visible'               , 'on'                      , ...
    'FontName'              , 'DroidSans'               , ...
    'FontSize'              , 16                        );

% Add legend. 
hx.legend = legend(hx.axesm(1));
set(hx.legend                                           , ...
    'EdgeColor'             , 'None'                    , ...
    'Position'              , [.515 .71 .04 .04]        , ...
    'String'                , {'Structural','Functional'});

% Save figure
if save_figures
    export_fig([figure_path filesep 'parcellations_kernel_' target_kernel '_manifold_' ...
        target_manifold '.png'], '-png', '-m2');
end