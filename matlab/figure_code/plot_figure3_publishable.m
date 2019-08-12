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
    figure_path = '/data/mica1/03_projects/reinder/figures/2019-BrainSpace/figure_3';
    mkdir(figure_path)
end

% Set the desired kernel and manifold for figure 1A and 1B. 
% Use P (Pearson), SM (Spearman), CS (Cosine Similarity), 
% NA (Normalized Angle), or G (Gaussian) for the kernel and 
% PCA (Principal Component Analysis), LE (Laplacian Eigenmap)
% or DM (diffusion map embedding) for the manifold.
target_kernel = 'CS';
target_manifold = 'DM';

% Set the desired parcellation scheme.
target_parcellation = 'schaefer_400';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Do not modify the code below this %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load Data 
% Add paths.
addpath(genpath([brainspace_path filesep 'matlab'])); 

% Load parcellation
parcellation = load([brainspace_path filesep 'shared' filesep 'parcellations', ...
                     filesep 'schaefer_400_conte69.csv']);

% Load brain surfaces and mask
surface_path = [brainspace_path filesep 'shared' filesep 'surfaces' filesep];
left_surface = convert_surface([surface_path 'conte69_32k_left_hemisphere.gii']);
right_surface = convert_surface([surface_path 'conte69_32k_right_hemisphere.gii']);

% Download the data if it doesn't exist. 
data_path = [brainspace_path filesep 'shared' filesep 'data' filesep 'online'];
data_file = [data_path filesep 'figure_data.mat'];
download_data(data_file)

% Load data
tmp = load(data_file,'figure3'); 
r_template = tmp.figure3.r_template.(target_parcellation);
r_subjects = tmp.figure3.r_singlesubjects.(target_parcellation);

%% Perform computations
% Generate template gradients 
G_template = GradientMaps('kernel',target_kernel,'manifold',target_manifold, ...
                          'n_components',10);
G_template = G_template.fit(r_template); 

G_subjects = GradientMaps('kernel',target_kernel,'manifold',target_manifold, ...
                          'n_components',10, 'alignment','PA');
G_subjects = G_subjects.fit(r_subjects,'first_alignment_target',G_template.gradients{1});

% Correlate each subject's (un)aligned gradients to the template
for ii = 1:numel(G_subjects.gradients)
    rho_unaligned(ii) = corr(G_subjects.gradients{ii}(:,1), ...
                             G_template.gradients{1}(:,1), ...
                             'type', 'spearman');
    rho_aligned(ii) = corr(G_subjects.aligned{ii}(:,1), ...
                           G_template.gradients{1}(:,1), ...
                           'type', 'spearman');
end


%% Plot figure
h.figure = figure('Units','Normalized','Position',[0 0 1 1],'Color','w');

% Box plot
h.axes = axes('Position',[.4, .38, .2, .2]);
rng(0)
hold on
jsize = .2;
jitter = (1:2) + rand(length(rho_unaligned),2)*jsize-jsize/2; 
h.plot = plot(jitter',[rho_unaligned;rho_aligned],'o--','color',[.8 .8 .8]);

h.box = boxplot([rho_unaligned(:),rho_aligned(:)]                       , ...
    'Jitter'                    , .3                            , ...
    'Labels'                    , {'Unaligned','Aligned'}       , ...
    'Symbol'                    , 'k.'                          , ...
    'colors'                    , [0 0 0]                       , ...
    'OutlierSize'               , 1                             , ...
    'Whisker'                   , 1000                          );
h.box = h.axes.Children(1);
h.box.Children(1).Visible = 'off';
h.box.Children(2).Visible = 'off';
h.axes.YLabel.String = 'Spearman Correlation';
set(h.axes                                                      , ...
    'Box'                       , 'off'                         , ...
    'YLim'                      , [-1 1]                        , ...
    'YTick'                     , [-1 0 1]                      , ...
    'XTick'                     , [1 2]                         , ...
    'XTickLabel'                , {'Unaligned','Aligned'}       , ...
    'FontName'                  , 'DroidSans'                   , ...
    'FontSize'                  , 14                            );

%%% Plot a few subjects' gradients.
idx = [find(max(rho_unaligned) == rho_unaligned), ...
       find(min(abs(rho_unaligned)) == abs(rho_unaligned)), ...
       find(min(rho_unaligned) == rho_unaligned)];

types = {'gradients','aligned'};

% Set some figure parameters
xinit = -.27;
xgap = .44;
yinit = 1.05;
ygap = .23;
sz = .13; 
xshift = [0 .08 0 .08];
yshift = [0 0 -.1 -.1]; 

% Build the hemispheres
for ii = 1:numel(idx) 
    for jj = 1:numel(types)
        z_gradient = zscore(G_subjects.(types{jj}){idx(ii)}(:,1));
        h_tmp = plot_hemispheres(z_gradient, ...
                                 {left_surface,right_surface}, ...
                                 parcellation);
        h.surf(ii,jj,:) = copyobj(h_tmp.axes,h.figure);
        delete(h_tmp.figure)
        
        for kk = 1:4
            h.surf(ii,jj,kk).Position = [xinit + xgap * jj + xshift(kk), ...
                                         yinit - ygap * ii + yshift(kk), ...
                                         sz sz];
        end
    end
end

% Plot the template
z_gradient = zscore(G_template.gradients{1}(:,1));
h_tmp = plot_hemispheres(z_gradient, ...
                         {left_surface,right_surface}, ...
                         parcellation);
h.template = copyobj(h_tmp.axes,h.figure);
delete(h_tmp.figure)
for kk = 1:4
    h.template(kk).Position = [xinit + xgap*1.5 + xshift(kk), ...
                               yinit - ygap*1.4 +  yshift(kk), ...
                               sz sz];
end

% Set the color map.
colormap([.7 .7 .7; parula]);
set([h.surf(:);h.template(:)],'CLim',[-2.8 2]);

% Add a colorbar. 
h.cb = colorbar(h.template(1),'north');
h.cb.Position = [h.template(1).Position(1) + .068, ...
                 h.template(1).Position(2) - .1, ...
                 .07, .008];
h.cb.Ticks = h.surf(1).CLim;   
h.cb.FontName = 'DroidSans';
h.cb.FontSize = 14;

% Add text labels
for ii = 1:3
    h.subjecttext(ii) = text(h.surf(ii,1,1),-.2, .08, ['Subject ' num2str(ii)], ...
                    'Units','Normalized', 'Rotation', 90, ...
                    'HorizontalAlignment', 'center');
end
  
str = {'Unaligned','Aligned'};
for ii = 1:2
    h.aligntext(ii) = text(h.surf(1,ii,1), 1.1, 1.1, str{ii}, ...
                    'Units','Normalized', 'HorizontalAlignment', 'center');
end

h.templatetext = text(h.template(1), 1.1, 1.1, 'Template', ...
                     'Units','Normalized', 'HorizontalAlignment', 'center'); 
                 
set([h.subjecttext(:);h.aligntext(:);h.templatetext]    , ...
    'FontName'              , 'DroidSans'               , ...
    'FontSize'              , 16                        ); 

if save_figures
    export_fig([figure_path filesep 'figure3.png'],'-m2','-png');
end
    