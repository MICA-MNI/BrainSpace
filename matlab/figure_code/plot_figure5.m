%% Load the data

% Set methods
kernel = 'NA';
manifold = 'DM';

% Figure path
addpath('/data_/mica1/03_projects/reinder/micasoft/BrainSpace/matlab');
addpath('/data_/mica1/03_projects/reinder/micasoft/BrainSpace/matlab/figure_code/export_fig');
figure_path = '/data/mica1/03_projects/reinder/figures/2019_brainspace';
mkdir(figure_path)
mkdir([figure_path '/figure_5']);

% Load surfaces
left_surface = convert_surface([getenv('MICASOFT_DIR') '/BrainSpace/shared/surfaces/conte69_32k_left_hemisphere.gii']);
right_surface = convert_surface([getenv('MICASOFT_DIR') '/BrainSpace/shared/surfaces/conte69_32k_right_hemisphere.gii']);

% Load parcellation
p = 'schaefer_400';
parcellation = load(['/data/mica1/03_projects/reinder/micasoft/BrainSpace/shared/parcellations/' p '_conte69.csv']);

% Generate template gradient.
T = load('/host/fladgate/local_raid/HCP_data/groupResults/subjectLists/subjectListUR2QC.mat'); 
template_subjects = T.subjectNames;

% Load template data
path = "/host/fladgate/local_raid/HCP_data/functionalData/vosdewael_schaefer_100_200_300_400/";
dir_c = dir(path);
all_files = {dir_c(3:end).name};
template_files = all_files(contains(all_files,template_subjects));
template_data = cellfun(@load,path + template_files);

% Load subject data.
S{1} = load('/host/fladgate/local_raid/HCP_data/groupResults/subjectLists/subjectListMT1QC.mat');
S{2} = load('/host/fladgate/local_raid/HCP_data/groupResults/subjectLists/subjectListUR1QC.mat');
target_subjects = [S{1}.subjectNames;S{2}.subjectNames];

addpath([getenv('MICASOFT_DIR') '/BrainSpace/matlab']);
target_files = all_files(contains(all_files,target_subjects));
data = cellfun(@load,path + target_files);

% Set parcellations
parcellations = ["vosdewael_";"schaefer_"] + (100:100:400);

for p = parcellations(:)'
    % Generate template gradients.
    for ii = 1:numel(template_data)
        subject_r = struct2cell(template_data(ii).r.(p));
        z_template{ii} = mean(atanh(cat(3,subject_r{:})),3); % Subject mean.
    end
    r_group.(p) = tanh(mean(cat(3,z_template{:}),3));
    G_t.(p) = GradientMaps('kernel',kernel,'manifold',manifold,'n_components',8,'alignment','procrustes');
    G_t.(p) = G_t.(p).fit(r_group.(p));
    
    % Generate individual gradients.
    for ii = 1:numel(data)
        subject_r = struct2cell(data(ii).r.(p));
        r_all.(p){ii} = tanh(mean(atanh(cat(3,subject_r{:})),3)); % Subject mean.
    end
    G.(p) = GradientMaps('kernel',kernel,'manifold',manifold,'n_components',8,'alignment','procrustes');
    G.(p) = G.(p).fit(r_all.(p));
    
    % Correlate each subject to the template. 
    gradients_unaligned_1.(p) = cellfun(@(x)x(:,1),G.(p).gradients,'Uniform',false);
    gradients_aligned_1.(p)   = cellfun(@(x)x(:,1),G.(p).aligned,'Uniform',false);
    
    gradients_unaligned_1.(p) = cat(2,gradients_unaligned_1.(p){:});
    gradients_aligned_1.(p) = cat(2,gradients_aligned_1.(p){:});
    
    r_unaligned.(p) = corr(gradients_unaligned_1.(p),G_t.(p).gradients{1}(:,1));
    r_aligned.(p) = corr(gradients_aligned_1.(p),G_t.(p).gradients{1}(:,1));
end

%% Plot gradient correlations
px = 'schaefer_400';
h.figure = figure('Units','Normalized','Position',[0 0 .3 .3],'Color','w');
h.axes = axes();
h.box = boxplot([rho_unaligned,rho_aligned]                                 , ...
    'Jitter'                    , .3                            , ...
    'Labels'                    , {'Unaligned','Aligned'}       , ...
    'Symbol'                    , 'k.'                          , ...
    'colors'                    , [0 0 0]                       , ...
    'OutlierSize'               , 1                             , ...
    'Whisker'                   , 1000                          );
h.box = h.axes.Children;
h.box.Children(1).Visible = 'off';
h.box.Children(2).Visible = 'off';
set(h.axes                                                      , ...
    'Box'                       , 'off'                         , ...
    'YLim'                      , [-1 1]                        , ...
    'YTick'                     , [-1 0 1]                      , ...
    'XTick'                     , [1 2]                         , ...
    'XTickLabel'                , {'Unaligned','Aligned'}       , ...
    'FontName'                  , 'DroidSans'                   , ...
    'FontSize'                  , 14                            );
h.axes.YLabel.String = 'Spearma Correlation';

export_fig([figure_path '/figure_5/boxplot_kernel_' kernel '_manifold_' manifold '.png'], '-png', '-m2');
%% Plot a few subjects before/after
toPlot = [find(r_unaligned.(px) == min(r_unaligned.(px))), ...
          find(r_unaligned.(px) == median(r_unaligned.(px))), ...
          find(r_unaligned.(px) == max(r_unaligned.(px)))];
clearvars h
x = 0;
for ii = toPlot
    for type = {'gradients','aligned'}
        z = (G.(type{1}){ii} - mean(G.(type{1}){ii})) ./ std(G.(type{1}){ii});
        h = data_on_surface(z(:,1), {left_surface,right_surface}, parcellation);
        h.axes(3).Position(1:2) = h.axes(1).Position(1:2) - [0 .26];
        h.axes(4).Position(1:2) = h.axes(2).Position(1:2) - [0 .26];
        set(h.axes,'Clim',[-2.5 2.5])
        h.cb = colorbar;
        h.cb.Position = [.25 .28 .005 .08];
        h.cb.Ticks = [-2.5 2.5];
        h.cb.FontName = 'DroidSans';
        h.cb.FontSize = 14;
        colormap([.7 .7 .7 ; parula])
        if x == 0 
            export_fig([figure_path '/figure_5/subject' num2str(ii) '_minimum_' type{1} '.png'],'-png','-m2');
        elseif x == 1
            export_fig([figure_path '/figure_5/subject' num2str(ii) '_median_' type{1} '.png'],'-png','-m2');
        elseif x == 2
            export_fig([figure_path '/figure_5/subject' num2str(ii) '_maximum_' type{1} '.png'],'-png','-m2');
        end
        delete(h.figure)
    end
    x = x + 1; 
end
    
%% Plot template
z = (G_t.(type{1}){1} - mean(G_t.(type{1}){1})) ./ std(G_t.(type{1}){1});
h = data_on_surface(z(:,1), {left_surface,right_surface}, parcellation);
h.axes(3).Position(1:2) = h.axes(1).Position(1:2) - [0 .26];
h.axes(4).Position(1:2) = h.axes(2).Position(1:2) - [0 .26];
set(h.axes,'Clim',[-2.5 2.5])
h.cb = colorbar;
h.cb.Position = [.25 .28 .005 .08];
h.cb.Ticks = [-2.5 2.5];
h.cb.FontName = 'DroidSans';
h.cb.FontSize = 14;
colormap([.7 .7 .7 ; parula])
export_fig([figure_path '/figure_5/template.png'],'-png','-m2');

