% Load surfaces
left_surface = convert_surface([getenv('MICASOFT_DIR') '/BrainSpace/shared/surfaces/conte69_5k_left_hemisphere.gii']);
right_surface = convert_surface([getenv('MICASOFT_DIR') '/BrainSpace/shared/surfaces/conte69_5k_right_hemisphere.gii']);
mask = load('/data_/mica1/03_projects/reinder/micasoft/BrainSpace/shared/surfaces/conte69_5k_midline_mask.csv');

% Load Data
load('/host/fladgate/local_raid/reinder/mpc_10k.mat');
tmp = load('/host/fladgate/local_raid/reinder/temp/UR1QC_MT1QC_10k_corr.mat');
r_all = tanh(tmp.z_all);
masked_mpc = vert_mpc_group_avg(~mask,~mask);
masked_r = r_all(~mask,~mask);

% Compute gradients with two different alignment methods. 
recomputeGradients = false;
G_p = GradientMaps('kernel','na','manifold','dm','alignment','procrustes','n_components',10);
G_p = G_p.fit({masked_mpc,masked_r},'sparsity',90);
G_p_alignReverse = procrustes_alignment({G_p.gradients{2},G_p.gradients{1}});
G_m = GradientMaps('kernel','na','manifold','dm','alignment','ma','n_components',10);
G_m = G_m.fit({masked_mpc,masked_r},'sparsity',90);

% Save
save('/host/fladgate/local_raid/reinder/mpc_gradients.mat','G_p','G_m'); 

%% Plot figure
load('/host/fladgate/local_raid/reinder/mpc_gradients.mat');
mpc_gradients = -inf(10000,size(G_m.gradients{1},2));
mpc_gradients(~mask,:) = G_m.gradients{1};
% Gradients
h = data_on_surface(mpc_gradients(:,1),{left_surface,right_surface});
colormap([.7 .7 .7; parula])
for ii = 1:4
    shading(h.axes(ii),'interp')
end
set(h.axes                                  , ...
    'CLim'              , [-.30 .30]         );

% Scree
h = scree_plot(G.lambda{1});
set(h.plot                                  , ...
    'LineStyle'             , '--'          , ...
    'Color'                 , 'k'           );  
h.axes.XLim = [0 20];

% Matrix representation of MPC.
clearvars h
h.fig = figure('Color','w');
h.axes = axes();
h.img = imagesc(masked_data);
set(h.axes                                  , ...
    'View'                  , [45 90]       , ...
    'DataAspectRatio'       , [1 1 1]       , ...
    'PlotBoxAspectRatio'    , [1 1 1]       , ...
    'XTick'                 , []            , ...
    'YTick'                 , []            , ...
    'CLim'                  , [0 1.5]       );

%% Tha massive figure.
clearvars h

allGradients = [G_p.gradients{1}(:,1),G_p.gradients{2}(:,1), ...
                G_p.aligned{1}(:,1),G_p.aligned{2}(:,1), ...
                G_p_alignReverse{2}(:,1),G_p_alignReverse{1}(:,1), ... % Note that {2} is MPC here and {1} is FC!
                G_m.aligned{1}(:,1),G_m.aligned{2}(:,1)];
allGradients = zscore(allGradients);            
allGradients_maskIncl = -inf(10000,8);
allGradients_maskIncl(~mask,:) = allGradients;

hx.fig = figure('Units','Normalized','Position',[0 0 1 1],'Color','White');
for ii = 1:size(allGradients,2)
    h{ii} = data_on_surface(allGradients_maskIncl(:,ii),{left_surface,right_surface});
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

for ii = 1:2:7
    hx.text1(ii) = text(hx.axes{ii,1},-.2,.5,'MPC-G1','Units','Normalized'); 
    hx.text1(ii+1) = text(hx.axes{ii+1,1},-.2,.5,'FC-G1','Units','Normalized'); 
end

names = {'Unaligned',{'Procrustes', 'target: MPC'},{'Procrustes', 'target: FC'},'Joint'};
for ii = 1:4
    hx.text2(ii) = text(hx.axes{2*ii-1,1},-.6,.07,names{ii},'Units','Normalized');
end
set([hx.text1,hx.text2]                     , ...
    'HorizontalAlignment'   , 'Center'      , ...
    'FontName'              , 'DroidSans'   , ...
    'Rotation'              , 90            , ...
    'FontSize'              , 16            );

% Correlation plots

% Create Gaussian filter matrix:
[xG, yG] = meshgrid(-5:5);
sigma = 1;
g = exp(-xG.^2./(2.*sigma.^2)-yG.^2./(2.*sigma.^2));
g = g./sum(g(:));
nBins = 100;
for ii = 1:4
    x = allGradients(:,2*ii-1); y = allGradients(:,2*ii);
    rho(ii) = corr(x,y,'type','Spearman');
    [N{ii},x_edges{ii},y_edges{ii}] = histcounts2(x,y,nBins); 
    hx.ax_img(ii) = axes('Position',[.40 .75-(ii-1)*.21 .12 .12]);
    hx.img(ii) = imagesc(conv2(N{ii},g,'same'));
    colormap(hx.ax_img(ii),flipud(gray))

    set(hx.ax_img(ii)                           , ...
        'DataAspectRatio'   , [1 1 1]           , ...
        'PlotBoxAspectRatio', [1 1 1]           , ...
        'CLim'              , [0 5]             , ...
        'XTick'             , [0 nBins]         , ...
        'XLim'              , [0 nBins]         , ...
        'XTickLabel'        , strsplit(num2str(x_edges{ii}([1,end]))), ...
        'YTick'             , [0 nBins]         , ...
        'YLim'              , [0 nBins]         , ...
        'YTickLabel'        , strsplit(num2str(y_edges{ii}([1,end]))), ...
        'FontName'          , 'DroidSans'       , ...
        'FontSize'          , 14                , ...
        'Box'               , 'off'             );
    hx.ax_img(ii).YAxis.Direction = 'normal';
    set(hx.ax_img(ii).XLabel                    , ...
        'String'            , 'MPC-G1'          , ...
        'Units'             , 'normalized'      , ...
        'Position'          , [.5 -.04 1]      );
    set(hx.ax_img(ii).YLabel                    , ...
        'String'            , 'FC-G1'           , ...
        'Units'             , 'normalized'      , ...
        'Position'          , [-.04 .5 1]      );
    hx.imgtext(ii) = text(hx.ax_img(ii),1.1,.1,['\rho = ' num2str(rho(ii),2)], ...
        'Units','Normalized', ...
        'FontName','DroidSans', ...
        'FontSize', 14);
end

export_fig('/data_/mica1/03_projects/reinder/figures/2019_brainspace/figure_2/fig2b.png', ...
           '-png', '-m2');




