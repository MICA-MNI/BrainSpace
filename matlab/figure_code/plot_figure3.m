%% Load the data

% Figure path
addpath('/data_/mica1/03_projects/reinder/micasoft/BrainSpace/matlab/figure_code/export_fig');
figure_path = '/data/mica1/03_projects/reinder/figures/2019_brainspace';
mkdir(figure_path)
mkdir([figure_path '/figure_3']);

% Load parcellated data.
addpath([getenv('MICASOFT_DIR') '/BrainSpace/matlab']);
S{1} = load('/host/fladgate/local_raid/HCP_data/groupResults/subjectLists/subjectListUR1QC.mat');
S{2} = load('/host/fladgate/local_raid/HCP_data/groupResults/subjectLists/subjectListMT1QC.mat');
subjects = [S{1}.subjectNames;S{2}.subjectNames];

addpath([getenv('MICASOFT_DIR') '/BrainSpace/matlab']);
path = "/host/fladgate/local_raid/HCP_data/functionalData/vosdewael_schaefer_100_200_300_400/";
dir_c = dir(path);
files = {dir_c(3:end).name};
files = files(contains(files,subjects));
data = cellfun(@load,path + files);

% Compute group level matrix
z_all = struct();
parcellation_names = ["vosdewael_";"schaefer_"] + (100:100:400);
for ii = 1:numel(data)
    for jj = 1:numel(parcellation_names)
        subject_r = struct2cell(data(ii).r.(parcellation_names{jj}));
        z_all.(parcellation_names{jj})(:,:,ii) = mean(atanh(cat(3,subject_r{:})),3); % Subject mean.
    end
end

z_m = structfun(@(x)mean(x,3),z_all,'uniform',false); % Group mean.
r_m = structfun(@(x)tanh(x),z_m,'uniform',false);

% Load parcellations
for ii = 1:numel(parcellation_names)
    tmp = load([getenv('MICASOFT_DIR') '/BrainSpace/shared/parcellations/' parcellation_names{ii} '_conte69.mat']);
    parcellations.(parcellation_names{ii}) = tmp.label;
end

% Load surfaces
left_surface = convert_surface([getenv('MICASOFT_DIR') '/BrainSpace/shared/surfaces/conte69_64k_left_hemisphere.gii']);
right_surface = convert_surface([getenv('MICASOFT_DIR') '/BrainSpace/shared/surfaces/conte69_64k_right_hemisphere.gii']);

% % Load Mesulam
% tmp = load([getenv('MICASOFT_DIR') '/BrainSpace/shared/parcellations/mesulam_conte69.mat']);
% mesulam = tmp.label;
%% Figure 3
% PCA-LE-DE Gradients
clearvars h G
addpath([getenv('MICASOFT_DIR') '/BrainSpace/matlab']);
kernels = {'Pearson','Spearman','CS','NA','Gaussian'};
manifolds = {'PCA','LE','DM'};
for xx = 1:numel(kernels)
    for yy = 1:numel(manifolds)
        kernel = kernels{xx};
        manifold = manifolds{yy};
        for ii = 1:numel(parcellation_names)
            G{ii} = GradientMaps( 'kernel',kernel, ...
                                'manifold',manifold, ...
                                'n_components',5);
            G{ii} = G{ii}.fit(r_m.(parcellation_names{ii}),'sparsity',90);
            
            gradients_full{ii,1} = parcel2full(G{ii}.gradients{1}(:,1),parcellations.(parcellation_names{ii}));
            gradients_full{ii,2} = parcel2full(G{ii}.gradients{1}(:,2),parcellations.(parcellation_names{ii}));
            
            gradients_z{ii} = (G{ii}.gradients{1} - mean(G{ii}.gradients{1})) ./ std(G{ii}.gradients{1});
        end
        
        % Crudely attempt to homogenize signs across gradients
        signs = zeros(numel(parcellation_names),2);
        signs(:,1) = cellfun(@(x)sign(corr(x,gradients_full{8,1},'rows','complete')),gradients_full(:,1));
        signs(:,2) = cellfun(@(x)sign(corr(x,gradients_full{8,2},'rows','complete')),gradients_full(:,2));
        
        % Plot data to separate figures.
        for ii = 1:numel(parcellation_names)
            h{ii,1} = data_on_surface(gradients_z{ii}(:,1).*signs(ii,1), {left_surface,right_surface}, parcellations.(parcellation_names{ii}));
            h{ii,2} = data_on_surface(gradients_z{ii}(:,2).*signs(ii,2), {left_surface,right_surface}, parcellations.(parcellation_names{ii}));
        end
        
        % Bring all data to one figure.
        hx.fig = figure('Color','w','Units','Normalized','Position',[0 0 1 1]);
        for ii = 1:numel(parcellation_names) % parcellations
            if mod(ii,2)==1; jmin=1;jmax=2; else; jmin=3;jmax=4; end
            ipos = ceil(ii/2);
            for jj = jmin:jmax % Axes
                for kk = 1:2 % Gradient numbers.
                    hx.axes(kk,jj,ii) = copyobj(h{ii,kk}.axes(jj),hx.fig);
                    hx.axes(kk,jj,ii).Position = [.1+.06*jj+floor((jj-1)/2)*.03 1.1-ipos*.18-kk*.08 .1 .1];
                end
            end
            close(h{ii,1}.figure);
            close(h{ii,2}.figure);
        end
        colormap([.7 .7 .7; parula])
        set(hx.axes(isgraphics(hx.axes))                        , ...
            'CLim'                      , [-2.5 2.5]                );
        hx.cb = colorbar;
        hx.cb.Position = [.315 .55 .005 .05];
        hx.cb.Ticks = [-2.5 2.5];
        hx.cb.FontName = 'DroidSans';
        hx.cb.FontSize = 12;
        % Add labels
        hx.columnText(1) = text(hx.axes(1,1,1),1.1,1.2,'Structural Parcellation', 'Units', 'Normalized');
        hx.columnText(2) = text(hx.axes(1,3,2),1.1,1.2,'Functional Parcellation', 'Units', 'Normalized');
        hx.rowText(1) = text(hx.axes(1,1,1),-.5,.1,'100 ROIs','Rotation',90','Units','Normalized');
        hx.rowText(2) = text(hx.axes(1,1,3),-.5,.1,'200 ROIs','Rotation',90','Units','Normalized');
        hx.rowText(3) = text(hx.axes(1,1,5),-.5,.1,'300 ROIs','Rotation',90','Units','Normalized');
        hx.rowText(4) = text(hx.axes(1,1,7),-.5,.1,'400 ROIs','Rotation',90','Units','Normalized');
        
        for ii = 1:4
            hx.gText(ii,1) = text(hx.axes(1,1,ii*2-1),-.2,.5,'G1','Rotation',90,'Units','Normalized');
            hx.gText(ii,2) = text(hx.axes(2,1,ii*2-1),-.2,.5,'G2','Rotation',90,'Units','Normalized');
        end
        
        set([hx.columnText(:);hx.rowText(:);hx.gText(:)]        , ...
            'HorizontalAlignment'   , 'Center'                  , ...
            'Visible'               , 'on'                      , ...
            'FontName'              , 'DroidSans'               , ...
            'FontSize'              , 16                        );
        
        export_fig(['/data_/mica1/03_projects/reinder/figures/2019_brainspace/figure_3/parcellations_kernel_' kernel '_manifold_' manifold '.png'], '-png', '-m2');
    end
end

%% Scree plots
clearvars h
h.fig = figure('Color','White','Units','Normalized');
for ii = 1:numel(G)
    h.ax(ii) = subplot(4,2,ii);
    h.sct(ii) = plot(G{ii}.lambda{1},'-o');
end