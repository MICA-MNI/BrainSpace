function h = plot_hemispheres(data,surface,parcellation)

% If parcellated data is provided, bring it to the full mesh.
if exist('parcellation','var')
    data = parcel2full(data,parcellation);
    data(isnan(data)) = -Inf;
end

% Make sure surfaces are in SurfStat format.
S = convert_surface(surface,'SurfStat');
if ~iscell(S); S = {S}; end
if numel(S) > 3; error('More than two surfaces are not accepted.'); end

% Check correct dimensions of data and surfaces
N1 = sum(cellfun(@(x)size(x.coord,2),S));
N2 = size(data,1);
if N1 ~= N2; error('Number of vertices on the surface and number of data points do not match.'); end
if size(data,2) > 5; warning('Plotting more than five data vectors results in off-screen axes'); end

% Split data over surfaces.
for ii = 1:size(data,2)
    D{1,ii} = data(1:size(S{1}.coord,2),ii);
    if numel(S) == 2
        D{2,ii} = data(size(S{1}.coord,2)+1:end,ii);
    end
end

% Plot the surface(s).
h.figure = figure('Color','white','Units','normalized','Position',[0 0 .9 .9]);
colormap(parula(256));
for ii = 1:size(data,2)
    clim(:,ii) = [min(data(~isinf(data(:,ii)),ii)); ...
                  max(data(~isinf(data(:,ii)),ii))];
end
for jj = 1:size(D,2)
    for ii = 1:numel(S)*2
        idx = ceil(ii/2);
        h.axes(ii,jj) = axes('Position',[-.133+ii*.133 1-.2*jj .2 .2]);
        h.trisurf(ii,jj) = trisurf(S{idx}.tri, ...
            S{idx}.coord(1,:), ...
            S{idx}.coord(2,:), ...
            S{idx}.coord(3,:), ...
            D{idx,jj}, ...
            'EdgeColor', 'None');
        material dull; lighting phong;
    end
    set(h.axes(:,jj)                    , ...
    'Visible'           , 'off'         , ...
    'DataAspectRatio'   , [1 1 1]       , ...
    'PlotBoxAspectRatio', [1 1 1]       , ...
    'CLim'              , clim(:,jj)     );
end

% Add axes embelishments
set(h.axes(1,:),'View',[-90 0]);
set(h.axes(2,:),'View',[90 0]);
if size(h.axes,1) == 4
    set(h.axes(3,:),'View',[-90 0]);
    set(h.axes(4,:),'View',[90 0]);
end

% Add a camlight. 
for ii = 1:numel(h.axes)
    axes(h.axes(ii));
    h.camlight(ii) = camlight();
end

% Add colorbars. 
for ii = 1:size(D,2)
    h.cb(ii) = colorbar(h.axes(4,ii)); 
    h.cb(ii).Position = [h.axes(4,ii).Position(1:2) + [.175 .065] ...
                         .007 .08];
    h.cb(ii).Ticks = clim(:,ii);
    h.cb(ii).FontName = 'DroidSans';
    h.cb(ii).FontSize = 14;
end