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
N2 = numel(data);
if N1 ~= N2; error('Number of vertices on the surface and number of data points do not match.'); end

% Split data over surfaces.
D{1} = data(1:size(S{1}.coord,2));
if numel(S) == 2
    D{2} = data(size(S{1}.coord,2)+1:end);
end

% Plot the surface(s).
h.figure = figure('Color','white','Units','normalized','Position',[0 0 1 1]);
colormap(parula(256));
for ii = 1:numel(S)*2
    idx = ceil(ii/2);
    h.axes(ii) = axes('Position',[-.2+ii*.2 .3 .3 .3]);
    h.trisurf(ii) = trisurf(S{idx}.tri, ...
        S{idx}.coord(1,:), ...
        S{idx}.coord(2,:), ...
        S{idx}.coord(3,:), ...
        D{idx}, ...
        'EdgeColor', 'None');
    material dull; lighting phong;
end

% Add some embelishments
set(h.axes                              , ...
    'Visible'           , 'off'         , ...
    'DataAspectRatio'   , [1 1 1]       , ...
    'PlotBoxAspectRatio', [1 1 1]       , ...
    'CLim'              , h.axes(1).CLim);
h.axes(1).View = [-90 0];
h.axes(2).View = [90 0];
if numel(h.axes) == 4
    h.axes(3).View = [-90 0];
    h.axes(4).View = [90 0];
end
for ii = 1:numel(h.axes)
    axes(h.axes(ii));
    h.camlight(ii) = camlight();
end