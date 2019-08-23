function [h,C,C_full] = gradient_in_euclidean(G,surface,parcellation)

if size(G,2) ~= 3 && size(G,2) ~= 2
    error('Input matrix must be numeric with two or three columns.');
end

h.figure = figure('Color','White','Units','Normalized','Position',[0 0 .9 .9]);
h.axes_scatter = axes('Position',[.38 .6 .3 .3]);

if size(G,2) == 2
    cart = (G - mean(G)) ./ max(abs(G(:)-mean(G(:))));
    [th,r] = cart2pol(cart(:,2),cart(:,1));
    r = r ./ 2 + .5;
    C = [cos(.75*(th + 0*pi)), cos(.75*th - .5*pi), cos(.75*th + .5*pi)];
    C = C .* r;
    C = C ./ max(C(:));
    C(C<0) = 0;
    h.scatter = scatter(G(:,1),G(:,2),200,C,'Marker','.');
else
    C = (G - min(G)) ./ max(G - min(G));
    h.scatter = scatter3(G(:,1),G(:,2),G(:,3),200,C,'Marker','.');
    zlabel('Gradient 3');
end
set(h.axes_scatter                              , ...
    'DataAspectRatio'       , [1 1 1]           , ...
    'PlotBoxAspectRatio'    , [1 1 1]           , ...
    'FontName'              , 'DroidSans'       , ...
    'FontSize'              , 14                );
xlabel('Gradient 1'); ylabel('Gradient 2');

if nargin > 1
    % If parcellated data is provided, bring it to the full mesh.
    if exist('parcellation','var')
        D_full = parcel2full((1:size(C,1))',parcellation);
    end
    
    % Deal with nans. 
    D_full(all(isnan(D_full),2)) = .7; 
    
    % Make sure surfaces are in SurfStat format.
    S = convert_surface(surface);
    if ~iscell(S); S = {S}; end
    if numel(S) > 3; error('More than two surfaces are not accepted.'); end
    
    % Check correct dimensions of C_full and surfaces
    N1 = sum(cellfun(@(x)size(x.coord,2),S));
    N2 = size(D_full,1);
    if N1 ~= N2; error('Number of vertices on the surface and number of data points do not match.'); end
    
    % Split C_full over surfaces.
    D{1} = D_full(1:size(S{1}.coord,2));
    if numel(S) == 2
        D{2} = D_full(size(S{1}.coord,2)+1:end);
    end
    
    % Plot the surface(s).
    for ii = 1:numel(S)*2
        idx = ceil(ii/2);
        h.axes(ii) = axes('Position',[-.1+ii*.2 .2 .3 .3]);
        h.trisurf(ii) = trisurf(S{idx}.tri, ...
            S{idx}.coord(1,:), ...
            S{idx}.coord(2,:), ...
            S{idx}.coord(3,:), ...
            D{idx}, ...
            'EdgeColor', 'None');
        material dull; lighting phong;
        colormap(h.axes(ii),[.7 .7 .7; C])
    end
    set(h.axes                              , ...
        'Visible'           , 'off'         , ...
        'DataAspectRatio'   , [1 1 1]       , ...
        'PlotBoxAspectRatio', [1 1 1]       , ...
        'CLim'              , [0,size(C,1)] );
    % Add axes embelishments
    set(h.axes(1),'View',[-90 0]);
    set(h.axes(2),'View',[90 0]);
    if numel(h.axes) == 4
        set(h.axes(3),'View',[-90 0]);
        set(h.axes(4),'View',[90 0]);
    end

    % Add a camlight. 
    for ii = 1:numel(h.axes)
        axes(h.axes(ii));
        h.camlight(ii) = camlight();
    end

end


end