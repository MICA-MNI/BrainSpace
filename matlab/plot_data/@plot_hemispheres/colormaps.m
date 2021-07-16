function colormaps(obj,maps)
% Sets colormaps for plot_hemispheres. Maps must either be a n-by-3 color
% map, or a cell array with the same number of elements as columns in
% obj.data, each containing n-by-3 colormaps.
%
% For more information, please consult our <a
% href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/visualization/plot_hemispheres.html">ReadTheDocs</a>.


% If a single map is provided, set the colormap for the entire figure. 
if numel(maps) == 1 || isnumeric(maps)
    if iscell(maps)
        maps = maps{1};
    end
    colormap(obj.handles.figure,maps);
    return
end

% If multiple maps are provided, set a map for each axis. 
if numel(maps) ~= size(obj.handles.axes,1)
    error('The number of maps must be one or equal to the number of axes.');
end

for ii = 1:size(obj.handles.axes,1)
    for jj = 1:size(obj.handles.axes,2)
        colormap(obj.handles.axes(ii,jj),maps{ii});
    end
end
drawnow
end