function colorlimits(obj,limits)
% Sets color limits for plot_hemispheres. Limits must be a 2-element
% numeric vector or n-by-2 matrix where n is the number of columns in
% obj.data. The first column of limits denotes the minimum color limit and
% the second the maximum. When limits is a 2-element vector, then the
% limits are applied to all axes, with limits(1) as minimum and limits(2)
% as maximum. 
%
% For more information, please consult our <a
% href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/visualization/plot_hemispheres.html">ReadTheDocs</a>.

% Check for correct input.
if size(limits,1) ~= size(obj.data,2) && numel(limits) == 1
    error('The number of color limits must be equal to the number of columns in the data matrix..');
end

% Set color limits for the axes and colorbar. 
if numel(limits)==2
    set(obj.handles.axes,'CLim',limits);
    set(obj.handles.colorbar,'Limits',limits, 'Ticks',limits);
else
    for ii = 1:size(limits,1)
        set(obj.handles.axes(ii,:),'CLim',limits(ii,:));
        set(obj.handles.colorbar(ii),'Limits',limits(ii,:), 'Ticks',limits(ii,:));
    end
end
drawnow
end