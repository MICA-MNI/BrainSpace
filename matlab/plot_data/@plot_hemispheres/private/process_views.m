function [view_angles,surf_id] = process_views(obj)
% Converts the input views character arrays to actual view angles and
% defines which surfaces to include. Valid views contains the view angle(s)
% in the top row, and the surface index (1,2, 3 for both surfaces) in the
% bottom row.
%
% View priority is as follows:
% Lateral/Medial for surface 1.
% Medial/Lateral for surface 2.
% Inferior
% Superior
% Anterior
% Posterior
%
%   For more information, please consult our <a
%   href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/visualization/plot_hemispheres.html">ReadTheDocs</a>.

% First take care of the lateral/medial views as these get
% mixed.
view_angles = {};
if contains(obj.views,'l') && contains(obj.views,'m')
    view_angles(1:4) = {[-90 0],[90 0],[-90 0], [90 0]};
    surf_id = {1,1,2,2};
elseif contains(obj.views,'l')
    view_angles(1:2) = {[-90 0],[90 0]};
    surf_id = {1,2};
elseif contains(obj.views,'m')
    view_angles(1:2) = {[90 0],[-90 0]};
    surf_id = {1,2};
end

% If only one surface is provided, remove the calls to surface
% 2.
if numel(obj.surface) == 1
    view_angles(:,cellfun(@(x)x==2,surf_id)) = [];
    surf_id(cellfun(@(x)(x==2),surf_id)) = []; 
end

% Now deal with the remaining views
valid_views = {'i', [180 -90]; ...
    's', [0 90]; ...
    'a', [180 0]; ...
    'p', [0 0]};
for ii = 1:size(valid_views,1)
    if contains(obj.views,valid_views{ii,1})
        view_angles{end+1} = valid_views{ii,2};
        surf_id{end+1} = 3; 
    end
end
end