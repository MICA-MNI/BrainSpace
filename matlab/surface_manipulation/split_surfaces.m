function varargout = split_surfaces(S)
% SPLIT_SURFACES   splits disconnected surfaces into separate variables.
%
%   varargout = SPLIT_SURFACES(S) takes the surface in S and returns all
%   connected components in separate variables. Order of the output
%   surfaces is identical to their order in the coordinate matrix. S can be
%   any surface accepted by convert_surface. 
%
%   Example usage:
%       Image S is a structure in SurfStat format containing a left and
%       right hemisphere, in that order.
%       [surface_left,surface_right] = SPLIT_SURFACES(S);
%
%   For more information, please consult our <a
%   href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/main_functionality/split_surfaces.html">ReadTheDocs</a>.

% Convert surface to graph.
G = surface_to_graph(S,'mesh'); 
C = conncomp(G); 
disp(['Found ' num2str(max(C)) ' surfaces.']);

% Check output arguments.
if nargout < max(C)
    warning(['Number of requested output arguments does not equal the number of surfaces found. Returning only the first ' num2str(nargout) ' surfaces.']);
elseif nargout > max(C)
    error('More output surfaces requested than surfaces found.');
end

% Check if surfaces are sorted
if any(C ~= sort(C))
    % This is actually possible to resolve - its just a pain. 
    error('Vertices do not appear to be ordered by surface. Cannot split these surfaces.');
end

% Split the surfaces. 
varargout = cell(max(C),1);
for ii = 1:max(C)
    varargout{ii}.coord = S.coord(:,C == ii);
    incl_tri = S.tri(all(ismember(S.tri,find(C==ii)),2),:);
    varargout{ii}.tri = incl_tri - min(incl_tri(:)) + 1; 
end
