function G = surface_to_graph(S,distance,mask,removeDegreeZero)
% SURFACE_TO_GRAPH   Converts a surface to a graph.
%
%   G = SURFACE_TO_GRAPH(S,distance) converts surface
%   S into graph G. Distance can either be "mesh", where edge lengths are 1
%   if a connection exists and 0 otherwise, or "geodesic", where edge
%   lengths are equal to the euclidean distance between vertices. S can be
%   in any format readable by convert_surface. 
%
%   G = SURFACE_TO_GRAPH(S,distance,mask) a mask can be added denoting
%   vertices that should have their edges removed from the graph. This mask
%   may be in logical format (true=remove) or numeric containing vertex
%   indices. Set mask to [] for no mask. 
%
%   G = SURFACE_TO_GRAPH(S,distance,mask,removeDegreeZero) if
%   removeDegreeZero is true, then vertices with degree zero are removed
%   from the graph (including masked vertices). Default is false.
%
%   For more information, pleasse consult our <a
% href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/support_functions/surface_to_graph.html">ReadTheDocs</a>.

% Make sure the surface in a matlab format. 
S = convert_surface(S,'format','matlab'); 

% Check mask format
if nargin > 2
    if islogical(mask)
        mask = find(mask);
    end
else
    mask = []; 
end

if nargin < 4
    removeDegreeZero = false;
end

% Convert triangles to edges.
faces = sort(S.faces,2);
edges = double(unique(sort([faces(:,[1 2]); faces(:,[1 3]); faces(:,[2 3])],2),'rows'));
edges(any(ismember(edges,mask),2),:) = [];

% Remove empty node
if removeDegreeZero
    [~,~,IC] = unique(edges);
    edges = reshape(IC,size(edges));
end

% Build a graph with either geodesic or mesh unit distances. 
if distance == "geodesic"
    distances = double(sqrt(sum((S.vertices(edges(:,1),:) - S.vertices(edges(:,2),:)).^2,2)));
    G = graph(edges(:,1),edges(:,2),distances);
elseif distance == "mesh"
    G = graph(edges(:,1),edges(:,2));
end

