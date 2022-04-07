function is_connected = graph_is_connected(G)
% GRAPH_IS_CONNECTED    Assesses whether a graph is fully connected.
%
%   is_connected = GRAPH_IS_CONNECTED(graph) tests whether the graph is
%   fully connected. Graph must be a graph, a 2D logical matrix, or a data type that
%   may be converted to a 2D logical matrix; is_connected is true if the graph is
%   connected, otherwise false.

if isa(G, 'graph')
    G = adjacency(G);
end

vertex = 1;
visited = zeros(size(G, 1), 1, 'logical');

visited = depth_first_search(G, vertex, visited);
is_connected = all(visited);
end

function visited = depth_first_search(graph, vertex, visited)
% Conducts a depth first search on the graph.
visited(vertex) = true;
adjacent_vertices = find(graph(vertex, :));
for adjacent_vertex = adjacent_vertices
    if ~visited(adjacent_vertex)
        visited = depth_first_search(graph, adjacent_vertex, visited);
    end
end
end