function h = gradient_in_euclidean(G)

if size(G,2) ~= 3
    error('Input matrix must be numeric with three columns.');
end

C = (G - min(G)) ./ max(G - min(G));
h.figure = figure('Color','White');
h.axes = axes(); 
h.scatter3 = scatter3(G(:,1),G(:,2),G(:,3),200,C,'Marker','.');

end