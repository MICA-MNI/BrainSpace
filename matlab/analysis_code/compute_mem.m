function V = compute_mem(W,varargin)

% Parse input
p = inputParser;
addParameter(p, 'n_ring', 1, @isnumeric);
addParameter(p, 'mask', [], @islogical);
addParameter(p, 'eigenvectors', 'all', @ischar);

parse(p, varargin{:});
R = p.Results;

% If input is cell array, combine surfaces.
if iscell(W)
    if numel(W) == 1
        W = W{1};
    elseif numel(W) > 2
        error('Did not recognize weight matrix or surface format.')
    else
        W = combine_surfaces(W{1},W{2});
    end
end

if isstruct(W) || ischar(W)   
    
    % Make sure the surface is in matlab format. 
    W = convert_surface(W,'format','matlab'); 
    
    % Check if mask is correct. 
    if numel(R.mask) > 0
        if numel(R.mask) == size(W.vertices,1)
            R.mask = find(R.mask);
        else
            error('Different number of elements in the mask than the number of vertices.')
        end
    end

    % Convert triangles to edges.
    faces = sort(W.faces,2);
    edges = double(unique(sort([faces(:,[1 2]); faces(:,[1 3]); faces(:,[2 3])],2),'rows'));
    edges(any(ismember(edges,R.mask),2),:) = [];
    
    % Compute nodes within n_ring - this can probably be much more efficient
    G = graph(edges(:,1),edges(:,2));
    G = rmnode(G,find(degree(G)==0));
    D_all = distances(G);
    D = (D_all > 0) & (D_all <= R.n_ring);
    
    % Compute euclidean distances
    dist = double(sqrt(sum((W.vertices(edges(:,1),:) - W.vertices(edges(:,2),:)).^2,2)));
    Gw = graph(edges(:,1),edges(:,2),dist);
    Gw = rmnode(Gw,find(degree(Gw)==0));
    D_weighted = distances(Gw);
    D_weighted(isinf(D_weighted)) = 0; 
    
    % Compute weights.
    W = (D .* D_weighted).^-1;
    W(isinf(W)) = 0; 
end

W = full(W - mean(W) - mean(W,2) + mean(W(:))); % See shortly after Eqn 1, Ref [1]. 

% Centering may destroy matrix symmetry due to floating point issues.
if max(max(abs(W - W'))) < 1e-8
    W = triu(W,1) + triu(W)';
else
    error('Centered matrix is not symmetric.'); 
end

% Eigenvalue decomposition of W. 
[V,lambda] = eig(full(W),'vector');

% Remove zero eigenvector
idx = find(abs(lambda) < 1e-10); 
if strcmp(R.eigenvectors,'all')
    if numel(idx) == 1
        V(:,idx) = [];
        lambda(idx) = []; 
    elseif numel(idx) > 1
        % See supplemental info 3 of Ref 1, function scores.listw().
        w = [ones(size(V,1),1),V(:,idx)];
        Q = qr(w);
        V(:,idx) = Q(:,1:end-1);
        V(:,idx(1)) = [];
        lambda(idx(1)) = []; 
    else
        error('Did not find a zero eigenvector');
    end
else
    V(:,idx) = [];
    lambda(:,idx) = []; 
end

% Sort eigenvectors and values.
[~, idx] = sort(lambda,'descend');
V = V(:,idx);

