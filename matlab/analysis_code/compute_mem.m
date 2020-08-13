function MEM = compute_mem(W,varargin)
% COMPUTE_MEM   computes the Moran eigenvectors. 
%
%   MEM = COMPUTE_MEM(W,varargin) computes the Moran eigenvectors for a
%   weight matrix or a surface. If a surface is provided, then the weights
%   are computed as inverse geodesic distance. 
%
%   Name-Value pairs:
%       'n_ring' Determines the distance (in mesh units) of the vertices
%       for which geodesic distance is computed. Only used when W is a
%       surface (Default: 1). 
%       'mask': A logical n-by-1 mask containing true for vertices thta
%       that shouold not be included in the weight matrix. Only used when W
%       is a surface (Default: []).
%       'eigenvectors: if set to 'all', then when multiple eigenvectors
%       have eigenvalues equal to zero, only one is removed and the
%       remainder is reorthogonalized. If set to anything else, removes all
%       eigenvector with eigenvalue zero.
% 
%   For complete documentation, please consult our <a
%   href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/main_functionality/compute_mem.html">ReadTheDocs</a>.

% Parse input
p = inputParser;
addParameter(p, 'n_ring', 1, @isnumeric);
addParameter(p, 'distance_metric', 'mesh', @ischar); 
addParameter(p, 'mask', [], @islogical);
addParameter(p, 'spectrum', 'all', @ischar);

parse(p, varargin{:});
R = p.Results;

% Check correct spectrum input
if ~ismember(lower(p.Results.spectrum),{'all','nonzero'})
    error('Spectrum must be ''all'' or ''nonzero''.');
end

% Check correct distance metric input. 
if ~ismember(lower(p.Results.distance_metric),{'mesh','geodesic'})
    error('distance_metric must be ''mesh'' or ''geodesic''.');
end

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
    
    % Compute euclidean distances
    dist = double(sqrt(sum((W.vertices(edges(:,1),:) - W.vertices(edges(:,2),:)).^2,2)));
    Gw = graph(edges(:,1),edges(:,2),dist);
    Gw = rmnode(Gw,find(degree(Gw)==0));
    % Check if graph is connected.
    if numel(unique(conncomp(Gw)))>1
        error('The graph is disconnected (after applying the mask). A path must exist between all vertices (excluding those in the mask).')
    end    
    D_weighted = distances(Gw);
    D_weighted(isinf(D_weighted)) = 0; 
    
    % Compute nodes within n_ring - this can probably be much more efficient
    if lower(R.distance_metric) == "mesh"
        G = graph(edges(:,1),edges(:,2));
        G = rmnode(G,find(degree(G)==0));

        % Check if graph is connected.
        if numel(unique(conncomp(G)))>1
            error('The graph is disconnected (after applying the mask). A path must exist between all vertices (excluding those in the mask).')
        end

        D_all = distances(G);
        D = (D_all > 0) & (D_all <= R.n_ring);
    elseif lower(R.distance_metric) == "geodesic"
        D = D_weighted <= R.n_ring; 
    end
    
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
[MEM,lambda] = eig(full(W),'vector');

% Remove zero eigenvector
idx = find(abs(lambda) < 1e-10); 
if strcmp(R.spectrum,'all')
    if numel(idx) == 1
        MEM(:,idx) = [];
        lambda(idx) = []; 
    elseif numel(idx) > 1
        % See supplemental info 3 of Ref 1, function scores.listw().
        w = [ones(size(MEM,1),1),MEM(:,idx)];
        Q = qr(w);
        MEM(:,idx) = Q(:,1:end-1);
        MEM(:,idx(1)) = [];
        lambda(idx(1)) = []; 
    else
        error('Did not find a zero eigenvector');
    end
else
    MEM(:,idx) = [];
    lambda(:,idx) = []; 
end

% Sort eigenvectors and values.
[~, idx] = sort(lambda,'descend');
MEM = MEM(:,idx);

