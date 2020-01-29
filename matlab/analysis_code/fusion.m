function fused_matrix = fusion(M)
% FUSION   Fuses matrices from multiple modalities for deep embedding.
%
%   fused_matrix = fusion(M) rank orders the input
%   matrices and scales them to the maximum of the sparsest matrix.
%   M must be a cell array of matrices with the same dimensionality.
%
%   For complete documentation please consult our <a
%   href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/main_functionality/gradientmaps.html">ReadTheDocs</a>.

% Check the input for nans, infs, and negatives.
if any(cellfun(@(x) any(x(:)<0) || any(isnan(x(:))) || any(isinf(x(:))), M))
   error('Negative numbers, nans, and infs are not allowed in the input matrices.');
end

% Check matrix sizes
sz = cellfun(@size,M,'Uniform',false);
if any(cellfun(@numel,sz) > 2)
   error('Input matrices may not have more than two dimensions.')
end
if ~all(cellfun(@(x)all(x==sz{1}),sz))
   error('Input matrices must have equal dimensions.')
end

% Reshape data to be a vector per input matrix
vectorized = cellfun(@(x)x(:),M,'Uniform',false);
data_matrix = cat(2,vectorized{:});
data_matrix(data_matrix==0) = nan;

% Get rank order and scale to 1 through the maximum of the smallest rank
ranking = tiedrank(data_matrix);
max_val = min(max(ranking));
rank_scaled = rescale(ranking,1,max_val,'InputMin',min(ranking),'InputMax',max(ranking));

% Reshape to output format and remove nans.
fused_matrix = reshape(rank_scaled, size(M{1},1), []);
fused_matrix(isnan(fused_matrix)) = 0;
end
