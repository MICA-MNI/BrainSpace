function [embedding,lambda] = diffusion_mapping(data, n_components, alpha, diffusion_time, random_state)
% DIFFUSION_MAPPING   Diffusion mapping decomposition of input matrix.
%   embedding = DIFFUSION_MAPPING(data,n_components,alpha,diffusion_time)
%   computes the first n_components diffusion components of matrix data
%   using parameters alpha and diffusion_time. Variable data must be an
%   n-by-n symmetric matrix containing only real non-negative values,
%   n_components is a natural number, alpha is a scalar in range [0,1], and
%   diffusion_time is a positive scalar. diffusion_time may also be set to
%   0 for automatic diffusion time estimation.
%
%   [embedding,lambda] = DIFFUSION_MAPPING(data,n_components,alpha, ...
%   diffusion_time) also returns the eigenvalues lambda. 
%
%   For complete documentation please consult our <a
%   href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/support_functions/diffusion_mapping.html">ReadTheDocs</a>.

if exist('random_state','var')
    rng(random_state);
end

% Parameter for later use.
sz = size(data);

d = sum(data,2) .^ -alpha;
L = data .* (d*d.');

d2 = sum(L,2) .^ -1;
M = bsxfun(@times, L, d2);

% Get the eigenvectors and eigenvalues
[eigvec,eigval] = eig(M);
eigval = diag(eigval);

% Sort eigenvectors and values.
[eigval, idx] = sort(eigval,'descend');
eigvec = eigvec(:,idx);

% Remove small eigenvalues.
n = max(2, floor(sqrt(sz(1))));
eigval = eigval(1:n);
lambda = eigval(2:end); 
eigvec = eigvec(:,1:n);

% Scale eigenvectors by the largest eigenvector.
psi = bsxfun(@rdivide, eigvec, eigvec(:,1));

% Automatically determines the diffusion time and scales the eigenvalues.
if diffusion_time == 0
    % diffusion_time = exp(1 - log(1 - eigval(2:end)) ./ log(eigval(2:end)));
    scaled_eigval = eigval(2:end) ./ (1 - eigval(2:end));
else
    scaled_eigval = eigval(2:end) .^ diffusion_time;
end

% Calculate embedding and bring the data towards output format.
try
    embedding = bsxfun(@times, psi(:,2:(n_components+1)), scaled_eigval(1:n_components).');
catch ME
    if strcmp(ME.identifier,'MATLAB:badsubscript')
        warning(['An error ocurred. Most likely you requested more components' ...
                 ' than could be computed. Attempting to return all available ' ...
                 'components.']);
        embedding = bsxfun(@times, psi(:,2:end),scaled_eigval(1:end).');
    else
        rethrow(ME);
    end
end
