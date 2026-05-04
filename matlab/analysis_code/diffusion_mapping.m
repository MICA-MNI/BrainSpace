function [embedding, scaled_eigval] = diffusion_mapping(data, n_components, alpha, diffusion_time, random_state)
% DIFFUSION_MAPPING   Diffusion mapping decomposition of input matrix.
%   embedding = DIFFUSION_MAPPING(data,n_components,alpha,diffusion_time)
%   computes the first n_components diffusion components of matrix data
%   using parameters alpha and diffusion_time. Variable data must be an
%   n-by-n symmetric matrix containing only real non-negative values,
%   n_components is a natural number, alpha is a scalar in range [0,1], and
%   diffusion_time is a positive scalar. diffusion_time may also be set to
%   0 for automatic diffusion time estimation.
%
%   [embedding, scaled_eigval] = DIFFUSION_MAPPING(data,n_components,alpha, ...
%   diffusion_time) also returns the eigenvalues scaled_eigval.
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

% Markov matrix M = D^{-1} * L is asymmetric in general, so eig(M) can
% return tiny imaginary parts and an unsorted spectrum (issue #98). Solve
% the eigenproblem on the symmetric similar matrix
%   Ms = D^{-1/2} * L * D^{-1/2}
% which has the same (real) eigenvalues; eigenvectors map back via
%   psi = D^{-1/2} * u.
d_sqrt_inv = sum(L,2) .^ -0.5;
Ms = (d_sqrt_inv * d_sqrt_inv.') .* L;
% Force exact symmetry to suppress numerical asymmetry from outer products.
Ms = (Ms + Ms.') / 2;

% Symmetric eigendecomposition: eigenvalues are real.
[eigvec_s, eigval] = eig(Ms, 'vector');
eigvec = eigvec_s .* d_sqrt_inv;

% Sort eigenvectors and values.
[eigval, idx] = sort(real(eigval),'descend');
eigvec = eigvec(:,idx);

% Scale eigenvectors by the largest eigenvector.
psi = bsxfun(@rdivide, eigvec, eigvec(:,1));

% Automatically determines the diffusion time and scales the eigenvalues.
if diffusion_time == 0
    scaled_eigval = eigval(2:end) ./ (1 - eigval(2:end));
else
    scaled_eigval = eigval(2:end) .^ diffusion_time;
end

% Calculate embedding and bring the data towards output format.
n_available = numel(scaled_eigval);
if n_components > n_available
    warning(['You requested %d components but only %d are available; ' ...
             'returning all available components.'], ...
            n_components, n_available);
    n_components = n_available;
end
embedding = bsxfun(@times, psi(:,2:(n_components+1)), ...
                   scaled_eigval(1:n_components).');
end
