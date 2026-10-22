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

% Reject zero-sum rows up front: ^-alpha or ^-0.5 of a zero row sum
% silently produces Inf/NaN and corrupts the embedding.
row_sum_data = sum(data, 2);
if any(row_sum_data <= 0)
    error('diffusion_mapping:zeroRowSum', ...
          'Affinity matrix has rows that sum to zero or below.');
end

d = row_sum_data .^ -alpha;
L = data .* (d*d.');

row_sum_L = sum(L, 2);
if any(row_sum_L <= 0)
    error('diffusion_mapping:zeroRowSum', ...
          'Normalized affinity has rows that sum to zero or below.');
end

% Solve L*v = lambda*D*v directly. Positive row sums make D positive
% definite, allowing the symmetric generalized eigensolver (PR #164).
% Retain the direct-call symmetry check from the previous implementation.
asym = norm(L - L.', 'fro');
sym_tol = 1e-8 * max(norm(L, 'fro'), 1);
assert(asym <= sym_tol, ...
       'diffusion_mapping:asymmetry', ...
       'Normalized affinity must be symmetric (||L - L''||_F = %g).', asym);
L = (L + L.') / 2;
[eigvec, eigval] = eig(L, diag(row_sum_L), 'vector');

% Sort the real eigenvalues and their corresponding eigenvectors.
[eigval, idx] = sort(real(eigval), 'descend');
eigvec = eigvec(:, idx);

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
