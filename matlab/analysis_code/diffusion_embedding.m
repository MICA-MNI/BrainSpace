function [embedding,eigval_out] = diffusion_embedding(data, nComponents, alpha, diffusion_time)

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
eigval_out = eigval(2:end); 
eigvec = eigvec(:,1:n);

% Scale eigenvectors by the largest eigenvector.
psi = bsxfun(@rdivide, eigvec, eigvec(:,1));

% Automatically determines the diffusion time and scales the eigenvalues.
if diffusion_time == 0
    diffusion_time = exp(1 - log(1 - eigval(2:end)) ./ log(eigval(2:end)));
    scaled_eigval = eigval(2:end) ./ (1 - eigval(2:end));
else
    scaled_eigval = eigval(2:end) .^ diffusion_time;
end

% Set the threshold for number of components.
%eigvalRatio = eigval ./eigval(1);
%threshold = max([.05,eigvalRatio(end)]);
%nComponentsAuto = min([sum(eigvalRatio > threshold)-1, sz(1)]);

% if isnan(nComponents)
%     nComponents = nComponentsAuto;
% end

% Calculate embedding and bring the data towards output format.
embedding = bsxfun(@times, psi(:,2:(nComponents+1)), scaled_eigval(1:nComponents).');
