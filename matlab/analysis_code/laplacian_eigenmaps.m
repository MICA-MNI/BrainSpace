function [embedding, lambdas] = laplacian_eigenmaps(data, n_components, random_state)
%LAPLACIAN_EIGENMAPS Performs non-linear dimensionality reduction using Laplacian Eigenmaps
%
% [embedding, lambda] = LAPLACIAN_EIGENMAPS(data, n_components)
%
% Performs non-linear dimensionality reduction using Laplacian Eigenmaps.
% The data is in matrix data, in which the rows are the observations and the
% columns the dimensions. The variable n_components indicates the preferred amount
% of dimensions to retain (default = 2).
% The reduced data is returned in the matrix embedding.
%
% This file is part of the Matlab Toolbox for Dimensionality Reduction.
% The toolbox can be obtained from http://homepage.tudelft.nl/19j49
% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you
% maintain the name of the original author.
%
% (C) Laurens van der Maaten, Delft University of Technology
%
% Changes for BrainSpace 
% - Added a check that the graph is fully connected. An error will be thrown
% for disconnected graphs.
% - Changed the connected component check to MATLAB's native conncomp. 
% - Enforced double input for the eigs function.
% - Only outputting lambdas as the second output. 
% - Changed some variable names. 
% - Computation of the Gaussian kernel removed.
% - Added random_state initialization.
%
% For complete documentation please consult our <a
% href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/support_functions/laplacian_eigenmaps.html">ReadTheDocs</a>.


if exist('random_state','var')
    rng(random_state);
end

if ~exist('n_components', 'var')
    n_components = 2;
end

% Only embed largest connected component of the neighborhood graph
blocks = conncomp(graph(data))';
if any(blocks > 1) 
    error('Graph is not connected; consider increasing the k parameter.');
end

% Construct diagonal weight matrix
D = diag(sum(data, 2));

% Compute Laplacian
L = D - data;
L(isnan(L)) = 0; D(isnan(D)) = 0;
L(isinf(L)) = 0; D(isinf(D)) = 0;

% Construct eigenmaps (solve Ly = lambda*Dy)
disp('Constructing Eigenmaps...');
tol = 0;

options.disp = 0;
options.isreal = 1;
% Add a v0 options i.e. random initialization. 
[embedding, lambdas] = eigs(double(L), double(D), n_components + 1, tol, options);			% only need bottom (no_dims + 1) eigenvectors

% Sort eigenvectors in ascending order
lambdas = diag(lambdas);
[lambdas, ind] = sort(lambdas, 'ascend');
lambdas = lambdas(2:n_components + 1);

% Final embedding
embedding = embedding(:,ind(2:n_components + 1));
end
