function [embedding, lambda] = laplacian_eigenmap(data, n_components)
%LAPLACIAN_EIGENMAP Performs non-linear dimensionality reduction using Laplacian Eigenmaps
%
% [embedding, lambda] = laplacian_eigenmap(data, n_components)
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
%
% For complete documentation please consult our <a
% href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/support_functions/laplacian_eigenmap.html">ReadTheDocs</a>.


if ~exist('no_dims', 'var')
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
[embedding, lambda] = eigs(double(L), double(D), n_components + 1, tol, options);			% only need bottom (no_dims + 1) eigenvectors

% Sort eigenvectors in ascending order
lambda = diag(lambda);
[lambda, ind] = sort(lambda, 'ascend');
lambda = lambda(2:n_components + 1);

% Final embedding
embedding = embedding(:,ind(2:n_components + 1));
end