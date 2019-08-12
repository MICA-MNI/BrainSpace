function [mappedX, mapping] = laplacian_eigen(G, no_dims)
%LAPLACIAN_EIGEN Performs non-linear dimensionality reduction using Laplacian Eigenmaps
%
%   [mappedX, mapping] = laplacian_eigen(X, no_dims, k, sigma, eig_impl)
%
% Performs non-linear dimensionality reduction using Laplacian Eigenmaps.
% The data is in matrix X, in which the rows are the observations and the
% columns the dimensions. The variable dim indicates the preferred amount
% of dimensions to retain (default = 2). The variable k is the number of
% neighbours in the graph (default = 12).
% The reduced data is returned in the matrix mappedX.
%

% This file is part of the Matlab Toolbox for Dimensionality Reduction.
% The toolbox can be obtained from http://homepage.tudelft.nl/19j49
% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you
% maintain the name of the original author.
%
% (C) Laurens van der Maaten, Delft University of Technology
%
% Added a check that the graph is fully connected. An error will be thrown
% for disconnected graphs (Jul 2019, Reinder Vos de Wael).
% Changed the connected component check to MATLAB's native conncomp. 
% Enforced double input for the eigs function.

if ~exist('no_dims', 'var')
    no_dims = 2;
end

% Only embed largest connected component of the neighborhood graph
blocks = conncomp(graph(G))';
if any(blocks > 1) 
    error('Graph is not connected; consider increasing the k parameter.');
end

% Construct diagonal weight matrix
D = diag(sum(G, 2));

% Compute Laplacian
L = D - G;
L(isnan(L)) = 0; D(isnan(D)) = 0;
L(isinf(L)) = 0; D(isinf(D)) = 0;

% Construct eigenmaps (solve Ly = lambda*Dy)
disp('Constructing Eigenmaps...');
tol = 0;

options.disp = 0;
options.isreal = 1;
% Add a v0 options i.e. random initialization. 
[mappedX, lambda] = eigs(double(L), double(D), no_dims + 1, tol, options);			% only need bottom (no_dims + 1) eigenvectors

% Sort eigenvectors in ascending order
lambda = diag(lambda);
[lambda, ind] = sort(lambda, 'ascend');
lambda = lambda(2:no_dims + 1);

% Final embedding
mappedX = mappedX(:,ind(2:no_dims + 1));

% Store data for out-of-sample extension
mapping.vec = mappedX;
mapping.val = lambda;
end