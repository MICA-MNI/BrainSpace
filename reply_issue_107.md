
Hi @Patricklv,

Thank you for raising this question. You are correct that for Diffusion Maps (DM) and Laplacian Eigenmaps (LE), the `lambdas_` attribute stores the eigenvalues of the affinity matrix (or the diffusion operator), which are not equivalent to "variance explained" as in PCA.

In PCA, the eigenvalues of the covariance matrix sum to the total variance, so dividing by the sum gives the percentage of variance explained. In DM, the eigenvalues relate to the decay of the diffusion process and the geometric structure of the data manifold. While they indicate the relative importance of the diffusion modes (gradients), they do not sum to a "total variance" in the same linear sense.

We have updated the documentation to clarify this distinction. Specifically, we've added notes to the `GradientMaps`, `DiffusionMaps`, and `LaplacianEigenmaps` docstrings to explicitly state that `lambdas_` for DM and LE do not represent variance explained.

Regarding your question about a workaround: using the relative magnitude of the eigenvalues (e.g., the ratio $\lambda_i / \lambda_1$ or simply the value of $\lambda_i$) is indeed a valid way to assess the relative importance or "spectral gap" of the gradients, as you suggested. However, interpreting them as a percentage of total variance is not mathematically accurate for non-linear manifold learning methods like DM.

The documentation updates will be available in the next release.
