Gradient construction and alignment FAQ
=======================================

Does ``normalized_angle`` build the affinity matrix?
-----------------------------------------------------

Yes. ``GradientMaps(kernel='normalized_angle').fit(fc)`` treats each row of
``fc`` as a connectivity profile. By default it first keeps the strongest
10% of entries in each row (``sparsity=0.9``), then computes cosine similarity
between profiles and converts that to normalized-angle affinity:

.. math::

   A_{ij} = 1 - \arccos(\mathrm{cosine}(x_i, x_j)) / \pi.

You do not need to compute the affinity separately. For example:

.. code-block:: python

    from brainspace.gradient import GradientMaps

    gm = GradientMaps(kernel='normalized_angle', random_state=0)
    gm.fit(fc, sparsity=0.9)
    gradients = gm.gradients_

If you already have an affinity matrix, use ``kernel=None`` and
``sparsity=None`` to avoid applying a second kernel or sparsification:

.. code-block:: python

    from brainspace.gradient.kernels import compute_affinity

    affinity = compute_affinity(fc, kernel='normalized_angle', sparsity=0.9)
    gm = GradientMaps(kernel=None, random_state=0)
    gm.fit(affinity, sparsity=None)

Connectivity and affinity are distinct: an FC matrix contains pairwise
correlations; the normalized-angle affinity measures similarity between rows
of connectivity profiles. Ensure the input and all subjects use the same
parcel or vertex ordering.

Why do my first two gradients differ from canonical gradients?
---------------------------------------------------------------

Components follow the embedding's spectral order, rather than a fixed
anatomical identity. Their signs are arbitrary, and components with similar
eigenvalues can exchange order or rotate within a similar subspace. Sample,
preprocessing, parcellation, sparsity, kernel and embedding parameters can
also change the result. A visual resemblance to canonical gradient 2 does
not by itself establish an implementation error.

Inspect the eigenvalues (``gm.lambdas_``), compare several components with an
appropriate reference, and check the input preprocessing and spatial ordering.
For comparison across subjects, fit enough components to capture the relevant
subspace and align to a common reference using Procrustes:

.. code-block:: python

    template = GradientMaps(n_components=10, kernel='normalized_angle',
                            random_state=0)
    template.fit(group_fc)

    subject = GradientMaps(n_components=10, kernel='normalized_angle',
                           alignment='procrustes', random_state=0)
    subject.fit(subject_fc, reference=template.gradients_)
    aligned = subject.aligned_

This example aligns one subject to a fixed reference. With a list of matrices,
``n_iter=1`` keeps the supplied reference fixed; larger values iteratively
update the common reference. Inspect the aligned maps: alignment cannot make
substantially different manifolds scientifically equivalent. After rotation,
aligned columns are reference coordinates and need not remain individual
eigenvectors of the subject's affinity matrix.

Should the group template use controls or all subjects?
--------------------------------------------------------

There is no template rule enforced by BrainSpace. Choose the reference to
match the analysis question and document its construction. A control-derived
or independent normative template is useful when the objective is to measure
deviation from controls. A pooled template can provide a shared descriptive
coordinate system for a combined cohort, but depends on group composition.
Use the same reference and spatial ordering for every group. For prediction
or held-out evaluation, construct the template using training or independent
data to avoid information leakage.

Group-mean connectivity followed by embedding, and generalized Procrustes
alignment of individual embeddings, are different template constructions.
State which one you use, along with preprocessing, participant selection and
embedding parameters.

Is Fisher r-to-z required before fitting gradients?
----------------------------------------------------

BrainSpace does not automatically apply Fisher's transform and does not
require it. The transform is a choice in constructing correlation-based FC,
particularly when averaging correlations across subjects. A common approach
is to average off-diagonal correlations in z space, then transform the mean
back to r space before building a connectivity-profile affinity. Applying a
kernel directly to z-valued profiles defines a different analysis; it should
be deliberate and consistent across subjects and the template.

For r-to-z conversion, handle the diagonal separately (self-correlations are
1, whose transform is infinite), check off-diagonal values are valid
correlations, and handle exact +/-1 values explicitly. Do not Fisher-transform
an already constructed normalized-angle affinity matrix. Always check that
the matrix passed to BrainSpace is finite and document diagonal handling.

Further reading
---------------

* `BrainSpace methods paper <https://doi.org/10.1038/s42003-020-0794-7>`_
  describes affinity construction, embedding and alignment.
* `Margulies et al. (2016) <https://doi.org/10.1073/pnas.1608282113>`_
  describes the canonical principal functional connectivity gradient.
