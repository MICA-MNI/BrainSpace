Contour overlays
================

VTK surface plots can show an isovalue contour over a scalar map, for example
the boundary of a binary significance mask. Values follow the same left-then-
right vertex ordering as the surface map:

.. code-block:: python

    from brainspace.plotting import plot_hemispheres

    plot_hemispheres(
        surf_lh, surf_rh, array_name=t_values,
        contour=significance_mask,
        contour_lwd=3, contour_lcol=(0, 0, 0), contour_lalpha=1,
        contour_col=(0, 0, 0), contour_alpha=0.2,
        contour_isovalue=0.5,
        layout_style='grid', cmap='RdBu_r', color_range='sym',
    )

``contour=None`` disables overlays. Fill is disabled by default
(``contour_alpha=0``); when enabled, it covers values above
``contour_isovalue``. Continuous finite maps are supported, as are point-data
array names present on both hemispheres. A contour at 0.5 interpolates the
boundary between binary 0/1 vertex values. The source surfaces and their
active scalars are not modified.

Lower-level layouts use a dictionary from surface keys to per-vertex values
or point-data array names:

.. code-block:: python

    from brainspace.plotting import build_plotter, add_contours_to_plotter

    plotter = build_plotter(
        {'lh': surf_lh, 'rh': surf_rh},
        [['lh', 'rh'], ['rh', 'lh']],
        array_name='t_values',
        view=[['dorsal', 'ventral'], ['medial', 'lateral']],
    )
    actors = add_contours_to_plotter(
        plotter, {'lh': mask_lh, 'rh': mask_rh}, lwd=3, alpha=0.2,
    )

``build_plotter`` also accepts the ``contour`` and ``contour_*`` keywords
directly. Overlays inherit the base actor's complete transformation in each
surface entry, so view rotations, custom actor transforms, repeated maps,
color bars, labels and empty layout entries are handled without renderer
index assumptions. For custom VTK pipelines, ``make_contour_actor`` returns a
standalone line actor in the input surface's coordinates.

Contour overlays currently require the VTK backend. The HTML report backend
raises an error when contour options are requested; see
:doc:`html_surface_reports` for its supported features.
