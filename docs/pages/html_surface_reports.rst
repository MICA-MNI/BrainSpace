.. _html_surface_reports:

HTML surface reports
====================

BrainSpace's existing plotting functions use VTK and remain the default path
for local interactive rendering, screenshots, and advanced VTK-based
visualization. For remote notebooks, containers, CI runners, and other
headless environments, BrainSpace also provides display-independent HTML
surface report writers that do not require a live VTK render window.

This path is intended for report export and result inspection: write the
surface geometry and scalar maps once, then open the resulting HTML file in a
browser.


Use ``plot_hemispheres`` with the HTML backend
----------------------------------------------

The smallest change for existing code is to keep using
:func:`brainspace.plotting.plot_hemispheres` and select the ``html`` backend:

.. code-block:: python

    from brainspace.datasets import load_conte69
    from brainspace.plotting import plot_hemispheres

    surf_lh, surf_rh = load_conte69()

    report = plot_hemispheres(
        surf_lh,
        surf_rh,
        array_name=my_surface_values,
        backend="html",
        filename="surface_report.html",
        title="My surface map",
        cmap="viridis",
        color_bar=True,
    )

``backend="vtk"`` remains the default. The HTML backend writes a browser
report to ``filename`` and returns the output path. It is useful when the goal
is to export a portable report rather than create a VTK window or screenshot.


Viewer controls and scope
-------------------------

The native report is a self-contained WebGL viewer with embedded geometry,
scalar maps and colormaps. It needs no network access or Python process after
export. Use a browser with WebGL enabled. Drag to rotate, shift-drag or
right-drag to pan, scroll to zoom, and use the map selector (or ``[`` / ``]``)
to switch maps. ``R`` resets the camera and ``H`` opens help.

The native viewer displays two hemispheres side by side with a shared color
bar. It does not reproduce VTK row/grid layouts, text labels, contour overlays,
screenshots or arbitrary actor/renderer options. The ``plot_hemispheres`` HTML
path rejects these unsupported requests. VTK-specific window size, camera
zoom, interaction and background settings do not configure the responsive
HTML viewer; use the controls in the exported report.


Write reports directly
----------------------

For scripts and reporting pipelines, call
:func:`brainspace.plotting.write_surface_report` directly:

.. code-block:: python

    from brainspace.plotting import write_surface_report

    report = write_surface_report(
        surfaces={"lh": surf_lh, "rh": surf_rh},
        arrays={
            "Gradient 1": gradient_1,
            "Gradient 2": {
                "values": gradient_2,
                "color_range": "sym",
                "cmap": "viridis",
            },
        },
        filename="gradients.html",
        title="Gradient report",
        metadata={"Surface": "Conte69", "Space": "fsLR"},
    )

``surfaces`` can be a ``{"lh": surf_lh, "rh": surf_rh}`` dictionary or a
two-item left/right sequence. Map values can be a combined vector ordered as
left hemisphere followed by right hemisphere, or a dictionary with separate
``"lh"`` and ``"rh"`` arrays.


Use the threemica viewer as an optional bridge
----------------------------------------------

BrainSpace can also call Zhengchen Cai's ``threemica`` package as an optional
dependency. This preserves threemica's original HTML template, JavaScript
viewer, atlas/query payloads, controls, and output layout. BrainSpace only
prepares compatible inputs and delegates report writing.

Install ``threemica`` separately, then call:

.. code-block:: python

    from brainspace.plotting import (
        available_threemica_surface_maps,
        write_brainspace_threemica_report,
    )

    print(available_threemica_surface_maps())

    reports = write_brainspace_threemica_report(
        "brainspace_threemica_report",
        maps="all",
    )

This example uses BrainSpace's bundled Conte69/fsLR-32k reference surfaces and
maps. The default ``subject="sub-brainspaceref"`` is only a synthetic label in
the temporary BIDS-style derivative; pass a real ``sub-*`` ID when exporting
participant-specific maps.

Built-in map selections are:

``"all"``
    Include the bundled cortical marker maps and gradients.

``"markers"``
    Include curvature, cortical thickness, and T1w/T2w myelin proxy maps.

``"gradients"``
    Include the bundled functional connectivity and MPC gradients.

You can also pass explicit tags or additional arrays:

.. code-block:: python

    write_brainspace_threemica_report(
        "subject_report",
        maps=["thickness", "fc_gradient1"],
        custom_maps={
            "aligned_gradient": {
                "values": aligned_gradients,
                "label": "Aligned gradient",
                "unit": "score",
                "cmap": "diverging",
            }
        },
    )

The custom map array can be one-dimensional, or two-dimensional with one map
per component. BrainSpace writes a temporary BIDS-style derivatives tree for
threemica and removes it after report generation unless ``clean_input=False``
is passed.

The lower-level :func:`brainspace.plotting.write_threemica_report` function is
available when users already have threemica-compatible BIDS derivatives and a
scope definition.


Choosing a rendering path
-------------------------

Use the VTK backend when you need BrainSpace's full local plotting behavior,
VTK actor customization, screenshots, or notebook embedding through the
existing plotting stack.

Use HTML reports when you need portable browser inspection, documentation
outputs, remote notebook exports, or a workflow that must not depend on an X
server, Xvfb, OSMesa, or EGL.

``threemica`` is an optional external package. BrainSpace does not vendor or
modify threemica; it only provides an adapter so users can call the original
viewer from BrainSpace-oriented workflows.
