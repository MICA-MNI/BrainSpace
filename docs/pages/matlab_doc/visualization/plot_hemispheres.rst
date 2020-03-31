.. _plot_hemispheres_matlab:

==================
plot_hemispheres
==================

------------------
Synopsis
------------------

Plots data on the cortical surface (`source code
<https://github.com/MICA-MNI/BrainSpace/blob/master/matlab/plot_data/%40plot_hemispheres/plot_hemispheres.m>`_).


------------------
Usage
------------------

::

   obj = plot_hemispheres(data,surface,varargin);

- *data*: an n-by-m vector of data to plot where n is the number of vertices or parcels, and m the number of markers to plot.
- *surface*: a surface readable by :ref:`convert_surface_matlab` or a two-element cell array containing left and right hemispheric surfaces in that order. 
- *varargin*: Name-Value Pairs (see below).
- *obj*: an object allowing for further modificatio of the figure (see below). 

------------------
Description
------------------

Plots any data vector onto cortical surfaces. Data is always provided as a
single vector; if two surfaces are provided then the *n* vertices of the first
surface will be assigned datapoints 1:*n* and the second surface is assigned the
remainder. If a parcellation scheme is provided, data should have as many
datapoints as there are parcels.  

BrainSpace only provides basic figure building functionality. For more
information on how to use MATLAB to create publication-ready figures we
recommend delving into `graphics object properties
<https://www.mathworks.com/help/matlab/graphics-object-properties.html>`_ (e.g.
`figure
<https://www.mathworks.com/help/matlab/ref/matlab.ui.figure-properties.html>`_,
`axes
<https://www.mathworks.com/help/matlab/ref/matlab.graphics.axis.axes-properties.html>`_,
`surface
<https://www.mathworks.com/help/matlab/ref/matlab.graphics.primitive.surface-properties.html>`_).
Also see the `source code
<https://github.com/MICA-MNI/BrainSpace/blob/master/matlab/plot_data/plot_hemispheres.m>`_
for basic graphic object property modifications.

Name-Value Pairs
^^^^^^^^^^^^^^^^^
- *parcellation*: an k-by-1 vector containing the parcellation scheme, where k is the number of vertices. Default [].
- *labeltext*: A cell array of m elements containing labels for each column of data. These will be printed next to the hemispheres. Default: [].
- *views*: A character vector containing the requested view angles. Options are: l(ateral), m(edial), i(nferior), s(uperior), a(nterior), and p(osterior). Default: 'lm'.

Public methods
^^^^^^^^^^^^^^^
Public methods can be used with obj.(method) e.g. obj.colorlimits to use the colorlimits method.

- *colorlimits(limits)*: Sets the color limits of each row. Limits must be a 2-element numeric vector or n-by-2 matrix where n is the number of columns in obj.data. The first column of limits denotes the minimum color limit and the second the maximum. When limits is a 2-element vector, then the limits are applied to all axes, with limits(1) as minimum and limits(2) as maximum. 
- *colorMaps(maps)*: Maps must either be an n-by-3 color map, or a cell array with the same number of elements as columns in obj.data, each containing n-by-3 colormaps.
- *labels(varargin)*: Modifies the properties of the labeltext. Varargin are valid name-value pairs for MATLAB's text function. e.g. `obj.labels('FontSize',25)`
