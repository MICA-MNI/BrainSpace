.. _split_surfaces_matlab:

====================
split_surfaces
====================

------------------
Synopsis
------------------

Splits surfaces in a single variable into separate variables. (`source code
<https://github.com/MICA-MNI/BrainSpace/blob/master/matlab/surface_manipulation/split_surfaces.m>`_).

------------------
Usage
------------------

::

    varargout = split_surfaces(surface);

- *surface*: surface loaded into MATLABB
- *varargout*: Variables to load the surfaces into. 

------------------
Description 
------------------

If you have a variable containing multiple disconnected surfaces, this function
will split them up and return each in a separate output variable. Accepts all surfaces
readable by :ref:`convert_surface_matlab`. 

