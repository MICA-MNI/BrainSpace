.. _combine_surfaces_matlab:

combine_surfaces
==============================

Synopsis
---------

Combines two surfaces into a single surface (`source code
<https://github.com/MICA-MNI/BrainSpace/blob/master/matlab/surface_manipulation/combine_surfaces.m>`_).


Usage 
----------
::

    SC = combine_surfaces(S1,S2);
    SC = combine_surfaces(S1,S2,format);

- *S1*, *S2*: two surfaces
- *format*: the output format; either 'SurfStat' (default) or 'MATLAB'.
- *S*: Combined surface.


Description 
------------
This can be used to merge left and right hemisphere surfaces into a single
surface. The input surfaces can be any file readable by
:ref:`read_surface_matlab` or a surface loaded into memory. 

A MATLAB structure consists of two fields: vertices, which is a n-by-3 list of
vertex coordinates, and faces, which is a m-by-3 matrix of triangles. A SurfStat
format consists of two fields: coord, which is a 3-by-n list of vertex
coordinates, and tri, which is identical to faces. 

