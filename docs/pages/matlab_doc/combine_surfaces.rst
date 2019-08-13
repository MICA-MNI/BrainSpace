.. _combine_surfaces:

combine_surfaces
==============================

Synopsis
---------

Combines two surfaces into a single surface. 

Usage 
----------
::

    combine_surfaces(S1,S2,format);

- *S1*, *S2*: two surfaces
- *format*: the output format; either 'SurfStat' (default) or 'MATLAB'.


Description 
------------
This can be used to merge left and right hemisphere surfaces into a single surface. The input surfaces can be any surface readable by ``convert_surface``. 

A MATLAB structure consists of two fields: vertices, which is a n-by-3 list of vertex coordinates, and faces, which is a m-by-3 matrix of triangles. A SurfStat format consists of two fields: coord, which is a 3-by-n list of vertex coordinates, and tri, which is identical to faces. 

