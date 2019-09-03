.. _convert_surface_matlab:

convert_surface
==============================

Synopsis
---------

Loads, converts, and writes surfaces (`source code
<https://github.com/MICA-MNI/BrainSpace/blob/master/matlab/surface_manipulation/convert_surface.m>`_).


Usage 
----------
::

    S = convert_surface(file);
    S = convert_surface(surface_struct,varargin);

- *file*: path to a gifti, freesurfer, or .obj file. 
- *surface_struct*: a surface stored either in MATLAB format or SurfStat format.
- *varargin*: see Name-Value pairs below.
- *S*: Output surface.


Description 
------------

This function is BrainSpace's surface loader/writer. When provided with the path
to a surface file it will load gifti files (provided the gifti library is
installed), freesurfer files, .mat files and .obj files. It can also be provided
with a structure variable containing a surface in either MATLAB format or
SurfStat format. When provided with a 'path' name-value pair, the input surface
will also be written to the disk.

A MATLAB surface is a structure consisting of two fields: 'vertices', which is
an n-by-3 list of vertex coordinates, and 'faces', which is an m-by-3 matrix of
triangles. A SurfStat format consists of two fields: 'coord', which is a 3-by-n
list of vertex coordinates, and 'tri', which is identical to faces in the MATLAB
format. 

Name-Value pairs
^^^^^^^^^^^^^^^^^^^
- *format*: the format to convert to; either SurfStat (default) or MATLAB
- *path*: a path to write the surface to (Default: ''). If used, only one surface can be provided in S. 
