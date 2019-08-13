.. _convert_surface:

convert_surface
==============================

Synopsis
---------

Loads surfaces and converts them to either MATLAB or SurfStat format. 

Usage 
----------
::

    convert_surface(file);
    convert_surface(surface_struct,format)

- *file*: path to a gifti, freesurfer, or .obj file. 
- *surface_struct*: a surface stored either in MATLAB format or SurfStat format.
- *format*: the format to convert to either SurfStat (default) or MATLAB.


Description 
------------
This function is BrainSpace's surface loader. When provided with the path to a file it'll load gifti files (provided the gifti library is installed), freesurfer files, and .obj files. It can also be provided with a structure variable containnig a surface in either MATLAB format or SurfStat format. 

A MATLAB surface is a structure consisting of two fields: 'vertices', which is a n-by-3 list of vertex coordinates, and 'faces', which is a m-by-3 matrix of triangles. A SurfStat format consists of two fields: 'coord', which is a 3-by-n list of vertex coordinates, and 'tri', which is identical to faces in the MATLAB format. 

