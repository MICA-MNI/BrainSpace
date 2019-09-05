.. _read_surface_matlab:

read_surface
==============================

Synopsis
---------

Reads surfaces (`source code
<https://github.com/MICA-MNI/BrainSpace/blob/master/matlab/surface_manipulation/read_surface.m>`_).


Usage 
----------
::

    surf_out = read_surface(file);

- *file*: File to read from.
- *surf_out*: Output surface.  


Description 
------------

Reads surfaces from the disk. Accepted formats are gifti files (provided the
gifti library is installed), freesurfer files, .mat files and .obj files. 

When provided with a .mat file, the file must contain variables corresponding to
a MATLAB surface i.e. 'vertices' and 'faces' or a SurfStat surface i.e. 'coord
and 'tri'. 'vertices' is an n-by-3 list of vertex coordinates, 'faces' is an
m-by-3 matrix of triangles, 'coord', is a 3-by-n list of vertex coordinates, and
'tri' is identical to faces in the MATLAB format. 

