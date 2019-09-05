.. _write_surface_matlab:

write_surface
==============================

Synopsis
---------

Writes surfaces (`source code
<https://github.com/MICA-MNI/BrainSpace/blob/master/matlab/surface_manipulation/write_surface.m>`_).


Usage 
----------
::

    write_surface(surface,path);

- *surface*: surface loaded into MATLABB
- *path*: path to write to 


Description 
------------

Writes surfaces to the designated output path. Accepted formats are .gii files
(provided the gifti library is installed), freesurfer files, .mat files and .obj
files. File type is determined by the extension; if the filetype is not
recognized the default is freesurfer format.

When saving a .mat file, the first will save the fields inside the surface
structure. 
