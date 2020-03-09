function surf_out = read_surface(file)
% READ_SURFACE   Reads surface from disk. 
%
%   surf_out = READ_SURFACE(file) reads the surface file and outputs it in
%   MATLAB format. The file can be (.gii, .mat, .obj, Freesurfer), a loaded
%   variable (in SurfStat or MATLAB format), or a cell array containing
%   multiple of the former.
%
%   For complete documentation, please consult our <a
%   href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/main_functionality/read_surface.html">ReadTheDocs</a>.

surf_out = convert_surface(file,'format','MATLAB');

