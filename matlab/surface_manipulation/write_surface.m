function write_surface(surface,file)
% WRITE_SURFACE   Reads surface from disk. 
%
%   surf_out = WRITE_SURFACE(surface,path) writes the surface to a file.
%   Accepted file formats are .gii, .mat, .obj. Any file that does not end
%   in one of those extensions will be encoded as a Freesurfer file.
%
%   For complete documentation, please consult our <a
%   href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/main_functionality/write_surface.html">ReadTheDocs</a>.

convert_surface(surface,'path',file); 
