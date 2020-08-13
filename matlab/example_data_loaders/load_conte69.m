function [surf_lh,surf_rh] = load_conte69(name)
% LOAD_CONTE69   loads conte69 surfaces.
%
%   [surf_lh,surf_rh] = LOAD_CONTE69(name) loads conte69-32k surfaces of
%   the left (surf_lh) and right (surf_rh) hemispheres. Name can be set to
%   'surfaces' for cortical surface or 'spheres' for corresponding spheres.
%
%   For more information, please consult our <a
%   href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/data_loaders/load_conte69.html">ReadTheDocs</a>.

if ~exist('name','var')
    name = 'surfaces';
end

P = mfilename('fullpath');
brainspace_path = fileparts(fileparts(fileparts(P)));
surface_path = [brainspace_path filesep 'shared' filesep 'surfaces' filesep]; 

if strcmpi(name,'spheres')
    surf_lh = convert_surface([surface_path 'conte69_32k_left_sphere.gii']);
    surf_rh = convert_surface([surface_path 'conte69_32k_right_sphere.gii']);
elseif strcmpi(name,'surfaces')
    surf_lh = convert_surface([surface_path 'conte69_32k_left_hemisphere.gii']);
    surf_rh = convert_surface([surface_path 'conte69_32k_right_hemisphere.gii']);
else
    error('Unknown surface requested.')
end


    
