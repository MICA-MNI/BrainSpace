function [surf_lh,surf_rh] = load_conte69(type)

if ~exist('type','var')
    type = 'surfaces';
end

P = mfilename('fullpath');
brainspace_path = regexp(P,'.*BrainSpace','match','once');
surface_path = [brainspace_path filesep 'shared' filesep 'surfaces' filesep]; 

if strcmp(type,'spheres')
    surf_lh = convert_surface([surface_path 'conte69_32k_left_sphere.gii']);
    surf_rh = convert_surface([surface_path 'conte69_32k_right_sphere.gii']);
elseif strcmp(type,'surfaces')
    surf_lh = convert_surface([surface_path 'conte69_32k_left_hemisphere.gii']);
    surf_rh = convert_surface([surface_path 'conte69_32k_right_hemisphere.gii']);
elseif strcmp(type,'5k_surfaces')
    surf_lh = convert_surface([surface_path 'conte69_5k_left_hemisphere.gii']);
    surf_rh = convert_surface([surface_path 'conte69_5k_right_hemisphere.gii']);
else
    error('Unknown type. Valid arguments are ''spheres'' and ''surfaces''.');
end


    