function [surf_lh,surf_rh] = load_conte69(name)

if ~exist('name','var')
    name = 'surfaces';
end

P = mfilename('fullpath');
brainspace_path = regexp(P,'.*BrainSpace','match','once');
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


    