function [surf_lh,surf_rh] = load_conte69()

P = mfilename('fullpath');
brainspace_path = regexp(P,'.*BrainSpace','match','once');
surface_path = [brainspace_path filesep 'shared' filesep 'surfaces' filesep]; 
surf_lh = convert_surface([surface_path 'conte69_32k_left_hemisphere.gii']);
surf_rh = convert_surface([surface_path 'conte69_32k_right_hemisphere.gii']);



    