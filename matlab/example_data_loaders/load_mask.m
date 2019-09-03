function [mask_lh, mask_rh] = load_mask(name)

if nargin < 1
    name = 'midline';
end

P = mfilename('fullpath');
brainspace_path = regexp(P,'.*BrainSpace','match','once');
surface_path = [brainspace_path filesep 'shared' filesep 'surfaces' filesep]; 

mask_lh = logical(load([surface_path filesep 'conte69_32k_lh_' name '_mask.csv']));
mask_rh = logical(load([surface_path filesep 'conte69_32k_rh_' name '_mask.csv']));
