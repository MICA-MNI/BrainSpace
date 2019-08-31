function [mask_lh, mask_rh] = load_mask(type)

if nargin < 1
    type = 'midline';
end

P = mfilename('fullpath');
brainspace_path = regexp(P,'.*BrainSpace','match','once');
surface_path = [brainspace_path filesep 'shared' filesep 'surfaces' filesep]; 

mask_lh = logical(load([surface_path filesep 'conte69_32k_lh_' type '_mask.csv']));
mask_rh = logical(load([surface_path filesep 'conte69_32k_rh_' type '_mask.csv']));
