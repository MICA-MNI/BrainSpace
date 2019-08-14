function [mask_lh mask_rh] = load_mask()

P = mfilename('fullpath');
brainspace_path = regexp(P,'.*BrainSpace','match','once');
surface_path = [brainspace_path filesep 'shared' filesep 'surfaces' filesep]; 

mask_lh = load([surface_path filesep 'conte69_32k_lh_mask.csv']);
mask_rh = load([surface_path filesep 'conte69_32k_rh_mask.csv']);