function [mask_lh, mask_rh] = load_mask(name)
% LOAD_MASK   loads cortical masks.
%
%   [mask_lh,mask_rh] = LOAD_MASK(name) loads masks on the conte69-32k surfaces of
%   the left (mask_lh) and right (mask_rh) hemispheres. Name can be set to
%   'midline' for a midline mask and 'temporal' for a temporal mask.
%
%   For more information, please consult our <a
%   href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/data_loaders/load_mask.html">ReadTheDocs</a>.

if nargin < 1
    name = 'midline';
end

P = mfilename('fullpath');
brainspace_path = fileparts(fileparts(fileparts(P)));
surface_path = [brainspace_path filesep 'shared' filesep 'surfaces' filesep]; 

mask_lh = logical(load([surface_path filesep 'conte69_32k_lh_' name '_mask.csv']));
mask_rh = logical(load([surface_path filesep 'conte69_32k_rh_' name '_mask.csv']));
