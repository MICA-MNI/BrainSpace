function conn_matrices = load_group_fc(name,parcel_number,group)
% LOAD_GROUP_FC   loads group level connectivity matrices.
%
%   conn_matrices = LOAD_GROUP_FC(name,parcel_number,group) loads sample
%   group level connectivity matrices of the HCP dataset. Name can be set
%   to 'vosdewael' for a subparcellation of the Desikan-Killiany atlas, or
%   'schaefer' for a functional parcellation; both may also be provided as
%   a cell/string array. Parcel_number denotes the resolution of the
%   parcellation. It is a vector containing any of the following values
%   [100,200,300,400]. Group is either 'main' (default) or 'holdout'. Data
%   from different subjects is loaded depending on the choice.
%   conn_matrices is a structure array containing all the requested
%   connectivity matrices. 
%
%   For more information, please consult our <a
%   href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/data_loaders/load_group_fc.html">ReadTheDocs</a>.

if nargin < 3
    group = 'main';
end

if ~iscell(name) && ~isstring(name)
    name = {name};
end    

P = mfilename('fullpath');
brainspace_path = fileparts(fileparts(fileparts(P)));
data_path = [brainspace_path filesep 'shared' filesep 'data' filesep group '_group']; 

for ii = 1:numel(name)
    for jj = 1:numel(parcel_number)
        label = char(name{ii} + "_" + parcel_number(jj)); 
        conn_matrices.(label) = load([data_path filesep label '_mean_connectivity_matrix.csv']);
    end
end
