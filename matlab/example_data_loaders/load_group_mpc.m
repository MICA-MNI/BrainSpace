function conn_matrices = load_group_mpc(name,parcel_number)
% LOAD_GROUP_MPC   loads group level microstructural profile covariance matrices.
%
%   conn_matrices = LOAD_GROUP_MPC(name, parcel_number) loads sample
%   microstructural profile covariance matrices of the HCP dataset. Name
%   can be set to 'vosdewael' for a subparcellation of the Desikan-Killiany
%   atlas. Parcel_number denotes the resolution of the parcellation. It
%   must be 200. conn_matrices is a structure array containing all the
%   requested connectivity matrices.
%
%   For more information, please consult our <a
%   href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/data_loaders/load_group_mpc.html">ReadTheDocs</a>.

if ~iscell(name) && ~isstring(name)
    name = {name};
end    

P = mfilename('fullpath');
brainspace_path = fileparts(fileparts(P));
data_path = [brainspace_path filesep 'datasets' filesep 'data' filesep ...
            'fusion_tutorial' filesep]; 

for ii = 1:numel(name)
    for jj = 1:numel(parcel_number)
        label = char(name{ii} + "_" + parcel_number(jj)); 
        conn_matrices.(label) = load([data_path filesep label '_mpc_matrix.csv']);
    end
end