function conn_matrices = load_group_fc(name,parcel_number,group)

if nargin < 3
    group = 'main';
end

if ~iscell(name) || ~isstring(name)
    name = {name};
end    

P = mfilename('fullpath');
brainspace_path = regexp(P,'.*BrainSpace','match','once');
data_path = [brainspace_path filesep 'shared' filesep 'data' filesep group '_group']; 

for ii = 1:numel(name)
    for jj = 1:numel(parcel_number)
        label = char(name{ii} + "_" + parcel_number(jj)); 
        conn_matrices.(label) = load([data_path filesep label '_mean_connectivity_matrix.csv']);
    end
end