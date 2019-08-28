function parcel_data = load_parcellation(name,parcel_number)

if ~iscell(name) && ~isstring(name)
    name = {name};
end    

P = mfilename('fullpath');
brainspace_path = regexp(P,'.*BrainSpace','match','once');
parcellation_path = [brainspace_path filesep 'shared' filesep 'parcellations']; 

for ii = 1:numel(name)
    for jj = 1:numel(parcel_number)
        label = char(name{ii} + "_" + parcel_number(jj)); 
        parcel_data.(label) = load([parcellation_path filesep label '_conte69.csv']);
    end
end