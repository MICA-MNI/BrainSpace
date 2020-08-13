function parcel_data = load_parcellation(name,parcel_number)
% LOAD_PARCELLATION   loads parcellation vectors.
%
%   parcel_data = LOAD_PARCELLATION(name) loads parcellations on
%   conte69-32k surfaces. Name can be set to 'vosdewael' for a
%   subparcellation of the Desikan-Killiany atlas, or 'schaefer' for a
%   functional parcellation; both may also be provided as a cell/string
%   array. Parcel_number denotes the resolution of the parcellation. It is
%   a vector containing any of the following values [100,200,300,400].
%   parcel_data is a structure array containing all the requested
%   parcellations. 
%
%   For more information, please consult our <a
%   href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/data_loaders/load_parcellation.html">ReadTheDocs</a>.

if ~iscell(name) && ~isstring(name)
    name = {name};
end    

P = mfilename('fullpath');
brainspace_path = fileparts(fileparts(fileparts(P)));
parcellation_path = [brainspace_path filesep 'shared' filesep 'parcellations']; 

for ii = 1:numel(name)
    for jj = 1:numel(parcel_number)
        label = char(name{ii} + "_" + parcel_number(jj)); 
        parcel_data.(label) = load([parcellation_path filesep label '_conte69.csv']);
    end
end
