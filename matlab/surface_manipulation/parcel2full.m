function full_data = parcel2full(parcellated_data,parcellation)
% PARCEL2FULL   Upsamples parcel data to vertex data.
%
%   full_data = parcel2full(parcellated_data,parcellation) takes n-by-1
%   vector parcellated_data and creates a new m-by-1 vector full_data,
%   where each value in full data corresponds to the parcellated data as
%   defined by the index in the m-by-1 parcellation vector i.e.
%   full_data(ii) = parcellated_data(parcellation(ii)).
%
%   Any 0s or NaNs in the parcellation are set to NaN in the full_data. 
%
%   For more information, please consult our <a
%   href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/main_functionality/parcel2full.html">ReadTheDocs</a>.


if iscell(parcellated_data)
    for ii = 1:numel(parcellated_data)
        full_data{ii} = parcel2full(parcellated_data{ii},parcellation);
    end
    return
end

% Check for correct size of parcellation.
sz = size(parcellated_data,1); 
if max(parcellation) ~= sz
    error('The highest number in the parcellation scheme must be equivalent to length of the first dimension of the parcellated data.');
end

% Check if all numbers are included. 
if ~all(ismember(1:sz,parcellation))
    warning('Some parcel numbers are missing, some data may not be included in the output.')
end

% Make sure 0's and NaNs are both correctly read as "missing data".
parcellation(parcellation == 0) = sz+1;
parcellation(isnan(parcellation)) = sz+1; 
parcellated_data(end+1,:) = nan; 

% Convert parcellated to full. 
full_data = parcellated_data(parcellation,:); 
end
