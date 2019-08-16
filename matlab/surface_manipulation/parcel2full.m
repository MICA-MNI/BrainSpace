function full_data = parcel2full(parcellated_data,parcellation)
% Converts parcellated data to full data. Any missing data in the full set
% (e.g. midline) should be denoted as either 0 or NaN. 

if iscell(parcellated_data)
    for ii = 1:numel(parcellated_data)
        full_data{ii} = parcel2full(parcellated_data{ii},parcellation);
    end
    return
end

% Check for correct size of parcellation.
sz = size(parcellated_data,1); 
if max(parcellation) ~= sz
    error('Parcellation number must be eqiuvalent to length of parcellated data.');
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
