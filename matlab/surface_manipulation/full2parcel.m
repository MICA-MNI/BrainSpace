function parcel_data = full2parcel(data,parcellation)
% Converts full data to parcellated data by meaning within each parcel.
% Data is sorted by ascending order of the parcellation numbers. If some
% numbers are missing, these are returned as NaN columns. 
parcel_data = labelmean(data,parcellation,'ignorewarning')';
end