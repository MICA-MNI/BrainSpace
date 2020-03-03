function parcel_data = full2parcel(data,parcellation)
% FULL2PARCEL   Downsamples vertex data to parcel data.
%
%   parcel_Data = full2parcel(data,parcellation) takes the mean of columns
%   in n-by-m matrix data that have the same value in corresponding 1-by-m
%   vector parcellation. If some numbers are missing in the parcellation
%   vector, these are returned as NaN columns.
%
%   For more information, please consult our <a
%   href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/main_functionality/full2parcel.html">ReadTheDocs</a>.

parcel_data = labelmean(data,parcellation,'ignorewarning')';
end
