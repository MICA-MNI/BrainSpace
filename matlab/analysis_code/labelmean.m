function out = labelmean(D,L,varargin)
% LABELMEAN    mean value within label
%   Averages columns with the same label. D must be a two-dimensional
%   matrix. L must be a row vector containing integers (NaNs allowed).
%   These integers are the labels of the corresponding column in D. The
%   sortbylabelvector flag will cause the output to be sorted in the same
%   order as the labels in L, rather than numeric order.
%
%   Written by Reinder Vos de Wael (May 2017)
%
%   - Updated to allow for missing labels (May 2018).  
%   - Ignore warning now also ignores the "Found 0s in label vector"
%       warning. 
%
%   For complete documentation please consult our <a
%   href="https://brainspace.readthedocs.io/en/latest/pages/matlab_doc/support_functions/labelmean.html">ReadTheDocs</a>.



% Deal with varargin
sortData=false;
ignoreWarning = false; 
for ii = 1:1:numel(varargin)
    switch lower(varargin{ii})
        case 'sortbylabelvector'
            sortData=true;
        case 'ignorewarning'
            ignoreWarning=true;
        otherwise
            error('Unknown name-value pair');
    end
end

% Display message to make user aware of function behavior. 
if ~ignoreWarning
    if sortData
        disp('Sorting data by order of appearance in the label! (labelmean.m)')
    else
        disp('Sorting data by smallest to largest number in the label! (labelmean.m)')
    end
end

% Check for correct input
if isvector(L)
    L = L(:)';
else
    error('Labels must be a row vector')
end

if size(L,2) ~= size(D,2)
    D = D';
    warning('Data and label did not have an equal number of columns, attempting after transposing D.');
    if size(L,2) ~= size(D,2)
        error('Transposing failed. Data and Labels must have an equal number of columns.');
    end
end
if ~isa(D,'double')
    D = double(D);
end
if ~isa(L,'double')
    L = double(L);
end
if any(L==0)
    if ~ignoreWarning
        warning('Found 0''s in the label vector. Replacing these with NaN.')
    end
    L(L==0) = nan;
end

% Remove NaN columns in D, and corresponding columns in L. 
toRemove = any(isnan(D),1); 
D(:,toRemove) = [];
L(:,toRemove) = [];

% Find indices corresponding to the non-NaNs
I = find(~isnan(L));

% Construct a sparse matrix that, when multiplied by D, gives the desired
% row sums. 
t = sparse(I, L(I), 1, size(D,2), max(L(I)));  

% Divide t by the sum of its columns to change the sum to a mean. 
t = t./sum(t,1); 

% Perform the multiplication. 
out = full(D*t); 

% Sort if requested
if sortData
    order = removeduplicates(L,'removeNaN');
    [~,ind] = sort(order);
    out = out(:,ind); 
end
end
